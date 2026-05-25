# Copyright (C) 2020, Xilinx, Inc.
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Transformation to create Zynq Vivado projects for FINN dataflow designs."""
import json
import math
import os
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.transformation.infer_data_layouts import InferDataLayouts
from shutil import copy
from subprocess import CalledProcessError

from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.instrumentation import GenerateInstrumentationIP
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import (
    launch_process_helper,
    make_build_dir,
    pynq_native_port_width,
    pynq_part_map,
)
from finn.util.exception import FINNError, FINNSynthesisError
from finn.util.settings import get_settings

from . import templates


def collect_ip_dirs(model, ipstitch_path):
    """Collect list of all IP directories required by the design."""
    ip_dirs = []
    need_memstreamer = False
    for node in model.graph.node:
        node_inst = getCustomOp(node)
        if node.op_type == "NodeContainer":
            if node_inst.get_nodeattr("multi_dnn_type") == "partial_reconfiguration":
                for id in range(node_inst.get_nodeattr("bodies")):
                    body_model = node_inst.get_nodeattr("body_" + str(id))
                    a = collect_ip_dirs(body_model, None)
                    ip_dirs += a
            else:
                code_gen_dir = node_inst.get_nodeattr("code_gen_dir_ipgen")
                if code_gen_dir and os.path.isdir(code_gen_dir):
                    ip_dirs.append(code_gen_dir)
                ip_dir_value = node_inst.get_nodeattr("ip_path")
                assert os.path.isdir(
                    ip_dir_value
                ), """The directory that should
                contain the generated ip blocks doesn't exist."""
                ip_dirs += [ip_dir_value]
        else:
            ip_dir_value = node_inst.get_nodeattr("ip_path")
            assert os.path.isdir(
                ip_dir_value
            ), """The directory that should
            contain the generated ip blocks doesn't exist."""
            ip_dirs += [ip_dir_value]
        if node.op_type.startswith("MVAU") or node.op_type == "Thresholding_hls":
            if node_inst.get_nodeattr("mem_mode") == "internal_decoupled":
                need_memstreamer = True
    ip_dirs += [ipstitch_path + "/ip"] if ipstitch_path else []
    if need_memstreamer:
        # add RTL streamer IP
        ip_dirs.append("$::env(FINN_RTLLIB)/memstream")
    return ip_dirs


class MakeZYNQProject(Transformation):
    """Create a Vivado overlay project (including the shell infrastructure)
    from the already-stitched IP block for this graph.
    All nodes in the graph must have the fpgadataflow backend attribute,
    and the CreateStitchedIP transformation must have been previously run on
    the graph. This is functionally equivalent with MakePYNQProject but does
    not use Pynq infrastructure and instead creates a fully custom block design.
    However, this transform requires DMAs in the accelerator design.

    Outcome if successful: sets the vivado_pynq_proj attribute in the ONNX
    ModelProto's metadata_props field, with the created project dir as the
    value.
    """

    def __init__(
        self,
        platform,
        period_ns,
        enable_debug=False,
        enable_finn_switch=False,
        live_fifo_sizing=False,
    ):
        """Initialize MakeZYNQProject with the target platform and clock period."""
        super().__init__()
        self.platform = platform
        self.period_ns = period_ns
        self.enable_finn_switch = enable_finn_switch
        self.live_fifo_sizing = live_fifo_sizing
        self.enable_debug = 1 if enable_debug else 0
        self.enable_gpio_reset = 0
        self.enable_selectable = 0

    def apply(self, model):
        """Apply the transformation to create a Zynq project."""
        config = []
        idma_idx = 0
        odma_idx = 0
        aximm_idx = 0
        nested_interconnect_count = 0
        master_axilite_idx = 0
        axilite_interconnect_idx = 0
        axilite_idx = 0
        instance_names = {}

        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        partial_reconfiguration = False
        for sdp_node in sdp_nodes:
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            if any(
                n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
                for n in kernel_model.graph.node
            ):
                partial_reconfiguration = True
                # Copy body_0 metadata to the SDP node
                pr_node = kernel_model.get_nodes_by_op_type("NodeContainer")[
                    0
                ]  # We can assume that we have only one NodeContainer
                pr_node_inst = getCustomOp(pr_node)
                body_model = pr_node_inst.get_nodeattr("body_0")
                kernel_model.set_metadata_prop(
                    "vivado_stitch_proj", body_model.get_metadata_prop("vivado_stitch_proj")
                )
                kernel_model.set_metadata_prop(
                    "wrapper_filename", body_model.get_metadata_prop("wrapper_filename")
                )
                kernel_model.set_metadata_prop(
                    "vivado_stitch_vlnv", body_model.get_metadata_prop("vivado_stitch_vlnv")
                )
                kernel_model.set_metadata_prop(
                    "vivado_stitch_ifnames", body_model.get_metadata_prop("vivado_stitch_ifnames")
                )
                kernel_model.save(dataflow_model_filename)

        sw_nodes = [
            getCustomOp(n)
            for sdp in sdp_nodes
            for n in ModelWrapper(getCustomOp(sdp).get_nodeattr("model")).graph.node
            if n.op_type == "NodeContainer"
            and getCustomOp(n).get_nodeattr("multi_dnn_type") == "selectable_weights"
        ]
        if sw_nodes:
            self.enable_selectable = True
        max_bodies = max((n.get_nodeattr("bodies") for n in sw_nodes), default=0)

        # instantiate instrumentation IP if it was generated
        instr_ip_dir = model.get_metadata_prop("instrumentation_ipgen")

        if self.enable_selectable:
            for import_name in [
                "/selector/selector_verilog.v",
                "/selector/selector.sv",
                "/axi/hdl/axilite.sv",
            ]:
                module_dir = os.environ["FINN_RTLLIB"] + import_name
                config.append(
                    "add_files -copy_to [get_property DIRECTORY [current_project]] -norecurse %s"
                    % module_dir
                )

        if self.enable_finn_switch:
            # TODO: Add ‑copy_to
            module_dir = os.path.join(get_settings().finn_rtllib, "finn_switch", "hdl", "switch.v")
            config.append(
                "add_files -copy_to [get_property DIRECTORY [current_project]] -norecurse %s"
                % module_dir
            )
            config.append("create_bd_cell -type module -reference finn_switch finn_switch")

        use_instrumentation = instr_ip_dir is not None and os.path.isdir(instr_ip_dir)
        if use_instrumentation:
            # instantiate GPIO IP to trigger reset
            self.enable_gpio_reset = 1
            # in the template this will connect to first port of interconnect_0
            master_axilite_idx += 1

            # update IP repository
            config.append(
                "set_property ip_repo_paths "
                "[concat [get_property ip_repo_paths [current_project]] [list %s]] "
                "[current_project]" % instr_ip_dir
            )
            config.append("update_ip_catalog -rebuild -scan_changes")
            # create instance
            config.append(
                "create_bd_cell -type ip -vlnv %s %s"
                % ("xilinx.com:hls:instrumentation_wrapper:1.0", "instrumentation_wrap_0")
            )
            # connect clock % reset
            config.append(
                "connect_bd_net [get_bd_pins instrumentation_wrap_0/ap_clk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            config.append(
                "connect_bd_net [get_bd_pins instrumentation_wrap_0/ap_rst_n] "
                "[get_bd_pins smartconnect_0/aresetn]"
            )
            # connect AXI-lite control interface
            config.append(
                "connect_bd_intf_net [get_bd_intf_pins instrumentation_wrap_0/s_axi_ctrl] "
                "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]" % (master_axilite_idx)
            )
            config.append("assign_axi_addr_proc instrumentation_wrap_0/s_axi_ctrl")
            master_axilite_idx += 1

        if self.live_fifo_sizing:
            # instantiate virtual FIFO controller
            rtl_path = get_settings().finn_rtllib
            files = [
                os.path.join(rtl_path, "axi/hdl/axilite.sv"),
                os.path.join(rtl_path, "fifo_virtual/hdl/fifo_gauge_pkg.sv"),
                os.path.join(rtl_path, "fifo_virtual/hdl/fifo_controller.sv"),
                os.path.join(rtl_path, "fifo_virtual/hdl/fifo_controller_wrapper.v"),
            ]
            for f in files:
                config.append(f"add_files -norecurse {f}")
            config.append(
                "create_bd_cell -type module -reference fifo_controller_wrapper fifo_controller_0"
            )

            # connect clock & reset
            config.append(
                "connect_bd_net [get_bd_pins fifo_controller_0/ap_clk] "
                "[get_bd_pins smartconnect_0/aclk]"
            )
            config.append(
                "connect_bd_net [get_bd_pins fifo_controller_0/ap_rst_n] "
                "[get_bd_pins smartconnect_0/aresetn]"
            )

            # connect AXI-lite control interface
            config.append(
                "connect_bd_intf_net [get_bd_intf_pins fifo_controller_0/s_axi] "
                "[get_bd_intf_pins axi_interconnect_0/M%02d_AXI]" % (master_axilite_idx)
            )
            # Do not use assign_axi_addr_proc here. It doesn't map the 32-bit aperture correctly.
            # Instead, let assign_bd_address command assign the address later.
            # TODO: Support 32-bit systems by making aperture smaller?
            # config.append("assign_axi_addr_proc fifo_controller_0/s_axi")
            master_axilite_idx += 1

        # instantiate nested AXI interconnects if required
        # only the nested interconnects and all interfaces connected before this line
        # will be connected to the original (master) interconnect
        total_axilite_count = 0
        for node in model.graph.node:
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))
            total_axilite_count += len(ifnames["axilite"])
        if total_axilite_count > (64 - master_axilite_idx):
            nested_interconnect_count = math.ceil(total_axilite_count / 64.0)
            for i in range(1, nested_interconnect_count + 1):
                # create instance
                config.append(
                    "create_bd_cell -type ip -vlnv $interconnect_vlnv axi_interconnect_%d" % (i)
                )
                # configure instance
                config.append(
                    "set_property -dict [list CONFIG.NUM_MI %d] [get_bd_cells axi_interconnect_%d]"
                    % (min(64, total_axilite_count), i)
                )
                # connect to master interconnect
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M%02d_AXI] "
                    "-boundary_type upper [get_bd_intf_pins axi_interconnect_%d/S00_AXI]"
                    % (master_axilite_idx, i)
                )
                # connect clocks/reset
                config.append(
                    "apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config "
                    '"Clk /zynq_ps/$zynq_ps_clkname" [get_bd_pins axi_interconnect_%d/ACLK]' % (i)
                )
                master_axilite_idx += 1
                total_axilite_count = max(0, total_axilite_count - 64)

            assert total_axilite_count == 0, "Not all AXI-lite interfaces connected!"

            # start populating the first nested interconnect
            axilite_interconnect_idx = 1
        else:
            axilite_idx = master_axilite_idx

        num_sdps = len(model.graph.node)
        prev_node_name = None
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            sdp_id = int(node.name.split("_")[-1])

            ipstitch_path = kernel_model.get_metadata_prop("vivado_stitch_proj")
            if ipstitch_path is None or (not os.path.isdir(ipstitch_path)):
                raise Exception(
                    "No stitched IPI design found for %s, apply CreateStitchedIP first." % node.name
                )

            vivado_stitch_vlnv = kernel_model.get_metadata_prop("vivado_stitch_vlnv")
            if vivado_stitch_vlnv is None:
                raise Exception("No vlnv found for %s, apply CreateStitchedIP first." % node.name)

            ip_dirs = ["list"]
            ip_dirs += collect_ip_dirs(kernel_model, ipstitch_path)
            ip_dirs_str = "[%s]" % (" ".join(ip_dirs))
            config.append(
                "set_property ip_repo_paths "
                "[concat [get_property ip_repo_paths [current_project]] %s] "
                "[current_project]" % ip_dirs_str
            )
            config.append("update_ip_catalog -rebuild -scan_changes")

            ifnames = eval(kernel_model.get_metadata_prop("vivado_stitch_ifnames"))

            # gather info on connectivity
            # assume each node connected to outputs/inputs is DMA:
            # has axis, aximm and axilite
            # everything else is axis-only
            # assume only one connection from each ip to the next
            # all aximm allocated to DDR[0]
            # all kernels allocated to SLR0
            if len(node.input) == 0:
                producer = None
            else:
                producer = model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])
            # define kernel instances
            # name kernels connected to graph inputs as idmaxx
            # name kernels connected to graph outputs as odmaxx
            # do not expect IDMA/ODMA when instrumentation is enabled
            if (not use_instrumentation or self.enable_finn_switch) and (
                (producer is None) or (consumer == [])
            ):
                # TODO not a good way of checking for external inp&out
                # should look at the list of top-level in/out instead
                if producer is None:
                    instance_names[node.name] = "idma" + str(idma_idx)
                    idma_idx += 1
                elif consumer == []:
                    instance_names[node.name] = "odma" + str(odma_idx)
                    odma_idx += 1
                config.append(
                    "create_bd_cell -type ip -vlnv %s %s"
                    % (vivado_stitch_vlnv, instance_names[node.name])
                )
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/m_axi_gmem0] "
                    "[get_bd_intf_pins smartconnect_0/S%02d_AXI]"
                    % (instance_names[node.name], aximm_idx)
                )
                assert len(ifnames["axilite"]) == 1, "Must have 1 AXI lite interface on IODMA nodes"
                axilite_intf_name = ifnames["axilite"][0]
                assert axilite_intf_name is not None
                config.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins axi_interconnect_%d/M%02d_AXI]"
                    % (
                        instance_names[node.name],
                        axilite_intf_name,
                        axilite_interconnect_idx,
                        axilite_idx,
                    )
                )
                # assign_bd_address with appropriate range/offset
                config.append(
                    "assign_axi_addr_proc %s/%s" % (instance_names[node.name], axilite_intf_name)
                )

                aximm_idx += 1
                axilite_idx += 1
                if axilite_idx == 64:
                    axilite_interconnect_idx += 1
                    axilite_idx = 0
                if axilite_interconnect_idx == 0:
                    master_axilite_idx += 1
            else:
                instance_names[node.name] = node.name
                config.append(
                    "create_bd_cell -type ip -vlnv %s %s"
                    % (vivado_stitch_vlnv, instance_names[node.name])
                )

                if self.enable_selectable:
                    config.append(templates.selector_zynq_shell_template_procs)
                    config.append(
                        templates.selector_zynq_shell_template
                        % (instance_names[node.name], max_bodies)
                    )
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                        "[get_bd_intf_pins axi_interconnect_%d/M%02d_AXI]"
                        % (
                            f"{instance_names[node.name]}_selector",
                            "s_axilite",
                            axilite_interconnect_idx,
                            axilite_idx,
                        )
                    )
                    axilite_idx += 1
                    if axilite_idx == 64:
                        axilite_interconnect_idx += 1
                        axilite_idx = 0
                    if axilite_interconnect_idx == 0:
                        master_axilite_idx += 1

                for axilite_intf_name in ifnames["axilite"]:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                        "[get_bd_intf_pins axi_interconnect_%d/M%02d_AXI]"
                        % (
                            instance_names[node.name],
                            axilite_intf_name,
                            axilite_interconnect_idx,
                            axilite_idx,
                        )
                    )
                    # assign_bd_address with appropriate range/offset
                    config.append(
                        "assign_axi_addr_proc %s/%s"
                        % (instance_names[node.name], axilite_intf_name)
                    )
                    axilite_idx += 1
                    if axilite_idx == 64:
                        axilite_interconnect_idx += 1
                        axilite_idx = 0
                    if axilite_interconnect_idx == 0:
                        master_axilite_idx += 1
            sdp_node.set_nodeattr("instance_name", instance_names[node.name])

            config.append(
                "connect_bd_net [get_bd_pins %s/ap_clk] "
                "[get_bd_pins smartconnect_0/aclk]" % instance_names[node.name]
            )
            config.append(
                "connect_bd_net [get_bd_pins %s/ap_rst_n] "
                "[get_bd_pins smartconnect_0/aresetn]" % instance_names[node.name]
            )
            # connect streams
            if self.enable_finn_switch:
                for i in range(len(node.input)):
                    if producer is not None:
                        producer = model.find_producer(node.input[i])
                        j = list(producer.output).index(node.input[i])
                        producer_model = ModelWrapper(getCustomOp(producer).get_nodeattr("model"))
                        producer_idma = any(
                            s.name.startswith("IODMA") for s in producer_model.graph.output
                        )
                        # node_model = ModelWrapper(getCustomOp(node).get_nodeattr("model"))
                        node_odma = any(
                            s.name.startswith("TLastMarker") for s in kernel_model.graph.input
                        )
                        if not (producer_idma or node_odma):
                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_%d] "
                                "[get_bd_intf_pins %s/m_axis_%d]"
                                % (
                                    instance_names[node.name],
                                    i,
                                    instance_names[producer.name],
                                    j,
                                )
                            )
                        elif producer_idma:
                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_%d] "
                                "[get_bd_intf_pins finn_switch/A_IN0]"
                                % (
                                    instance_names[producer.name],
                                    j,
                                )
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins finn_switch/A_IN1] "
                                "[get_bd_intf_pins instrumentation_wrap_0/finnix]"
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_0] "
                                "[get_bd_intf_pins finn_switch/A_OUT]" % (instance_names[node.name])
                            )

                            ifnames = kernel_model.get_metadata_prop("vivado_stitch_ifnames")
                            ifnames = json.loads(ifnames)
                            width = ifnames["s_axis"][0][1]
                            config.append(
                                "set_property CONFIG.DATA_WIDTH_A {%d} [get_bd_cells finn_switch]"
                                % width
                            )
                        else:
                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_%d] "
                                "[get_bd_intf_pins finn_switch/B_OUT0]"
                                % (
                                    instance_names[node.name],
                                    i,
                                )
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins finn_switch/B_OUT1] "
                                "[get_bd_intf_pins instrumentation_wrap_0/finnox]"
                            )

                            config.append(
                                "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_0] "
                                "[get_bd_intf_pins finn_switch/B_IN]"
                                % (instance_names[producer.name])
                            )

                            ifnames = kernel_model.get_metadata_prop("vivado_stitch_ifnames")
                            ifnames = json.loads(ifnames)
                            width = ifnames["s_axis"][0][1]
                            config.append(
                                "set_property CONFIG.DATA_WIDTH_B {%d} [get_bd_cells finn_switch]"
                                % width
                            )
            else:
                for i in range(len(node.input)):
                    producer = model.find_producer(node.input[i])
                    if producer is not None:
                        j = list(producer.output).index(node.input[i])
                        config.append(
                            "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_%d] "
                            "[get_bd_intf_pins %s/m_axis_%d]"
                            % (
                                instance_names[node.name],
                                i,
                                instance_names[producer.name],
                                j,
                            )
                        )

            # connect first/last dataflow partition to instrumentation wrapper
            if use_instrumentation and not self.enable_finn_switch:
                if producer is None:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/s_axis_0] "
                        "[get_bd_intf_pins instrumentation_wrap_0/finnix]"
                        % (instance_names[node.name])
                    )
                if consumer == []:
                    config.append(
                        "connect_bd_intf_net [get_bd_intf_pins %s/m_axis_0] "
                        "[get_bd_intf_pins instrumentation_wrap_0/finnox]"
                        % (instance_names[node.name])
                    )

            # connect ring bus for live FIFO sizing
            if self.live_fifo_sizing:
                if "icfg" not in ifnames["ap_none"] or "ocfg" not in ifnames["ap_none"]:
                    raise FINNError(
                        "Live FIFO sizing requested but no icfg/ocfg interfaces found "
                        "on SDP %s" % node.name
                    )
                if sdp_id == 0:
                    # connect first SDP to fifo_controller
                    config.append(
                        "connect_bd_net [get_bd_pins fifo_controller_0/ocfg] "
                        f"[get_bd_pins {instance_names[node.name]}/icfg]"
                    )
                else:
                    # connect previous SDP to this SDP
                    config.append(
                        f"connect_bd_net [get_bd_pins {instance_names[prev_node_name]}/ocfg] "
                        f"[get_bd_pins {instance_names[node.name]}/icfg]"
                    )
                if sdp_id == num_sdps - 1:
                    # connect last SDP to fifo_controller
                    config.append(
                        f"connect_bd_net [get_bd_pins {instance_names[node.name]}/ocfg] "
                        "[get_bd_pins fifo_controller_0/icfg]"
                    )
                prev_node_name = node.name

        # TODO: WORKAROUND, do not instantiate smartconnect when not needed!
        if use_instrumentation and not self.enable_finn_switch:
            config.append("delete_bd_objs [get_bd_cells smartconnect_0]")
            aximm_idx = 1

        # finalize nested interconnect clock/reset
        for i in range(1, nested_interconnect_count + 1):
            config.append(
                "apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config "
                '"Clk /zynq_ps/$zynq_ps_clkname"  [get_bd_pins axi_interconnect_%d/M*_ACLK]' % (i)
            )

        # create a temporary folder for the project
        vivado_pynq_proj_dir = make_build_dir(prefix="vivado_zynq_proj_")
        model.set_metadata_prop("vivado_pynq_proj", vivado_pynq_proj_dir)

        fclk_mhz = int(1 / (self.period_ns * 0.001))

        pr_config = self._generate_pr_flow(model) if partial_reconfiguration else ""

        # create a TCL recipe for the project
        ipcfg = vivado_pynq_proj_dir + "/ip_config.tcl"
        config = "\n".join(config) + "\n"
        with open(ipcfg, "w") as f:
            f.write(
                (
                    templates.custom_zynq_shell_template
                    % (
                        fclk_mhz,
                        master_axilite_idx,
                        aximm_idx,
                        self.platform,
                        pynq_part_map[self.platform],
                        config,
                        self.enable_debug,
                        self.enable_gpio_reset,
                        self.enable_finn_switch,
                    )
                )
                .replace("$BOARDFILES$", str(get_settings().finn_deps / "board_files"))
                .replace("$PR_CONFIG$", pr_config)
            )

        # create a TCL recipe for the project
        synth_project_sh = vivado_pynq_proj_dir + "/synth_project.sh"
        working_dir = os.getcwd()
        with open(synth_project_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(vivado_pynq_proj_dir))
            f.write("vivado -mode batch -source %s\n" % ipcfg)
            f.write("cd {}\n".format(working_dir))

        # call the synthesis script
        bash_command = ["bash", synth_project_sh]
        try:
            launch_process_helper(bash_command, print_stdout=False)
        except CalledProcessError as e:
            raise FINNSynthesisError(
                f"Synthesis failed. Check {vivado_pynq_proj_dir} for details.",
                Path(vivado_pynq_proj_dir) / "vivado.log",
            ) from e

        bitfile_name = vivado_pynq_proj_dir + "/finn_zynq_link.runs/impl_1/top_wrapper.bit"
        if not os.path.isfile(bitfile_name):
            raise FINNSynthesisError(
                "Synthesis failed, no bitfile found. Check logs under %s" % vivado_pynq_proj_dir,
                Path(vivado_pynq_proj_dir) / "vivado.log",
            )
        deploy_bitfile_name = vivado_pynq_proj_dir + "/resizer.bit"
        copy(bitfile_name, deploy_bitfile_name)
        # set bitfile attribute
        model.set_metadata_prop("bitfile", deploy_bitfile_name)
        hwh_name_alts = [
            vivado_pynq_proj_dir + "/finn_zynq_link.srcs/sources_1/bd/top/hw_handoff/top.hwh",
            vivado_pynq_proj_dir + "/finn_zynq_link.gen/sources_1/bd/top/hw_handoff/top.hwh",
        ]
        hwh_name = None
        for hwh_name_cand in hwh_name_alts:
            if os.path.isfile(hwh_name_cand):
                hwh_name = hwh_name_cand
        if not os.path.isfile(hwh_name):
            raise FINNSynthesisError(
                "Synthesis failed, no bitfile found. Check logs under %s" % vivado_pynq_proj_dir,
                Path(vivado_pynq_proj_dir) / "vivado.log",
            )
        deploy_hwh_name = vivado_pynq_proj_dir + "/resizer.hwh"
        copy(hwh_name, deploy_hwh_name)
        model.set_metadata_prop("hw_handoff", deploy_hwh_name)
        # filename for the synth utilization report
        synth_report_filename = vivado_pynq_proj_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        if partial_reconfiguration:
            partial_bs_dir = vivado_pynq_proj_dir + "/partial_bitstreams"
            if os.path.isdir(partial_bs_dir):
                model.set_metadata_prop("partial_bitfiles_dir", partial_bs_dir)
        return (model, False)

    def _generate_pr_flow(self, model):
        """Generate partial reconfiguration hardware and bitstreams."""
        pr_config = []
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        pr_sdp_nodes = []
        for sdp_node in sdp_nodes:
            sdp_node_inst = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node_inst.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            if [
                n
                for n in kernel_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            ]:
                pr_sdp_nodes.append(sdp_node)

        for pr_sdp_node in pr_sdp_nodes:
            pr_sdp_node_inst = getCustomOp(pr_sdp_node)
            dataflow_model_filename = pr_sdp_node_inst.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            pr_node = [
                n
                for n in kernel_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            ][0]
            pr_node_inst = getCustomOp(pr_node)
            sdp_name = pr_sdp_node.name
            for id in range(pr_node_inst.get_nodeattr("bodies")):
                body_model = pr_node_inst.get_nodeattr("body_" + str(id))
                if id == 0:
                    # Special case, as this block is in the main bd
                    pr_config.append(
                        "group_bd_cells Hier_%s [get_bd_cells %s]" % (sdp_name, sdp_name)
                    )

                    # Validate before creating a Block Design Container
                    pr_config.append("validate_bd_design")

                    pr_config.append("startgroup")
                    pr_config.append("set curdesign [current_bd_design]")
                    pr_config.append(
                        "create_bd_design -cell [get_bd_cells /Hier_%s] Hier_%s"
                        % (sdp_name, sdp_name)
                    )
                    pr_config.append("current_bd_design $curdesign")

                    pr_config.append(
                        "set new_cell "
                        "[create_bd_cell -type container -reference Hier_%s Hier_%s_temp]"
                        % (sdp_name, sdp_name)
                    )
                    pr_config.append("replace_bd_cell [get_bd_cells /Hier_%s] $new_cell" % sdp_name)

                    pr_config.append("catch {delete_bd_objs [get_bd_cells /Hier_%s]}" % sdp_name)
                    pr_config.append("set_property name Hier_%s $new_cell" % sdp_name)
                    pr_config.append("endgroup")

                    # Enable DFX on the BDC
                    pr_config.append(
                        "set_property CONFIG.ENABLE_DFX {true} [get_bd_cells Hier_%s]" % sdp_name
                    )
                else:
                    # For each additional body create a Reconfigurable Module BD
                    # boundary ports are pre-defined by the container
                    body_vlnv = body_model.get_metadata_prop("vivado_stitch_vlnv")
                    body_ipstitch_path = body_model.get_metadata_prop("vivado_stitch_proj")
                    body_ifnames = eval(body_model.get_metadata_prop("vivado_stitch_ifnames"))

                    body_ip_dirs = ["list"]
                    body_ip_dirs += collect_ip_dirs(body_model, body_ipstitch_path)
                    body_ip_dirs_str = "[%s]" % (" ".join(body_ip_dirs))

                    bd_name = "Hier_%s_%d" % (sdp_name, id)
                    instance_name = "body_%d_ip" % id

                    pr_config.append(
                        "create_bd_design -boundary_from_container "
                        "[get_bd_cells /Hier_%s] %s" % (sdp_name, bd_name)
                    )

                    pr_config.append("current_bd_design [get_bd_designs %s]" % bd_name)
                    pr_config.append(
                        "set_property ip_repo_paths "
                        "[concat [get_property ip_repo_paths [current_project]] %s] "
                        "[current_project]" % body_ip_dirs_str
                    )
                    pr_config.append("update_ip_catalog -rebuild -scan_changes")
                    pr_config.append(
                        "create_bd_cell -type ip -vlnv %s %s" % (body_vlnv, instance_name)
                    )
                    pr_config.append(
                        "connect_bd_net [get_bd_pins %s/ap_clk] "
                        "[get_bd_ports ap_clk]" % instance_name
                    )
                    pr_config.append(
                        "connect_bd_net [get_bd_pins %s/ap_rst_n] "
                        "[get_bd_ports ap_rst_n]" % instance_name
                    )
                    for s_axis_name, _ in body_ifnames.get("s_axis", []):
                        pr_config.append(
                            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                            "[get_bd_intf_ports %s]" % (instance_name, s_axis_name, s_axis_name)
                        )
                    for m_axis_name, _ in body_ifnames.get("m_axis", []):
                        pr_config.append(
                            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                            "[get_bd_intf_ports %s]" % (instance_name, m_axis_name, m_axis_name)
                        )
                    for axilite_name in body_ifnames.get("axilite", []):
                        pr_config.append(
                            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                            "[get_bd_intf_ports %s]" % (instance_name, axilite_name, axilite_name)
                        )
                    for aximm_name, _ in body_ifnames.get("aximm", []):
                        pr_config.append(
                            "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                            "[get_bd_intf_ports %s]" % (instance_name, aximm_name, aximm_name)
                        )
                    pr_config.append("save_bd_design")
                    pr_config.append("validate_bd_design")
                    pr_config.append("current_bd_design $curdesign")

        # DFX Controller & ICAP

        pr_config.append("current_bd_design $curdesign")
        for dfx_file in [
            os.path.join(get_settings().finn_rtllib, "icap", "icape3_wrapper.v"),
            os.path.join(
                get_settings().finn_rtllib, "dfx", "dfx_finn_decouple", "dfx_finn_decouple.sv"
            ),
            os.path.join(
                get_settings().finn_rtllib,
                "dfx",
                "dfx_finn_decouple",
                "dfx_finn_decouple_wrapper.v",
            ),
            os.path.join(get_settings().finn_rtllib, "dfx", "dfx_schedule", "dfx_schedule.sv"),
            os.path.join(
                get_settings().finn_rtllib, "dfx", "dfx_schedule", "dfx_schedule_wrapper.v"
            ),
            os.path.join(get_settings().finn_rtllib, "axi", "hdl", "axilite.sv"),
        ]:
            pr_config.append(
                "add_files -copy_to [get_property DIRECTORY [current_project]] -norecurse %s"
                % dfx_file
            )
        pr_config.append("create_bd_cell -type module -reference icape3_wrapper icape3_wrapper")
        pr_config.append(
            "create_bd_cell -type ip -vlnv xilinx.com:ip:dfx_controller:1.0 dfx_controller_0"
        )
        pr_config.append(
            "source [get_property REPOSITORY "
            "[get_ipdefs *dfx_controller:1.0]]"
            "/xilinx/dfx_controller_v1_0/tcl/api.tcl -notrace"
        )
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins dfx_controller_0/ICAP] "
            "[get_bd_intf_pins icape3_wrapper/ICAP]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins icape3_wrapper/clk] [get_bd_pins zynq_ps/$zynq_ps_clkname]"
        )
        for pr_sdp in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp)
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer = [
                n
                for n in pr_sdp_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            ][0]
            pr_nodecontainer_inst = getCustomOp(pr_nodecontainer)
            num_bodies = pr_nodecontainer_inst.get_nodeattr("bodies")
            dfx_cont_vs_config = []

            vs_name = pr_sdp.name
            dfx_cont_vs_config.append("CONFIG.VS.%s.NUM_RMS_ALLOCATED %d" % (vs_name, num_bodies))
            for rm_idx in range(num_bodies):
                dfx_cont_vs_config.append("CONFIG.VS.%s.RM.%d.BS.0.ADDRESS 0x0" % (vs_name, rm_idx))
            dfx_cont_vs_config.append(
                "CONFIG.VS.%s.NUM_TRIGGERS_ALLOCATED %d" % (vs_name, num_bodies)
            )
            dfx_cont_vs_config.append("CONFIG.VS.%s.NUM_HW_TRIGGERS %d" % (vs_name, num_bodies))
            for rm_idx in range(num_bodies):
                dfx_cont_vs_config.append(
                    "CONFIG.VS.%s.TRIGGER%d_TO_RM %d" % (vs_name, rm_idx, rm_idx)
                )
            pr_config.append(
                "dfx_controller_v1_0::set_property -dict [list %s] [get_bd_cells dfx_controller_0]"
                % " ".join(dfx_cont_vs_config)
            )

        pr_config.append("set_property CONFIG.PSU__USE__S_AXI_GP3 {1} [get_bd_cells zynq_ps]")

        # Create dedicated reset controller for DFX controller
        # (reset 1, independent of main system reset 0)
        pr_config.append(
            "create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_dfx"
        )
        pr_config.append(
            "apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config "
            '"Clk /zynq_ps/$zynq_ps_clkname" '
            "[get_bd_pins proc_sys_reset_dfx/slowest_sync_clk]"
        )

        pr_config.append("set_property CONFIG.PSU__NUM_FABRIC_RESETS {2} [get_bd_cells zynq_ps]")

        pr_config.append(
            "connect_bd_net [get_bd_pins zynq_ps/pl_resetn1] "
            "[get_bd_pins proc_sys_reset_dfx/ext_reset_in]"
        )

        # Connect DFX controller clock and reset
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_controller_0/clk] " "[get_bd_pins smartconnect_0/aclk]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_controller_0/clk] "
            "[get_bd_pins dfx_controller_0/icap_clk]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_controller_0/reset] "
            "[get_bd_pins proc_sys_reset_dfx/peripheral_aresetn]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_controller_0/icap_reset] "
            "[get_bd_pins proc_sys_reset_dfx/peripheral_aresetn]"
        )

        # Connect DFX controller s_axi_reg to PS master via axi_interconnect_0
        # (extend axi_interconnect_0 with one extra master port)
        pr_config.append(
            "set dfx_mi_idx [get_property CONFIG.NUM_MI [get_bd_cells axi_interconnect_0]]"
        )
        pr_config.append(
            "set_property CONFIG.NUM_MI [expr {$dfx_mi_idx + 1}] [get_bd_cells axi_interconnect_0]"
        )
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins dfx_controller_0/s_axi_reg] "
            "[get_bd_intf_pins axi_interconnect_0/[format M%02d_AXI $dfx_mi_idx]]"
        )
        pr_config.append(
            "apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config "
            '"Clk /zynq_ps/$zynq_ps_clkname" '
            "[get_bd_pins axi_interconnect_0/[format M%02d_ACLK $dfx_mi_idx]]"
        )
        pr_config.append("assign_bd_address [get_bd_addr_segs {dfx_controller_0/s_axi_reg/Reg}]")

        pr_config.append("save_bd_design")

        # Create a new SmartConnect to route dfx_controller AXI master zynq_ps/S_AXI_HP1_FPD
        pr_config.append(
            "set smartconnect_dfx_vlnv "
            "[get_property VLNV [get_ipdefs xilinx.com:ip:smartconnect:*]]"
        )
        pr_config.append("create_bd_cell -type ip -vlnv $smartconnect_dfx_vlnv smartconnect_dfx")
        pr_config.append(
            "set_property -dict [list CONFIG.NUM_SI {1}] [get_bd_cells smartconnect_dfx]"
        )
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins dfx_controller_0/M_AXI_MEM] "
            "[get_bd_intf_pins smartconnect_dfx/S00_AXI]"
        )
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins smartconnect_dfx/M00_AXI] "
            "[get_bd_intf_pins zynq_ps/S_AXI_HP1_FPD]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins smartconnect_dfx/aclk] [get_bd_pins smartconnect_0/aclk]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins smartconnect_dfx/aresetn] "
            "[get_bd_pins smartconnect_0/aresetn]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins zynq_ps/saxihp1_fpd_aclk] "
            "[get_bd_pins smartconnect_0/aclk]"
        )

        pr_config.append("create_bd_cell -type module -reference dfx_schedule_wrapper dfx_schedule")

        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_schedule/controller_trigger] "
            "[get_bd_pins dfx_controller_0/vsm_*_hw_triggers]"
        )

        pr_config.append(
            "set gpio_mi_idx [get_property CONFIG.NUM_MI [get_bd_cells axi_interconnect_0]]"
        )
        pr_config.append(
            "set_property CONFIG.NUM_MI [expr {$gpio_mi_idx + 1}] [get_bd_cells axi_interconnect_0]"
        )

        # Dfx Scheule Wrapper
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins dfx_schedule/s_axilite] "
            "[get_bd_intf_pins axi_interconnect_0/[format M%02d_AXI $gpio_mi_idx]]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_schedule/aclk] " "[get_bd_pins smartconnect_0/aclk]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_schedule/aresetn] "
            "[get_bd_pins proc_sys_reset_dfx/peripheral_aresetn]"
        )

        pr_config.append(
            "set_property -dict [list CONFIG.MAX_TRIGGERS {%d} CONFIG.SCHEDULE_SLOTS {%d}] "
            "[get_bd_cells dfx_schedule]" % (num_bodies, len(pr_sdp_nodes))
        )

        pr_config.append("assign_bd_address [get_bd_addr_segs {dfx_schedule/S_AXI/Reg}]")

        pr_config.append(
            "apply_bd_automation -rule xilinx.com:bd_rule:clkrst -config "
            '"Clk /zynq_ps/$zynq_ps_clkname" '
            "[get_bd_pins axi_interconnect_0/[format M%02d_ACLK $gpio_mi_idx]]"
        )
        pr_config.append("assign_bd_address")

        pr_config.append(
            "set decouple_pins "
            "[get_bd_pins -of_objects [get_bd_cells dfx_controller_0]"
            ' -filter "name =~ *vsm_*_rm_decouple"]'
        )

        pr_config.append(
            "create_bd_cell -type ip -vlnv xilinx.com:ip:xlconcat:2.1 xlconcat_decouple"
        )
        pr_config.append(
            "set_property CONFIG.NUM_PORTS "
            "[llength $decouple_pins] [get_bd_cells xlconcat_decouple]"
        )
        pr_config.append(
            "for {set i 0} {$i < [llength $decouple_pins]} {incr i} {"
            " connect_bd_net [lindex $decouple_pins $i]"
            " [get_bd_pins xlconcat_decouple/In$i]"
            " }"
        )

        pr_config.append(
            "create_bd_cell -type ip -vlnv "
            "xilinx.com:ip:util_reduced_logic:2.0 util_reduced_logic_decouple"
        )
        pr_config.append(
            "set_property -dict [list CONFIG.C_OPERATION {or} "
            "CONFIG.C_SIZE [llength $decouple_pins]] "
            "[get_bd_cells util_reduced_logic_decouple]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins xlconcat_decouple/dout] "
            "[get_bd_pins util_reduced_logic_decouple/Op1]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins util_reduced_logic_decouple/Res] "
            "[get_bd_pins dfx_schedule/controller_decouple]"
        )

        # DFX FINN decoupler
        pr_config.append(
            "create_bd_cell -type module -reference dfx_finn_decouple_wrapper dfx_finn_decouple"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_finn_decouple/decouple] "
            "[get_bd_pins dfx_schedule/accel_decouple_input]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_finn_decouple/aclk] "
            "[get_bd_pins smartconnect_0/aclk]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_finn_decouple/aresetn] "
            "[get_bd_pins proc_sys_reset_dfx/peripheral_aresetn]"
        )

        idma_node_inst = None
        for sdp in model.get_nodes_by_op_type("StreamingDataflowPartition"):
            sdp_model = ModelWrapper(getCustomOp(sdp).get_nodeattr("model"))
            for n in sdp_model.graph.node:
                if n.op_type == "IODMA_hls":
                    inst = getCustomOp(n)
                    if inst.get_nodeattr("direction") == "in":
                        idma_node_inst = inst
                        break
        decouple_data_width = idma_node_inst.get_nodeattr("streamWidth")
        decouple_cnt = math.prod(idma_node_inst.get_nodeattr("numInputVectors"))
        pr_config.append(
            "set_property -dict [list CONFIG.DATA_WIDTH {%d} CONFIG.DECOUPLE_CNT {%d}] "
            "[get_bd_cells dfx_finn_decouple]" % (decouple_data_width, decouple_cnt)
        )

        # DFX decoupler has its own api for setting properties
        pr_config.append(
            "source [get_property REPOSITORY "
            "[get_ipdefs *dfx_decoupler:1.0]]"
            "/xilinx/dfx_decoupler_v1_0/tcl/api.tcl -notrace"
        )
        pr_config.append(
            "create_bd_cell -type ip -vlnv xilinx.com:ip:dfx_decoupler:1.0 dfx_decoupler"
        )
        pr_config.append(
            "dfx_decoupler_v1_0::set_property -dict "
            "[list CONFIG.INTF.intf_0.VLNV "
            "xilinx.com:interface:axis_rtl:1.0] "
            "[get_bd_cells dfx_decoupler]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins dfx_decoupler/decouple] "
            "[get_bd_pins dfx_schedule/accel_decouple_output]"
        )

        # Reset Block
        pr_config.append(
            "create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_accel"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins proc_sys_reset_accel/ext_reset_in] "
            "[get_bd_pins zynq_ps/pl_resetn0]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins proc_sys_reset_accel/slowest_sync_clk] "
            "[get_bd_pins zynq_ps/$zynq_ps_clkname]"
        )
        pr_config.append(
            "connect_bd_net [get_bd_pins proc_sys_reset_accel/aux_reset_in] "
            "[get_bd_pins dfx_schedule/accel_reset]"
        )

        pr_config.append(
            "set finn_switch_A_OUT_slave "
            "[get_bd_intf_pins -of_objects "
            "[get_bd_intf_nets -of_objects [get_bd_intf_pins finn_switch/A_OUT]] "
            "-filter {mode == Slave}]"
        )
        pr_config.append(
            "delete_bd_objs [get_bd_intf_nets -of_objects [get_bd_intf_pins finn_switch/A_OUT]]"
        )
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins finn_switch/A_OUT] "
            "[get_bd_intf_pins dfx_finn_decouple/s_axis]"
        )
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins dfx_finn_decouple/m_axis] "
            "$finn_switch_A_OUT_slave"
        )

        pr_config.append(
            "set finn_switch_B_IN_master "
            "[get_bd_intf_pins -of_objects "
            "[get_bd_intf_nets -of_objects [get_bd_intf_pins finn_switch/B_IN]] "
            "-filter {mode == Master}]"
        )
        pr_config.append(
            "delete_bd_objs [get_bd_intf_nets -of_objects [get_bd_intf_pins finn_switch/B_IN]]"
        )
        pr_config.append(
            "connect_bd_intf_net $finn_switch_B_IN_master "
            "[get_bd_intf_pins dfx_decoupler/rp_intf_0]"
        )
        pr_config.append(
            "connect_bd_intf_net [get_bd_intf_pins dfx_decoupler/s_intf_0] "
            "[get_bd_intf_pins finn_switch/B_IN]"
        )

        pr_config.append(
            "foreach rst_pin [get_bd_pins -of_objects "
            "[get_bd_cells *StreamingDataflowPartition_*] -filter {name == ap_rst_n}] {"
            " set rst_net [get_bd_nets -of_objects $rst_pin];"
            " if {$rst_net ne {}} { disconnect_bd_net $rst_net $rst_pin };"
            " connect_bd_net [get_bd_pins proc_sys_reset_accel/peripheral_aresetn] $rst_pin"
            " }"
        )

        for pr_sdp in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp)
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer = [
                n
                for n in pr_sdp_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            ][0]
            pr_nodecontainer_inst = getCustomOp(pr_nodecontainer)
            sdp_name = pr_sdp.name
            num_bodies = pr_nodecontainer_inst.get_nodeattr("bodies")
            bd_list = ":".join(
                ["Hier_%s.bd" % sdp_name]
                + ["Hier_%s_%d.bd" % (sdp_name, i) for i in range(1, num_bodies)]
            )
            pr_config.append(
                "set_property -dict [list "
                "CONFIG.LIST_SIM_BD {%s} "
                "CONFIG.LIST_SYNTH_BD {%s} "
                "] [get_bd_cells Hier_%s]" % (bd_list, bd_list, sdp_name)
            )

        pr_config.append("save_bd_design")
        pr_config.append("make_wrapper -files [get_files top.bd] -import -fileset sources_1 -top")
        pr_config.append("set_property top top_wrapper [get_filesets sources_1]")
        pr_config.append("update_compile_order -fileset sources_1")
        pr_config.append("set_property PR_FLOW 1 [current_project]")
        pr_config.append("generate_target all [get_files top.bd]")

        pr_sdp_bodies = []
        pr_sdp_names = []
        for pr_sdp_node in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp_node)
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer_inst = getCustomOp(
                [
                    n
                    for n in pr_sdp_model.graph.node
                    if n.op_type == "NodeContainer"
                    and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
                ][0]
            )
            pr_sdp_names.append(pr_sdp_node.name)
            pr_sdp_bodies.append(pr_nodecontainer_inst.get_nodeattr("bodies"))
        assert all(
            n == pr_sdp_bodies[0] for n in pr_sdp_bodies
        ), "All NodeContainers must have the same number of bodies for pr"
        num_bodies = pr_sdp_bodies[0]

        for body_id in range(num_bodies):
            config_name = "config_%d" % body_id
            partitions = " ".join(
                "top_i/Hier_%s:Hier_%s_inst_0" % (sdp_name, sdp_name)
                if body_id == 0
                else "top_i/Hier_%s:Hier_%s_%d_inst_0" % (sdp_name, sdp_name, body_id)
                for sdp_name in pr_sdp_names
            )
            pr_config.append(
                "create_pr_configuration -name %s -partitions [list %s]" % (config_name, partitions)
            )
            if body_id == 0:
                pr_config.append("set_property PR_CONFIGURATION config_0 [get_runs impl_1]")
            else:
                impl_run = "impl_body_%d" % body_id
                pr_config.append(
                    "create_run %s -parent_run impl_1 "
                    "-flow {Vivado Implementation 2020} -pr_config %s" % (impl_run, config_name)
                )

        pr_config.append("launch_runs synth_1 -jobs 4")
        pr_config.append("wait_on_run synth_1")

        pr_config.append("open_run synth_1 -name synth_1")
        for pr_sdp in pr_sdp_nodes:
            pr_sdp_inst = getCustomOp(pr_sdp)
            sdp_name = pr_sdp.name
            pr_sdp_model = ModelWrapper(pr_sdp_inst.get_nodeattr("model"))
            pr_nodecontainer = [
                n
                for n in pr_sdp_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            ][0]
            pr_nodecontainer_inst = getCustomOp(pr_nodecontainer)
            pblock = pr_nodecontainer_inst.get_nodeattr("pblock")
            assert pblock, "NodeContainer %s has no 'pblock' attribute set" % sdp_name
            pblock_name = "pblock_Hier_%s" % sdp_name
            cell_path = "top_i/Hier_%s" % sdp_name
            pr_config.append("create_pblock %s" % pblock_name)
            pr_config.append(
                "add_cells_to_pblock [get_pblocks %s] [get_cells %s]" % (pblock_name, cell_path)
            )
            pr_config.append("resize_pblock [get_pblocks %s] -add {%s}" % (pblock_name, pblock))
            pr_config.append("set_property SNAPPING_MODE ON [get_pblocks %s]" % pblock_name)
        pr_config.append("save_constraints -force")
        pr_config.append("close_design")

        for body_id in range(num_bodies):
            run_name = "impl_1" if body_id == 0 else "impl_body_%d" % body_id
            pr_config.append(
                "set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs %s]" % run_name
            )

        pr_config.append("launch_runs impl_1 -to_step write_bitstream -jobs 4")
        pr_config.append("wait_on_run impl_1")
        for body_id in range(1, num_bodies):
            impl_run = "impl_body_%d" % body_id
            pr_config.append("launch_runs %s -to_step write_bitstream -jobs 4" % impl_run)
            pr_config.append("wait_on_run %s" % impl_run)

        pr_config.append(
            "set partial_bs_dir "
            "[file join [get_property DIRECTORY [current_project]] partial_bitstreams]"
        )
        pr_config.append("file mkdir $partial_bs_dir")
        for body_id in range(num_bodies):
            impl_run = "impl_1" if body_id == 0 else "impl_body_%d" % body_id
            pr_config.append(
                "file copy -force "
                "[file join [get_property DIRECTORY [get_runs %s]] top_wrapper.bit] "
                "[file join $partial_bs_dir config_%d.bit]" % (impl_run, body_id)
            )
            for sdp_name in pr_sdp_names:
                if body_id == 0:
                    partial_bit_name = "top_i_Hier_%s_Hier_%s_inst_0_partial.bit" % (
                        sdp_name,
                        sdp_name,
                    )
                else:
                    partial_bit_name = "top_i_Hier_%s_Hier_%s_%d_inst_0_partial.bit" % (
                        sdp_name,
                        sdp_name,
                        body_id,
                    )
                pr_config.append(
                    "file copy -force "
                    "[file join [get_property DIRECTORY [get_runs %s]] %s] "
                    "[file join $partial_bs_dir partial_%s_%d.bit]"
                    % (impl_run, partial_bit_name, sdp_name, body_id)
                )
                partial_bin_name = partial_bit_name.replace(".bit", ".bin")
                pr_config.append(
                    "file copy -force "
                    "[file join [get_property DIRECTORY [get_runs %s]] %s] "
                    "[file join $partial_bs_dir partial_%s_%d.bin]"
                    % (impl_run, partial_bin_name, sdp_name, body_id)
                )
                pr_config.append(
                    "dfx_controller_v1_0::format_bin_for_icap "
                    "-bs 1 "
                    "-i [file join $partial_bs_dir partial_%s_%d.bin] "
                    "-o [file join $partial_bs_dir partial_%s_%d_icap.bin]"
                    % (sdp_name, body_id, sdp_name, body_id)
                )

        pr_config.append("set pr_flow 1")
        pr_config.append("save_bd_design")
        pr_config = "\n".join(pr_config) + "\n"
        return pr_config


class ZynqBuild(Transformation):
    """Best-effort attempt at building the accelerator for Zynq.
    It assumes the model has only fpgadataflow nodes

    """

    def __init__(
        self,
        platform,
        period_ns,
        enable_debug=False,
        enable_instrumentation=False,
        instrumentation_no_dma=False,
        live_fifo_sizing=False,
        partition_model_dir=None,
    ):
        """Initialize ZynqBuild with platform and build settings."""
        super().__init__()
        self.fpga_part = pynq_part_map[platform]
        self.axi_port_width = pynq_native_port_width[platform]
        self.period_ns = period_ns
        self.platform = platform
        self.enable_debug = enable_debug
        self.enable_instrumentation = enable_instrumentation
        self.instrumentation_no_dma = instrumentation_no_dma
        self.live_fifo_sizing = live_fifo_sizing
        self.partition_model_dir = partition_model_dir

    def apply(self, model):
        """Apply the ZynqBuild transformation to create a complete Zynq accelerator."""
        model = model.transform(InferDataLayouts())
        # prepare at global level, then break up into kernels
        enable_finn_switch = (
            self.enable_instrumentation
            and (not self.instrumentation_no_dma)
            and (not self.live_fifo_sizing)
        )
        if self.enable_instrumentation:
            if self.instrumentation_no_dma is True or self.live_fifo_sizing is True:
                prep_transforms = [
                    GenerateInstrumentationIP(self.fpga_part, self.period_ns),
                    Floorplan(),
                    CreateDataflowPartition(partition_model_dir=self.partition_model_dir),
                ]
            else:
                # DMA & Instrumentation Wrapper Case
                prep_transforms = [
                    GenerateInstrumentationIP(self.fpga_part, self.period_ns),
                    InsertIODMA(self.axi_port_width),
                    InsertDWC(),
                    SpecializeLayers(self.fpga_part),
                    Floorplan(),
                    CreateDataflowPartition(partition_model_dir=self.partition_model_dir),
                ]
        else:
            prep_transforms = [
                InsertIODMA(self.axi_port_width),
                InsertDWC(),
                SpecializeLayers(self.fpga_part),
                Floorplan(),
                CreateDataflowPartition(partition_model_dir=self.partition_model_dir),
            ]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
        # Build each kernel individually
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        for sdp_node in sdp_nodes:
            prefix = sdp_node.name + "_"
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)

            if not kernel_model.get_nodes_by_op_type("IODMA_hls"):
                del kernel_model.model.graph.metadata_props[:]
                kernel_model.save(dataflow_model_filename)

            prcont = [
                n
                for n in kernel_model.graph.node
                if n.op_type == "NodeContainer"
                and getCustomOp(n).get_nodeattr("multi_dnn_type") == "partial_reconfiguration"
            ]
            if prcont:
                assert (
                    kernel_model.graph.node[0].op_type == "NodeContainer"
                    and getCustomOp(kernel_model.graph.node[0]).get_nodeattr("multi_dnn_type")
                    == "partial_reconfiguration"
                ), "Expected NodeContainer in SDP when using partial reconfiguration"
                assert (
                    len(kernel_model.graph.node) == 1
                ), "Only one NodeContainer per SDP when using partial reconfiguration"
                prcontainer = prcont[0]
                pr_container_inst = getCustomOp(prcontainer)
                for id in range(pr_container_inst.get_nodeattr("bodies")):
                    body_model = pr_container_inst.get_nodeattr("body_" + str(id))

                    if not self.enable_instrumentation:
                        body_model = body_model.transform(InsertFIFO())
                    body_model = body_model.transform(SpecializeLayers(self.fpga_part))

                    body_model.save(dataflow_model_filename)
                    body_model = body_model.transform(PrepareIP(self.fpga_part, self.period_ns))
                    body_model = body_model.transform(HLSSynthIP())
                    body_model = body_model.transform(
                        CreateStitchedIP(
                            self.fpga_part,
                            self.period_ns,
                            f"sdp_{pr_container_inst.onnx_node.name}_{id}",
                            vitis=False,
                        )
                    )
                    body_model.set_metadata_prop("platform", "zynq-iodma")
                    pr_container_inst.set_nodeattr("body_" + str(id), body_model)
                    body_model.save(dataflow_model_filename)

                kernel_model.set_metadata_prop("platform", "zynq-iodma")
                kernel_model.save(dataflow_model_filename)

            else:
                # InsertFIFO at this stage interferes with tLastMarker
                # TODO: is this really needed here at all?
                if not self.enable_instrumentation:
                    kernel_model = kernel_model.transform(InsertFIFO())
                kernel_model = kernel_model.transform(SpecializeLayers(self.fpga_part))

                nodecontiner = kernel_model.get_nodes_by_op_type("NodeContainer")
                if not nodecontiner:
                    kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
                kernel_model.save(dataflow_model_filename)
                kernel_model = kernel_model.transform(PrepareIP(self.fpga_part, self.period_ns))
                kernel_model = kernel_model.transform(HLSSynthIP())
                kernel_model = kernel_model.transform(
                    CreateStitchedIP(
                        self.fpga_part, self.period_ns, sdp_node.onnx_node.name, vitis=False
                    )
                )
                kernel_model.set_metadata_prop("platform", "zynq-iodma")
                kernel_model.save(dataflow_model_filename)
        # Assemble design from IPs
        model = model.transform(
            MakeZYNQProject(
                self.platform,
                self.period_ns,
                enable_debug=self.enable_debug,
                enable_finn_switch=enable_finn_switch,
                live_fifo_sizing=self.live_fifo_sizing,
            )
        )

        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "zynq-iodma")

        return (model, False)
