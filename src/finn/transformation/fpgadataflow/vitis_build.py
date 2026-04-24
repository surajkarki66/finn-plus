"""Build a Vitis accelerator from a completed FINN design."""

# Copyright (c) 2020, Xilinx, Inc.
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
from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import (
    GiveReadableTensorNames,
    GiveUniqueNodeNames,
    RemoveUnusedTensors,
)
from subprocess import CalledProcessError

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    FpgaMemoryType,
    MFCommunicationKernel,
    VitisOptStrategy,
)
from finn.transformation.fpgadataflow.build_xo import CreateVitisXO
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.floorplan import Floorplan
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.insert_dwc import InsertDWC
from finn.transformation.fpgadataflow.insert_fifo import InsertFIFO
from finn.transformation.fpgadataflow.insert_iodma import InsertIODMA
from finn.transformation.fpgadataflow.multifpga.aurora_metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.metadata import DataDirection
from finn.transformation.fpgadataflow.multifpga.utils import get_device_id
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.util.basic import launch_process_helper, make_build_dir
from finn.util.exception import (
    FINNConfigurationError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNSynthesisError,
    FINNVitisLinkConfigError,
)
from finn.util.logging import log
from finn.util.settings import get_settings
from finn.util.vivado import check_vitis_envvars

from . import templates


class VitisLinkConfiguration:
    """Manages XO files, CU instantiations, stream connections,
    port connections, Vivado props, etc.
    It can output a linking configuration to pass to v++ and
    create a shell script to run it. Tries to be as strict and careful as possible,
    and depending on the issue raises an Exception, logs an error or warning
    or continues silently."""

    def __init__(self, platform: str, optimization_level: str, f_mhz: int) -> None:
        self.cu: list[str] = []
        self.nk: list[tuple[str, str]] = []
        self.sc: dict[str, list[str]] = {}
        self.sp: dict[str, str] = {}
        self.xo: list[Path] = []
        self.connects: list[tuple[str, str]] = []
        self.vivado_section: str = "[vivado]\n"
        self.connectivity_section: str = ""
        self.platform: str = platform
        self.optimization_level: str = optimization_level
        self.f_mhz: int = f_mhz

    def add_cu(self, kernel_name: str, cu_name: str) -> None:
        """Add a compute unit (instance of a kernel)"""
        if cu_name in self.cu:
            kern = next(kname for kname, cname in self.nk if cname == cu_name)
            raise FINNVitisLinkConfigError(
                f"Tried creating CU {cu_name}, but a CU of this "
                f"name of kernel {kern} already exists!"
            )
        self.cu.append(cu_name)
        self.nk.append((kernel_name, cu_name))

    def add_sc(self, cu_sender: str, cu_receiver: str) -> None:
        """Add a Streaming Connection between two CUs:
        >>> lc = VitisLinkConfiguration("", "", 100)
        >>> lc.add_cu("A", "a")
        >>> lc.add_cu("B", "b")
        >>> lc.add_sc("a.out", "b.in")
        >>> lc.sc["a.out"]
        ['b.in']
        """
        # Check formatting
        for cu in [cu_sender, cu_receiver]:
            splits = cu.split(".")
            if len(splits) != 2:
                raise FINNVitisLinkConfigError(
                    f"{cu} is incorrectly formatted. Required "
                    f"syntax to add a streaming connection from CU "
                    f'a on port out is "a.out".'
                )

        # Yield warning if the direction seems wrong
        sender_port = cu_sender.split(".")[1]
        receiver_port = cu_receiver.split(".")[1]
        if sender_port.lower() in ["s_axis", "in"] or receiver_port.lower() in ["m_axis", "out"]:
            log.error(
                f"Adding connection sc={cu_sender}:{cu_receiver}. The port "
                "names suggest that the order of sender and receiver might be "
                "swapped. Proceeding now."
            )

        # Add the connection
        if cu_sender not in self.sc.keys():
            self.sc[cu_sender] = []
        self.sc[cu_sender].append(cu_receiver)

    def add_sp(self, cu_port_name: str, mem_type: str) -> None:
        """Add an SP assignment."""
        self.sp[cu_port_name] = mem_type

    def add_connect(self, a: str, b: str) -> None:
        """Add a connect assignment. Not to be confused with stream_connect (sc)!"""
        self.connects.append((a, b))

    def add_vivado_line(self, line: str) -> None:
        """Add a custom line to the vivado section."""
        self.vivado_section += line

    def add_xo(self, xo_files: Path | list[Path]) -> None:
        """Add an XO file. This will emit an error if the XO file is not found, but it will
        NOT raise an exception. Ignores duplicate calls"""
        all_xos = []
        if type(xo_files) is Path:
            all_xos = [xo_files]
        elif type(xo_files) is list:
            all_xos = xo_files
        else:
            all_xos = [Path(xo_files)]

        for xo_file in all_xos:
            if xo_file in self.xo:
                log.warning(f"Ignoring duplicate addition of .xo: {xo_file.name}")
                continue
            if not xo_file.exists():
                log.error(
                    f"Tried adding non-existing file {xo_file.absolute()}. "
                    f"Continuing in case this is on purpose."
                )
            self.xo.append(xo_file)

    def add_connectivity(self, txt: str) -> None:
        """Add further lines to the connectivity section. For example to assign clocks or ports"""
        self.connectivity_section += txt

    def get_config_validation_errors(self) -> None | list[FINNVitisLinkConfigError]:
        """Check the configuration and if errors are found, return them"""
        errors = []
        # All CUs in SCs exist and CU ports are correctly formatted
        for cu_sender, receivers in self.sc.items():
            for cu_receiver in receivers:
                sender_split = cu_sender.split(".")
                if len(sender_split) != 2:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} "
                            f"incorrectly formatted. "
                            "Use the syntax CU.PORT"
                        )
                    )
                sender_name = sender_split[0]
                if sender_name not in self.cu:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} uses the unknown CU {sender_name}"
                        )
                    )
                receiver_split = cu_receiver.split(".")
                if len(receiver_split) != 2:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} "
                            f"incorrectly formatted. "
                            "Use the syntax CU.PORT"
                        )
                    )
                receiver_name = receiver_split[0]
                if receiver_name not in self.cu:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:"
                            f"{cu_receiver} uses the unknown "
                            f"CU {receiver_name}"
                        )
                    )
        # No two same named CUs
        if len(set(self.cu)) != len(self.cu):
            errors.append(
                FINNVitisLinkConfigError(
                    "It seems that there are one or more CUs with the same name!"
                )
            )
        for kernel, cu in self.nk:
            for kernel2, cu2 in self.nk:
                if cu == cu2 and kernel != kernel2:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"There are 2 or more CUs named {cu} "
                            f"from different kernels ({kernel} "
                            f"and {kernel2})"
                        )
                    )
        if len(errors) > 0:
            return errors
        return None

    def generate_config(self, path: Path) -> None:
        """Write the complete config to the given path. Raises an error if the
        config is invalid"""
        errors = self.get_config_validation_errors()
        if errors is not None:
            for err in errors:
                log.error(f"{path}: {err}")
            if len(errors) == 1:
                raise errors[0]
            raise FINNVitisLinkConfigError(
                "Multiple VitisLinkConfig errors ocurred. " "Please check your logs to fix them."
            )
        with path.open("w+") as f:
            f.write("[connectivity]\n")
            for kernel_name, cu_name in self.nk:
                f.write(f"nk={kernel_name}:1:{cu_name}\n")

            # origin_cu and target_cu already require the ports already being in the str
            for origin_cu in self.sc.keys():
                for target_cu in self.sc[origin_cu]:
                    f.write(f"sc={origin_cu}:{target_cu}\n")

            for sp_cu, sp_mem in self.sp.items():
                f.write(f"sp={sp_cu}:{sp_mem}\n")

            for a, b in self.connects:
                f.write(f"connect={a}:{b}\n")

            if self.connectivity_section != "":
                f.write(self.connectivity_section + "\n")

            f.write(self.vivado_section)

        if not path.exists():
            raise FINNMultiFPGAError(f"Failed to create vitis config at {path}.")

    def generate_run_script(self, config_path: Path, target: Path | None = None) -> None:
        """Generate a shell script to start v++ with the correct parameters.
        Produces the shell script next to the path of the config file
        unless a path is specified"""
        xo_string = " ".join([str(xo) for xo in self.xo])
        if not config_path.exists():
            log.error(
                f"Writing compilation / v++ script for non-existing configuration "
                f"in {config_path.absolute()}. Continuing in case this is on purpose."
            )
        runner_path = config_path.parent / "run_vitis_link.sh"
        if target is not None:
            runner_path = target
        with runner_path.open("w+") as f:
            f.write("#!/bin/bash\n")
            f.write(
                f"v++ --target hw --platform {self.platform} --link {xo_string} "
                f"--config {config_path} --optimize {self.optimization_level} "
                f"--report_level estimate --save-temps --kernel_frequency {self.f_mhz}"
            )

        if not runner_path.exists():
            raise FINNConfigurationError(f"Failed to create config run script at {runner_path}")


class MultiVitisLink(Transformation):
    """Vitis linking transformation explicitly for Multi-FPGA."""

    # TODO: Pass args explicitly, not the whole config
    def __init__(self, cfg: DataflowBuildConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def get_aurora_xos(self, model: ModelWrapper, device: int) -> list[Path]:
        """Get a list of all aurora XO paths for a device"""
        storage_path = model.get_metadata_prop("aurora_storage")
        if storage_path is None:
            raise Exception("Run Aurora kernel packaging beforehand!")  # TODO: Exception
        storage_path = Path(storage_path)
        metadata = AuroraNetworkMetadata(model)
        return [
            storage_path / Path(kernelname + ".xo")
            for kernelname in metadata.get_aurora_kernels(device)
        ]

    def package_dummy_kernels(self) -> tuple[Path, Path]:
        """Prepare dummy kernels that might be needed when a kernel is in duplex mode
        but only needs one connected port. Returns a tuple containing the path to
        the RX kernel .xo and the TX kernel .xo"""
        dummy_kernel_dir = get_settings().finn_deps / "vitis_dummy_kernel"
        rx_dummy = dummy_kernel_dir / "rx_dummy_kernel.xo"
        tx_dummy = dummy_kernel_dir / "tx_dummy_kernel.xo"
        if not rx_dummy.exists() or not tx_dummy.exists():
            subprocess.run(["make"], cwd=dummy_kernel_dir, stdout=subprocess.DEVNULL)
        return rx_dummy, tx_dummy

    def execute_synthesis_parallel(
        self, configs: list[VitisLinkConfiguration], workers: int
    ) -> None:
        """Execute the list of synthesis in parallel. Can be used for faster design space
        exploration or for Multi-FPGA applications.
        This creates the necessary temp dirs by itself as well"""

        if workers < 1:
            raise FINNMultiFPGAConfigError(
                f"Number of synthesis workers set to {workers}. " "Needs to be atleast 1!"
            )
        if workers == 1 and len(configs) > 1:
            log.warning(
                "The number of parallel synthesis workers was set to 1, despite having "
                "multiple synthesis queued up. This may take a long time!"
            )

        def run_link_config(config: VitisLinkConfiguration, index: int) -> None:
            link_dir = Path(make_build_dir(f"parallel_link{index}_"))
            config.generate_config(link_dir / "config.txt")
            config.generate_run_script(link_dir / "config.txt")
            subprocess.run("bash run_vitis_link.sh", shell=True, cwd=link_dir)
            if not (link_dir / "a.xclbin").exists():
                log.critical(
                    f"a.xclbin not found in link directory. "
                    f"Synthesis / implementation (probably) failed. Check {link_dir} "
                    f"and the logs."
                )
            # TODO: Move bitstreams after synthesis

        with ThreadPoolExecutor(max_workers=workers) as tpe:
            tpe.map(run_link_config, configs, list(range(len(configs))))
        tpe.shutdown(wait=True)

    # TODO: Refactor / remove when merging with single-fpga case
    # TODO: since IODMAs might be part of the graph?
    def check_all_sdp(self, model: ModelWrapper, allow_dmas: bool = False) -> None:
        """Check if all nodes in the graph are SDPs. If not raise an error."""
        for sdp in model.graph.node:
            if allow_dmas and sdp.op_type == "IODMA_hls":
                continue
            if sdp.op_type != "StreamingDataflowPartition":
                raise FINNMultiFPGAError(
                    f"Detected non-SDP node in graph when " f"trying to link. Node: {sdp.name}"
                )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        # TODO: Change this when merging with single-fpga case, since
        # TODO: the transform still can be successful without a partitioning config
        # TODO: in that case
        if self.cfg.partitioning_configuration is None:
            raise FINNMultiFPGAError(
                "Cannot do Multi-FPGA link when no partitioning " "configuration is given"
            )

        if (
            self.cfg.vitis_opt_strategy is None
            or self.cfg.vitis_opt_strategy != VitisOptStrategy.PERFORMANCE_BEST
        ):
            log.warning(
                "Consider setting vitis_opt_strategy to PERFORMANCE_BEST to get the best"
                "synthesis / implementation results. Setting strategy to default values."
            )
            self.cfg.vitis_opt_strategy = VitisOptStrategy.DEFAULT

        # Check if all nodes are StreamingDataflowPartitions
        self.check_all_sdp(model, allow_dmas=False)

        # Prepare dummy kernels
        rx_dummy_xo, tx_dummy_xo = self.package_dummy_kernels()

        # All configs, one per device
        configs: dict[int, VitisLinkConfiguration] = {}

        # Switch on the communication kernel used
        match self.cfg.partitioning_configuration.communication_kernel:
            case MFCommunicationKernel.AURORA:
                metadata = AuroraNetworkMetadata(model)
                dummy_kernels_per_device: dict[int, int] = {}
                for i, sdp in enumerate(model.graph.node):
                    this_device = get_device_id(sdp)
                    if this_device is None:
                        raise FINNMultiFPGAConfigError(
                            f"The node {sdp.name} does not have a set "
                            f"device_id attribute. Make sure that "
                            f"CreateMultiFPGAStreamingDataflowPartition,"
                            f" or another SDP creating transformation "
                            f"was run before calling VitisLink!"
                        )

                    # Create a VitisLinkConfiguration
                    if this_device not in configs.keys():
                        configs[this_device] = VitisLinkConfiguration(
                            self.cfg._resolve_vitis_platform(),  # noqa
                            self.cfg.vitis_opt_strategy.value,
                            round(1000 / self.cfg.synth_clk_period_ns),
                        )
                    this_config = configs[this_device]
                    this_config.add_xo(
                        ModelWrapper(getCustomOp(sdp).get_nodeattr("model")).get_metadata_prop(
                            "vitis_xo"
                        )
                    )

                    # Initialize SDP kernel
                    this_config.add_cu(sdp.name, sdp.name)
                    if i in [0, len(model.graph.node)]:
                        # TODO: I/ODMA might not necessarily be on first or last node
                        this_config.add_sp(sdp.name + ".m_axi_gmem0", "HBM[0]")

                    # Get the metadata entry for >this< device and >this< SDP
                    # (there might be multiple SDP on a single device)
                    sending_to = metadata.sends_to_aurora(sdp.name, this_device)
                    receiving_from = metadata.receives_from_aurora(sdp.name, this_device)
                    if len(set(sending_to)) != len(sending_to):
                        raise FINNMultiFPGAError(
                            f"There are multiple Aurora kernels of the same "
                            f"name in the Aurora kernels that SDP {sdp.name} "
                            f"sends to! ({sending_to})"
                        )
                    if len(set(receiving_from)) != len(receiving_from):
                        raise FINNMultiFPGAError(
                            f"There are multiple Aurora kernels of the same "
                            f"name in the Aurora kernels that SDP {sdp.name} "
                            f"receives from!"
                        )

                    # Add aurora kernels
                    this_config.add_xo(self.get_aurora_xos(model, this_device))
                    # Save kernel_name + instance name for each filename
                    aurora_names: dict[str, str] = {}
                    # TODO: Depends on the order of aurora kernels.
                    # TODO: Should not be a problem, but still improve at some point
                    for aurora_number, aurora_kernel in enumerate(sending_to + receiving_from):
                        # TODO: Force aurora to be QSFP attached SLR (SLR2 for U280 for example)
                        aurora_names[aurora_kernel] = f"aurora_flow_{aurora_number}"
                        this_config.add_cu(
                            f"aurora_flow_{aurora_number}", f"aurora_flow_{aurora_number}"
                        )
                        this_config.add_connect(
                            f"io_clk_qsfp{aurora_number}_refclkb_00",
                            f"aurora_flow_{aurora_number}/gt_refclk_{aurora_number}",
                        )
                        this_config.add_connect(
                            f"aurora_flow_{aurora_number}/gt_port", f"io_gt_qsfp{aurora_number}_00"
                        )
                        this_config.add_connect(
                            f"aurora_flow_{aurora_number}/init_clk",
                            "ii_level0_wire/ulp_m_aclk_freerun_ref_00",
                        )

                        if aurora_kernel in sending_to:
                            this_config.add_sc(
                                sdp.name + ".m_axis_0", aurora_names[aurora_kernel] + ".tx_axis"
                            )
                        if aurora_kernel in receiving_from:
                            this_config.add_sc(
                                aurora_names[aurora_kernel] + ".rx_axis", sdp.name + ".s_axis_0"
                            )

                    # Add dummy kernels
                    if this_device not in dummy_kernels_per_device.keys():
                        dummy_kernels_per_device[this_device] = 0
                    open_rx_connections = metadata.get_open_duplex_connections(
                        DataDirection.RX, this_device
                    )
                    open_tx_connections = metadata.get_open_duplex_connections(
                        DataDirection.TX, this_device
                    )
                    for rx_missing_aurora in open_rx_connections:
                        aurora_name = aurora_names[rx_missing_aurora]
                        dummy_instance = f"rx_dummy_kernel_{dummy_kernels_per_device[this_device]}"
                        dummy_kernels_per_device[this_device] += 1
                        this_config.add_cu("rx_dummy_kernel", dummy_instance)
                        this_config.add_sc(aurora_name + ".rx_axis", dummy_instance + ".A")
                        this_config.add_xo(rx_dummy_xo)
                    for tx_missing_aurora in open_tx_connections:
                        aurora_name = aurora_names[tx_missing_aurora]
                        dummy_instance = f"tx_dummy_kernel_{dummy_kernels_per_device[this_device]}"
                        dummy_kernels_per_device[this_device] += 1
                        this_config.add_cu("tx_dummy_kernel", dummy_instance)
                        this_config.add_sc(dummy_instance + ".A", aurora_name + ".tx_axis")
                        this_config.add_xo(tx_dummy_xo)

                    # Add performance optimization directives
                    if self.cfg.vitis_opt_strategy == VitisOptStrategy.PERFORMANCE_BEST:
                        this_config.add_vivado_line(
                            "prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=ExploreWithRemap\n"
                            "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                            "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true\n"
                            "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore\n"
                            "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                        )

            case _:
                raise NotImplementedError()

        # Run synthesis
        self.execute_synthesis_parallel(
            list(configs.values()), self.cfg.partitioning_configuration.parallel_synthesis_workers
        )
        return model, False


class MultiVitisBuild(Transformation):
    """Build Multi-FPGA designs. Will eventually be merged with VitisBuild to unify both single
    and multi FPGA flows"""

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        model = model.transform(MultiVitisLink(self.cfg))
        return model, False


class VitisLink(Transformation):
    """Create an XCLBIN with Vitis.

    Outcome if successful: sets the bitfile attribute in the ONNX
    ModelProto's metadata_props field with the XCLBIN full path as value.
    """

    def __init__(
        self,
        platform,
        f_mhz=200,
        strategy=VitisOptStrategy.PERFORMANCE,
        enable_debug=False,
        fpga_memory_type="default",
    ):
        """Initialize VitisLink transformation with platform and build settings."""
        super().__init__()
        self.platform = platform
        self.f_mhz = f_mhz
        self.strategy = strategy
        self.enable_debug = enable_debug
        self.fpga_memory_type = fpga_memory_type

    def apply(self, model):
        """Apply VitisLink transformation to create XCLBIN."""
        check_vitis_envvars()
        # create a config file and empty list of xo files
        config = ["[connectivity]"]
        object_files = []
        idma_idx = 0
        odma_idx = 0
        mem_idx = 0
        instance_names = {}
        for node in model.graph.node:
            assert node.op_type == "StreamingDataflowPartition", "Invalid link graph"
            sdp_node = getCustomOp(node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_xo = kernel_model.get_metadata_prop("vitis_xo")
            object_files.append(kernel_xo)
            # gather info on connectivity
            # assume each node connected to outputs/inputs is DMA:
            # has axis, aximm and axilite
            # everything else is axis-only
            # assume only one connection from each ip to the next
            if len(node.input) == 0:
                producer = None
            else:
                producer = model.find_producer(node.input[0])
            consumer = model.find_consumers(node.output[0])
            # define kernel instances
            # name kernels connected to graph inputs as idmaxx
            # name kernels connected to graph inputs as odmaxx
            # TODO not a good way of checking for external in/out
            # check top-level in/out list instead
            if producer is None:
                instance_names[node.name] = "idma" + str(idma_idx)
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
                idma_idx += 1
            elif consumer == []:
                instance_names[node.name] = "odma" + str(odma_idx)
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
                odma_idx += 1
            else:
                instance_names[node.name] = node.name
                config.append("nk=%s:1:%s" % (node.name, instance_names[node.name]))
            sdp_node.set_nodeattr("instance_name", instance_names[node.name])
            # explicitly assign SLRs if the slr attribute is not -1
            node_slr = sdp_node.get_nodeattr("slr")
            if node_slr != -1:
                config.append("slr=%s:SLR%d" % (instance_names[node.name], node_slr))
            # assign memory banks
            if producer is None or consumer is None or consumer == []:
                node_mem_port = sdp_node.get_nodeattr("mem_port")
                if node_mem_port == "":
                    if self.fpga_memory_type == FpgaMemoryType.DEFAULT:
                        # configure good defaults based on board
                        if (
                            "u50" in self.platform
                            or "u280" in self.platform
                            or "u55c" in self.platform
                        ):
                            # Use HBM where available (also U50 does not have DDR)
                            mem_type = "HBM"
                            mem_idx = 0
                        elif "u200" in self.platform:
                            # Use DDR controller in static region of U200
                            mem_type = "DDR"
                            mem_idx = 1
                        elif "u250" in self.platform:
                            # Use DDR controller on the node's SLR if set, otherwise 0
                            mem_type = "DDR"
                            if node_slr == -1:
                                mem_idx = 0
                            else:
                                mem_idx = node_slr
                        else:
                            mem_type = "DDR"
                            mem_idx = 1
                    elif self.fpga_memory_type == FpgaMemoryType.HOST_MEM:
                        mem_type = "HOST"
                        mem_idx = 0
                    else:
                        raise RuntimeError(
                            "Unknown fpga memory type: "
                            + str(self.fpga_memory_type)
                            + ". Aborting!"
                        )
                    node_mem_port = "%s[%d]" % (mem_type, mem_idx)
                config.append("sp=%s.m_axi_gmem0:%s" % (instance_names[node.name], node_mem_port))
            # connect streams
            if producer is not None:
                for i in range(len(node.input)):
                    producer = model.find_producer(node.input[i])
                    if producer is not None:
                        j = list(producer.output).index(node.input[i])
                        config.append(
                            "stream_connect=%s.m_axis_%d:%s.s_axis_%d"
                            % (
                                instance_names[producer.name],
                                j,
                                instance_names[node.name],
                                i,
                            )
                        )

        # create a temporary folder for the project
        link_dir = make_build_dir(prefix="vitis_link_proj_")
        model.set_metadata_prop("vitis_link_proj", link_dir)

        # add Vivado physopt directives if desired
        if self.strategy == VitisOptStrategy.PERFORMANCE_BEST:
            config.append("[vivado]")
            config.append("prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=ExploreWithRemap")
            config.append("prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore")
            config.append("prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true")
            config.append("prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore")
            config.append("prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore")

        config = "\n".join(config) + "\n"
        with open(link_dir + "/config.txt", "w") as f:
            f.write(config)

        # create tcl script to generate resource report in XML format
        gen_rep_xml = templates.vitis_gen_xml_report_tcl_template
        gen_rep_xml = gen_rep_xml.replace("$VITIS_PROJ_PATH$", link_dir)
        with open(link_dir + "/gen_report_xml.tcl", "w") as f:
            f.write(gen_rep_xml)

        debug_commands = []
        if self.enable_debug:
            for inst in list(instance_names.values()):
                debug_commands.append("--dk chipscope:%s" % inst)

        # create a shell script and call Vitis
        script = link_dir + "/run_vitis_link.sh"
        working_dir = os.getcwd()
        with open(script, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("set -e\n")
            f.write("cd {}\n".format(link_dir))
            f.write(
                "v++ -t hw --platform %s --link %s"
                " --kernel_frequency %d --config config.txt --optimize %s"
                " --save-temps -R2 %s\n"
                % (
                    self.platform,
                    " ".join(object_files),
                    self.f_mhz,
                    self.strategy.value,
                    " ".join(debug_commands),
                )
            )
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", script]

        try:
            launch_process_helper(bash_command, print_stdout=False)
        except CalledProcessError as e:
            raise FINNSynthesisError(
                f"Linking failed. Check {link_dir} for further details.",
                Path(link_dir) / "vivado.log",
            ) from e
        xclbin = link_dir + "/a.xclbin"
        if not os.path.isfile(xclbin):
            raise FINNSynthesisError(
                "Vitis .xclbin file not created, check logs under %s" % link_dir,
                Path(link_dir) / "vivado.log",
            )

        # TODO rename xclbin appropriately here?
        model.set_metadata_prop("bitfile", xclbin)

        # run Vivado to gen xml report
        gen_rep_xml_sh = link_dir + "/gen_report_xml.sh"
        working_dir = os.getcwd()
        with open(gen_rep_xml_sh, "w") as f:
            f.write("#!/bin/bash \n")
            f.write("cd {}\n".format(link_dir))
            f.write("set -e\n")
            f.write("vivado -mode batch -source %s\n" % (link_dir + "/gen_report_xml.tcl"))
            f.write("cd {}\n".format(working_dir))
        bash_command = ["bash", gen_rep_xml_sh]
        try:
            launch_process_helper(bash_command, print_stdout=False)
        except CalledProcessError:
            log.error(f"Creation of XML reports failed. Check {link_dir} for details. Continuing..")
        # filename for the synth utilization report
        synth_report_filename = link_dir + "/synth_report.xml"
        model.set_metadata_prop("vivado_synth_rpt", synth_report_filename)
        return (model, False)


class VitisBuild(Transformation):
    """Best-effort attempt at building the accelerator with Vitis.
    It assumes the model has only fpgadataflow nodes

    :parameter fpga_part: string identifying the target FPGA
    :parameter period_ns: target clock period
    :parameter platform: target Alveo platform, one of ["U50", "U200", "U250", "U280"]
    :parameter strategy: Vitis optimization strategy
    :parameter enable_debug: add Chipscope to all AXI interfaces
    :parameter floorplan_file: path to a JSON containing a dictionary with
        SLR assignments for each node in the ONNX graph.
        Must be parse-able by the ApplyConfig transform.
    :parameter enable_link: enable linking kernels (.xo files),
        otherwise just synthesize them independently.
    :parameter fpga_memory_type: Specify whether Host or FPGA memory such as DDR/HBM should be used
    """

    def __init__(
        self,
        fpga_part,
        period_ns,
        platform,
        strategy=VitisOptStrategy.PERFORMANCE,
        enable_debug=False,
        floorplan_file=None,
        enable_link=True,
        partition_model_dir=None,
        fpga_memory_type=FpgaMemoryType.DEFAULT,
    ):
        """Initialize VitisBuild transformation with FPGA and build settings."""
        super().__init__()
        self.fpga_part = fpga_part
        self.period_ns = period_ns
        self.platform = platform
        self.strategy = strategy
        self.enable_debug = enable_debug
        self.floorplan_file = floorplan_file
        self.enable_link = enable_link
        self.partition_model_dir = partition_model_dir
        self.fpga_memory_type = fpga_memory_type

    def apply(self, model):
        """Apply VitisBuild transformation to create complete Vitis accelerator."""
        check_vitis_envvars()
        # prepare at global level, then break up into kernels
        prep_transforms = [InsertIODMA(512), InsertDWC(), SpecializeLayers(self.fpga_part)]
        for trn in prep_transforms:
            model = model.transform(trn)
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())

        model = model.transform(Floorplan(floorplan=self.floorplan_file))

        model = model.transform(
            CreateDataflowPartition(partition_model_dir=self.partition_model_dir)
        )
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())

        # Build each kernel individually
        sdp_nodes = model.get_nodes_by_op_type("StreamingDataflowPartition")
        for sdp_node in sdp_nodes:
            prefix = sdp_node.name + "_"
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            kernel_model = ModelWrapper(dataflow_model_filename)
            kernel_model = kernel_model.transform(InsertFIFO())
            kernel_model = kernel_model.transform(SpecializeLayers(self.fpga_part))
            kernel_model = kernel_model.transform(RemoveUnusedTensors())
            kernel_model = kernel_model.transform(GiveUniqueNodeNames(prefix))
            kernel_model.save(dataflow_model_filename)
            kernel_model = kernel_model.transform(PrepareIP(self.fpga_part, self.period_ns))
            kernel_model = kernel_model.transform(HLSSynthIP())
            kernel_model = kernel_model.transform(
                CreateStitchedIP(self.fpga_part, self.period_ns, sdp_node.onnx_node.name, True)
            )
            kernel_model = kernel_model.transform(CreateVitisXO(sdp_node.onnx_node.name))
            kernel_model.set_metadata_prop("platform", "alveo")
            kernel_model.save(dataflow_model_filename)
        # Assemble design from kernels
        if self.enable_link:
            model = model.transform(
                VitisLink(
                    self.platform,
                    round(1000 / self.period_ns),
                    strategy=self.strategy,
                    enable_debug=self.enable_debug,
                    fpga_memory_type=self.fpga_memory_type,
                )
            )
        # set platform attribute for correct remote execution
        model.set_metadata_prop("platform", "alveo")

        return (model, False)
