"""Create SDPs for Multi-FPGA usage."""

from __future__ import annotations

import yaml
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import TYPE_CHECKING

from finn.builder.build_dataflow_config import MFVerbosity
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.transformation.fpgadataflow.multifpga.utils import (
    get_device_id,
    get_submodel,
    set_device_id,
)
from finn.util.basic import make_build_dir
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


class CreateMultiFPGAStreamingDataflowPartition(Transformation):
    """Operates like CreateDataflowPartition but using the nodes device id as a key. Additionally,
    two non consecutive instances on the same device create
    different SDPs (think for example about a there-and-back topology).

    IMPORTANT: Currently this assumes that every branch is split and joined on the same device.
    """

    def __init__(self, verbosity: MFVerbosity) -> None:  # noqa
        super().__init__()
        self.verbosity = verbosity

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:  # noqa
        current_device = get_device_id(model.graph.node[0])
        current_max = 0
        mapping = {}
        for node in model.graph.node:
            assert node.op_type not in ["StreamingDataflowPartition", "GenericPartition"]
            device = get_device_id(node)
            assert device is not None, f"Node {node.name} of type {node.op_type} does not have"
            "a device_id attribute"
            # TODO: Setting partition_id and calling CreateDataflowPartitions might not be
            # the best way to do it. Maybe change at some point
            if device != current_device:
                current_device = device
                current_max += 1
            getCustomOp(node).set_nodeattr("partition_id", current_max)
            if current_max not in mapping:
                mapping[current_max] = []
            mapping[current_max].append({"device": current_device, "node": node.name})

        if self.verbosity.value > MFVerbosity.NONE.value:
            log.info(f"Creating a total of {current_max} StreamingDataflowPartitions...")

        # Write partition ID <-> Device+Node name mapping into a human readable file for
        # debugging
        cdfp_dir = Path(make_build_dir("dataflow_multifpga_partition"))
        sdp_logfile = cdfp_dir / "partition_id_mapping.yaml"
        with sdp_logfile.open("w+") as f:
            yaml.dump(mapping, f, yaml.Dumper)

        if self.verbosity.value > MFVerbosity.LOW.value:
            log.info(f"Storing SDP mapping at: {sdp_logfile}")

        # Create the SDFPs
        model = model.transform(CreateDataflowPartition(str(cdfp_dir)))
        model = model.transform(GiveUniqueNodeNames())

        # Set the SDP's device_id
        for node in model.graph.node:
            device_id = get_device_id(get_submodel(node).graph.node[0])
            assert device_id is not None
            set_device_id(node, device_id)
        return model, False
