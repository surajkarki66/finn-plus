"""Create SDPs for Multi-FPGA usage."""

from __future__ import annotations

import yaml
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
from finn.util.exception import FINNMultiFPGAUserError
from finn.util.logging import log

if TYPE_CHECKING:
    from pathlib import Path
    from qonnx.core.modelwrapper import ModelWrapper


class CreateMultiFPGAStreamingDataflowPartition(Transformation):
    """Operates like CreateDataflowPartition but using the nodes device id as a key. Additionally,
    two non consecutive instances on the same device create
    different SDPs (think for example about a there-and-back topology).

    IMPORTANT: Currently this assumes that every branch is split and joined on the same device.
    """

    def __init__(  # noqa
        self, separate_iodmas: bool, dataflow_partition_directory: Path, verbosity: MFVerbosity
    ) -> None:
        """Create one SDP per all consecutive layers.

        Arguments:
        ---------
            `separate_iodmas`: If true, IODMA nodes (if detected) receive their own partition
                ID / SDP. Their device ID stays unchanged.

            `dataflow_partition_directory`: Directory in which to store logs and kernel partitions.

            `verbosity`: How verbose the transform should be.
        """
        super().__init__()
        self.verbosity = verbosity
        self.separate_iodmas = separate_iodmas
        self.cdfp_dir = dataflow_partition_directory

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:  # noqa
        for node in model.graph.node:
            # Check that all nodes have an device ID
            if get_device_id(node) is None:
                raise FINNMultiFPGAUserError(
                    f"Cannot create StreamingDataflowParititions "
                    f"for Multi-FPGA without an assigned device "
                    f"ID. (Node: {node.name}). Did you forget "
                    f"to run partitioning first?"
                )
            # Check that we have no SDPs yet
            if node.op_type in ["StreamingDataflowPartition", "GenericPartition"]:
                raise FINNMultiFPGAUserError(
                    f"Cannot create SDPs in graph: Node "
                    f"{node.name} is already a dataflow partition."
                )

        # Prepare everything
        current_device = get_device_id(model.graph.node[0])
        current_max = 0 if not self.separate_iodmas else 1
        mapping = {}
        total = 0

        # Go through every node
        for node in model.graph.node:
            if current_max not in mapping:
                mapping[current_max] = []

            # Consider IODMAs
            if "IODMA" in node.op_type and self.separate_iodmas:
                if model.find_direct_predecessors(node) is None:
                    getCustomOp(node).set_nodeattr("partition_id", 0)
                    mapping[0] = [{"device": get_device_id(node), "node": node.name}]
                    total += 1
                if model.find_direct_successors(node) is None:
                    last_id = len(model.graph.node) + 1
                    getCustomOp(node).set_nodeattr("partition_id", last_id)
                    mapping[last_id] = [{"device": get_device_id(node), "node": node.name}]
                    total += 1
                continue  # without changing the device number

            # Save for logging purposes
            mapping[current_max].append({"device": current_device, "node": node.name})

            # Get the new device number to check whether it changed
            device = get_device_id(node)

            # TODO: Setting partition_id and calling CreateDataflowPartitions might not be
            # the best way to do it. Maybe change at some point
            if device != current_device:
                current_device = device
                current_max += 1

            # Set the partition ID
            getCustomOp(node).set_nodeattr("partition_id", current_max)
            total += 1

        if self.verbosity.value > MFVerbosity.NONE.value:
            log.info(f"Creating a total of {total} StreamingDataflowPartitions...")

        # Write partition ID <-> Device+Node name mapping into a human readable file for
        # debugging
        sdp_logfile = self.cdfp_dir / "partition_id_mapping.yaml"
        if not self.cdfp_dir.exists():
            self.cdfp_dir.mkdir(parents=True)
        with sdp_logfile.open("w+") as f:
            yaml.dump(mapping, f, yaml.Dumper)

        if self.verbosity.value > MFVerbosity.LOW.value:
            log.info(f"Storing SDP mapping at: {sdp_logfile}")

        # Create the SDFPs
        model = model.transform(CreateDataflowPartition(str(self.cdfp_dir)))
        model = model.transform(GiveUniqueNodeNames())

        # Set the SDP's device_id
        for node in model.graph.node:
            device_id = get_device_id(get_submodel(node).graph.node[0])
            assert device_id is not None
            set_device_id(node, device_id)
        return model, False
