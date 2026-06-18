"""Create SDPs for Multi-FPGA usage."""

from __future__ import annotations

from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from typing import TYPE_CHECKING

from finn.builder.build_dataflow_config import MFVerbosity
from finn.transformation.fpgadataflow.create_dataflow_partition import CreateDataflowPartition
from finn.util.exception import FINNInternalError
from finn.util.fpgadataflow import get_device_id, get_submodel, set_device_id
from finn.util.logging import log

if TYPE_CHECKING:
    from pathlib import Path
    from qonnx.core.modelwrapper import ModelWrapper


class ClusterByNodeattribute(Transformation):
    """Cluster nodes by a "comparing attribute". If two nodes are clustered together, they
    have their "partition attribute" set to the same value. Can be run by `transform` until no more
    merges happen, then the graph is fully partitioned by the given attribute.

    Requirements:
        - An attribute that can be compared on equality (==)
        - A partition attribute of type `int`. Every node will receive a unique partition ID
            upon first call.
    """

    def __init__(self, compare_attribute: str, partition_attribute: str = "partition_id") -> None:
        """Cluster by comparison attribute, by setting partition_attribute."""
        self.comp = compare_attribute
        self.part = partition_attribute
        self.first_call = True

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Merge nodes with their neighbors. If atleast one merge happens, return modified=True."""
        # Check that all compare attributes exist
        for i, node in enumerate(model.graph.node):
            try:
                _ = getCustomOp(node).get_nodeattr(self.comp)
            except AttributeError as e:
                raise FINNInternalError(
                    f"Cannot cluster because "
                    f"comparison node attribute "
                    f"'{self.comp}' is  missing "
                    f"on at least one node ({node.name})."
                ) from e
            if self.first_call:
                try:
                    getCustomOp(node).set_nodeattr(self.part, i)
                except AttributeError as e:
                    raise FINNInternalError(
                        f"Cannot initialize partition attribute '{self.part}' "
                        f"on at least one node ({node.name}). Make sure the "
                        f"attribute exists and is of type 'int'."
                    ) from e

        # Dont re-initialize partition attributes again
        self.first_call = False

        # Merge
        modified = False
        for node in model.graph.node:
            node_op = getCustomOp(node)
            pre = model.find_direct_predecessors(node)
            suc = model.find_direct_successors(node)
            if pre is None:
                pre = []
            if suc is None:
                suc = []
            neighbors = pre + suc
            for neighbor in neighbors:
                neighbor_op = getCustomOp(neighbor)
                if node_op.get_nodeattr(self.comp) == neighbor_op.get_nodeattr(
                    self.comp
                ) and node_op.get_nodeattr(self.part) != neighbor_op.get_nodeattr(self.part):
                    neighbor_op.set_nodeattr(self.part, node_op.get_nodeattr(self.part))
                    modified = True
        return model, modified


class CreateMultiFPGAStreamingDataflowPartition(Transformation):
    """Create SDPs for Multi-FPGA models. This is done by clustering all nodes according to their
    device ID. The nodes are then packed into SDPs which have the
    same device ID as their submodel nodes.

    For different, separated sections of the model with the same device ID,
    the transformation creates separate partitions. Thus a model with
    devices A -> A -> B -> A will have 3 SDPs: 0 (A) -> 1 (B) -> 2 (A).
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
        # Cluster the partition IDs
        model = model.transform(ClusterByNodeattribute("device_id", "partition_id"))

        # Create separate IODMAs
        all_ids = [getCustomOp(node).get_nodeattr("partition_id") for node in model.graph.node]
        if self.separate_iodmas:
            for node in model.graph.node:
                if node.op_type == "IODMA_hls":
                    current_max = max(all_ids)
                    getCustomOp(node).set_nodeattr("partition_id", current_max + 1)
                    all_ids.append(current_max + 1)
        if self.verbosity.value > MFVerbosity.MEDIUM.value:
            log.info(
                f"Clustered the graph into {len(set(all_ids))} partitions according to device ID."
            )

        # Create the SDFPs
        model = model.transform(CreateDataflowPartition(str(self.cdfp_dir)))
        model = model.transform(GiveUniqueNodeNames())

        # Set the SDP's device_id
        for node in model.graph.node:
            device_id = get_device_id(get_submodel(node)[0].graph.node[0])
            assert device_id is not None
            set_device_id(node, device_id)
        return model, False
