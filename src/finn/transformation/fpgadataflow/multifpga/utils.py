"""Utility functions for Multi-FPGA uses."""

from __future__ import annotations

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from typing import TYPE_CHECKING, cast

from finn.util.exception import FINNMultiFPGAError

if TYPE_CHECKING:
    from onnx import NodeProto


def get_device_id(node: NodeProto) -> int | None:
    """Return the node's device ID. If no nodeattribute exists returns None."""
    try:
        return cast("int", (getCustomOp(node).get_nodeattr("device_id")))
    except ValueError:
        return None


def set_device_id(node: NodeProto, value: int) -> None:
    """Set the device_id nodeattribute of the given node."""
    getCustomOp(node).set_nodeattr("device_id", value)


def get_submodel(node: NodeProto) -> ModelWrapper:
    """Attempt to get the submodule of the given node."""
    try:
        modelname = cast("str", getCustomOp(node).get_nodeattr("model"))
    except ValueError as e:
        raise FINNMultiFPGAError(
            f"Node {node.name} has no submodel " f"(a 'model' nodeattribute to be specific)."
        ) from e
    return ModelWrapper(modelname)


def get_last_submodel_node(sdp_node: NodeProto) -> NodeProto:
    """Return the last node of the submodel of the parent node.
    IMPORTANT: This is not necessarily the only output/end-node.
    """
    return get_submodel(sdp_node).graph.node[-1]


def get_first_submodel_node(sdp_node: NodeProto) -> NodeProto:
    """Return the frist node of the submodel of the parent node.
    IMPORTANT: This is not necessarily the only input/start-node.
    """
    return get_submodel(sdp_node).graph.node[0]
