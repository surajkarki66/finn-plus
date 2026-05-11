"""Utils for Multi-FPGA testing."""
import pytest

import brevitas.nn as qnn
import configparser
import onnx.helper as oh
import torch
from brevitas.export import export_qonnx
from brevitas_examples.bnn_pynq.models.resnet import quant_resnet18
from networkx.classes.digraph import DiGraph
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper

from finn.builder.build_dataflow import resolve_build_steps
from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.builder.custom_step_library.resnet import (
    step_resnet_convert_to_hw,
    step_resnet_streamline,
    step_resnet_tidy,
)
from finn.util.basic import make_build_dir


# ---- RESNET 18 ----


class RN18(torch.nn.Module):
    """A simple ResNet-18 with an input quantizer."""

    def __init__(self, cfg: configparser.ConfigParser) -> None:
        """Create a simple ResNet-18 from scratch with the given config."""
        super().__init__()
        self.inpQuantizer = qnn.QuantIdentity(bit_width=8, return_quant_tensor=True)
        self.resnet = quant_resnet18(cfg)

    def forward(self, x):  # noqa
        x = self.inpQuantizer(x)
        return self.resnet(x)


def _create_rn18_model(w: int, a: int, classes: int = 100) -> RN18:
    """Create a ResNet-18 (Brevitas model) with the given weight and activation bitwidths."""
    cfg = configparser.ConfigParser()
    cfg["MODEL"] = {"NUM_CLASSES": str(classes)}
    cfg["QUANT"] = {"WEIGHT_BIT_WIDTH": str(w), "ACT_BIT_WIDTH": str(a)}
    return RN18(cfg)


def _create_rn18_onnx(path: Path, w: int, a: int, classes: int = 100) -> None:
    """Create a ResNet-18 and export as a QONNX model to the given path."""
    model = _create_rn18_model(w, a, classes)
    model.eval()
    inp = torch.zeros((1, 3, 32, 32))
    _ = model(inp)
    export_qonnx(model, (inp,), str(path.absolute()))


def generate_rn18(
    generation_dir_prefix: str, w: int, a: int, classes: int = 100
) -> tuple[ModelWrapper, Path]:
    """Generate a new ResNet-18 in a fresh build directory, usable for testing."""
    rn18_path = Path(make_build_dir(generation_dir_prefix + "_")) / "rn18.onnx"
    _create_rn18_onnx(rn18_path, w, a, classes)
    return ModelWrapper(str(rn18_path)), rn18_path


@pytest.mark.multifpga
@pytest.mark.parametrize("w", [1, 2, 4, 8])
@pytest.mark.parametrize("a", [1, 2, 4, 8])
def test_rn18_generation(w: int, a: int) -> None:
    """Test that the RN18 is created correctly and can be loaded by a
    QONNX modelwrapper.
    """
    model, path = generate_rn18("test_rn18_generation", w, a)
    assert path.exists()
    assert model is not None


# Steps for a resnet to execute before reaching the point at which Multi-FPGA can be started
rn18_pre_multifpga_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    step_resnet_tidy,
    step_resnet_streamline,
    "step_convert_to_hw",
    step_resnet_convert_to_hw,
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]
"""Steps required to prepare the ResNet18 model for Multi-FPGA usage."""


def prepare_resnet_for_multifpga(
    model: ModelWrapper, cfg: DataflowBuildConfig, skip_fifo_sizing: bool = False
) -> tuple[ModelWrapper, DataflowBuildConfig]:
    """Run all steps until the model is ready for Multi-FPGA. For ResNets.
    Changes the configs steps and returns the new configuration.
    """
    cfg.steps = rn18_pre_multifpga_steps
    if skip_fifo_sizing:
        cfg.steps.remove("step_set_fifo_depths")
    steps = resolve_build_steps(cfg)
    for step in steps:
        model = step(model, cfg)
    return model, cfg


# ---- GRAPH UTILITY ----


def list_contains_all_elements(this: list, other: list) -> bool:
    """Return whether a list contain all elements of another list."""
    return all(n in this for n in other)


def networkx_to_onnx(g: DiGraph) -> ModelWrapper:
    """Convert a networkx graph into a QONNX ModelWrapper.
    All nodes will be StreamingDataflowPartitions.
    """
    nodes = [oh.make_node("StreamingDataflowPartition", [], [], n) for n in g.nodes]
    get_node_by_name = lambda name: [n for n in nodes if n.name == name][0]  # noqa
    for i, edge in enumerate(g.edges):
        source_node = get_node_by_name(edge[0])
        target_node = get_node_by_name(edge[1])
        source_node.output.append(f"edge_{i}")
        target_node.input.append(f"edge_{i}")
    # TODO: Make graph inputs and outputs
    graph = oh.make_graph(nodes, "graph", [], [])
    model = oh.make_model(graph)
    return ModelWrapper(model)
