"""Utils for Multi-FPGA testing."""
import pytest

import brevitas.nn as qnn
import configparser
import hashlib
import onnx.helper as oh
import random
import torch
from brevitas.export import export_qonnx
from brevitas_examples.bnn_pynq.models.resnet import quant_resnet18
from copy import deepcopy
from networkx.classes.digraph import DiGraph
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from testing_util.test import get_test_model
from typing import Any, Callable

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
    model_onnx_path: Path, w: int, a: int, classes: int = 100
) -> tuple[ModelWrapper, Path]:
    """Generate a new ResNet-18 in a fresh build directory, usable for testing."""
    _create_rn18_onnx(model_onnx_path, w, a, classes)
    return ModelWrapper(str(model_onnx_path)), model_onnx_path


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

# ---- MOBILENET ----

mn_pre_multifpga_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_streamline",  # Custom step
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_lower_convs",  # Custom step
    "finn.builder.custom_step_library.mobilenet.step_mobilenet_convert_to_hw_layers_separate_th",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_transpose_decomposition",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
]


def generate_mobilenet(
    model_onnx_path: Path, wbits: int, abits: int, pretrained: bool
) -> ModelWrapper:
    """Provide a mobilenet modelwrapper."""
    fc = get_test_model("mobilenet", wbits, abits, pretrained)
    export_qonnx(fc, torch.randn((1, 3, 224, 224)), str(model_onnx_path))
    qonnx_cleanup(str(model_onnx_path), out_file=str(model_onnx_path))
    return ModelWrapper(str(model_onnx_path))


# ---- BASIC MODELS ----

basic_pre_multifpga_steps = [
    "finn.builder.custom_step_library.general.add_preproc_divide_by_255",  # Custom step
    "finn.builder.custom_step_library.general.add_postproc_top1",  # Custom step
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
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


def generate_basic_model(
    model_onnx_path: Path, typename: str, wbits: int, abits: int, pretrained: bool
) -> ModelWrapper:
    """Provide a basic modelwrapper. This can be one of the TFC, CNV, etc. models."""
    fc = get_test_model(typename, wbits, abits, pretrained)
    ishape = (1, 1, 28, 28)
    if typename == "CNV":
        ishape = (1, 3, 32, 32)
    export_qonnx(fc, torch.randn(ishape), str(model_onnx_path))
    qonnx_cleanup(str(model_onnx_path), out_file=str(model_onnx_path))
    return ModelWrapper(str(model_onnx_path))


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


# ---- CACHE ----


def get_model(
    model: str,
    wbits: int,
    abits: int,
    pretrained: bool,
    until_step: str | None,
    skip_fifo_sizing: bool,
    cfg: DataflowBuildConfig,
    pytestconfig: pytest.Config | None,
    identifier: str | None = None,
) -> tuple[ModelWrapper, DataflowBuildConfig]:  # noqa
    """Get a prepared model.

    Arguments:
    ---------
        `model`: The model to generate. Will be created.
        `wbits`: Weight-bitwidth of the model.
        `abits`: Act-bitwidth of the model.
        `pretrained`: Whether the model should be pretrained.
        `until_step`: The last step until which the FINN flow is executed (inclusive).
            If it is None instead, the model will be returned without any modification.
        `cfg`: The config to use for the transformations.
        `pytestconfig`: If given, this is used to build an identifying string and retrieve
            cached models matching the requested specification.
        `identifier`: (Optional) unique identifier for cache requests, which is appended to the
            internally generated identifier.
        `skip_fifo_sizing`: Whether to skip step_set_fifo_depths.

    Returns:
    -------
        `ModelWrapper, DataflowBuildConfig`: The prepared model and the modified dataflow config.

    Raises:
    ------
        `FINNError`: If the last step is unknown, no valid model(type) is passed, etc.
    """

    def get_test_identifier(steps: list[str | Any]) -> str:
        return (
            f"{identifier}_{model}_{wbits}_{abits}_{pretrained}_"
            f"{skip_fifo_sizing}_{cfg.target_fps}_{cfg.mvau_wwidth_max}_"
            f"{cfg.folding_two_pass_relaxation}_"
            f"{cfg.shell_flow_type.name if cfg.shell_flow_type is not None else 'SHELL'}_"
            f"{cfg.board}_{cfg._resolve_fpga_part()}_{cfg.hls_clk_period_ns}_"
            f"{'_'.join([str(step) for step in steps])}"
        )

    def get_cache_key(identifier: str) -> str:
        """Get the sha256 hash of the identifier, otherwise the cache key is too long (Errno 36)."""
        h = hashlib.new("sha256")
        h.update(identifier.encode("utf-8"))
        return h.hexdigest()

    filename = f"{model}_w{wbits}_a{abits}_pretrained-{pretrained}.onnx"

    # Create the model and fetch the steps
    modelwrapper = None
    steps = []
    match model:
        case "resnet18":
            modelwrapper = generate_rn18(Path(cfg.output_dir) / filename, wbits, abits)[0]
            steps = deepcopy(rn18_pre_multifpga_steps)
        case "mobilenetv1":
            modelwrapper = generate_mobilenet(
                Path(cfg.output_dir) / filename, wbits, abits, pretrained
            )
            steps = deepcopy(mn_pre_multifpga_steps)
        case "LFC" | "TFC" | "SFC" | "CNV":
            modelwrapper = generate_basic_model(
                Path(cfg.output_dir) / filename, model, wbits, abits, pretrained
            )
            steps = deepcopy(basic_pre_multifpga_steps)
        case _:
            raise NotImplementedError(f"Unknown test model: {model}.")

    # Check that we have the steps
    if len(steps) == 0:
        raise NotImplementedError(f"Could not find steps to prepare model of type {model}..")

    # If the last step is not given, we return the unmodified model
    if until_step is None:
        return modelwrapper, cfg

    # Check that the step can be found
    if until_step not in steps:
        raise NotImplementedError(
            f"Cannot prepare until given step {until_step} which "
            f"was not found in the default preparation steps: {steps}"
        )
    # Resolve steps
    cfg.steps = steps
    if skip_fifo_sizing:
        # If the set fifo step is the selected last one and we skip, set the last step to be
        # the one before
        if until_step == "step_set_fifo_depths":
            until_step = str(cfg.steps[cfg.steps.index(until_step) - 1])
        cfg.steps.remove("step_set_fifo_depths")

    # Test the cache
    # Remove steps one by one to find the latest model
    cached_steps = cfg.steps[: cfg.steps.index(until_step) + 1]
    leftover_steps = []
    if pytestconfig is not None:
        while True:
            value = pytestconfig.cache.get(
                get_cache_key(get_test_identifier(cached_steps)), default=None
            )
            if value is not None:
                modelpath = Path(value)
                if modelpath.exists():
                    modelwrapper = ModelWrapper(str(modelpath.absolute()))
                    break

            # If we have checked all steps, break
            if len(cached_steps) == 0:
                break

            # Add the latest step to the list yet to do
            leftover_steps.insert(0, cached_steps.pop())
    else:
        leftover_steps = cached_steps
        cached_steps = []

    # Run all steps
    cache_dir = Path(make_build_dir("model_cache_"))
    all_steps = deepcopy(cfg.steps)
    cfg.steps = leftover_steps
    steps = resolve_build_steps(cfg)
    done = []

    # Build cache.
    # If A and B were already done, and C, D and E are left over,
    # we start with cached_steps = [A,B], leftover_steps = [C,D,E] and done = []
    # We have loaded the model with A and B, and now execute C
    # Afterwards we store the cache key "identifier_A_B_C"
    # and have cached_steps = [A,B], leftover_steps = [D,E] and done = [C]
    # ...
    # cached_steps = [A,B], leftover_steps = [], done = [C,D,E], last model saved
    # is then "identifier_A_B_C_D_E"
    for i, step in enumerate(steps):
        # Execute the new step
        modelwrapper = step(modelwrapper, cfg)

        # From this generate the newest identifier
        done.append(leftover_steps[i])
        test_identifier = get_test_identifier(cached_steps + done)

        # Store the model with its identifier in the cache, using a pseudo-random filename
        fn = cache_dir / ("".join([random.choice("ABCDEF0123456789") for _ in range(30)]) + ".onnx")
        modelwrapper.save(str(fn))
        if pytestconfig is not None:
            pytestconfig.cache.set(get_cache_key(test_identifier), str(fn))

    # Restore steps
    cfg.steps = all_steps
    return modelwrapper, cfg
