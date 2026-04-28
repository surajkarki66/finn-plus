"""Unit tests for SimulationBuilder isolated node model generation."""

from __future__ import annotations

import pytest

import numpy as np
import os
import sys
import types
from onnx import GraphProto, NodeProto, TensorProto, ValueInfoProto, helper
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from typing import Protocol


class _SimulationBuilderProtocol(Protocol):
    def __init__(self, model: ModelWrapper, fpgapart: str, clk_ns: float) -> None:
        ...

    def _isolated_node_model(self, by_node: int | str) -> ModelWrapper:
        ...


def _import_simulation_build_types() -> tuple[type[_SimulationBuilderProtocol], type[Exception]]:
    finn_xsi_stub_dir = Path("/tmp/finn_xsi_stub")
    finn_xsi_stub_dir.mkdir(parents=True, exist_ok=True)
    (finn_xsi_stub_dir / "xsi.so").touch(exist_ok=True)
    os.environ.setdefault("FINN_XSI", str(finn_xsi_stub_dir))

    finn_xsi_module = types.ModuleType("finn_xsi")
    finn_xsi_module.__path__ = []
    finn_xsi_adapter_module = types.ModuleType("finn_xsi.adapter")

    def _get_simkernel_so() -> str:
        return ""

    finn_xsi_adapter_module.__dict__["get_simkernel_so"] = _get_simkernel_so
    finn_xsi_sim_engine_module = types.ModuleType("finn_xsi.sim_engine")

    class _SimEngine:
        pass

    finn_xsi_sim_engine_module.__dict__["SimEngine"] = _SimEngine
    finn_xsi_module.__dict__["adapter"] = finn_xsi_adapter_module
    finn_xsi_module.__dict__["sim_engine"] = finn_xsi_sim_engine_module
    sys.modules.setdefault("finn_xsi", finn_xsi_module)
    sys.modules.setdefault("finn_xsi.adapter", finn_xsi_adapter_module)
    sys.modules.setdefault("finn_xsi.sim_engine", finn_xsi_sim_engine_module)

    scipy_module = types.ModuleType("scipy")
    scipy_special_module = types.ModuleType("scipy.special")

    def _softmax(x: np.ndarray, axis: int | None = None) -> np.ndarray:
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    scipy_special_module.__dict__["softmax"] = _softmax
    scipy_module.__dict__["special"] = scipy_special_module
    sys.modules.setdefault("scipy", scipy_module)
    sys.modules.setdefault("scipy.special", scipy_special_module)

    from finn.transformation.fpgadataflow.simulation_build import SimulationBuilder
    from finn.util.exception import FINNInternalError

    return SimulationBuilder, FINNInternalError


def _vi(name: str, shape: list[int]) -> ValueInfoProto:
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _make_dwc(name: str, inp: str, out: str, shape: list[int]) -> NodeProto:
    return helper.make_node(
        "StreamingDataWidthConverter",
        [inp],
        [out],
        domain="finn.custom_op.fpgadataflow",
        backend="fpgadataflow",
        inShape=list(shape),
        outShape=list(shape),
        inWidth=8,
        outWidth=8,
        dataType="INT8",
        preferred_impl_style="rtl",
        name=name,
    )


def _make_add_hls(name: str, lhs: str, rhs: str, out: str, shape: list[int]) -> NodeProto:
    return helper.make_node(
        "ElementwiseAdd_hls",
        [lhs, rhs],
        [out],
        domain="finn.custom_op.fpgadataflow.hls",
        backend="fpgadataflow",
        numInputVectors=[1],
        lhs_shape=list(shape),
        rhs_shape=list(shape),
        out_shape=list(shape),
        lhs_dtype="INT8",
        rhs_dtype="INT8",
        out_dtype="INT9",
        lhs_style="input",
        rhs_style="input",
        PE=1,
        name=name,
    )


def _make_mvau_rtl(name: str, inp: str, weights: str, out: str) -> NodeProto:
    return helper.make_node(
        "MVAU_rtl",
        [inp, weights],
        [out],
        domain="finn.custom_op.fpgadataflow.rtl",
        backend="fpgadataflow",
        MW=4,
        MH=4,
        SIMD=1,
        PE=1,
        inputDataType="INT8",
        weightDataType="INT8",
        outputDataType="INT32",
        ActVal=0,
        binaryXnorMode=0,
        noActivation=1,
        name=name,
    )


def _wrap_model(graph: GraphProto) -> ModelWrapper:
    return ModelWrapper(qonnx_make_model(graph, producer_name="simulation-build-test"))


def _build_unary_target_model(pre_binary: bool = False, succ_binary: bool = False) -> ModelWrapper:
    shape = [1, 4]
    nodes = []
    graph_inputs = []
    graph_outputs = [_vi("graph_out", shape)]
    value_info = [_vi("target_in", shape), _vi("target_out", shape)]

    if pre_binary:
        graph_inputs.extend([_vi("pre_in0", shape), _vi("pre_in1", shape)])
        nodes.append(_make_add_hls("pre_add", "pre_in0", "pre_in1", "target_in", shape))
    else:
        graph_inputs.append(_vi("pre_in0", shape))
        nodes.append(_make_dwc("pre_dwc", "pre_in0", "target_in", shape))

    nodes.append(_make_dwc("target_dwc", "target_in", "target_out", shape))

    if succ_binary:
        graph_inputs.append(_vi("succ_in1", shape))
        nodes.append(_make_add_hls("succ_add", "target_out", "succ_in1", "graph_out", shape))
    else:
        nodes.append(_make_dwc("succ_dwc", "target_out", "graph_out", shape))

    graph = helper.make_graph(
        nodes=nodes,
        name="unary_target_graph",
        inputs=graph_inputs,
        outputs=graph_outputs,
        value_info=value_info,
    )
    return _wrap_model(graph)


def _build_binary_target_model(
    initializer_side: str | None = None, mlo: bool = False
) -> ModelWrapper:
    shape = [1, 4]
    lhs_name = "lhs_in"
    rhs_name = "rhs_in"
    nodes = [_make_add_hls("target_add", lhs_name, rhs_name, "target_out", shape)]
    nodes.append(_make_dwc("succ_dwc", "target_out", "graph_out", shape))

    graph_inputs = []
    if initializer_side != "lhs":
        graph_inputs.append(_vi(lhs_name, shape))
    if initializer_side != "rhs":
        graph_inputs.append(_vi(rhs_name, shape))

    value_info = [_vi("target_out", shape)]
    if initializer_side == "lhs":
        value_info.append(_vi(lhs_name, shape))
    if initializer_side == "rhs":
        value_info.append(_vi(rhs_name, shape))

    graph = helper.make_graph(
        nodes=nodes,
        name="binary_target_graph",
        inputs=graph_inputs,
        outputs=[_vi("graph_out", shape)],
        value_info=value_info,
    )
    model = _wrap_model(graph)

    if initializer_side is not None:
        init_name = lhs_name if initializer_side == "lhs" else rhs_name
        model.set_initializer(init_name, np.ones(shape, dtype=np.float32))

    if mlo:
        model.set_metadata_prop("is_mlo", "1")
        mlo_inputs = [rhs_name] if initializer_side == "rhs" else [lhs_name]
        model.set_metadata_prop("mlo_input_parameter_names", str(mlo_inputs))

    return model


def _build_mvau_target_model(mlo: bool = False) -> ModelWrapper:
    shape_ifm = [1, 1, 1, 4]
    shape_out = [1, 1, 1, 4]
    shape_w = [4, 4]

    graph = helper.make_graph(
        nodes=[
            _make_mvau_rtl("target_mvau", "ifm", "weights", "mvau_out"),
            _make_dwc("succ_dwc", "mvau_out", "graph_out", shape_out),
        ],
        name="mvau_target_graph",
        inputs=[_vi("ifm", shape_ifm), _vi("weights", shape_w)],
        outputs=[_vi("graph_out", shape_out)],
        value_info=[_vi("mvau_out", shape_out)],
    )
    model = _wrap_model(graph)

    if mlo:
        model.set_metadata_prop("is_mlo", "1")
        model.set_metadata_prop("mlo_input_parameter_names", str(["weights"]))

    return model


def _assert_isolated_model(
    isolated_model: ModelWrapper,
    target_name: str,
    expected_graph_inputs: list[str],
    expected_graph_outputs: list[str],
    expected_initializer_inputs: list[str],
    expected_input_node_flag: bool,
    expected_target_inputs: list[str],
    expected_target_outputs: list[str],
) -> None:
    graph = isolated_model.graph
    graph_input_names = [x.name for x in graph.input]
    graph_output_names = [x.name for x in graph.output]

    assert graph_input_names == expected_graph_inputs
    assert graph_output_names == expected_graph_outputs

    input_dummy_nodes = [
        n for n in graph.node if n.op_type == "RemoveDataPath_rtl" and "_input_dummy_" in n.name
    ]
    output_dummy_nodes = [
        n for n in graph.node if n.op_type == "RemoveDataPath_rtl" and "_output_dummy_" in n.name
    ]
    target_nodes = [n for n in graph.node if n.name == target_name]

    assert len(target_nodes) == 1
    assert len(input_dummy_nodes) == len(expected_graph_inputs)
    assert len(output_dummy_nodes) == 1

    initializer_names = [x.name for x in graph.initializer]
    assert initializer_names == expected_initializer_inputs

    target_node = target_nodes[0]
    assert list(target_node.input) == expected_target_inputs
    assert list(target_node.output) == expected_target_outputs
    target_dummy_inputs = [inp for inp in target_node.input if inp.endswith("_dummy")]
    target_initializer_inputs = [inp for inp in target_node.input if inp in initializer_names]
    assert len(target_dummy_inputs) == len(expected_graph_inputs)
    assert target_initializer_inputs == expected_initializer_inputs

    assert isolated_model.get_metadata_prop("predecessors") == str(expected_graph_inputs)
    assert isolated_model.get_metadata_prop("successors") == str(graph_output_names)
    assert isolated_model.get_metadata_prop("input_node") == str(expected_input_node_flag).lower()
    assert isolated_model.get_metadata_prop("output_node") == "false"


def _isolate_node_model(builder: _SimulationBuilderProtocol, by_node: int | str) -> ModelWrapper:
    return builder._isolated_node_model(by_node)  # noqa: SLF001


@pytest.mark.parametrize(
    "pre_binary,succ_binary",
    [
        (False, False),
        (True, True),
    ],
)
def test_isolated_node_model_unary_target_with_varied_other_node_inputs(
    pre_binary: bool, succ_binary: bool
) -> None:
    """Isolate unary target with unary/binary surrounding nodes."""
    simulation_builder_cls, _ = _import_simulation_build_types()
    model = _build_unary_target_model(pre_binary=pre_binary, succ_binary=succ_binary)
    builder = simulation_builder_cls(model, "xc7z020clg400-1", 5.0)

    isolated = _isolate_node_model(builder, 1)

    _assert_isolated_model(
        isolated_model=isolated,
        target_name="target_dwc",
        expected_graph_inputs=["target_in"],
        expected_graph_outputs=["target_out"],
        expected_initializer_inputs=[],
        expected_input_node_flag=False,
        expected_target_inputs=["target_in_dummy"],
        expected_target_outputs=["target_out_dummy"],
    )


def test_isolated_node_model_select_by_name() -> None:
    """Selecting node by name returns the correct isolated model."""
    simulation_builder_cls, _ = _import_simulation_build_types()
    model = _build_unary_target_model(pre_binary=False, succ_binary=False)
    builder = simulation_builder_cls(model, "xc7z020clg400-1", 5.0)

    isolated = _isolate_node_model(builder, "target_dwc")

    _assert_isolated_model(
        isolated_model=isolated,
        target_name="target_dwc",
        expected_graph_inputs=["target_in"],
        expected_graph_outputs=["target_out"],
        expected_initializer_inputs=[],
        expected_input_node_flag=False,
        expected_target_inputs=["target_in_dummy"],
        expected_target_outputs=["target_out_dummy"],
    )


@pytest.mark.parametrize(
    "initializer_side,mlo,expected_graph_inputs,expected_initializer_inputs,expected_target_inputs",
    [
        (None, False, ["lhs_in", "rhs_in"], [], ["lhs_in_dummy", "rhs_in_dummy"]),
        ("rhs", False, ["lhs_in"], ["rhs_in"], ["lhs_in_dummy", "rhs_in"]),
        ("lhs", False, ["rhs_in"], ["lhs_in"], ["lhs_in", "rhs_in_dummy"]),
        ("rhs", True, ["lhs_in"], ["rhs_in"], ["lhs_in_dummy", "rhs_in"]),
        (None, True, ["rhs_in"], ["lhs_in"], ["lhs_in", "rhs_in_dummy"]),
    ],
)
def test_isolated_node_model_binary_target_with_dynamic_and_fixed_inputs(
    initializer_side: str | None,
    mlo: bool,
    expected_graph_inputs: list[str],
    expected_initializer_inputs: list[str],
    expected_target_inputs: list[str],
) -> None:
    """Isolate binary target for dynamic/fixed lhs-rhs and MLO/non-MLO cases."""
    simulation_builder_cls, _ = _import_simulation_build_types()
    model = _build_binary_target_model(initializer_side=initializer_side, mlo=mlo)
    builder = simulation_builder_cls(model, "xc7z020clg400-1", 5.0)

    isolated = _isolate_node_model(builder, 0)

    _assert_isolated_model(
        isolated_model=isolated,
        target_name="target_add",
        expected_graph_inputs=expected_graph_inputs,
        expected_graph_outputs=["target_out"],
        expected_initializer_inputs=expected_initializer_inputs,
        expected_input_node_flag=True,
        expected_target_inputs=expected_target_inputs,
        expected_target_outputs=["target_out_dummy"],
    )


def test_isolated_node_model_elementwise_sets_const_style_for_mlo_initializer() -> None:
    """Elementwise ops set lhs_style/rhs_style=const for remapped MLO initializer inputs."""
    simulation_builder_cls, _ = _import_simulation_build_types()
    model = _build_binary_target_model(initializer_side=None, mlo=True)
    builder = simulation_builder_cls(model, "xc7z020clg400-1", 5.0)

    isolated = _isolate_node_model(builder, 0)
    target_node = next(n for n in isolated.graph.node if n.name == "target_add")
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in target_node.attribute}
    lhs_style = (
        attrs["lhs_style"].decode() if isinstance(attrs["lhs_style"], bytes) else attrs["lhs_style"]
    )
    rhs_style = (
        attrs["rhs_style"].decode() if isinstance(attrs["rhs_style"], bytes) else attrs["rhs_style"]
    )

    assert lhs_style == "const"
    assert rhs_style == "input"


def test_isolated_node_model_mvau_sets_internal_decoupled_for_initializer_input() -> None:
    """MVAU ops set mem_mode=internal_decoupled when an input is remapped to initializer."""
    simulation_builder_cls, _ = _import_simulation_build_types()
    model = _build_mvau_target_model(mlo=True)
    builder = simulation_builder_cls(model, "xc7z020clg400-1", 5.0)

    isolated = _isolate_node_model(builder, 0)
    target_node = next(n for n in isolated.graph.node if n.name == "target_mvau")
    attrs = {attr.name: helper.get_attribute_value(attr) for attr in target_node.attribute}
    mem_mode = (
        attrs["mem_mode"].decode() if isinstance(attrs["mem_mode"], bytes) else attrs["mem_mode"]
    )

    assert mem_mode == "internal_decoupled"


def test_isolated_node_model_rejects_bad_mlo_metadata() -> None:
    """Reject invalid mlo_input_parameter_names metadata values."""
    simulation_builder_cls, finn_internal_error_cls = _import_simulation_build_types()
    model = _build_binary_target_model(initializer_side="rhs", mlo=False)
    model.set_metadata_prop("is_mlo", "1")
    model.set_metadata_prop("mlo_input_parameter_names", "42")
    builder = simulation_builder_cls(model, "xc7z020clg400-1", 5.0)

    with pytest.raises(finn_internal_error_cls, match="mlo_input_parameter_names"):
        _isolate_node_model(builder, 0)
