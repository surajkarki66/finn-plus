"""Test Multi-FPGA Metadata creation and correctness."""

from __future__ import annotations

import pytest

from fpgadataflow.multifpga.utils import generate_rn18, prepare_resnet_for_multifpga
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from typing import TYPE_CHECKING, cast

from finn.builder import build_dataflow_steps
from finn.builder.build_dataflow import resolve_build_steps
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    MFCommunicationKernel,
    MFTopology,
    PartitioningConfiguration,
    PartitioningStrategy,
    ShellFlowType,
)
from finn.builder.build_dataflow_steps import step_partition_for_multifpga
from finn.builder.custom_step_library.resnet import (
    step_resnet_convert_to_hw,
    step_resnet_streamline,
    step_resnet_tidy,
)
from finn.transformation.fpgadataflow.multifpga.utils import get_estimated_model_resources
from finn.util.basic import make_build_dir
from finn.util.exception import FINNError
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "partition_config", [PartitioningConfiguration(10), PartitioningConfiguration(1), None]
)
def test_multifpga_metadata_info_set_after_partitioning(
    partition_config: PartitioningConfiguration | None,
) -> None:
    """Test whether running partitioning on a config with a PartitioningConfiguration sets the
    model's 'is_multifpga' flag to True.
    """
    dt = DataType["BINARY"]
    model = make_multi_fclayer_model(3, dt, dt, dt, 10)
    output_dir = str(make_build_dir("test_multifpga_metadata_info_output_dir"))
    cfg = DataflowBuildConfig(output_dir=Path(output_dir), synth_clk_period_ns=5.0)
    cfg.partitioning_configuration = partition_config

    # TODO: As soon as partitioning move this whole test into the proper partitioning test
    with pytest.raises(FINNError):
        model = step_partition_for_multifpga(model, cfg)
    if partition_config is not None and partition_config.num_fpgas > 1:
        assert model.get_metadata_prop("is_multifpga") == "True"
    else:
        assert model.get_metadata_prop("is_multifpga") == "False"


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("model_type", ["rn18", "multi-fclayer"])
@pytest.mark.parametrize(
    "platform", [("U280", ShellFlowType.VITIS_ALVEO), ("Pynq-Z1", ShellFlowType.VIVADO_ZYNQ)]
)
@pytest.mark.parametrize("num_layers", [2, 10])
@pytest.mark.parametrize("bitwidth", [4, 8, 2, 3])
def test_resource_est_for_all_layers(
    model_type: str, platform: tuple[str, ShellFlowType], num_layers: int, bitwidth: int
) -> None:
    """Test that resource estimtates for all layers can be found."""
    board, shell = platform

    # Create a dataflow config
    output_dir = make_build_dir("test_res_estimation_")
    cfg = DataflowBuildConfig(
        output_dir=str(output_dir),
        synth_clk_period_ns=5.0,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS],
        board=board,
        steps=[],
        target_fps=3000,
        shell_flow_type=shell,
    )

    match model_type:
        case "multi-fclayer":
            dt = DataType["UINT" + str(bitwidth)]
            model = make_multi_fclayer_model(3, dt, dt, dt, num_layers)
            steps = [
                "step_qonnx_to_finn",
                "step_tidy_up",
                "step_streamline",
                "step_convert_to_hw",
                "step_specialize_layers",
                "step_target_fps_parallelization",
                "step_apply_folding_config",
                "step_minimize_bit_width",
                "step_generate_estimate_reports",
                "step_hw_codegen",
                "step_hw_ipgen",
                "step_set_fifo_depths",
            ]
            # Run the first half of the FINN flow
            steps_to_execute = resolve_build_steps(cfg)
            for step in steps_to_execute:
                model = step(model, cfg)
            cfg.steps = steps

        case "rn18":
            model, modelpath = generate_rn18("test_resource_est_all_layers", w=bitwidth, a=bitwidth)
            assert modelpath.exists()
            # TODO, DEBUG: Set skip_fifo_sizing to False
            model, cfg = prepare_resnet_for_multifpga(model, cfg, skip_fifo_sizing=True)
        case _:
            raise NotImplementedError(
                f"Invalid test configuration. " f"Unknown model type: {model_type}"
            )

    # Run the resource estimation
    estimates: dict[int, dict[str, int | float]] = get_estimated_model_resources(
        model, fpga_part=cfg._resolve_fpga_part()  # noqa
    )
    model = cast("ModelWrapper", model)
    for node in model.graph.node:
        index = model.get_node_index(node)
        assert index in estimates.keys(), (
            f"No estimate found for layer " f"{node.name} (index: {index})"
        )
        for est in estimates[index].values():
            assert type(est) in [int, float]  # Efficiency measures use floats

        assert any(est > 0 for est in estimates[index].values()), (
            f"Layer {node.name} (index: {index}) does not use "
            f"any resources at all: {estimates[index]}"
        )


@pytest.mark.parametrize("w", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("a", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_fpgas", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("board", ["U280", "U55C"])
def test_partition_aurora_chain_rn18(w: int, a: int, num_fpgas: int, board: str) -> None:
    """Test that the ResNet-18 model can be partitioned using the AuroraFlow partitioner."""
    model, modelpath = generate_rn18("test_partition_aurora_chain_rn18", w=w, a=a)
    assert modelpath.exists()
    cfg = DataflowBuildConfig(
        output_dir=make_build_dir("test_partition_aurora_chain_rn18_build"),
        board=board,
        target_fps=1000,
        synth_clk_period_ns=5.0,
        partitioning_configuration=PartitioningConfiguration(
            num_fpgas=num_fpgas,
            communication_kernel=MFCommunicationKernel.AURORA,
            topology=MFTopology.CHAIN,
            partition_strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
            partition_solver_timeout=120,
        ),
    )
    model, cfg = prepare_resnet_for_multifpga(model, cfg)

    # Run partitioning
    model = step_partition_for_multifpga(model, cfg)

    raise NotImplementedError("Asserts missing for testing partitioning results.")


def test_partition_solution_found() -> None:
    """Test some known model - fpga combinations that should
    be solveable.
    """
    raise NotImplementedError()


def test_aurora_partition_valid() -> None:
    """Test known model - fpga combination solutions and check
    that they are valid for constraints that the Aurora
    Partitioner requires.
    """
    raise NotImplementedError()


def test_platform_resources() -> None:
    raise NotImplementedError()
