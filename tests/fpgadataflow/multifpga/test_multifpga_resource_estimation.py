import pytest

from pathlib import Path
from qonnx.core.datatype import DataType
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow import resolve_build_steps
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    MFTopology,
    MFVerbosity,
    PartitioningStrategy,
    ShellFlowType,
)
from finn.transformation.fpgadataflow.multifpga.partitioner import (
    AuroraPartitioner,
    Partitioner,
    PartitionForMultiFPGA,
)
from finn.transformation.fpgadataflow.multifpga.utils import (
    available_resources,
    get_estimated_model_resources,
)
from finn.util import platforms
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGANoPartitionerSolutionError
from tests.fpgadataflow.multifpga.utils import generate_rn18, prepare_resnet_for_multifpga
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@pytest.mark.parametrize(
    "platform", [platforms.Alveo_NxU280_Platform(), platforms.Zynq7020_Platform()]
)
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize("network_ports", [2])
@pytest.mark.parametrize("ideal_max_util", [(0.8, 0.9), (0.9, 1.0), (0.2, 0.8), (0.2, 1.0)])
def test_enforce_utilization_limit_aurora(
    platform: platforms.Platform,
    topology: MFTopology,
    network_ports: int,
    ideal_max_util: tuple[float, float],
) -> None:
    """Test that the partitioner upholds the resource utilization limit."""
    test_dir_identifier = f"test_util_limit_{platform.__class__.__name__}_{topology.name}"
    diff = 0.05
    max_util = ideal_max_util[1]
    ideal_util = ideal_max_util[0]
    devices = 2
    nodes = 2
    considered_resources = ["LUT", "FF", "DSP", "BRAM_18K"]
    res_per_device = available_resources(platform, considered_resources)
    # Device0 is underutilized, Device1 is overutilized
    resource_estimates = {
        0: {res: res_per_device[res] * (max_util - diff) for res in considered_resources},
        1: {res: res_per_device[res] * (max_util + diff) for res in considered_resources},
    }
    with pytest.raises(FINNMultiFPGANoPartitionerSolutionError):
        part = AuroraPartitioner(
            output_dir=Path(make_build_dir(test_dir_identifier + "_")),
            network_ports_per_device=network_ports,
            strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
            devices=devices,
            nodes=nodes,
            considered_resources=considered_resources,
            resources_per_device=res_per_device,
            inseperable_nodes=[],
            topology=topology,
            max_utilization=max_util,
            ideal_utilization=ideal_util,
            resource_estimates=resource_estimates,
            verbosity=MFVerbosity.NONE,
        )
        solution = part.solve(100)
        assert solution is None


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
