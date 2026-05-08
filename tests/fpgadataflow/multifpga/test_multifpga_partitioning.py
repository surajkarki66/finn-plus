from __future__ import annotations

import pytest

import torch
from brevitas.export import export_qonnx
from fpgadataflow.multifpga.utils import generate_rn18, prepare_resnet_for_multifpga
from pathlib import Path
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from testing_util.test import get_test_model

from finn.builder import build_dataflow_steps
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    DataflowOutputType,
    MFCommunicationKernel,
    MFTopology,
    MFVerbosity,
    PartitioningConfiguration,
    PartitioningStrategy,
    default_build_dataflow_steps,
)
from finn.transformation.fpgadataflow.multifpga.partitioner import (
    AuroraPartitioner,
    PartitionForMultiFPGA,
)
from finn.transformation.fpgadataflow.multifpga.utils import available_resources
from finn.util import platforms
from finn.util.basic import make_build_dir
from finn.util.exception import FINNError
from tests.fpgadataflow.test_set_folding import make_multi_fclayer_model


# TODO: Add mobilenet
@pytest.mark.auroraflow
@pytest.mark.parametrize(
    "model_type",
    [
        ("CNV", 1, 1, True),
        ("CNV", 1, 2, True),
        ("CNV", 2, 2, True),
        ("LFC", 1, 1, True),
        ("LFC", 1, 2, True),
        ("SFC", 1, 1, True),
        ("SFC", 1, 2, True),
        ("SFC", 2, 2, True),
        ("TFC", 1, 1, True),
        ("TFC", 1, 2, True),
    ],
)
@pytest.mark.parametrize("devices", [2, 3, 4, 10])
@pytest.mark.parametrize("max_util", [0.95, 0.85])
@pytest.mark.parametrize("ideal_util", [0.80, 0.75])
@pytest.mark.parametrize(
    "partition_strategy",
    [PartitioningStrategy.LAYER_COUNT, PartitioningStrategy.RESOURCE_UTILIZATION],
)
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize("board", ["Pynq-Z1"])
def test_aurora_partition_solution_found(
    model_type: tuple[str, int, int, bool],
    devices: int,
    partition_strategy: PartitioningStrategy,
    topology: MFTopology,
    board: str,
    max_util: float,
    ideal_util: float,
) -> None:
    """Test some known model - fpga combinations that should
    be solveable.
    """
    # TODO: Fix: Certain model types fail during streamlining

    typename, wbits, abits, pretrained = model_type
    test_dir_identifier = (
        f"test_partition_solution_{typename}_{wbits}_{abits}"
        f"_p{pretrained}_dev{devices}_board{board}_topology{topology}"
    )
    model_onnx_path = Path(make_build_dir(test_dir_identifier + "_")) / "fc.onnx"
    fc = get_test_model(typename, wbits, abits, pretrained)
    ishape = (1, 1, 28, 28)
    if typename == "CNV":
        ishape = (1, 3, 32, 32)
    elif typename == "mobilenet":
        ishape = (1, 3, 224, 224)
    export_qonnx(fc, torch.randn(ishape), str(model_onnx_path))
    qonnx_cleanup(str(model_onnx_path), out_file=str(model_onnx_path))
    model = ModelWrapper(str(model_onnx_path))

    cfg = DataflowBuildConfig(
        output_dir=str(model_onnx_path.parent / "out_dir"),
        synth_clk_period_ns=5.0,
        generate_outputs=[DataflowOutputType.ESTIMATE_REPORTS, DataflowOutputType.STITCHED_IP],
        board=board,
        target_fps=2000,
        save_intermediate_models=True,
        partitioning_configuration=PartitioningConfiguration(
            num_fpgas=devices,
            partition_strategy=partition_strategy,
            max_utilization=max_util,
            ideal_utilization=ideal_util,
            communication_kernel=MFCommunicationKernel.AURORA,
            topology=topology,
        ),
    )
    for transform_step in default_build_dataflow_steps:
        model = build_dataflow_steps.build_dataflow_step_lookup[transform_step](model, cfg)
        if transform_step == "step_set_fifo_depths":
            break

    assert cfg.partitioning_configuration is not None
    model = model.transform(
        PartitionForMultiFPGA(
            cfg.partitioning_configuration,
            cfg._resolve_fpga_part(),  # noqa
            board,
            Path(cfg.output_dir),
        )
    )


def test_aurora_chain_partitioning_regression() -> None:
    """Test that the partitioning of certain models has not regressed in recent commits."""
    raise NotImplementedError()


@pytest.mark.auroraflow
@pytest.mark.parametrize(
    "distribution",
    ["equal-LUT", "equal-FF"],
)
@pytest.mark.parametrize("distribution_args", [{"level": 1000}, {"level": 10000}, {"level": 10e6}])
@pytest.mark.parametrize("nodes", [*list(range(1, 10))])
@pytest.mark.parametrize("devices", [*list(range(1, 10))])
@pytest.mark.parametrize("considered_resources", [["LUT", "FF", "DSP", "BRAM_18K"]])
@pytest.mark.parametrize("board", ["U280", "Pynq-Z1"])
@pytest.mark.parametrize("max_util", [0.85])
@pytest.mark.parametrize("ideal_util", [0.75])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize("inseperable_nodes", [[]])
@pytest.mark.parametrize("network_ports", [2])
def test_aurora_partitioning_pure_resource_optimize(
    distribution: str,
    distribution_args: dict,
    nodes: int,
    devices: int,
    considered_resources: list[str],
    board: str,
    ideal_util: float,
    max_util: float,
    topology: MFTopology,
    inseperable_nodes: list[int],
    network_ports: int,
) -> None:
    """Test partitioning with the Aurora model based on constructed data instead of real models."""
    dist_type = distribution.split("-")[0]
    dist_res = distribution.split("-")[1]

    # Mock resources. The given resource type is set to the passed level, all others to 0
    # TODO: Change how the test is configured
    resource_estimates = {}
    for node in range(nodes):
        resource_estimates[node] = dict(
            zip(considered_resources, [0 for _ in range(len(considered_resources))], strict=True)
        )
        if dist_type == "equal":
            resource_estimates[node][dist_res] = distribution_args["level"]

    test_dir_identifier = (
        f"test_pure_aurora_resource_opt_device{devices}"
        f"_node{nodes}_topo{topology.name}_{board}_dist{distribution}"
        f"_{max_util}_{ideal_util}_ins{len(inseperable_nodes)}_topology{topology}"
    )
    res_per_device = available_resources(platforms.platforms[board](), considered_resources)

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
        index_node_name_map={i: str(i) for i in range(nodes)},
    )
    solution = part.solve(100)

    # Check if a solution was found
    # If the model is impossible assert that no solution was found
    # and return to skip the rest if the test
    overutilized_overall = nodes * distribution_args["level"] > res_per_device[dist_res] * devices
    overutilized_per_device = distribution_args["level"] > max_util * res_per_device[dist_res]
    if devices > nodes or overutilized_overall or overutilized_per_device:
        assert solution is None
        return
    assert solution is not None

    # Get the resource of the partitioned design.
    # maps: device -> {LUT: 0.3, BRAM: 0.5, ...}, ...
    usage = part.get_resource_use_relative()
    assert usage is not None

    # Every device is utilized
    for device in usage.keys():
        # Only one resource type has to be above 0
        assert any(usage[device][restype] > 0 for restype in usage[device].keys())

    # max_utilization not overstepped
    for device in usage.keys():
        for restype, res in usage[device].items():
            assert res <= max_util * res_per_device[restype]

    # Atleast 2 different device IDs (to catch the qonnx nodeattr bug)
    if devices >= 2:
        assert len(set(solution.values())) >= 2

    # All nodes have an assignment
    for i in range(nodes):
        assert str(i) in solution.keys()
        assert solution[str(i)] is not None

    # Consecutive assignments
    for i in range(nodes - 1):
        assert abs(solution[str(i)] - solution[str(i + 1)]) <= 1

    # No device was visited twice
    if topology == MFTopology.CHAIN:
        visited = [solution["0"]]
        for i in range(nodes - 1):
            if solution[str(i + 1)] != solution[str(i)]:
                visited.append(solution[str(i + 1)])
    assert len(set(visited)) == len(visited)

    # TODO: Test same asserts on real models
    raise NotImplementedError()


@pytest.mark.parametrize("devices", [2, 3, 10, 100])
@pytest.mark.parametrize("nodes", [1, 2, 3, 10, 100, 110])
@pytest.mark.parametrize("network_ports", [2, 3, 4])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize(
    "platform", [platforms.Alveo_NxU280_Platform(), platforms.Zynq7020_Platform()]
)
def test_impossible_inseperable_nodes(
    devices: int, nodes: int, network_ports: int, topology: MFTopology, platform: platforms.Platform
) -> None:
    """Check that impossible device node combinations are caught."""
    # TODO: Check all conditions that can fail in the partitioner with regards to
    # inseperable groups

    test_dir_identifier = f"test_all_inseperable_{devices}_topology{topology.name}"
    max_util = 0.9
    ideal_util = 0.8
    considered_resources = ["LUT", "FF", "DSP", "BRAM_18K"]
    res_per_device = available_resources(platform, considered_resources)
    resource_estimates = {
        node: {res: ideal_util * res_per_device[res] for res in considered_resources}
        for node in range(nodes)
    }

    # Ignore working configs, only test wrong configurations
    if (nodes < devices) or ((devices == nodes) and (devices > 1)):
        with pytest.raises(FINNError):
            part = AuroraPartitioner(
                network_ports_per_device=network_ports,
                strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
                devices=devices,
                nodes=nodes,
                considered_resources=considered_resources,
                resources_per_device=res_per_device,
                inseperable_nodes=[list(range(devices))],
                topology=topology,
                max_utilization=max_util,
                ideal_utilization=ideal_util,
                resource_estimates=resource_estimates,
                verbosity=MFVerbosity.NONE,
                output_dir=Path(make_build_dir(test_dir_identifier + "_")),
            )
            solution = part.solve(100)
            assert solution is None


@pytest.mark.auroraflow
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
    assert cfg.partitioning_configuration is not None
    model = model.transform(
        PartitionForMultiFPGA(
            cfg.partitioning_configuration,
            cfg._resolve_fpga_part(),  # noqa
            board,
            Path(cfg.output_dir),
        )
    )

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
