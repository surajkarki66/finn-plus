"""Tests regarding the specific AuroraFlow Multi-FPGA implementation, separate from
the general tests.
"""

from __future__ import annotations

import pytest

import contextlib
import mip
from dataclasses import dataclass
from fpgadataflow.multifpga.utils import MockGraph, MockModelWrapper, get_model, mock_model
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from test_multifpga_sdp_creation import create_sdp_ready_model_no_branches
from typing import cast

import finn
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFCommunicationKernel,
    MFTopology,
    MFVerbosity,
    PartitioningConfiguration,
    PartitioningStrategy,
    ShellFlowType,
)
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import AuroraPartitioner
from finn.transformation.fpgadataflow.multifpga.communication_kernels import PrepareAuroraFlow
from finn.transformation.fpgadataflow.multifpga.create_multi_sdp import (
    CreateMultiFPGAStreamingDataflowPartition,
)
from finn.transformation.fpgadataflow.multifpga.create_network_metadata import CreateNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.partition_model import PartitionForMultiFPGA
from finn.util.basic import make_build_dir
from finn.util.exception import (
    FINNError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGAPartitionerError,
    FINNMultiFPGAUserError,
)
from finn.util.fpgadataflow import get_device_id
from finn.util.platforms import platforms
from finn.util.resources import (
    ResourceEstimates,
    available_resources_on_platform,
    get_estimated_model_resources,
)


@pytest.mark.auroraflow
@pytest.mark.multifpga
@pytest.mark.parametrize("board", ["U280", "U55C"])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
@pytest.mark.parametrize(
    "communication_kernel_args",
    [{}, {"FIFO_WIDTH": 32, "TX_FIFO_SIZE": 8192, "RX_FIFO_SIZE": 65536}],
)
class TestAuroraFlowPreparationAndMetadata:
    """Tests about the kernel preparation and metadata creation."""

    @pytest.mark.slow
    @pytest.mark.parametrize("device_node_combinations", [(1, 2), (1, 3), (2, 2), (5, 10), (5, 11)])
    @pytest.mark.parametrize("assignment_type", ["random", "equal"])
    @pytest.mark.parametrize("shuffle_devices", [True, False])
    def test_aurora_packaging_integrated(
        self,
        device_node_combinations: tuple[int, int],
        assignment_type: str,
        topology: MFTopology,
        shuffle_devices: bool,
        communication_kernel_args: dict[str, str],
        board: str,
    ) -> None:
        """Test the whole AuroraFlow preparation pipeline:
        - Create a model from scratch
        - Create the SDP partitions
        - Create the metadata based on the SDP partitions
        - Check that the metadata, XOs and model metadata props exist.
        """
        devices, nodes = device_node_combinations

        # Create an SDP ready branchless model
        model = create_sdp_ready_model_no_branches(nodes, devices, assignment_type, shuffle_devices)

        # Create a config based on the test parameters
        cfg = DataflowBuildConfig(
            output_dir=make_build_dir("test_aurora_packaging_integrated_build"),
            board=board,
            partitioning_configuration=PartitioningConfiguration(
                num_fpgas=devices,
                communication_kernel=MFCommunicationKernel.AURORA,
                topology=topology,
                communication_kernel_arguments=communication_kernel_args,
            ),
        )
        assert cfg.partitioning_configuration is not None

        # Execute the whole Aurora packaging flow
        model = model.transform(
            CreateMultiFPGAStreamingDataflowPartition(
                separate_iodmas=True,
                dataflow_partition_directory=Path(make_build_dir("test_aurora_package_")),
                verbosity=MFVerbosity.NONE,
            )
        )
        model = model.transform(
            CreateNetworkMetadata(
                cfg.partitioning_configuration.communication_kernel, MFVerbosity.NONE
            )
        )

        # No kernels packaged yet
        meta = AuroraNetworkMetadata.from_model(model)
        unprepared_aurora_kernels = len(meta.get_unprepared_aurora_kernels())
        if devices > 1:
            assert unprepared_aurora_kernels == (devices - 1) * 2
        else:
            assert unprepared_aurora_kernels == 0

        model = model.transform(
            PrepareAuroraFlow(
                cfg._resolve_vitis_platform(),  # noqa
                cfg._resolve_fpga_part(),  # noqa
                cfg.partitioning_configuration,
            )
        )

        # Try and load the previously generated metadata from the models metadata prop
        meta = AuroraNetworkMetadata.from_model(model)
        unprepared_aurora_kernels = len(meta.get_unprepared_aurora_kernels())
        assert unprepared_aurora_kernels == 0

        # Check that the AuroraFlow storage directory got saved in the model metadata
        aurora_storage = model.get_metadata_prop("aurora_storage")
        assert aurora_storage is not None

        # Check that this directory actually exists
        aurora_storage = Path(aurora_storage)
        assert aurora_storage.exists()

        # Check if each device had its respective kernels packaged
        if devices == 1:
            assert len(meta.data.keys()) == 0
        else:
            assert len(meta.data.keys()) == devices
        for kerneldata in meta.data.values():
            for aurora in kerneldata:
                assert aurora.aurora_xo is not None
                assert aurora.aurora_xo.exists()

    @pytest.mark.multifpga
    @pytest.mark.slow
    def test_aurora_package_single(
        self, communication_kernel_args: dict[str, str], board: str, topology: MFTopology
    ) -> None:
        """Test Aurora packaging. In detail:
        - Check that the names of the XO files produced by AuroraFlow didn't change.
        - Check that the transformation creates a build dir to store the AuroraFlow XO files in.
        - Check that the files were created at the correct path and moved to the correct target.
        """
        cfg = DataflowBuildConfig(
            output_dir=make_build_dir("test_aurora_package_single_output_dir"),
            board=board,
            partitioning_configuration=PartitioningConfiguration(
                num_fpgas=2,
                topology=topology,
                communication_kernel=MFCommunicationKernel.AURORA,
                communication_kernel_arguments=communication_kernel_args,
            ),
        )
        assert cfg.partitioning_configuration is not None
        prep = PrepareAuroraFlow(
            cfg._resolve_vitis_platform(),  # noqa
            cfg._resolve_fpga_part(),  # noqa
            cfg.partitioning_configuration,
        )
        assert prep.aurora_storage.exists()
        res = prep.package_single("", 0, 0)
        build_dir = prep.aurora_storage / "auroraflow_build_dev0_ind0"
        assert build_dir.exists()
        assert res.exists()

        # On devices like the U280, there are only 2 QSFPs, so an index 2 (third kernel) cannot
        # be produced, hence the kernel should not be there.
        with pytest.raises(FINNError):
            res = prep.package_single("", 1, 2)
            build_dir = prep.aurora_storage / "auroraflow_build_dev1_ind2"
            assert build_dir.exists()
            assert res.exists()


@pytest.mark.auroraflow
@pytest.mark.parametrize("board", ["U280", "Pynq-Z1"])
@pytest.mark.parametrize("topology", [MFTopology.CHAIN])
class TestAuroraFlowPartitioning:
    """Tests regarding the partitioning using the AuroraFlow kernel and partitioner class."""

    def get_shell_flow_type(self, board: str) -> ShellFlowType:
        """Return the shell flow type for the given board.
        Errors if a not assigned board is passed.
        """
        if board in ["U280"]:
            return ShellFlowType.VITIS_ALVEO
        elif board in ["Pynq-Z1"]:  # noqa
            return ShellFlowType.VIVADO_ZYNQ
        raise NotImplementedError(
            f"Unknown board type ({board}) for this test. Unsure which shell type to use."
        )

    def requires_too_many_resources(self, model: ModelWrapper, cfg: DataflowBuildConfig) -> bool:
        """Check if a model requires more resources than the given number of devices supplies.
        Considers the number of devices, as well as the max utilization percentage.
        """
        assert cfg.partitioning_configuration is not None, "No partitioning configuration found!"
        assert cfg.board is not None, (
            "Partitioning requires the 'board' " "parameter to be set in the dataflow config."
        )
        resource_estimates = get_estimated_model_resources(
            model,
            cfg._resolve_fpga_part(),  # noqa
            cfg.partitioning_configuration.considered_resources,
            True,
        )
        device_resources = available_resources_on_platform(
            platforms[cfg.board](), cfg.partitioning_configuration.considered_resources
        )
        for restype in cfg.partitioning_configuration.considered_resources:
            total_required = sum([rv[restype] for rv in resource_estimates.values()])
            total_on_devices = (
                cfg.partitioning_configuration.max_utilization
                * cfg.partitioning_configuration.num_fpgas
                * device_resources[restype]
            )
            if total_required > total_on_devices:
                return True
        return False

    @pytest.mark.parametrize("network_ports", [2])
    @pytest.mark.parametrize("ideal_max_util", [(0.8, 0.9), (0.9, 1.0), (0.2, 0.8), (0.2, 1.0)])
    def test_enforce_utilization_limit(
        self,
        board: str,
        topology: MFTopology,
        network_ports: int,
        ideal_max_util: tuple[float, float],
    ) -> None:
        """Test that the partitioner upholds the resource utilization limit."""
        raise NotImplementedError()

    @pytest.mark.parametrize(
        "model_type",
        [
            ("CNV", 1, 1, True),
            ("CNV", 1, 2, True),
            ("CNV", 2, 2, True),
            ("LFC", 1, 1, True),
            ("LFC", 1, 2, True),
            ("SFC", 1, 2, True),
            ("SFC", 2, 2, True),
            ("TFC", 1, 1, True),
            ("TFC", 1, 2, True),
            ("mobilenetv1", 4, 4, True),
            ("resnet18", 4, 4, True),
        ],
    )
    @pytest.mark.parametrize("devices", [2, 3, 4, 10])
    @pytest.mark.parametrize("max_util", [0.95, 0.85])
    @pytest.mark.parametrize("ideal_util", [0.80, 0.75])
    @pytest.mark.parametrize(
        "partition_strategy",
        [PartitioningStrategy.LAYER_COUNT, PartitioningStrategy.RESOURCE_UTILIZATION],
    )
    def test_partitioning(
        self,
        model_type: tuple[str, int, int, bool],
        devices: int,
        partition_strategy: PartitioningStrategy,
        topology: MFTopology,
        board: str,
        max_util: float,
        ideal_util: float,
        pytestconfig: pytest.Config,
    ) -> None:
        """Test some known model - fpga combinations that should
        be solveable.
        """
        typename, wbits, abits, pretrained = model_type
        test_identifier = (
            f"test_partitioning_{typename}_{wbits}_{abits}_{pretrained}_"
            f"{devices}_{partition_strategy.name}_{topology.name}_{board}_{max_util}_{ideal_util}"
        )

        # Skip tests for models that don't fit on certain devices
        if typename in ["mobilenetv1", "resnet18"] and board not in ["U280"]:
            pytest.skip(reason=f"Model {typename} too large for board {board}!")  # type: ignore

        # Skip the same tests that the end2end tests skip
        if typename.lower() == "lfc" and wbits == 1 and devices == 2 and board == "Pynq-Z1":
            pytest.skip(
                "Not running LFC models with W1A1/W1A2 on "  # type: ignore
                "2 Pynq-Z1 devices - there is no "
                "way to partition this model for this particular config."
            )

        # Build the config
        flow_type = self.get_shell_flow_type(board)
        cfg = DataflowBuildConfig(
            output_dir=make_build_dir(test_identifier),
            board=board,
            mvau_wwidth_max=512,
            target_fps=5000,
            synth_clk_period_ns=5.0,
            standalone_thresholds=True,
            minimize_bit_width=True,
            shell_flow_type=flow_type,
            partitioning_configuration=PartitioningConfiguration(
                num_fpgas=devices,
                topology=topology,
                communication_kernel=MFCommunicationKernel.AURORA,
                partition_strategy=partition_strategy,
                max_utilization=max_util,
                ideal_utilization=ideal_util,
                ports_per_device=2,
                separate_iodmas=True,
                partition_solver_timeout=180,
                verbosity=MFVerbosity.EXTRA_HIGH,
            ),
        )

        # Get the model
        model, cfg = get_model(
            typename,
            wbits,
            abits,
            pretrained,
            "step_set_fifo_depths",
            True,
            cfg,
            pytestconfig,
            identifier="test_partitioning",
        )

        # Skip tests for models that simply require too many resources
        if self.requires_too_many_resources(model, cfg):
            pytest.skip(
                "Requires more resources than are available in "
                "this partitioning configuration."  # type: ignore
            )

        # Catch if there are more devices than nodes
        context = contextlib.nullcontext()
        if len(model.graph.node) < devices:
            context = pytest.raises(FINNMultiFPGAConfigError)

        # Do the partitioning
        # IMPORTANT: Large device counts open a very large search space for the model,
        # which is why it could happen that the solver does not find a solution in the given time,
        # despite the model being feasible. The current timeout should prevent this from happening.
        with context:
            assert cfg.partitioning_configuration is not None
            part = PartitionForMultiFPGA(cfg)
            model = model.transform(part)

        # If there are more devices than nodes, we are done here, since we already tested,
        # that the error is raised
        if len(model.graph.node) < devices:
            return

        # Check that partitioning was successful
        assert part.partitioner is not None
        assert part.partitioner.status in [
            mip.OptimizationStatus.FEASIBLE,
            mip.OptimizationStatus.OPTIMAL,
        ]
        assert cfg.partitioning_configuration is not None

        # Get the solution data
        solution = part.mapping
        assert solution is not None

        # Test that only apply for resource utilization strategy
        if partition_strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            # Get the resource of the partitioned design.
            # maps: device -> {LUT: 0.3, BRAM: 0.5, ...}, ...
            usage = part.partitioner.get_resource_use_relative()
            assert usage is not None

            # Available resources on this device
            res_per_device = available_resources_on_platform(
                platforms[board](), cfg.partitioning_configuration.considered_resources
            )

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

        # All fork/join nodes need to have the same device ID as the nodes they are connected to
        # Otherwise this SDP requires 2+ IO ports
        for node in model.graph.node:
            node_dev = get_device_id(node)
            assert node_dev is not None
            if model.is_fork_node(node):
                successors = model.find_direct_successors(node)
                if successors is None:
                    continue
                for successor_node in successors:
                    successor_device = get_device_id(successor_node)
                    assert successor_device is not None
                    assert successor_device == node_dev, (
                        f"Expected fork node {node.name} (device: {node_dev}) and successor node "
                        f"{successor_node.name} (device: {successor_device}) to have "
                        f"the same device ID!"
                    )
            elif model.is_join_node(node):
                predecessors = model.find_direct_predecessors(node)
                if predecessors is None:
                    continue
                for predecessor_node in predecessors:
                    predecessor_device = get_device_id(predecessor_node)
                    assert predecessor_device is not None
                    assert predecessor_device == node_dev, (
                        f"Expected join node {node.name} (device: {node_dev}) and predecessor node "
                        f"{predecessor_node.name} (device: {predecessor_device}) to have "
                        f"the same device ID!"
                    )

        # All nodes have an assignment
        for node in model.graph.node:
            assert node.name in solution.keys()
            assert solution[node.name] is not None

        # Consecutive assignments
        for node in model.graph.node:
            suc = model.find_direct_successors(node)
            if suc is None:
                continue
            for successor_node in suc:
                assert abs(solution[successor_node.name] - solution[node.name]) <= 1

        # No device was visited twice

        # TODO: This is slightly difficult for non-linear models!
        # TODO: Also not necessarily required

    @pytest.mark.parametrize(
        "model_type",
        [
            ("mobilenet", 4, 4, True),
            ("resnet18", 4, 4, True),
        ],
    )
    @pytest.mark.parametrize(
        "strategy", [PartitioningStrategy.LAYER_COUNT, PartitioningStrategy.RESOURCE_UTILIZATION]
    )
    def test_objective_regression(
        self,
        topology: MFTopology,
        board: str,
        model_type: tuple[str, int, int, bool],
        strategy: PartitioningStrategy,
    ) -> None:
        """Test that the partitioning of certain models has not regressed in recent commits.
        Specifically, test the final value of the objective function.

        TODO: Save recent CI run data in proper infrastructure as soon as we have set it
        up, instead of hardcoding the values.
        """
        # flow_type = self.get_shell_flow_type(board)

        # TODO
        raise NotImplementedError()

        # assert part.partitioner is not None
        # assert part.partitioner.model.objective.x is not None
        # assert part.partitioner.model.objective.x < 0

    @pytest.mark.parametrize(
        "distribution",
        ["equal-LUT", "equal-FF"],
    )
    @pytest.mark.parametrize(
        "distribution_args", [{"level": 1000}, {"level": 10000}, {"level": 10e6}]
    )
    @pytest.mark.parametrize("nodes", [*list(range(1, 10))])
    @pytest.mark.parametrize("devices", [*list(range(1, 10))])
    @pytest.mark.parametrize("considered_resources", [["LUT", "FF", "DSP", "BRAM_18K"]])
    @pytest.mark.parametrize("max_util", [0.85])
    @pytest.mark.parametrize("ideal_util", [0.75])
    @pytest.mark.parametrize("inseparable_nodes", [[]])
    @pytest.mark.parametrize("network_ports", [2])
    def test_resource_balancing_with_artificial_data(
        self,
        distribution: str,
        distribution_args: dict,
        nodes: int,
        devices: int,
        considered_resources: list[str],
        board: str,
        ideal_util: float,
        max_util: float,
        topology: MFTopology,
        inseparable_nodes: list[int],
        network_ports: int,
    ) -> None:
        """Test partitioning with the Aurora model based
        on constructed data instead of real models.
        """
        # This specific case creates an infeasible model. The available
        # LUTs per device are ~53k. At 85% utilization, there is overall space for roughly 90k LUTs.
        # This could theoretically hold all 90k LUTs required, however the 90k are split
        # across devices. Since a layer is 10k, and each device can hold roughly 45k alone, the 5k
        # leftover LUTs are not utilized, giving effectively 80k LUTs overall,
        # making the model requiring 90k LUTs infeasible.
        if (
            max_util <= 0.85
            and devices == 2
            and distribution_args["level"] == 10000
            and board == "Pynq-Z1"
            and distribution == "equal-LUT"
        ):
            pytest.skip("Infeasible model configuration.")  # type: ignore

        dist_type = distribution.split("-")[0]
        dist_res = distribution.split("-")[1]

        # Mock resources. The given resource type is set to the passed level, all others to 0
        # TODO: Change how the test is configured
        resource_estimates = {}
        for node in range(nodes):
            resource_estimates[node] = dict(
                zip(
                    considered_resources, [0 for _ in range(len(considered_resources))], strict=True
                )
            )
            if dist_type == "equal":
                resource_estimates[node][dist_res] = distribution_args["level"]

        test_dir_identifier = (
            f"test_pure_aurora_resource_opt_device{devices}"
            f"_node{nodes}_topo{topology.name}_{board}_dist{distribution}"
            f"_{max_util}_{ideal_util}_ins{len(inseparable_nodes)}_topology{topology}"
        )
        res_per_device = available_resources_on_platform(platforms[board](), considered_resources)

        # Check if we expect an error regarding usage of resources
        # If more resources than allowed are utilized, the partitioner will
        # raise an error upon intialization. Otherwise we have a nullcontext as a no-op
        exception_context = contextlib.nullcontext()
        for estimates in resource_estimates.values():
            for resource in considered_resources:
                # Check if we overshoot the limit
                if estimates[resource] > res_per_device[resource] * max_util:
                    exception_context = pytest.raises(FINNMultiFPGAPartitionerError)

        # Check if we expect an error because the total resource amount is larger than
        # what we have available over all devices
        total_estimates = {
            rt: sum([rvs[rt] for rvs in resource_estimates.values()]) for rt in considered_resources
        }
        for rt in considered_resources:
            if total_estimates[rt] > max_util * res_per_device[rt] * devices:
                exception_context = pytest.raises(FINNMultiFPGAPartitionerError)

        with exception_context:
            part = AuroraPartitioner(
                output_dir=Path(make_build_dir(test_dir_identifier + "_")),
                network_ports_per_device=network_ports,
                strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
                devices=devices,
                nodes=nodes,
                considered_resources=considered_resources,
                resources_per_device=res_per_device,  # type: ignore
                inseparable_nodes=[],
                topology=topology,
                max_utilization=max_util,
                ideal_utilization=ideal_util,
                resource_estimates=resource_estimates,
                verbosity=MFVerbosity.NONE,
                index_node_name_map={i: str(i) for i in range(nodes)},
            )

        # If the model is infeasbile due to resource constraints end the test here
        if type(exception_context) is not contextlib.nullcontext:
            return

        # Try to solve the model
        solution = part.solve(100)

        # Check if a solution was found
        # If the model is impossible assert that no solution was found
        # and return to skip the rest if the test
        overutilized_overall = (
            nodes * distribution_args["level"] > res_per_device[dist_res] * devices
        )
        overutilized_per_device = distribution_args["level"] > max_util * res_per_device[dist_res]
        if devices > nodes or overutilized_overall or overutilized_per_device:
            assert solution is None
            return

        # Check that a solution was found
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
        # TODO: For non-linear models this needs to check predecessors, not topology - 1 indexes
        for i in range(nodes - 1):
            assert abs(solution[str(i)] - solution[str(i + 1)]) <= 1

        # No device was visited twice
        if topology == MFTopology.CHAIN:
            visited = [solution["0"]]
            for i in range(nodes - 1):
                if solution[str(i + 1)] != solution[str(i)]:
                    visited.append(solution[str(i + 1)])
        assert len(set(visited)) == len(visited)

    @pytest.mark.parametrize(
        "fail_type", ["node_larger_than_device", "all_nodes_larger_than_device", "too_many_devices"]
    )
    def test_bad_resource_requirements(
        self, fail_type: str, board: str, topology: MFTopology, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that configurations that require too many resources fail properly."""
        devices = 0
        nodes = 0
        resource_table = {}
        model = MockModelWrapper(MockGraph([]))
        device_resources = available_resources_on_platform(platforms[board](), ["LUT"])
        max_util = 0.7

        # Set up the config / mock the resource estimator so that the specified fail type
        # appears
        match fail_type:
            case "node_larger_than_device":
                devices = 2
                nodes = 2
                resource_table = {
                    "0": {"LUT": int(max_util * device_resources["LUT"]) + 10},
                    "1": {"LUT": 1},
                }
                model = mock_model(nodes)
            case "all_nodes_larger_than_device":
                devices = 2
                nodes = 5
                resource_table = {
                    "0": {"LUT": int(max_util / 2 * device_resources["LUT"]) + 10},
                    "1": {"LUT": int(max_util / 2 * device_resources["LUT"]) + 10},
                    "2": {"LUT": int(max_util / 2 * device_resources["LUT"]) + 10},
                    "3": {"LUT": int(max_util / 2 * device_resources["LUT"]) + 10},
                }
                model = mock_model(nodes)
            case "too_many_devices":
                devices = 10
                nodes = 2
                resource_table = {"0": {"LUT": 1}, "1": {"LUT": 1}}
                model = mock_model(nodes)
            case _:
                raise AssertionError(f"Unknown fail_type: {fail_type}")

        # Patch the resource estimator
        monkeypatch.setattr(
            "finn.transformation.fpgadataflow.multifpga"
            ".aurora.partitioner.get_estimated_model_resources",
            lambda _a, _b, _c, _d: resource_table,
        )

        # Create the config
        cfg = DataflowBuildConfig(
            output_dir=make_build_dir(f"test_bad_config_{fail_type}_"),
            board=board,
            partitioning_configuration=PartitioningConfiguration(
                num_fpgas=devices,
                max_utilization=max_util,
                ideal_utilization=0.4,
                verbosity=MFVerbosity.EXTRA_HIGH,
                topology=topology,
                ports_per_device=2,
                single_stream_network=False,
                considered_resources=["LUT"],
                partition_strategy=PartitioningStrategy.RESOURCE_UTILIZATION,
            ),
        )

        # Assert the error that should be raised
        with pytest.raises(FINNMultiFPGAPartitionerError):
            _ = AuroraPartitioner(cfg, cast("ModelWrapper", model))
