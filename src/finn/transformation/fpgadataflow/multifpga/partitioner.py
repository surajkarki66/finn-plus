"""Partitioners for Multi-FPGA usage."""

from __future__ import annotations

import mip
import yaml
from abc import ABC, abstractmethod
from math import ceil
from mip import Model, xsum
from pathlib import Path, PosixPath, PurePath, WindowsPath
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from rich import box
from rich.layout import Layout
from rich.table import Table
from typing import TYPE_CHECKING, Any

from finn.builder.build_dataflow_config import (
    MFCommunicationKernel,
    MFTopology,
    MFVerbosity,
    PartitioningStrategy,
)
from finn.transformation.fpgadataflow.multifpga.utils import (
    available_resources,
    get_estimated_model_resources,
    get_inseparable_nodes,
    is_single_in_out_model,
    set_device_id,
)
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGAConfigError, FINNMultiFPGAError, FINNMultiFPGAUserError
from finn.util.logging import LogDisabledConsole, log
from finn.util.platforms import platforms

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


class Partitioner(ABC):
    """Models a linear problem that can be used to solve Multi-FPGA partitioning. The idea to solve
    this in general using an LP was first devised by the AMD team for Elastic-DF and implemented as
    a prototype in finn-experimental.
    (https://github.com/Xilinx/finn-experimental/blob/main/src/finnexperimental/analysis/partitioning.py)

    We use a slightly different approach to modelling the problem and the objective function,
    however the partitioner from finn-experimental should be relativly easy to swap in
    if needed.

    Parameters
    ----------
        inseperable_nodes: Nodes that need to stay together because they are in a split

        considered_resources: What types of resources are used in the objective
            function to determine load.
    """  # noqa

    def __init__(  # noqa
        self,
        strategy: PartitioningStrategy,
        topology: MFTopology | None,
        devices: int,
        nodes: int,
        inseperable_nodes: list[list[int]],
        verbosity: MFVerbosity,
        resources_per_device: dict,
        output_dir: Path,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
        network_ports_per_device: int = 2,
        max_utilization: float | None = None,
        ideal_utilization: float | None = None,
    ) -> None:
        # MIP member variables first
        self.status: mip.OptimizationStatus | None
        try:
            self.model = Model()
        except OSError:
            log.warning(
                "Creation of mip.Model failed. This might be known bug "
                "(LD_LIBRARY_PATH only modified at runtime to point to "
                "libgurobi instead of before). Falling back to CBC"
            )  # See finn-plus issue #67
            self.model = Model(solver_name=mip.CBC)

        # Details about the partitioning
        self.strategy = strategy
        self.max_utilization = max_utilization
        self.topology = topology
        self.ideal_util = ideal_utilization
        self.inseperable_nodes = inseperable_nodes
        self.resource_estimates = resource_estimates
        self.resources_per_device = resources_per_device
        self.considered_resources = (
            ["LUT", "FF", "BRAM_18K", "DSP"]
            if considered_resources is None
            else considered_resources
        )
        self.verbosity = verbosity
        self.device_count = devices
        self.node_count = nodes
        self.network_ports_per_device = network_ports_per_device
        self.output_dir = output_dir
        self.working_directory = Path(make_build_dir(prefix="partitioning_")).absolute()
        self.latest_snapshot_path: Path | None = None
        if self.verbosity.value > MFVerbosity.NONE.value:
            log.info(f"Starting network partitioning. Selected strategy: {self.strategy.name}")
        if self.verbosity.value > MFVerbosity.LOW.value:
            if self.topology is not None:
                log.info(f"Network topology: {self.topology.name}")
            log.info(f"Devices: {self.device_count}")
            log.info(f"Nodes: {self.node_count}")
            log.info(f"Considered resource types: {self.considered_resources}")
            log.info(f"Ideal resource utilization: {self.ideal_util}")
            log.info(f"Maximum resource utilization: {self.max_utilization}")
            log.info(f"Groups of inseparable nodes: {len(self.inseperable_nodes)}")
            log.info(f"Network ports per device: {self.network_ports_per_device}")

        if (self.strategy == PartitioningStrategy.RESOURCE_UTILIZATION) and (
            None in [self.max_utilization, self.ideal_util, self.resource_estimates]
        ):
            raise FINNMultiFPGAError(
                f"One of the required partitioner parameters for the strategy "
                f"{self.strategy.name} was not found. Please provide max_utilization, "
                "ideal_utilization and resource_estimates!"
            )

    @abstractmethod
    def create_result(self) -> dict[str, int]:
        """Method that is used to generate a uniform solution type from the internal model.
        Any model / solver can implement its constraints and variables differently and overwrite
        this method, so that any class inheriting from the base can have a uniform result type.
        This type should map node-names to devices.
        The method should also error, if it is called before a solution was found.
        """

    def solve(
        self,
        solver_timeout: int,
    ) -> dict[str, int] | None:
        """Try to optimize the objective function. If no feasible solution is found
        return None, otherwise return a mapping of nodes to their device. After trying
        to solve, creates a snapshot description of the model in a temp build dir, as well
        as a solution in the same dir, if one was found.
        """
        self.status = self.model.optimize(solver_timeout)  # type: ignore
        if self.status == mip.OptimizationStatus.ERROR:
            raise FINNMultiFPGAUserError("The solver returned an error status!")
        if self.status in [
            mip.OptimizationStatus.INFEASIBLE,
            mip.OptimizationStatus.NO_SOLUTION_FOUND,
        ]:
            return None
        return self.create_result()

    def write_results(self, p: Path) -> None:
        """Write the partition results as a YAML to the given directory."""
        results = self.create_result()
        with p.open("w+") as f:
            yaml.dump(results, f, yaml.Dumper)

    @abstractmethod
    def _get_resource_use_relative(self) -> dict[int, dict[str, Any]]:
        """Get resources used by the device in percent. Must fail if no
        partition was calculated yet.
        """
        pass  # noqa

    def get_resource_use_relative(self) -> dict[int, dict[str, Any]] | None:
        """Return the resources used by a device. This only works if the optimization goal was
        resource usage. If no optimization was done, the dict will contain None's
        Actual implementation is left to the subclasses.
        """
        if self.strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            return self._get_resource_use_relative()
        return None


class AuroraPartitioner(Partitioner):  # noqa
    def __init__(  # noqa
        self,
        network_ports_per_device: int,
        strategy: PartitioningStrategy,
        devices: int,
        nodes: int,
        inseperable_nodes: list[list[int]],
        resources_per_device: dict,
        verbosity: MFVerbosity,
        topology: MFTopology,
        output_dir: Path,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
        limit_nodes_per_device: int | None = None,
        max_utilization: float | None = None,
        ideal_utilization: float | None = None,
    ) -> None:
        super().__init__(
            strategy,
            topology,
            devices,
            nodes,
            inseperable_nodes,
            verbosity,
            resources_per_device,
            output_dir,
            resource_estimates,
            considered_resources,
            network_ports_per_device,
            max_utilization,
            ideal_utilization,
        )
        self.verbosity = verbosity
        if self.model is None or type(self.model) is not Model:
            raise FINNMultiFPGAError("Creation of partitioner model unexpectedly failed")
        self.limit_nodes_per_device = limit_nodes_per_device
        self.model.verbose = 0

        # The status will be set by the _solve method
        # This way we can catch if someone tries to request resource usage before
        # the model was solved
        self.status: mip.OptimizationStatus | None = None

        if (
            ideal_utilization is not None
            and max_utilization is not None
            and ideal_utilization > max_utilization
        ):
            raise FINNMultiFPGAConfigError(
                "Cannot create Multi-FPGA partition if the requested ideal utilization"
                "is greater than the requested max allowed utilization "
                f"({ideal_utilization:.2%} > {max_utilization:.2%})"
            )

        log.debug("Creating partitioning model")

        # self.devices[node][device] = 1: Node <node> is on device <device>
        self.devices = [
            [
                self.model.add_var(name=f"node{node}_on_device{device}", var_type=mip.BINARY)
                for device in range(self.device_count)
            ]
            for node in range(self.node_count)
        ]

        # Every layer can only be on one device
        for node in range(self.node_count):
            self.model += (
                xsum(self.devices[node][device] for device in range(len(self.devices[node]))) == 1
            )

        # Helper to see what device a node is on
        self.chosen_device = [
            self.model.add_var(name=f"chosen_device_of_node_{node}", var_type=mip.INTEGER)
            for node in range(self.node_count)
        ]
        for node in range(self.node_count):
            self.model += self.chosen_device[node] == xsum(
                self.devices[node][device] * device  # type: ignore
                for device in range(self.device_count)
            )

        # Grouped nodes need to stay together
        # First check that no group is too large
        if len(self.inseperable_nodes) > 0:
            nodes_in_groups = sum(len(group) for group in self.inseperable_nodes)
            max_devices_possible = nodes - nodes_in_groups + len(self.inseperable_nodes)
            for i, group in enumerate(self.inseperable_nodes):
                # 1. Single group larger than the model itself
                if len(group) > nodes:
                    raise FINNMultiFPGAError(
                        f"Group {i} of inseperable nodes is larger than the total set of all "
                        f"nodes in the model. (Has {len(group)} nodes, but only {nodes} "
                        "nodes in the graph!)"
                    )
                # 2. Num. nodes == Num. groups. Leads to atleast 1 empty device
                if len(group) == nodes and devices > 1:
                    raise FINNMultiFPGAError(
                        f"Group {i} has the same number of nodes as the graph in total. However "
                        "since more than 1 device is used, this would result in one device "
                        "being completely empty, leading to an invalid partitioning model."
                    )
            # 3. Not enough devices to have this many nodes in groups
            if devices > max_devices_possible:
                raise FINNMultiFPGAError(
                    f"Requested number of FPGAs ({devices}) is larger than the number of "
                    f"devices possible. {nodes - nodes_in_groups} nodes can be alone on a "
                    f"device, and {len(self.inseperable_nodes)} groups of nodes can be on a "
                    f"device. The largest possible device count partitioning would "
                    f"result in {max_devices_possible} devices"
                )

        # Nodes in groups stay together
        for group in self.inseperable_nodes:
            for node in range(len(group) - 1):
                self.model += self.chosen_device[group[node]] == self.chosen_device[group[node + 1]]

        # Number of nodes having this device as their ID
        self.nodesperdevice = [
            self.model.add_var(name=f"lpd_{device}", var_type=mip.INTEGER)
            for device in range(self.device_count)
        ]
        for device in range(self.device_count):
            self.model += self.nodesperdevice[device] == xsum(
                self.devices[node][device] for node in range(self.node_count)
            )

        # Optionally limit number of nodes per device
        # (This may be necessary in some cases to avoid the maximum number of compute units allowed
        # on the FPGAs)
        if self.limit_nodes_per_device is not None:
            log.info(f"Number of nodes per device limited to: {self.limit_nodes_per_device}")
            for device in range(self.device_count):
                self.model += (
                    self.nodesperdevice[device] <= self.limit_nodes_per_device  # type: ignore
                )

        # Connections that a device has with other devices
        self.connections_per_device_helper = [
            [
                self.model.add_var(
                    name=f"nodes_{node}_{node+1}_on_diff_devices", var_type=mip.INTEGER
                )
                for node in range(self.node_count)
            ]
            for device in range(self.device_count)
        ]
        self.connections_per_device = [
            self.model.add_var(name=f"connections_on_device_{device}", var_type=mip.INTEGER)
            for device in range(self.device_count)
        ]
        for device in range(self.device_count):
            for node in range(self.node_count - 1):
                # Variable is 1, if the next node is on a different device
                self.model += (
                    self.connections_per_device_helper[device][node]
                    >= self.devices[node][device] - self.devices[node + 1][device]
                )
                self.model += (
                    self.connections_per_device_helper[device][node]
                    >= self.devices[node + 1][device] - self.devices[node][device]
                )

            # Number of nodes that leave a device
            self.model += self.connections_per_device[device] == xsum(
                self.connections_per_device_helper[device][node] for node in range(self.node_count)
            )

        # Limit the number of connections per device (depends on the FPGAs QSFP ports)
        for device in range(self.device_count):
            self.model += (
                self.connections_per_device[device] <= self.network_ports_per_device
            )  # type: ignore

        # Helper for device difference
        self.device_diff = []
        for i in range(self.node_count):
            self.device_diff.append(
                self.model.add_var(
                    name=f"device_difference_node{i}_to_node{i+1}", var_type=mip.INTEGER
                )
            )

        # Consecutive nodes must be on consecutive devices
        for node in range(self.node_count - 1):
            self.model += (
                self.device_diff[node] >= self.chosen_device[node] - self.chosen_device[node + 1]
            )
            self.model += (
                self.device_diff[node] >= self.chosen_device[node + 1] - self.chosen_device[node]
            )
            self.model += self.device_diff[node] <= 1
            self.model += self.device_diff[node] >= 0

        # Setting topology requirements
        match self.topology:
            case MFTopology.CHAIN:
                self.model += self.chosen_device[0] == 0
                self.model += self.chosen_device[-1] == self.device_count - 1

            case MFTopology.RETURNCHAIN:
                self.model += self.chosen_device[0] == 0
                self.model += self.chosen_device[-1] == self.chosen_device[0]

            case _:
                raise FINNMultiFPGAConfigError(
                    f"Invalid communication scheme for Aurora partitioner: {self.topology}"
                )

        # Objective Function
        if self.strategy == PartitioningStrategy.LAYER_COUNT:
            # Calculcate the difference to the "ideal" load
            # (All devices have the same number of layers)
            avg_diff = [
                self.model.add_var(var_type=mip.CONTINUOUS) for i in range(self.device_count)
            ]
            avg_ideal_load = self.node_count / self.device_count
            for i in range(self.device_count):
                self.model += avg_diff[i] >= self.nodesperdevice[i] - avg_ideal_load  # type: ignore
                self.model += avg_diff[i] >= avg_ideal_load - self.nodesperdevice[i]  # type: ignore

            # Get the largest of those differences
            max_diff = self.model.add_var(name="max_diff", var_type=mip.CONTINUOUS)
            for device in range(self.device_count):
                self.model += max_diff >= avg_diff[device]

            # Try to minimize the max difference to ideal
            self.model.objective = max_diff
            self.model.sense = mip.MINIMIZE

        elif self.strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            assert self.resource_estimates is not None

            # Collect the resource usage of all nodes on a device
            self.resource_use_int = {}
            for device in range(self.device_count):
                self.resource_use_int[device] = {}
                for resource_name in self.considered_resources:
                    self.resource_use_int[device][resource_name] = self.model.add_var(
                        f"resource_use_int_device{device}_resource{resource_name}",
                        var_type=mip.INTEGER,
                    )
                    self.model += self.resource_use_int[device][resource_name] == xsum(
                        [
                            self.devices[node][device]
                            * self.resource_estimates[node][resource_name]
                            for node in range(self.node_count)
                            if resource_name in self.resource_estimates[node].keys()
                        ]
                    )

            # Limit resource usage to (available resources * max usage in percent)
            total_per_device = self.resources_per_device
            for device in range(self.device_count):
                for resource_name in self.considered_resources:
                    assert (
                        self.max_utilization is not None
                    )  # Should be caught in constructor before
                    max_resources = int(total_per_device[resource_name] * self.max_utilization)
                    self.model += self.resource_use_int[device][resource_name] <= max_resources

            # Give a warning if resource usage gets close to its maximum
            for node in range(self.node_count):
                for restype in self.resource_estimates[node].keys():
                    assert self.max_utilization is not None
                    thresh_percentage = 0.1 if self.max_utilization >= 0.1 else self.max_utilization
                    if restype not in total_per_device.keys():
                        continue
                    warn_threshold = (self.max_utilization - thresh_percentage) * total_per_device[
                        restype
                    ]
                    max_util = total_per_device[restype] * self.max_utilization
                    if self.resource_estimates[node][restype] >= warn_threshold:
                        if self.resource_estimates[node][restype] < max_util:
                            log.warning(
                                f"Node {node}'s usage of {restype} is within "
                                f"{thresh_percentage:2.2%} of the maximum allowed utilization "
                                f"({self.resource_estimates[node][restype]} / "
                                f"{max_util}) "
                                "on a single device. Partitioning might fail!"
                            )
                        else:
                            raise FINNMultiFPGAConfigError(
                                f"Node {node}'s usage of {restype} is above "
                                f"the allowed utilization "
                                f"({self.resource_estimates[node][restype]} > "
                                f"{max_util}). "
                                "Theoretical max per device would "
                                f"be {total_per_device[restype]} "
                                "on a single device. Partitioning will fail!"
                            )

            # Balance so that the maximum difference to the ideal load over all devices and
            # resources is as low as possible
            # Use relative values because resources are available at vastly different scales
            self.resource_diff = {}
            self.resource_use_relative = {}
            for device in range(self.device_count):
                self.resource_diff[device] = {}
                self.resource_use_relative[device] = {}
                for resource_name in self.considered_resources:
                    self.resource_diff[device][resource_name] = self.model.add_var(
                        name=f"resource_diff_to_ideal_device{device}_resource{resource_name}",
                        var_type=mip.CONTINUOUS,
                    )
                    self.resource_use_relative[device][resource_name] = self.model.add_var(
                        name=f"resource_use_cont_device{device}_resource{resource_name}",
                        var_type=mip.CONTINUOUS,
                    )

                    # Resource util in relative terms (0-1)
                    self.model += self.resource_use_relative[device][resource_name] == (
                        self.resource_use_int[device][resource_name]
                        / total_per_device[resource_name]
                    )

                    # Convert to float and get diff
                    self.model += self.resource_diff[device][resource_name] >= (
                        self.resource_use_relative[device][resource_name] - self.ideal_util
                    )
                    self.model += self.resource_diff[device][resource_name] >= (
                        self.ideal_util - self.resource_use_relative[device][resource_name]
                    )

            # A device cannot be completely empty
            for device in range(self.device_count):
                self.model += (
                    xsum(
                        [
                            self.resource_use_relative[device][res]
                            for res in self.considered_resources
                        ]
                    )
                    # Needs to be really small so the model is still valid for very small designs
                    >= 0.0000001
                )  # type: ignore

            # The min resource diff to ideal on a device, regardless of resource type
            # (If ideal is 70%, and we have LUT: 61% and DSP: 32%,
            # then we use 61%, so diff is 70%-61%=9%)
            self.min_resource_diff = []
            for device in range(self.device_count):
                self.min_resource_diff.append(
                    self.model.add_var(f"min_resource_diff_device{device}", var_type=mip.CONTINUOUS)
                )
                for res in self.considered_resources:
                    self.model += self.min_resource_diff[device] <= self.resource_diff[device][res]

                    # If we dont specify this, it will stay at the initial value of 0,
                    # since 0 is smaller than all the resource_diffs
                    self.model += self.min_resource_diff[device] >= 0.000000001

            # Maximum of the min resource diff of all devices
            max_diff = self.model.add_var("max_diff", var_type=mip.CONTINUOUS)
            for device in range(self.device_count):
                self.model += max_diff >= xsum(
                    [self.resource_diff[device][res] for res in self.considered_resources]
                )

            # Set objective function
            self.model.objective = max_diff
            self.model.sense = mip.MINIMIZE

        else:
            raise FINNMultiFPGAConfigError(f"Unknown partitioning strategy: {self.strategy}")

    def create_result(self) -> dict[str, int]:
        """Create a uniform result mapping from the internal model. Refer to baseclass docstring
        for more information.
        """
        mapping = {}
        for i in range(self.node_count):
            mapping[i] = self.chosen_device[i].x
        return mapping

    def _get_resource_use_relative(self) -> dict[int, dict[str, Any]]:
        # TODO: Return string-> mapping instead of int->
        if self.status is None:
            raise FINNMultiFPGAError(
                "Resource utilization per device was requested "
                "before the model was solved. Please call solve() first."
            )
        data = {}
        for device in range(self.device_count):
            data[device] = {}
            for restype in self.resource_use_relative[device].keys():
                data[device][restype] = self.resource_use_relative[device][restype].x
        return data


class ApplyPartitioning(Transformation):
    """Apply partitioning from a YAML file to the graph. Can be used to load an existing
    configuration. Afterwards every node has their device_id node attribute set.
    """

    def __init__(self, mapping: dict[str, int] | Path) -> None:
        """Load the mapping either directly or from the specified path."""
        super().__init__()
        self.mapping: dict[str, int] = {}
        if type(mapping) is dict:
            self.mapping = mapping
        elif type(mapping) in [PurePath, Path, PosixPath, WindowsPath]:
            mapping = Path(mapping)  # type: ignore
            if not mapping.exists():
                raise FINNMultiFPGAUserError(
                    f"Cannot read partitioning from " f"{mapping}. No such file exists."
                )
            with mapping.open("r") as f:
                self.mapping = yaml.load(f, yaml.Loader)
        else:
            raise FINNMultiFPGAError(f"Cannot read partitioning config of type {type(mapping)}.")

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Assign the device ids to the nodes."""
        for nodename, device in self.mapping.items():
            node = model.get_node_from_name(nodename)
            if node is None:
                raise FINNMultiFPGAError(
                    f"Tried assigning node {nodename} "
                    f"to device {device}, but the model does "
                    f"not contain any nodes of that name. "
                    f"Is your partitioning config outdated?"
                )
            set_device_id(node, device)
        return model, False


class PartitionForMultiFPGA(Transformation):
    """Receive a model with only FPGADataflow nodes and partition it by assigning it's
    device node attribute. Partitioning is done with respect to the chosen strategy.
    To determine how partitioning is done, pass the partitioner type yourself.

    The resulting IDs are assigned to the nodes
    and additionally stored in .../output_dir/partitioning.yaml.
    """

    def __init__(  # noqa
        self,
        partitioning_strategy: PartitioningStrategy,
        num_fpgas: int,
        topology: MFTopology,
        communication_kernel: MFCommunicationKernel,
        considered_resources: list[str],
        max_utilization: float,
        ideal_utilization: float,
        ports_per_device: int,
        fpga_part: str,
        board: str,
        timeout: int,
        output_dir: Path,
        verbosity: MFVerbosity,
    ) -> None:
        self.verbosity = verbosity
        self.board = board
        self.topology = topology
        self.output_dir = output_dir
        self.part = fpga_part
        self.num_fpgas = num_fpgas
        self.considered_resources = considered_resources
        self.max_utilization = max_utilization
        self.ideal_utilization = ideal_utilization
        self.ports_per_device = ports_per_device
        self.communication_kernel = communication_kernel
        self.partitioning_strategy = partitioning_strategy
        self.timeout = timeout

        # Select the partitioner class based on the communication kernel
        partitioners = {MFCommunicationKernel.AURORA: AuroraPartitioner}
        try:
            self.partitioner_type = partitioners[communication_kernel]
        except KeyError as ke:
            raise FINNMultiFPGAConfigError(
                f"There is currently no partitioner implementation "
                f"for usage with the communication kernel "
                f"{communication_kernel.name}"
            ) from ke
        if verbosity.value > MFVerbosity.LOW.value:
            log.info(
                f"Based on the communication kernel, "
                f'partitioner "{self.partitioner_type.__name__}" was chosen!'
            )

        # Run some checks
        # Needed to resolve platform
        if board is None:
            raise FINNMultiFPGAConfigError(
                "Parameter 'board' is required in config for MultiFPGA partitioning"
            )

        # Set the target device count
        if num_fpgas < 0:
            self.devices = self.estimate_required_fpgas()
            raise NotImplementedError()
        else:  # noqa
            self.devices = num_fpgas

    def estimate_required_fpgas(self) -> int:
        """Use resource utilization to estimate how many FPGAs will be needed to
        partition the given model.
        """
        raise NotImplementedError()

    def check_missing_estimates(self, estimates: dict[int, dict[str, int | float]]) -> None:
        """Check that all layers have some resource estimation associated.
        The test is only done if the RESOURCE_UTILIZATION partitioning strategy is used.
        """
        if self.partitioning_strategy == PartitioningStrategy.RESOURCE_UTILIZATION:
            missing_estimates = False
            for layer in estimates.keys():
                if all(estimates[layer][res] <= 0 for res in estimates[layer].keys()):
                    missing_estimates = True
                    log.critical(
                        f"Layer {layer} has an all-0 resource estimation for all "
                        "resource types. Cannot partition using resource estimates!"
                    )
            if missing_estimates:
                raise FINNMultiFPGAError(
                    "Cannot partition with faulty resource estimations and "
                    "RESOURCE_UTILIZATION as PartitioningStrategy. Check logs to find information"
                    "about which layers have missing resource estimates!"
                )

    def show_mapping(self, model: ModelWrapper, mapping: dict[str, int]) -> None:
        """Display mapping either as table or prints, depending on console size."""
        # TODO: Make dependent on verbose info flag in partitioning config
        with LogDisabledConsole() as cons:
            required_tables = ceil(len(model.graph.node) / (cons.height - 5))
            allowed_tables = cons.width / 20
            if required_tables < allowed_tables:
                entries_per_table = (len(model.graph.node) // required_tables) + 1
                tables = [Table(box=box.SIMPLE) for _ in range(required_tables)]
                layout = Layout()
                layout.split_row(*tables)
                for table in tables:
                    table.add_column("Index", justify="center", header_style="bold")
                    table.add_column("Node Name", justify="left", header_style="bold")
                    table.add_column("Dev", justify="left", header_style="bold", style="bold green")
                for i, node in enumerate(model.graph.node):
                    log.info(str(i) + ": " + str(i // entries_per_table))
                    tables[i // entries_per_table].add_row(
                        str(i), node.name, str(int(mapping[node.name]))
                    )
                cons.print(layout)
                return
        for node in model.graph.node:
            log.info(f"Mapping {node.name} -> {int(mapping[node.name])}")
        return

    def _log_pre_solve_information(self, partitioner: Partitioner) -> None:
        """Log some information before starting to solve the LP."""
        if self.verbosity.value > MFVerbosity.LOW.value:
            log.info(
                f"[bold green]Starting solver [/bold green][Name: [bold blue]"
                f"{partitioner.model.solver_name}[/bold blue], "
                f"Timeout: {self.timeout}]...",
                extra={"markup": True},
            )
        if self.verbosity.value > MFVerbosity.MEDIUM.value:
            log.info(f"Number of variables in model: {len(partitioner.model.vars)}")
            log.info(f"Number of constraints in model: {len(partitioner.model.constrs)}")

    def _log_post_solve_information(
        self,
        model: ModelWrapper,
        mapping: dict[str, int],
        util: dict[int, dict[str, Any]] | None,
    ) -> None:
        """Log some information after the solver is done. Also shows the partitioning results."""
        if self.verbosity.value > MFVerbosity.LOW.value:
            log.info("[bold green]Solver done.[/bold green]", extra={"markup": True})
        # Resource utilization
        if util is not None and self.verbosity.value > MFVerbosity.MEDIUM.value:
            log.info("Relative resource utilization")
            for device, device_util in util.items():
                log.info(
                    f"Device {device}:  "
                    + ", ".join(f"{k}: {v:.1%}" for k, v in device_util.items())
                )

        # Report results
        # TODO: This currently does not store the mapping in the log. This is
        # TODO: currently done via solution.txt, which should be put into the output_dir
        if self.verbosity.value == MFVerbosity.EXTRA_HIGH.value:
            self.show_mapping(model, mapping)
        if self.verbosity.value > MFVerbosity.NONE.value:
            device_nodes = {}
            for i in range(len(model.graph.node)):
                dev = int(mapping[model.graph.node[i].name])
                if dev not in device_nodes.keys():
                    device_nodes[dev] = 0
                device_nodes[dev] += 1
            log.info("Partitioning results:")
            for dev in device_nodes.keys():
                percentage = float(device_nodes[dev]) / float(len(model.graph.node))
                log.info(f"Device {dev}: {device_nodes[dev]} nodes ({percentage:.1%})")

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:  # noqa
        if self.devices > len(model.graph.node):
            # Stop if there are more devices than nodes
            raise FINNMultiFPGAConfigError(
                f"Model infeasible: Cannot partition a model with "
                f"{len(model.graph.node)} to {self.devices} devices!"
            )
        if not is_single_in_out_model(model):
            # Dont split during branches. Find all layers that should be on the same device.
            raise FINNMultiFPGAConfigError(
                "The model has either more than 1 input or more than 1 output nodes. "
                "This might cause issue during partitioning. Please check your ONNX file."
            )

        # Start by gathering node groups that cannot be split (due to branching) TODO: MultiFPGA 2.0
        if self.verbosity.value > MFVerbosity.MEDIUM.value:
            log.info("Gathering inseparable node groups...")
        inseperable_nodes = get_inseparable_nodes(model)

        # Calculate resource estimates for the solver objective function (if needed)
        model = model.transform(GiveUniqueNodeNames())
        estimates = get_estimated_model_resources(model, self.part)
        self.check_missing_estimates(estimates)

        # Create the partitioner itself
        device_resources = available_resources(platforms[self.board](), self.considered_resources)
        partitioner = self.partitioner_type(
            devices=self.devices,
            topology=self.topology,
            strategy=self.partitioning_strategy,
            inseperable_nodes=inseperable_nodes,
            nodes=len(model.graph.node),
            verbosity=self.verbosity,
            output_dir=self.output_dir,
            resources_per_device=device_resources,
            considered_resources=self.considered_resources,
            resource_estimates=estimates,
            max_utilization=self.max_utilization,
            ideal_utilization=self.ideal_utilization,
            network_ports_per_device=self.ports_per_device,
        )

        # Temporary dir to store information regarding the partitioning
        logdir = Path(make_build_dir("partition_solver_"))

        # Print information
        self._log_pre_solve_information(partitioner)

        # Actually try to solve the model
        mapping = partitioner.solve(
            solver_timeout=self.timeout,
        )
        if mapping is None:
            raise FINNMultiFPGAConfigError(
                f"No feasible partitioning solution could be found for "
                f"the given model and configuration. If you are sure "
                f"that everything is set up correctly, try using a "
                f"different solver. Reports can be found at: "
                f"{logdir.absolute()}"
            )

        # Apply results back to the model
        # TODO: Warning for very low resource usage
        model = model.transform(ApplyPartitioning(mapping))

        # Write results
        partitioner.write_results(self.output_dir / "partitioning.yaml")

        # Print results to console
        self._log_post_solve_information(model, mapping, partitioner.get_resource_use_relative())

        return model, False
