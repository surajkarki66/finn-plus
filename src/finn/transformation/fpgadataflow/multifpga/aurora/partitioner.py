import mip
from mip import Model, xsum
from pathlib import Path
from typing import Any, Literal

from finn.builder.build_dataflow_config import (
    MFTopology,
    MFVerbosity,
    MIPSolver,
    PartitioningStrategy,
)
from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner
from finn.util.exception import (
    FINNInternalError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGANoPartitionerSolutionError,
)
from finn.util.logging import log


class AuroraPartitioner(Partitioner):  # noqa
    def __init__(  # noqa
        self,
        network_ports_per_device: int,
        strategy: PartitioningStrategy,
        devices: int,
        nodes: int,
        inseparable_nodes: list[list[int]],
        resources_per_device: dict,
        verbosity: MFVerbosity,
        topology: MFTopology,
        output_dir: Path,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
        limit_nodes_per_device: int | None = None,
        max_utilization: float | None = None,
        ideal_utilization: float | None = None,
        index_node_name_map: dict[int, str] | None = None,
        solver: Literal[MIPSolver.CBC, MIPSolver.GUROBI, MIPSolver.HIGHS] | None = None,
        solver_emphasis: mip.SearchEmphasis = mip.SearchEmphasis.DEFAULT,
    ) -> None:
        super().__init__(
            strategy,
            topology,
            devices,
            nodes,
            inseparable_nodes,
            verbosity,
            resources_per_device,
            output_dir,
            resource_estimates,
            considered_resources,
            network_ports_per_device,
            max_utilization,
            ideal_utilization,
            index_node_name_map,
            solver=solver,
            solver_emphasis=solver_emphasis,
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
        if len(self.inseparable_nodes) > 0:
            nodes_in_groups = sum(len(group) for group in self.inseparable_nodes)
            max_devices_possible = nodes - nodes_in_groups + len(self.inseparable_nodes)
            for i, group in enumerate(self.inseparable_nodes):
                # 1. Single group larger than the model itself
                if len(group) > nodes:
                    raise FINNMultiFPGAError(
                        f"Group {i} of inseparable nodes is larger than the total set of all "
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
                    f"device, and {len(self.inseparable_nodes)} groups of nodes can be on a "
                    f"device. The largest possible device count partitioning would "
                    f"result in {max_devices_possible} devices"
                )

        # Nodes in groups stay together
        for group in self.inseparable_nodes:
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
                            raise FINNMultiFPGANoPartitionerSolutionError(
                                f"Node {node}'s usage of {restype} is above "
                                f"the allowed utilization per device "
                                f"({self.resource_estimates[node][restype]} > "
                                f"{max_util}). "
                                "Theoretical max per device would "
                                f"be {total_per_device[restype]}. "
                                "The node cannot fit on a device of this type."
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
        if self.index_node_map is None:
            raise FINNInternalError(
                "Cannot create result mapping, because no "
                "Index-Nodename mapping was passed to the partitioner object."
            )
        mapping = {}
        for i in range(self.node_count):
            mapping[self.index_node_map[i]] = int(self.chosen_device[i].x)  # type: ignore
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
