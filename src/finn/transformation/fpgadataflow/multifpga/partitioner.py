"""Partitioners for Multi-FPGA usage."""

from __future__ import annotations

import mip
import yaml
from abc import ABC, abstractmethod
from mip import Model
from pathlib import Path
from typing import Any

from finn.builder.build_dataflow_config import MFTopology, MFVerbosity, PartitioningStrategy
from finn.util.basic import make_build_dir
from finn.util.exception import (
    FINNInternalError,
    FINNMultiFPGAError,
    FINNMultiFPGAUserError,
    FINNUserError,
)
from finn.util.logging import log


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
        inseparable_nodes: Nodes that need to stay together because they are in a split

        considered_resources: What types of resources are used in the objective
            function to determine load.
    """  # noqa

    def __init__(  # noqa
        self,
        strategy: PartitioningStrategy,
        topology: MFTopology | None,
        devices: int,
        nodes: int,
        inseparable_nodes: list[list[int]],
        verbosity: MFVerbosity,
        resources_per_device: dict,
        output_dir: Path,
        resource_estimates: dict | None = None,
        considered_resources: list[str] | None = None,
        network_ports_per_device: int = 2,
        max_utilization: float | None = None,
        ideal_utilization: float | None = None,
        index_node_name_map: dict[int, str] | None = None,
        solver: str | None = None,
        solver_emphasis: mip.SearchEmphasis = mip.SearchEmphasis.DEFAULT,
    ) -> None:
        # MIP member variables first
        self.status: mip.OptimizationStatus | None
        self.solver = solver
        self.solver_emphasis = solver_emphasis
        if solver is None:
            try:
                self.model = Model()
            except OSError:
                log.warning(
                    "Creation of mip.Model failed. This might be known bug "
                    "(LD_LIBRARY_PATH only modified at runtime to point to "
                    "libgurobi instead of before). Falling back to CBC"
                )  # See finn-plus issue #67
                self.model = Model(solver_name=mip.CBC)
            except mip.exceptions.InterfacingError as e:
                log.warning(
                    f"Could not create a default-initialized mip.Model. "
                    f"The error encountered was: {e}. Trying to fallback to a CBC based model."
                )
                self.model = Model(solver_name=mip.CBC)
        else:
            try:
                self.model = Model(solver_name=solver)
            except mip.exceptions.InterfacingError as e:
                raise FINNUserError(
                    f"Cannot create mip solver of type {solver}. Original error: {e}"
                ) from e
        self.model.name = "FINN_MultiFPGA_Partitioning_Model"

        # Set model emphasis
        self.model.emphasis = self.solver_emphasis

        # Document setup
        if verbosity.value > MFVerbosity.NONE.value:
            log.info(f"Using solver: {self.model.solver_name}")
        if verbosity.value > MFVerbosity.LOW.value:
            log.info(f"Solver emphasis: {self.solver_emphasis.name}")

        # Details about the partitioning
        self.index_node_map = index_node_name_map
        if self.index_node_map is not None:
            for i in range(nodes):
                if i not in self.index_node_map.keys():
                    raise FINNInternalError(
                        f"Cannot use index-node map, since name of node index {i} is missing!"
                    )
        self.strategy = strategy
        self.max_utilization = max_utilization
        self.topology = topology
        self.ideal_util = ideal_utilization
        self.inseparable_nodes = inseparable_nodes
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
            log.info(f"Groups of inseparable nodes: {len(self.inseparable_nodes)}")
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
        """  # noqa

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
