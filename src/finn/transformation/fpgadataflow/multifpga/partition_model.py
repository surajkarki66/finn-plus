import mip
import rich.box
import yaml
from math import ceil
from pathlib import Path, PosixPath, PurePath, WindowsPath
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import GiveUniqueNodeNames
from rich.layout import Layout
from rich.table import Table
from typing import Any

from finn.builder.build_dataflow_config import (
    MFCommunicationKernel,
    MFVerbosity,
    PartitioningConfiguration,
    PartitioningStrategy,
)
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import AuroraPartitioner
from finn.transformation.fpgadataflow.multifpga.graph import (
    get_inseparable_nodes,
    is_single_in_out_model,
)
from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner
from finn.util.basic import make_build_dir
from finn.util.exception import (
    FINNInternalError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGANoPartitionerSolutionError,
    FINNMultiFPGAUserError,
)
from finn.util.fpgadataflow import set_device_id
from finn.util.logging import LogDisabledConsole, log
from finn.util.platforms import platforms
from finn.util.resources import available_resources, get_estimated_model_resources


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
                    f"Cannot read partitioning from {mapping}. No such file exists."
                )
            with mapping.open("r") as f:
                self.mapping = yaml.load(f, yaml.Loader)
        else:
            raise FINNMultiFPGAError(f"Cannot read partitioning config of type {type(mapping)}.")

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Assign the device ids to the nodes."""
        done = 0
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
            done += 1
        if done != len(model.graph.node):
            raise FINNInternalError(
                f"Something went wrong when partitioning. "
                f"Some nodes did not receive a device ID. Set "
                f"device IDs: {done} / Total expected: {len(model.graph.node)}"
            )
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
        partitioning_configuration: PartitioningConfiguration,
        fpga_part: str,
        board: str,
        output_dir: Path,
    ) -> None:
        self.verbosity = partitioning_configuration.verbosity
        self.board = board
        self.topology = partitioning_configuration.topology
        self.output_dir = output_dir
        self.part = fpga_part
        self.num_fpgas = partitioning_configuration.num_fpgas
        self.considered_resources = partitioning_configuration.considered_resources
        self.max_utilization = partitioning_configuration.max_utilization
        self.ideal_utilization = partitioning_configuration.ideal_utilization
        self.ports_per_device = partitioning_configuration.ports_per_device
        self.communication_kernel = partitioning_configuration.communication_kernel
        self.partitioning_strategy = partitioning_configuration.partition_strategy
        self.timeout = partitioning_configuration.partition_solver_timeout
        self.solver = partitioning_configuration.partition_solver
        self.solver_emphasis = partitioning_configuration.partition_solver_emphasis

        # Select the partitioner class based on the communication kernel
        partitioners = {MFCommunicationKernel.AURORA: AuroraPartitioner}
        try:
            self.partitioner_type = partitioners[self.communication_kernel]
        except KeyError as ke:
            raise FINNMultiFPGAConfigError(
                f"There is currently no partitioner implementation "
                f"for usage with the communication kernel "
                f"{self.communication_kernel.name}"
            ) from ke
        if self.verbosity.value > MFVerbosity.LOW.value:
            log.info(
                f"Based on the communication kernel, "
                f'partitioner "{self.partitioner_type.__name__}" was chosen!'
            )

        self.partitioner: Partitioner | None = None
        self.mapping: dict[str, int] | None = None

        # Run some checks
        # Needed to resolve platform
        if board is None:
            raise FINNMultiFPGAConfigError(
                "Parameter 'board' is required in config for MultiFPGA partitioning"
            )

        # Set the target device count
        if self.num_fpgas < 0:
            self.devices = self.estimate_required_fpgas()
            raise NotImplementedError()
        else:  # noqa
            self.devices = self.num_fpgas

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
                tables = [Table(box=rich.box.SIMPLE) for _ in range(required_tables)]
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
        inseparable_nodes = get_inseparable_nodes(model)

        # Calculate resource estimates for the solver objective function (if needed)
        model = model.transform(GiveUniqueNodeNames())
        estimates = get_estimated_model_resources(model, self.part, self.considered_resources, True)
        self.check_missing_estimates(estimates)

        # Create the partitioner itself
        device_resources = available_resources(platforms[self.board](), self.considered_resources)
        self.partitioner = self.partitioner_type(
            devices=self.devices,
            topology=self.topology,
            strategy=self.partitioning_strategy,
            inseparable_nodes=inseparable_nodes,
            nodes=len(model.graph.node),
            verbosity=self.verbosity,
            output_dir=self.output_dir,
            resources_per_device=device_resources,
            considered_resources=self.considered_resources,
            resource_estimates=estimates,
            max_utilization=self.max_utilization,
            ideal_utilization=self.ideal_utilization,
            network_ports_per_device=self.ports_per_device,
            index_node_name_map={i: model.graph.node[i].name for i in range(len(model.graph.node))},
            solver=self.solver,
            solver_emphasis=self.solver_emphasis,
        )

        # Temporary dir to store information regarding the partitioning
        logdir = Path(make_build_dir("partitioning_model_data_"))

        # Store the model definition. This is useful for debugging
        self.partitioner.model.write(str((logdir / "model.lp").absolute()))

        # Print information
        self._log_pre_solve_information(self.partitioner)

        # Actually try to solve the model
        self.mapping = self.partitioner.solve(
            solver_timeout=self.timeout,
        )
        if self.mapping is None:
            raise FINNMultiFPGANoPartitionerSolutionError(
                f"No feasible partitioning solution could be found for "
                f"the given model and configuration. If you are sure "
                f"that everything is set up correctly, try using a "
                f"different solver. The generated model can be found at: "
                f"{logdir.absolute()}"
            )
        if self.partitioner.status is not None and self.verbosity.value > MFVerbosity.LOW.value:
            if self.partitioner.status == mip.OptimizationStatus.OPTIMAL:
                log.info("OPTIMAL solution found!")
            elif self.partitioner.status == mip.OptimizationStatus.FEASIBLE:
                log.info("FEASIBLE solution found.")
            else:
                log.info(f"Model optimization status: {self.partitioner.status.name}")

        # Apply results back to the model
        # TODO: Warning for very low resource usage
        model = model.transform(ApplyPartitioning(self.mapping))

        # Write results
        self.partitioner.write_results(self.output_dir / "partitioning.yaml")

        # Print results to console
        self._log_post_solve_information(
            model, self.mapping, self.partitioner.get_resource_use_relative()
        )

        return model, False
