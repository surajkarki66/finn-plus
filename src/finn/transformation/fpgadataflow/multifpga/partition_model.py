import mip
import rich.box
import yaml
from collections import Counter
from math import ceil
from pathlib import Path, PosixPath, PurePath, WindowsPath
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from rich.layout import Layout
from rich.table import Table

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFCommunicationKernel,
    MFVerbosity,
    PartitioningConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.aurora.partitioner import AuroraPartitioner
from finn.transformation.fpgadataflow.multifpga.partitioner import Partitioner
from finn.util.basic import make_build_dir
from finn.util.exception import (
    FINNInternalError,
    FINNMultiFPGAConfigError,
    FINNMultiFPGAError,
    FINNMultiFPGAPartitionerError,
    FINNMultiFPGAUserError,
)
from finn.util.fpgadataflow import set_device_id
from finn.util.logging import LogDisabledConsole, log


class ApplyPartitioning(Transformation):
    """Apply partitioning from a YAML file to the graph. Can be used to load an existing
    configuration. Afterwards every node has their device_id node attribute set.

    This expects a YAML file:
    ```
    MVAU_hls_0: 1
    FMPadding_rtl_2: 19
    ...
    ```
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

    The resulting IDs are assigned to the nodes
    and additionally stored in .../output_dir/partitioning.yaml.
    """

    def __init__(self, cfg: DataflowBuildConfig) -> None:  # noqa
        self.cfg = cfg
        if self.cfg.partitioning_configuration is None:
            raise FINNMultiFPGAConfigError(
                "Partitioning config is None, but " "'PartitionForMultiFPGA' was called. "
            )
        self.pcfg: PartitioningConfiguration = cfg.partitioning_configuration  # type: ignore
        self.verbosity = self.pcfg.verbosity

        # Select the partitioner class based on the communication kernel
        partitioners = {MFCommunicationKernel.AURORA: AuroraPartitioner}
        try:
            self.partitioner_type = partitioners[self.pcfg.communication_kernel]
        except KeyError as ke:
            raise FINNMultiFPGAConfigError(
                f"There is currently no partitioner implementation "
                f"for usage with the communication kernel "
                f"{self.pcfg.communication_kernel.name}"
            ) from ke

        # These will be filled out after partitioning
        self.partitioner: Partitioner | None = None
        self.mapping: dict[str, int] | None = None

        # Set the target device count
        if self.pcfg.num_fpgas < 0:
            self.pcfg.num_fpgas = self.estimate_required_fpgas()

    def estimate_required_fpgas(self) -> int:
        """Use resource utilization to estimate how many FPGAs will be needed to
        partition the given model.
        """
        raise NotImplementedError()

    def show_mapping(self, mapping: dict[str, int]) -> None:
        """Display mapping either as table or prints, depending on console size."""
        keys = list(mapping.keys())
        with LogDisabledConsole() as cons:
            required_tables = ceil(len(keys) / (cons.height - 5))
            allowed_tables = cons.width / 20
            if required_tables < allowed_tables:
                entries_per_table = (len(keys) // required_tables) + 1
                tables = [Table(box=rich.box.SIMPLE) for _ in range(required_tables)]
                layout = Layout()
                layout.split_row(*tables)
                for table in tables:
                    table.add_column("Index", justify="center", header_style="bold")
                    table.add_column("Node Name", justify="left", header_style="bold")
                    table.add_column("Dev", justify="left", header_style="bold", style="bold green")
                for i, key in enumerate(keys):
                    log.info(str(i) + ": " + str(i // entries_per_table))
                    tables[i // entries_per_table].add_row(str(i), key, str(int(mapping[key])))
                cons.print(layout)
                return
        for key in keys:
            log.info(f"Mapping {key} -> {mapping[key]}")
        return

    def _log_pre_solve_information(self, partitioner: Partitioner) -> None:
        """Log some information before starting to solve the LP."""
        if self.verbosity.value > MFVerbosity.LOW.value:
            solver = (
                self.pcfg.partition_solver.value
                if self.pcfg.partition_solver is not None
                else "Default"
            )
            log.info(f"Using solver: {solver}")
            log.info(f"Solver emphasis: {self.pcfg.partition_solver_emphasis.name}")
        if self.verbosity.value > MFVerbosity.MEDIUM.value:
            log.info(f"Number of variables in model: {len(partitioner.model.vars)}")
            log.info(f"Number of constraints in model: {len(partitioner.model.constrs)}")

    def _log_post_solve_information(
        self,
        mapping: dict[str, int],
        partitioner: Partitioner,
    ) -> None:
        """Log some information after the solver is done. Also shows the partitioning results."""
        if self.partitioner is None or self.partitioner.status is None:
            raise FINNInternalError(
                "Cannot log post-solving information before the model was solved."
            )
        if self.verbosity.value > MFVerbosity.LOW.value:
            log.info("[bold green]Solver done.[/bold green]", extra={"markup": True})
        # Status
        if self.partitioner.status is not None and self.verbosity.value > MFVerbosity.LOW.value:
            if self.partitioner.status == mip.OptimizationStatus.OPTIMAL:
                log.info("OPTIMAL solution found!")
            elif self.partitioner.status == mip.OptimizationStatus.FEASIBLE:
                log.info("FEASIBLE solution found.")
            else:
                log.info(f"Model optimization status: {self.partitioner.status.name}")
        # Resource utilization
        util = partitioner.get_resource_use_relative()
        if util is not None and self.verbosity.value > MFVerbosity.NONE.value:
            log.info("Relative resource utilization")
            for device, device_util in util.items():
                log.info(
                    f"Device {device}:  "
                    + ", ".join(f"{k}: {v:.1%}" for k, v in device_util.items())
                )
        # Report results
        if self.verbosity.value == MFVerbosity.EXTRA_HIGH.value:
            log.info(f"Model objective value: {partitioner.model.objective.x}")
            self.show_mapping(mapping)
        # Show layer-wise partitioning results
        if self.verbosity.value > MFVerbosity.NONE.value:
            counter = Counter(mapping.values())
            log.info("Partitioning results:")
            for dev in counter.keys():
                percentage = counter[dev] / counter.total()
                log.info(f"Device {dev}: {counter[dev]} nodes ({percentage:.1%})")

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:  # noqa
        # Create the partitioner
        self.partitioner = self.partitioner_type(self.cfg, model)

        logdir = Path(make_build_dir("partitioning_model_data_"))
        self.partitioner.model.write(str((logdir / "model.lp").absolute()))
        self._log_pre_solve_information(self.partitioner)
        self.mapping = self.partitioner.solve(solver_timeout=self.pcfg.partition_solver_timeout)
        if self.mapping is None:
            raise FINNMultiFPGAPartitionerError(
                f"No feasible partitioning solution could be found for "
                f"the given model and configuration. If you are sure "
                f"that everything is set up correctly, try using a "
                f"different solver. The generated model can be found at: "
                f"{logdir.absolute()}"
            )

        # Apply results back to the model
        model = model.transform(ApplyPartitioning(self.mapping))

        # Write results to build dir and log dir
        self.partitioner.write_results(logdir / "partitioning.yaml")
        self.partitioner.write_results(Path(self.cfg.output_dir) / "partitioning.yaml")

        # Print results to console
        self._log_post_solve_information(self.mapping, self.partitioner)

        return model, False
