"""Dataclass to store a linking configuration for vitis builds. This is essential to enable multiple
transformations / steps to edit the same configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from mashumaro.mixins.yaml import DataClassYAMLMixin
from pathlib import Path
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, cast

from finn.builder.build_dataflow_config import FpgaMemoryType, VitisOptStrategy
from finn.templates import get_jinja_environment
from finn.transformation.fpgadataflow.multifpga.utils import get_device_id
from finn.util.basic import make_build_dir
from finn.util.exception import FINNInternalError, FINNVitisLinkConfigError
from finn.util.fpgadataflow import (
    check_all_sdp_nodes,
    check_graph_is_line,
    get_submodel,
    get_vitis_xo,
)
from finn.util.logging import log

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper


@dataclass
class VitisLinkConfiguration(DataClassYAMLMixin):
    """Manages XO files, CU instantiations, stream connections,
    port connections, Vivado props, etc.
    It can output a linking configuration to pass to v++ and
    create a shell script to run it. Tries to be as strict and careful as possible,
    and depending on the issue raises an Exception, logs an error or warning
    or continues silently.
    """

    config_path: Path
    f_mhz: int
    optimization_level: str
    platform: str
    run_script_path: Path = field(init=False, default_factory=lambda: Path())
    cu: list[str] = field(default_factory=list)
    nk: list[tuple[str, str]] = field(default_factory=list)
    sc: dict[str, list[str]] = field(default_factory=dict)
    sp: dict[str, str] = field(default_factory=dict)
    xo: list[Path] = field(default_factory=list)
    slr: dict[str, str] = field(default_factory=dict)
    connects: list[tuple[str, str]] = field(default_factory=list)
    vivado_section: str = ""
    connectivity_section: str = ""

    def __post_init__(self) -> None:  # noqa
        self.config_path.parent.mkdir(exist_ok=True, parents=True)
        self.run_script_path = self.config_path.parent / "run_link.sh"
        self.run_script_path.parent.mkdir(exist_ok=True, parents=True)

    def store(self, p: Path) -> None:
        """Store the config as a YAML file, so that it can be loaded
        by other transformations again.
        """
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(self.to_yaml()))

    @staticmethod
    def load(p: Path) -> VitisLinkConfiguration:
        """Load a VitisLinkConfiguration from the given YAML file."""
        return VitisLinkConfiguration.from_yaml(p.read_text())

    @staticmethod
    def load_from_model(model: ModelWrapper) -> dict[int, VitisLinkConfiguration]:
        """Load all VitisLinkConfigurations from a modelwrapper. The path to this
        directory should be stored in the "vitis_link_configs" metadata prop.
        The function expects the configs to be stored in directories
            `link_configs/0/config.yaml`
        where 0 is the device ID and link_configs the stored path.

        Returns
        -------
            dict[int, VitisLinkConfiguration]: Maps device-IDs to their respective
                linking configurations.
        """
        storage_path = model.get_metadata_prop("vitis_link_configs")
        if storage_path is None:
            raise FINNVitisLinkConfigError(
                "Cannot load VitisLinkConfig from model, "
                "since the metadata prop "
                "'vitis_link_configs' was not found!"
            )
        storage_path = Path(storage_path)
        if not storage_path.exists():
            raise FINNVitisLinkConfigError(
                f"Cannot load VitisLinkConfigs from invalid path: {storage_path}"
            )

        configs = {}
        for device_path in storage_path.iterdir():
            configs[int(str(device_path.name))] = VitisLinkConfiguration.load(
                storage_path / device_path / "config.yaml"
            )
        return configs

    @staticmethod
    def store_to_model(  # noqa
        configs: dict[int, VitisLinkConfiguration], model: ModelWrapper, path: Path | None = None
    ) -> ModelWrapper:
        """Store all VitisLinkConfigurations into a modelwrapper. The path to this
        directory will be stored in the "vitis_link_configs" metadata prop.
        The function stores the configs in directories
            `link_configs/0/config.yaml`
        where 0 is the device ID and link_configs the stored path.

        Arguments:
        ---------
            `path`: Path to a directory in which the configs are stored. This path may contain no
                other files or directories. If no path is given, the function tries to read it
                from the model itself. This only works if the path has been set once manually.
            `configs`: The configuration objects to store.
            `model`: The model to update when the configs are stored.

        Returns:
        -------
            ModelWrapper: The modified modelwrapper with the updated metadata prop.
        """
        if path is None:
            existing_path = model.get_metadata_prop("vitis_link_configs")
            if existing_path is None:
                raise FINNInternalError(
                    "Can only store link config without explicit path, "
                    "if the config has been saved at a given path once "
                    "before. If this is the first call to this function "
                    "for this config, provide a path!"
                )
            path = Path(existing_path)
        path.mkdir(exist_ok=True, parents=True)
        model.set_metadata_prop("vitis_link_configs", str(path.absolute()))
        for device, config in configs.items():
            config.store(path / Path(str(device)) / "config.yaml")
        return model

    def add_cu(self, kernel_name: str, cu_name: str) -> None:
        """Add a compute unit (instance of a kernel)."""
        if cu_name in self.cu:
            kern = next(kname for kname, cname in self.nk if cname == cu_name)
            raise FINNVitisLinkConfigError(
                f"Tried creating CU {cu_name}, but a CU of this "
                f"name of kernel {kern} already exists!"
            )
        self.cu.append(cu_name)
        self.nk.append((kernel_name, cu_name))

    def add_sc(self, cu_sender: str, cu_receiver: str) -> None:
        """Add a Streaming Connection between two CUs:
        >>> lc = VitisLinkConfiguration("", "", 100)
        >>> lc.add_cu("A", "a")
        >>> lc.add_cu("B", "b")
        >>> lc.add_sc("a.out", "b.in")
        >>> lc.sc["a.out"]
        ['b.in']
        """  # noqa
        # Check formatting
        for cu in [cu_sender, cu_receiver]:
            splits = cu.split(".")
            if len(splits) != 2:
                raise FINNVitisLinkConfigError(
                    f"{cu} is incorrectly formatted. Required "
                    f"syntax to add a streaming connection from CU "
                    f'a on port out is "a.out".'
                )

        # Yield warning if the direction seems wrong
        sender_port = cu_sender.split(".")[1]
        receiver_port = cu_receiver.split(".")[1]
        if any(n in sender_port.lower() for n in ["s_axis", "in"]) or any(
            n in receiver_port.lower() for n in ["m_axis", "out"]
        ):
            log.error(
                f"Adding connection sc={cu_sender}:{cu_receiver}. The port "
                "names suggest that the order of sender and receiver might be "
                "swapped. Proceeding now."
            )

        # Add the connection
        if cu_sender not in self.sc.keys():
            self.sc[cu_sender] = []
        self.sc[cu_sender].append(cu_receiver)

    def add_slr(self, cu: str, slr: str) -> None:
        """Place the given CU on the given SLR."""
        self.slr[cu] = slr

    def add_sp(self, cu_port_name: str, mem_type: str) -> None:
        """Add an SP assignment."""
        known_sp_names = ["DDR", "HBM", "PLRAM", "HOST"]
        if all(sp_name not in mem_type for sp_name in known_sp_names):
            log.warning(
                f"Adding system port connection {cu_port_name}:{mem_type}. "
                f"System port tag {mem_type} might be unknown."
            )
        self.sp[cu_port_name] = mem_type

    def add_connect(self, a: str, b: str) -> None:
        """Add a connect assignment. Not to be confused with stream_connect (sc)."""
        self.connects.append((a, b))

    def add_vivado_line(self, line: str) -> None:
        """Add a custom line to the vivado section."""
        self.vivado_section += line + ("" if line[-1] != "\n" else "\n")

    def add_xo(self, xo_files: Path | list[Path] | str) -> None:
        """Add an XO file to the list of XO files that will be passed upon linking.
        Ignores duplicate calls.
        """
        all_xos = []
        if type(xo_files) in [Path, str]:
            all_xos = [Path(xo_files)]  # type: ignore
        elif type(xo_files) is list:
            all_xos = xo_files
        else:
            all_xos = [Path(xo_files)]  # type: ignore

        for xo_file in all_xos:
            if xo_file in self.xo:
                log.warning(f"Ignoring duplicate addition of .xo: {xo_file.name}")
                continue
            if not xo_file.exists():
                raise FINNVitisLinkConfigError(
                    f"Tried adding .xo file which " f"does not exist: {xo_file}"
                )
            self.xo.append(xo_file)

    def add_connectivity(self, txt: str) -> None:
        """Add further lines to the connectivity section. For example to assign clocks or ports."""
        self.connectivity_section += txt + ("" if txt[-1] != "\n" else "\n")

    def get_config_validation_errors(self) -> None | list[FINNVitisLinkConfigError]:
        """Check the configuration and if errors are found, return them."""
        errors = []
        # All CUs in SCs exist and CU ports are correctly formatted
        for cu_sender, receivers in self.sc.items():
            for cu_receiver in receivers:
                sender_split = cu_sender.split(".")
                if len(sender_split) != 2:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} "
                            f"incorrectly formatted. "
                            "Use the syntax CU.PORT"
                        )
                    )
                sender_name = sender_split[0]
                if sender_name not in self.cu:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} uses the unknown CU {sender_name}"
                        )
                    )
                receiver_split = cu_receiver.split(".")
                if len(receiver_split) != 2:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:{cu_receiver} "
                            f"incorrectly formatted. "
                            "Use the syntax CU.PORT"
                        )
                    )
                receiver_name = receiver_split[0]
                if receiver_name not in self.cu:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"SC {cu_sender}:"
                            f"{cu_receiver} uses the unknown "
                            f"CU {receiver_name}"
                        )
                    )
        # No two same named CUs
        if len(set(self.cu)) != len(self.cu):
            errors.append(
                FINNVitisLinkConfigError(
                    "It seems that there are one or more CUs with the same name!"
                )
            )
        for kernel, cu in self.nk:
            for kernel2, cu2 in self.nk:
                if cu == cu2 and kernel != kernel2:
                    errors.append(
                        FINNVitisLinkConfigError(
                            f"There are 2 or more CUs named {cu} "
                            f"from different kernels ({kernel} "
                            f"and {kernel2})"
                        )
                    )
        if len(errors) > 0:
            return errors
        return None

    def generate_config(self) -> None:
        """Write the complete config. Raises an error if the
        config is invalid.
        """
        # Checking for errors
        errors = self.get_config_validation_errors()
        if errors is not None:
            for err in errors:
                log.error(f"{self.config_path}: {err}")
            if len(errors) == 1:
                # TODO: When we have switched to Python 3.11 use an exception group
                # TODO: to raise all exceptions at once, instead of one per run.
                raise errors[0]
            raise FINNVitisLinkConfigError(
                f"Multiple configuration errors occurred. First one is: {errors[0]}"
            )

        # Template rendering
        env = get_jinja_environment()
        template = env.get_template("vitis_link/link_config.txt.jinja")
        rendered = template.render(
            nk=self.nk,
            sc=self.sc,
            sp=self.sp,
            slr=self.slr,
            connects=self.connects,
            connectivity_extras=self.connectivity_section,
            vivado_extras=self.vivado_section,
        )
        self.config_path.write_text(rendered)

    def generate_run_script(self) -> None:
        """Generate a shell script to start v++ with the correct parameters.
        Produces the shell script next to the path of the config file
        unless a path is specified.
        """
        # Accumulate all xos
        xo_string = " ".join([str(xo) for xo in self.xo])

        # Check that a config for this link script exists
        if not self.config_path.exists():
            log.error(
                f"Writing compilation / v++ script for non-existing configuration "
                f"in {self.config_path.absolute()}. Continuing in case this is on purpose."
            )

        # Rendering the template
        env = get_jinja_environment()
        template = env.get_template("vitis_link/run_link.sh.jinja")
        rendered = template.render(
            platform=self.platform,
            xos=xo_string,
            config=self.config_path.absolute(),
            optimization=self.optimization_level,
            f_mhz=self.f_mhz,
        )
        self.run_script_path.write_text(rendered)


class BuildBasicVitisLinkConfig(Transformation):
    """Build basic configs for an SDP-only graph. If multiple devices are used, generate a config
    per device. The directory with all configurations is stored in the metadata
    prop `vitis_link_configs`. If a config already exists, emits an error and does nothing.

    Refer to `VitisLinkConfiguration.load_from_model` and
    `VitsLinkConfiguration.store_to_model` for more information.

    When done, the config is ready to link (for Single-FPGA).

    To find the path of the final config, one can load the config from the model,
    then check the `config_path` and `run_script_path` fields of the `VitisLinkConfiguration`.
    """

    def __init__(
        self,
        platform: str,
        mem_type: FpgaMemoryType,
        vitis_opt_strategy: VitisOptStrategy,
        optimization_level: str,
        f_mhz: int,
    ) -> None:  # noqa
        super().__init__()
        self.platform = platform
        self.fpga_memory_type = mem_type
        self.optimization_level = optimization_level
        self.vitis_opt_strategy = vitis_opt_strategy
        self.f_mhz = f_mhz

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:  # noqa
        configs: dict[int, VitisLinkConfiguration] = {}
        config_storage: Path = Path(make_build_dir("vitis_link_configs_"))

        # Check that we are the first to edit link configs
        vitis_link_configs = model.get_metadata_prop("vitis_link_configs")
        if vitis_link_configs is not None:
            log.error(
                f"Detected existing linking configurations in {vitis_link_configs}."
                "BuildBasicVitisLinkConfig should be the first "
                "transformation to create the initial configuration. "
                "No changes will be made."
            )
            return model, False

        # Differentiate Multi- and Single-FPGA cases
        number_of_device_ids = 0
        for node in model.graph.node:
            if get_device_id(node) is not None:
                number_of_device_ids += 1

        if number_of_device_ids != 0 and number_of_device_ids < len(model.graph.node):
            raise FINNVitisLinkConfigError(
                f"{number_of_device_ids} / "
                f"{len(model.graph.node)} nodes in the graph "
                f"have an associated device ID. Either all nodes "
                f"have an ID (Multi-FPGA) or none (Single-FPGA). "
                f"Stopping."
            )

        # Set up all configs
        is_multifpga = False
        if number_of_device_ids == 0:
            # Single FPGA
            configs[0] = VitisLinkConfiguration(
                config_path=Path(make_build_dir("vitis_single_link_")) / "config.txt",
                platform=self.platform,
                optimization_level=self.optimization_level,
                f_mhz=self.f_mhz,
            )
        else:
            # Multi-FPGA
            is_multifpga = True
            for node in model.graph.node:
                device = get_device_id(node)
                assert device is not None
                if device not in configs:
                    configs[device] = VitisLinkConfiguration(
                        config_path=Path(make_build_dir(f"vitis_multi_link_device_{device}_"))
                        / "config.txt",
                        platform=self.platform,
                        optimization_level=self.optimization_level,
                        f_mhz=self.f_mhz,
                    )

        # Already add optimization strategies for all devices
        for device in configs.keys():
            if self.vitis_opt_strategy == VitisOptStrategy.PERFORMANCE_BEST:
                configs[device].add_vivado_line(
                    "prop=run.impl_1.STEPS.OPT_DESIGN.ARGS.DIRECTIVE=ExploreWithRemap\n"
                    "prop=run.impl_1.STEPS.PLACE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.IS_ENABLED=true\n"
                    "prop=run.impl_1.STEPS.PHYS_OPT_DESIGN.ARGS.DIRECTIVE=Explore\n"
                    "prop=run.impl_1.STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE=Explore\n"
                )

        # Some temporary variables needed to construct the configs
        current_device: int
        cu_names: dict[str, str] = {}

        # Loop through all SDPs
        check_all_sdp_nodes(model)
        check_graph_is_line(model)
        for node in model.graph.node:
            current_device = cast("int", get_device_id(node)) if is_multifpga else 0
            submodel, _ = get_submodel(node)
            predecessors = model.find_direct_predecessors(node)
            successors = model.find_direct_successors(node)
            is_input = predecessors is not None and len(predecessors) == 1
            is_output = successors is not None and len(successors) == 1
            node_slr = getCustomOp(node).get_nodeattr("slr")

            # Add the SDPs XO file
            configs[current_device].add_xo(get_vitis_xo(node))

            # Instantiate the kernel
            if len(submodel.graph.node) == 1 and "IODMA" in submodel.graph.node[0].name:
                if is_input:
                    configs[current_device].add_cu(node.name, "idma")
                    cu_names[node.name] = "idma"
                if is_output:
                    configs[current_device].add_cu(node.name, "odma")
                    cu_names[node.name] = "odma"
            else:
                configs[current_device].add_cu(node.name, node.name)
                cu_names[node.name] = node.name

            # Add connection between kernels. For this we need a predecessor and be either
            # Single-FPGA or Multi-FPGA and on the same device
            if predecessors is not None and (
                not is_multifpga or get_device_id(predecessors[0]) == current_device
            ):
                predecessor_cu = cu_names[predecessors[0].name] + ".m_axis_0"
                this_cu = cu_names[node.name] + ".s_axis_0"
                configs[current_device].add_sc(predecessor_cu, this_cu)

            # Add system ports
            mem_type: str
            mem_idx: int
            if self.fpga_memory_type == FpgaMemoryType.HOST_MEM:
                mem_type = "HOST"
                mem_idx = 0
            else:
                match self.platform.lower():
                    case "u50" | "u280" | "u55c":
                        mem_type = "HBM"
                        mem_idx = 0
                    case "u250":
                        mem_type = "DDR"
                        mem_idx = 0 if node_slr == -1 else cast("int", node_slr)
                    case _:
                        mem_type = "DDR"
                        mem_idx = 1
            if cu_names[node.name] in ["idma", "odma"]:
                configs[current_device].add_sp(
                    cu_names[node.name] + ".m_axi_gmem0", f"{mem_type}[{mem_idx}]"
                )

        # Store everything, generate scripts and return
        model = VitisLinkConfiguration.store_to_model(configs, model, config_storage)
        for config in configs.values():
            config.generate_config()
            config.generate_run_script()
        return model, False
