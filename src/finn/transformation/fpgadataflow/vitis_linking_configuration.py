"""Dataclass to store a linking configuration for vitis builds. This is essential to enable multiple
transformations / steps to edit the same configuration.
"""

from __future__ import annotations

from pathlib import Path

from finn.util.exception import FINNConfigurationError, FINNMultiFPGAError, FINNVitisLinkConfigError
from finn.util.logging import log


class VitisLinkConfiguration:
    """Manages XO files, CU instantiations, stream connections,
    port connections, Vivado props, etc.
    It can output a linking configuration to pass to v++ and
    create a shell script to run it. Tries to be as strict and careful as possible,
    and depending on the issue raises an Exception, logs an error or warning
    or continues silently.
    """

    def __init__(  # noqa
        self,
        config_path: Path,
        platform: str,
        optimization_level: str,
        f_mhz: int,
        run_script_path: Path | None = None,
    ) -> None:
        """Create a new configuration with the given parameters.

        Arguments:
        ---------
            `config_path`: Path at which the configuration file will be stored.
            `platform`: FPGA platform to link for.
            `optimization_level`: v++ optimization level to use during linking.
            `f_mhz`: Target clock frequency in MHz.
            `run_script_path`: Path at which the linker start script will be stored.
                If left empty, it is placed next to the config file as "run_link.sh".
        """
        self.cu: list[str] = []
        self.nk: list[tuple[str, str]] = []
        self.sc: dict[str, list[str]] = {}
        self.sp: dict[str, str] = {}
        self.xo: list[Path] = []
        self.connects: list[tuple[str, str]] = []
        self.vivado_section: str = "[vivado]\n"
        self.connectivity_section: str = ""
        self.platform: str = platform
        self.optimization_level: str = optimization_level
        self.f_mhz: int = f_mhz
        self.config_path = config_path
        config_path.parent.mkdir(exist_ok=True, parents=True)
        if run_script_path is not None:
            self.run_script_path = run_script_path
        else:
            self.run_script_path = config_path.parent / "run_link.sh"
        self.run_script_path.parent.mkdir(exist_ok=True, parents=True)

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
        if sender_port.lower() in ["s_axis", "in"] or receiver_port.lower() in ["m_axis", "out"]:
            log.error(
                f"Adding connection sc={cu_sender}:{cu_receiver}. The port "
                "names suggest that the order of sender and receiver might be "
                "swapped. Proceeding now."
            )

        # Add the connection
        if cu_sender not in self.sc.keys():
            self.sc[cu_sender] = []
        self.sc[cu_sender].append(cu_receiver)

    def add_sp(self, cu_port_name: str, mem_type: str) -> None:
        """Add an SP assignment."""
        known_sp_names = ["DDR", "HBM", "PLRAM"]
        if mem_type not in known_sp_names:
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
        with self.config_path.open("w+") as f:
            f.write("[connectivity]\n")
            for kernel_name, cu_name in self.nk:
                f.write(f"nk={kernel_name}:1:{cu_name}\n")

            # origin_cu and target_cu already require the ports already being in the str
            for origin_cu in self.sc.keys():
                for target_cu in self.sc[origin_cu]:
                    f.write(f"sc={origin_cu}:{target_cu}\n")

            for sp_cu, sp_mem in self.sp.items():
                f.write(f"sp={sp_cu}:{sp_mem}\n")

            for a, b in self.connects:
                f.write(f"connect={a}:{b}\n")

            if self.connectivity_section != "":
                f.write(self.connectivity_section + "\n")

            f.write(self.vivado_section)

        if not self.config_path.exists():
            raise FINNMultiFPGAError(f"Failed to create vitis config at {self.config_path}.")

    def generate_run_script(self) -> None:
        """Generate a shell script to start v++ with the correct parameters.
        Produces the shell script next to the path of the config file
        unless a path is specified.
        """
        xo_string = " ".join([str(xo) for xo in self.xo])
        if not self.config_path.exists():
            log.error(
                f"Writing compilation / v++ script for non-existing configuration "
                f"in {self.config_path.absolute()}. Continuing in case this is on purpose."
            )
        with self.run_script_path.open("w+") as f:
            f.write("#!/bin/bash\n")
            f.write(
                f"v++ --target hw --platform {self.platform} --link {xo_string} "
                f"--config {self.config_path.absolute()} --optimize {self.optimization_level} "
                f"--report_level estimate --save-temps --kernel_frequency {self.f_mhz}"
            )

        if not self.run_script_path.exists():
            raise FINNConfigurationError(
                f"Failed to create config run script " f"at {self.run_script_path}"
            )
