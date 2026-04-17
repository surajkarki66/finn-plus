"""Prepare communication kernels for usage (e.g. packaging IP cores)."""
import shlex
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import DataflowBuildConfig, MFVerbosity
from finn.transformation.fpgadataflow.multifpga.metadata import AuroraNetworkMetadata
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGAConfigError, FINNMultiFPGAError
from finn.util.logging import log
from finn.util.settings import get_settings


class PrepareAuroraFlow(Transformation):
    """Use the AuroraNetworkMetadata to package all necessary kernels. Sets the aurora_storage
    metadata prop of the model to point to the directory containing the object files. This does
    nothing else.
    """

    def __init__(self, cfg: DataflowBuildConfig) -> None:
        """Prepare AuroraFlow."""
        super().__init__()
        # TODO: non-vitis platforms?
        self.platform = cfg._resolve_vitis_platform()  # noqa
        self.part = cfg._resolve_fpga_part()  # noqa
        if cfg.partitioning_configuration is None:
            raise FINNMultiFPGAConfigError(
                "Cannot run AuroraFlow preparation on " "a run without partitioning configuration!"
            )
        self.verbosity = cfg.partitioning_configuration.verbosity
        self.make_args = " ".join(
            f"{k}={v}"
            for k, v in cfg.partitioning_configuration.communication_kernel_arguments.items()
        )
        self.aurora_storage = Path(make_build_dir("aurora_storage_")).absolute()
        self.aurora_path = get_settings().finn_deps / "AuroraFlow"
        if not self.aurora_path.exists():
            raise FINNMultiFPGAConfigError(
                "Could not find AuroraFlow in FINN+'s dependency "
                "directory. Are all dependencies downloaded "
                "and installed?"
            )

    def package_single(self, args: str, kernel_xo: str, save_as_xo: str) -> Path:
        """Package a single aurora core and put it into the given location with the given name.
        Copies aurora so that multiple packaging processes can happen at once.

        >>> prep = PrepareAuroraFlow()
        >>> output = prep.package_single("", "aurora_flow_0.xo", "mykernel.xo")
        >>> output.exists()
        True
        """
        if self.verbosity == MFVerbosity.HIGH:
            log.info(f"Packaging AuroraFlow core ({kernel_xo} -> {save_as_xo})")

        # Copy the AuroraFlow project into a build directory
        temp_dir = Path(make_build_dir("aurora_temp_builddir_"))
        shutil.copytree(self.aurora_path, temp_dir, dirs_exist_ok=True)

        # Create the aurora kernel xo file
        if self.make_args != "" and self.verbosity.value > MFVerbosity.LOW.value:
            log.info(
                f"Packaging AuroraFlow kernel with additional make arguments: {self.make_args}"
            )
        result = subprocess.run(
            shlex.split(f"make aurora_hw {args} {self.make_args}"),
            cwd=temp_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FINNMultiFPGAError(
                f"Error during creation of the " f"AuroraFlow kernels: {result.stderr}"
            )
        p_origin = temp_dir / "build" / kernel_xo
        if not p_origin.exists():
            raise FINNMultiFPGAError(
                f"Packaging AuroraFlow failed. Expected "
                f"kernel at path {p_origin}. Check logs in {temp_dir}"
            )

        # Rename / Move the created xo to the given target
        p_target = self.aurora_storage / save_as_xo
        shutil.move(p_origin, p_target)
        if not p_target.exists():
            raise FINNMultiFPGAError(f"Failed to move aurora xo from {p_origin} to {p_target}!")

        # We can now safely delete the temp build dir
        shutil.rmtree(temp_dir)
        return p_target.absolute()

    def package_all_from_metadata(self, metadata: AuroraNetworkMetadata) -> None:
        """Use the passed metadata to package all required Aurora kernels at once."""
        # List all auroras that need to be packaged
        auroras = []
        for device in metadata.table.keys():
            # TODO: Here we simply assume the function
            # returns the kernel names in the right order
            auroras += enumerate(metadata.get_aurora_kernels(device))

        def _package_aurora(d: tuple[int, str]) -> None:
            """Package a single aurora core with the given index and name."""
            i, aurora_name = d
            origin = f"aurora_flow_hw_{i}.xo"
            target = f"{aurora_name}.xo"
            # TODO: args?
            self.package_single(f"PART={self.part} PLATFORM={self.platform}", origin, target)

        if self.verbosity == MFVerbosity.HIGH:
            log.info(f"Packaging kernels with PART={self.part} and PLATFORM={self.platform}")

        # Package all Aurora kernels concurrently
        futures: list[Future] = []
        with ThreadPoolExecutor(max_workers=get_settings().num_default_workers) as tpe:
            for aurora in auroras:
                futures.append(tpe.submit(_package_aurora, aurora))
            tpe.shutdown()

        # Check results to propagate exceptions from threads
        for future in futures:
            future.result()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Package all aurora kernels required by the model in parallel.
        Sets the "aurora_storage" metadata prop to the path where the xo files
        are stored.
        """
        metadata = AuroraNetworkMetadata(model)
        model.set_metadata_prop("aurora_storage", str(self.aurora_storage.absolute()))
        self.package_all_from_metadata(metadata)
        return model, False
