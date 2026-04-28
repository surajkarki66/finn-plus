"""Prepare communication kernels for usage (e.g. packaging IP cores)."""
import shlex
import shutil
import subprocess
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation

from finn.builder.build_dataflow_config import MFVerbosity, PartitioningConfiguration
from finn.transformation.fpgadataflow.multifpga.aurora_metadata import AuroraNetworkMetadata
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGAConfigError, FINNMultiFPGAError, FINNMultiFPGAUserError
from finn.util.logging import log
from finn.util.settings import get_settings


class PrepareAuroraFlow(Transformation):
    """Prepares all AuroraFlow kernel XO files and stores them in the network
    metadata of the ModelWrapper.

    Requires: A metadata creation transformation needs to be run beforehand, so that this transform
    knows, which kernels it needs to package.

    Afterwards: The models metadata contains only valid paths to AuroraFlow kernel XOs.
    """

    def __init__(
        self, platform: str, part: str, partitioning_configuration: PartitioningConfiguration
    ) -> None:
        """Prepare AuroraFlow."""
        super().__init__()
        # TODO: non-vitis platforms?
        self.platform = platform
        self.part = part
        self.verbosity = partitioning_configuration.verbosity
        self.make_args = " ".join(
            f"{k}={v}" for k, v in partitioning_configuration.communication_kernel_arguments.items()
        )
        self.aurora_storage = Path(make_build_dir("aurora_storage_")).absolute()
        self.aurora_path = get_settings().finn_deps / "AuroraFlow"
        if not self.aurora_path.exists():
            raise FINNMultiFPGAConfigError(
                "Could not find AuroraFlow in FINN+'s dependency "
                "directory. Are all dependencies downloaded "
                f"and installed? (Searched at {self.aurora_path})."
            )

    def package_single(self, args: str, device: int, index: int) -> Path:
        """Package a single AuroraFlow kernel. Stored in a subdirectory of the
        aurora_storage directory,
        which is saved in the metadata prop of the model.
        The directory is named 'aurora_device_<device>_index_<index>'.
        Inside are the packaged xo files.
        """
        if self.verbosity == MFVerbosity.HIGH:
            log.info(f"Packaging AuroraFlow core (Device: {device}, Index: {index})")

        # Copy the AuroraFlow project into a build directory
        build_dir = self.aurora_storage / f"auroraflow_build_dev{device}_ind{index}"
        shutil.copytree(self.aurora_path, build_dir, dirs_exist_ok=True)

        # Create the aurora kernel xo file
        if self.make_args != "" and self.verbosity.value > MFVerbosity.LOW.value:
            log.info(
                f"Packaging AuroraFlow kernel with additional make arguments: {self.make_args}"
            )
        result = subprocess.run(
            shlex.split(f"make aurora_hw {args} {self.make_args}"),
            cwd=build_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FINNMultiFPGAUserError(
                f"Error during creation of the AuroraFlow kernels:\n{result.stderr}"
            )
        xo_path = build_dir / "build" / f"aurora_flow_hw_{index}.xo"
        if not xo_path.exists():
            raise FINNMultiFPGAError(
                f"Packaging AuroraFlow failed. Expected "
                f"kernel at path {xo_path}. Check logs in {build_dir}"
            )
        return xo_path.absolute()

    def package_all_from_metadata(self, metadata: AuroraNetworkMetadata) -> None:
        """Package all AuroraFlow kernels required by the metadata. If a path is missing or
        doesn't point to a valid object, it is generated and filled out.
        """
        if self.verbosity == MFVerbosity.HIGH:
            log.info(f"Packaging kernels with PART={self.part} and PLATFORM={self.platform}")

        # Package all Aurora kernels concurrently
        futures: list[Future] = []
        dev_index: list[tuple[int, int]] = []
        with ThreadPoolExecutor(max_workers=get_settings().num_default_workers) as tpe:
            for device, index in metadata.get_unprepared_aurora_kernels():
                if self.verbosity.value > MFVerbosity.LOW.value:
                    log.info(f"Packaging AuroraFlow core number {index} for device {device}.")
                futures.append(
                    tpe.submit(
                        self.package_single,
                        f"PART={self.part} PLATFORM={self.platform}",
                        device,
                        index,
                    )
                )
                dev_index.append((device, index))
            tpe.shutdown()

        # Store results in the metadata
        for i in range(len(futures)):
            device, index = dev_index[i]
            metadata[device][index].aurora_xo = futures[i].result()

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:
        """Package all aurora kernels required by the model in parallel.
        Sets the "aurora_storage" metadata prop to the path where the xo files
        are stored.
        """
        # Load the metadata from the model's metadata prop
        metadata = AuroraNetworkMetadata.from_model(model)

        # Store the location of the aurora kernels in the model as well - just in case
        model.set_metadata_prop("aurora_storage", str(self.aurora_storage.absolute()))

        # Package the kernels and modify the metadata to contain the paths
        # of the packaged kernels
        self.package_all_from_metadata(metadata)

        # Store the updated metadata
        metadata.save()
        return model, False
