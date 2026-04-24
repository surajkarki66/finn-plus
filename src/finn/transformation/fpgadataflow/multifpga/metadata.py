"""Contains the base NetworkMetadata class, as well as creators
for metadatas for various kinds of topologies.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from finn.builder.build_dataflow_config import MFVerbosity
from finn.transformation.fpgadataflow.multifpga.utils import get_device_id
from finn.util.basic import make_build_dir
from finn.util.exception import FINNMultiFPGAError
from finn.util.logging import log

if TYPE_CHECKING:
    from collections.abc import Callable
    from qonnx.core.modelwrapper import ModelWrapper


class DataDirection(str, Enum):
    """Data movement direction."""

    TX = "TX"
    RX = "RX"
    BIDIRECTIONAL = "BIDIRECTIONAL"


@dataclass
class NetworkMetadata(ABC):
    """Metadata baseclass for storage of Multi-FPGA connections. Defines connections between
    devices, as well as which nodes on the devices are responsible for communication.
    """

    @staticmethod
    @abstractmethod
    def from_model(model: ModelWrapper) -> NetworkMetadata:
        """Load the metadata from a modelwrapper."""
        raise NotImplementedError()

    @abstractmethod
    def __getitem__(self, key: Any) -> Any:
        """Getter base method for internal data."""
        pass  # noqa

    @abstractmethod
    def __setitem__(self, key: Any, value: Any) -> Any:
        """Setter base method for internal data."""
        pass  # noqa

    @abstractmethod
    def add_connection(
        self,
        sender_device: int,
        sender_node: str,
        receiver_device: int,
        receiver_node: str,
    ) -> None:
        """Add a connection to the metadata."""
        pass  # noqa

    @abstractmethod
    def save(self, p: Path | None) -> None:
        """Save the internal data to the given path. If no path is given,
        the metadata is stored at the location it was loaded from before.
        """

    @staticmethod
    @abstractmethod
    def load(p: Path) -> NetworkMetadata:
        """Load the metadata from the given path."""


class CreateNetworkMetadata(ABC):
    """Run this transformation to create metadata necessary for Multi-FPGA settings. Pass the type
    of metadata as a type to the constructor, or a factory producing one.
    """

    def __init__(  # noqa
        self,
        save_as_format: type[NetworkMetadata] | Callable[[], NetworkMetadata],
        verbosity: MFVerbosity,
    ) -> None:
        super().__init__()
        self.verbosity = verbosity
        self.metadata_type = save_as_format
        self.metadata = self.metadata_type()

    def save_metadata(self, model: ModelWrapper, suffix: str = "yaml") -> Path:
        """Save the metadata and store the path as a metadata prop in the modelwrapper instance."""
        metadata_dir = Path(make_build_dir("network_metadata_")).absolute()
        metadata_path = metadata_dir / ("metadata." + suffix)
        self.metadata.save(metadata_path)
        model.set_metadata_prop("network_metadata", str(metadata_path))
        return metadata_path

    @abstractmethod
    def create_metadata(self, model: ModelWrapper) -> None:
        """Create the metadata and assign it to the object variable.
        When creating a new type of network metadata this has to be implemented.
        """
        raise NotImplementedError()


class CreateChainNetworkMetadata(CreateNetworkMetadata):
    """Create a simple network with FPGAs connected in a simple line."""

    def __init__(  # noqa
        self, save_as_format: type[NetworkMetadata], verbosity: MFVerbosity
    ) -> None:
        super().__init__(save_as_format, verbosity)

    def create_metadata(self, model: ModelWrapper) -> None:
        """Create network metadata from the given model."""
        for n1 in model.graph.node:
            sucs = model.find_direct_successors(n1)
            if sucs is None:
                continue
            for n2 in sucs:
                d1 = get_device_id(n1)
                d2 = get_device_id(n2)
                if d1 is None:
                    raise FINNMultiFPGAError(f"Node {n1.name} does not have a device id!")
                if d2 is None:
                    raise FINNMultiFPGAError(f"Node {n2.name} does not have a device id!")
                if d1 != d2:
                    if self.verbosity.value > MFVerbosity.MEDIUM.value:
                        log.info(f"Adding connection:  {n1.name} [{d1}] ----> {n2.name} [{d2}]")
                    self.metadata.add_connection(int(d1), n1.name, int(d2), n2.name)


class CreateReturnChainNetworkMetadata(CreateNetworkMetadata):
    """Create a network with a chain that returns across the devices."""

    pass  # noqa
