"""Assign metadata transformation."""

from __future__ import annotations

from qonnx.transformation.base import Transformation
from typing import TYPE_CHECKING, Final

from finn.builder.build_dataflow_config import MFCommunicationKernel, MFTopology, MFVerbosity
from finn.transformation.fpgadataflow.multifpga.aurora.metadata import AuroraNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.metadata import (
    CreateChainNetworkMetadata,
    CreateNetworkMetadata,
    CreateReturnChainNetworkMetadata,
)
from finn.util.exception import FINNUserError

if TYPE_CHECKING:
    from qonnx.core.modelwrapper import ModelWrapper

    from finn.transformation.fpgadataflow.multifpga.metadata import NetworkMetadata


class AssignNetworkMetadata(Transformation):
    """Create and store metadata for the given type of metadata and a topology.

    Requires: Partitioning must have happened before (device IDs for all nodes need to be set).

    Afterwards: Metadata with the given parameters was created and stored. The path can be found
    in the `network_metadata` prop of the ModelWrapper.
    """

    TOPOLOGY_CREATOR_MAP: Final[dict[MFTopology, type[CreateNetworkMetadata]]] = {
        MFTopology.CHAIN: CreateChainNetworkMetadata,
        MFTopology.RETURNCHAIN: CreateReturnChainNetworkMetadata,
    }

    COMMUNICATION_KERNEL_METADATA_MAP: Final[dict[MFCommunicationKernel, type[NetworkMetadata]]] = {
        MFCommunicationKernel.AURORA: AuroraNetworkMetadata
    }

    def __init__(  # noqa
        self,
        communication_kernel: MFCommunicationKernel,
        topology: MFTopology,
        verbosity: MFVerbosity,
    ) -> None:
        """Create metadata.

        Arguments:
        ---------
            `communication_kernel`: Determines the format of metadata to store the connections in.
            `topology`: Type of topology/connections to create in metadata.
            `verbosity`: Determines how much information is printed during metadata creation.

        The function automatically selects the required classes:
        >>> t = AssignNetworkMetadata(MFCommunicationKernel.AURORA, MFTopology.CHAIN, MFVerbosity.LOW)
        >>> t.creator_type.__name__
        'CreateChainNetworkMetadata'
        >>> t.metadata_type.__name__
        'AuroraNetworkMetadata'
        """  # noqa
        super().__init__()
        self.verbosity = verbosity
        try:
            self.creator_type = AssignNetworkMetadata.TOPOLOGY_CREATOR_MAP[topology]
            self.metadata_type = AssignNetworkMetadata.COMMUNICATION_KERNEL_METADATA_MAP[
                communication_kernel
            ]
        except KeyError as e:
            raise FINNUserError(
                f"Cannot create metadata for topology {topology.name}"
                f" and communication kernel {communication_kernel.name}, "
                f"since no creator/metadata type was written/connected for it yet. "
            ) from e
        self.creator: CreateNetworkMetadata = self.creator_type(
            save_as_format=self.metadata_type, verbosity=verbosity
        )

    def apply(self, model: ModelWrapper) -> tuple[ModelWrapper, bool]:  # noqa
        self.creator.create_metadata(model)
        self.creator.save_metadata(model)
        return model, False
