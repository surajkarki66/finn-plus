import pytest

from pathlib import Path
from qonnx.core.modelwrapper import ModelWrapper
from test_multifpga_sdp_creation import create_sdp_ready_model_no_branches

from finn.builder.build_dataflow_config import MFCommunicationKernel, MFTopology, MFVerbosity
from finn.transformation.fpgadataflow.multifpga.assign_metadata import AssignNetworkMetadata
from finn.transformation.fpgadataflow.multifpga.create_multi_sdp import (
    CreateMultiFPGAStreamingDataflowPartition,
)
from finn.transformation.fpgadataflow.multifpga.utils import get_device_id
from finn.util.basic import make_build_dir


@pytest.mark.multifpga
@pytest.mark.parametrize(
    "device_node_combinations", [(2, 2), (5, 10), (40, 100), (50, 50), (1, 10)]
)
@pytest.mark.parametrize("assignment_type", ["random", "equal"])
@pytest.mark.parametrize("shuffle_devices", [True, False])
@pytest.mark.parametrize("assignment_order", ["linear"])
@pytest.mark.parametrize("communication_kernel", [MFCommunicationKernel.AURORA])
def test_chain_metadata(
    device_node_combinations: tuple[int, int],
    assignment_type: str,
    assignment_order: str,
    shuffle_devices: bool,
    communication_kernel: MFCommunicationKernel,
) -> None:
    """Test that creating the metadata for a model in the chain topology works."""
    device_count, node_count = device_node_combinations
    # TODO: Not ideal, better pass an own, self constructed model
    model = create_sdp_ready_model_no_branches(
        node_count, device_count, assignment_type, assignment_order, shuffle_devices
    )
    model = model.transform(
        CreateMultiFPGAStreamingDataflowPartition(
            separate_iodmas=True,
            dataflow_partition_directory=Path(
                make_build_dir("multi_sdp_test_aurora_chain_metadata")
            ),
            verbosity=MFVerbosity.NONE,
        )
    )
    model = model.transform(
        AssignNetworkMetadata(
            communication_kernel=communication_kernel,
            topology=MFTopology.CHAIN,
            verbosity=MFVerbosity.NONE,
        )
    )

    # Check that the assignments worked as expected
    metadata_path = model.get_metadata_prop("network_metadata")
    assert metadata_path is not None, "network_metadata metadataprop not assigned"
    metadata_path = Path(metadata_path)
    metadata_type = AssignNetworkMetadata.COMMUNICATION_KERNEL_METADATA_MAP[communication_kernel]
    _ = metadata_type.from_model(model)

    for node in model.graph.node:
        successors = model.find_direct_successors(node)
        if successors is None:
            d1 = get_device_id(node)
            assert d1 is not None

            raise NotImplementedError()
