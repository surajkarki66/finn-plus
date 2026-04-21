"""Tests for the communication kernels used by the Multi-FPGA extension."""

from __future__ import annotations

import pytest

from pathlib import Path
from test_multifpga_sdp_creation import create_sdp_ready_model_no_branches
from typing import Any

from finn.builder.build_dataflow_config import (
    DataflowBuildConfig,
    MFCommunicationKernel,
    MFTopology,
    PartitioningConfiguration,
)
from finn.transformation.fpgadataflow.multifpga.communication_kernels import PrepareAuroraFlow
from finn.transformation.fpgadataflow.multifpga.create_multi_sdp import (
    CreateMultiFPGAStreamingDataflowPartition,
)
from finn.transformation.fpgadataflow.multifpga.metadata import (
    AssignNetworkMetadata,
    AuroraNetworkMetadata,
    CreateChainNetworkMetadata,
)
from finn.util.basic import make_build_dir


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("device_node_combinations", [(1, 2), (1, 3), (2, 2), (5, 10), (5, 11)])
@pytest.mark.parametrize("assignment_type", ["random", "equal"])
@pytest.mark.parametrize("metadata_creator", [CreateChainNetworkMetadata])
@pytest.mark.parametrize("shuffle_devices", [True, False])
@pytest.mark.parametrize("board", ["U280", "U55C"])
def test_aurora_packaging_integrated(
    device_node_combinations: tuple[int, int],
    assignment_type: str,
    metadata_creator: type,
    shuffle_devices: bool,
    board: str,
) -> None:
    """Test the whole AuroraFlow preparation pipeline:
    - Create a model from scratch
    - Create the SDP partitions
    - Create the metadata based on the SDP partitions
    - Check that the metadata, XOs and model metadata props exist.
    """
    devices, nodes = device_node_combinations
    assignment_order = ""

    # Check which creator we use
    if metadata_creator is CreateChainNetworkMetadata:
        assignment_order = "linear"
        topology = MFTopology.CHAIN
    else:
        raise NotImplementedError()

    # Create an SDP ready branchless model
    model = create_sdp_ready_model_no_branches(
        nodes, devices, assignment_type, assignment_order, shuffle_devices
    )

    # Create a config based on the test parameters
    cfg = DataflowBuildConfig(
        output_dir=make_build_dir("test_aurora_packaging_integrated_build"),
        board=board,
        partitioning_configuration=PartitioningConfiguration(
            num_fpgas=devices, communication_kernel=MFCommunicationKernel.AURORA, topology=topology
        ),
    )

    # Execute the whole Aurora packaging flow
    prepare_aurora = PrepareAuroraFlow(cfg)
    model = model.transform(CreateMultiFPGAStreamingDataflowPartition())
    model = model.transform(AssignNetworkMetadata(AuroraNetworkMetadata, metadata_creator))
    model = model.transform(prepare_aurora)

    # Try and load the previously generated metadata from the models metadata prop
    meta = AuroraNetworkMetadata(model)

    # Check that the AuroraFlow storage directory got saved in the model metadata
    aurora_storage = model.get_metadata_prop("aurora_storage")
    assert aurora_storage is not None

    # Check that this directory actually exists
    aurora_storage = Path(aurora_storage)
    assert aurora_storage.exists()

    # Check if each device had its respective kernels packaged
    for device in meta.get_devices():
        for kernel in meta.get_aurora_kernels(device):
            assert (aurora_storage / (kernel + "xo")).exists()


@pytest.fixture
def create_aurora_metadata(request: Any) -> AuroraNetworkMetadata | None:
    """Testfixture to create an Aurora metadata object for certain topologies (minimal
    working examples).
    """
    match request.param:
        case "empty":
            return AuroraNetworkMetadata()
        case "chain":
            m_chain = AuroraNetworkMetadata()
            m_chain.add_connection(0, "sdp0", 1, "sdp1")
            m_chain.add_connection(1, "sdp1", 2, "sdp2")
            m_chain.add_connection(2, "sdp2", 3, "sdp3")
            return m_chain
        case "returnchain":
            m_returnchain = AuroraNetworkMetadata()
            m_returnchain.add_connection(0, "sdp0", 1, "sdp1")
            m_returnchain.add_connection(1, "sdp1", 2, "sdp2")
            m_returnchain.add_connection(2, "sdp2", 3, "sdp3")
            m_returnchain.add_connection(3, "sdp3", 2, "sdp4")
            m_returnchain.add_connection(2, "sdp4", 1, "sdp5")
            m_returnchain.add_connection(1, "sdp5", 0, "sdp6")
            return m_returnchain
        case _:
            return None


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("board", ["U280", "U55C"])
@pytest.mark.parametrize("create_aurora_metadata", ["empty", "chain", "returnchain"], indirect=True)
def test_aurora_packaging_isolated(
    board: str, create_aurora_metadata: AuroraNetworkMetadata | None
) -> None:
    """Using the fixture 'create_aurora_metadata' create a metadata object with the given topology
    and test that all required XO files exist after packaging the kernel based on this metadata.
    """
    assert create_aurora_metadata is not None, "Invalid testconfig: Unknown metadata type"
    prep = PrepareAuroraFlow(
        cfg=DataflowBuildConfig(
            board=board, output_dir=make_build_dir("test_aurora_packaging_isolated_build")
        )
    )
    prep.package_all_from_metadata(create_aurora_metadata)
    for device in create_aurora_metadata.table.keys():
        for aurora_name in create_aurora_metadata.table[device].keys():
            assert (prep.aurora_storage / (aurora_name + ".xo")).exists()


@pytest.mark.multifpga
@pytest.mark.slow
@pytest.mark.parametrize("args", ["", "FIFO_WIDTH=32 TX_FIFO_SIZE=8192 RX_FIFO_SIZE=65536"])
@pytest.mark.parametrize("board", ["U280", "U55C"])
def test_aurora_package_single(args: str, board: str) -> None:
    """Test Aurora packaging. In detail:
    - Check that the names of the XO files produced by AuroraFlow didn't change.
    - Check that the transformation creates a build dir to store the AuroraFlow XO files in.
    - Check that the files were created at the correct path and moved to the correct target.
    """
    FROM_XO_PREFIX = "aurora_flow_hw"  # noqa
    cfg = DataflowBuildConfig(
        output_dir=make_build_dir("test_aurora_package_single_output_dir"),
        board=board,
        partitioning_configuration=PartitioningConfiguration(
            num_fpgas=2, communication_kernel=MFCommunicationKernel.AURORA
        ),
    )
    prep = PrepareAuroraFlow(cfg)
    assert prep.aurora_storage.exists()
    moved = prep.package_single(args, FROM_XO_PREFIX + "_0.xo", "tested0.xo")
    assert moved.exists()
    moved = prep.package_single(args, FROM_XO_PREFIX + "_1.xo", "tested1.xo")
    assert moved.exists()
