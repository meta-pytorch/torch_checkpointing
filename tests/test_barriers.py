# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures as cf
import socket
import unittest.mock as mock

import pytest
import torch
from torch_checkpointing.barriers import (
    TCPStoreBarrierConfig,
)
from torch_checkpointing.types import RankInfo


@pytest.fixture
def master_rank_info():
    """Set up master rank info."""
    return RankInfo(
        global_rank=0,
        global_world_size=4,
        role_rank=0,
        role_world_size=4,
    )


@pytest.fixture
def non_master_rank_info():
    """Set up non-master rank info."""
    return RankInfo(
        global_rank=1,
        global_world_size=4,
        role_rank=1,
        role_world_size=4,
    )


@mock.patch("torch.distributed.TCPStore")
@mock.patch("torch.distributed.elastic.utils.store.barrier")
def test_tcpstore_barrier_initialization_master_rank(
    _, mock_tcpstore, master_rank_info
):
    """Test that TCPStoreBarrier initializes correctly."""
    # Create the barrier using the new elegant config system
    barrier_config = TCPStoreBarrierConfig(
        # Common fields
        timeout_barrier_init_sec=60,
        # TCPStore-specific fields
        use_checkpoint_barrier_tcpstore_libuv=True,
        tcpstore_port=12345,
        master_address="localhost",
    )
    barrier_config.create_barrier(master_rank_info)

    # Verify TCPStore was called with correct parameters
    mock_tcpstore.assert_called_once_with(
        "localhost",  # master_address from barrier_args
        12345,  # tcpstore_port from barrier_args
        world_size=4,  # world_size from rank_info
        is_master=True,
        timeout=mock.ANY,
        use_libuv=True,  # use_checkpoint_barrier_tcpstore_libuv from barrier_args
        wait_for_workers=False,
    )


@mock.patch("torch.distributed.TCPStore")
@mock.patch("torch.distributed.elastic.utils.store.barrier")
def test_tcpstore_barrier_initialization_non_master_rank(
    _, mock_tcpstore, non_master_rank_info
):
    """Test that TCPStoreBarrier initializes correctly."""
    # Create the barrier using the new elegant config system
    barrier_config = TCPStoreBarrierConfig(
        # Common fields
        timeout_barrier_init_sec=60,
        # TCPStore-specific fields
        use_checkpoint_barrier_tcpstore_libuv=True,
        tcpstore_port=12345,
        master_address="localhost",
    )
    barrier_config.create_barrier(non_master_rank_info)

    # Verify TCPStore was NOT called on non-master ranks; this happens lazily
    assert mock_tcpstore.call_count == 0


@mock.patch("torch.distributed.TCPStore")
@mock.patch("torch.distributed.elastic.utils.store.barrier")
def test_execute_barrier(mock_barrier, mock_tcpstore, master_rank_info):
    """Test that execute_barrier calls the barrier function correctly."""
    # Mock the TCPStore instance
    mock_tcpstore_instance = mock.MagicMock()
    mock_tcpstore.return_value = mock_tcpstore_instance

    # Create the barrier using the new elegant config system
    barrier_config = TCPStoreBarrierConfig(
        # Common fields
        timeout_barrier_init_sec=60,
        # TCPStore-specific fields
        use_checkpoint_barrier_tcpstore_libuv=True,
        tcpstore_port=12345,
        master_address="localhost",
    )
    barrier = barrier_config.create_barrier(master_rank_info)

    # Execute the barrier
    timeout_secs = 30
    barrier.execute_barrier(timeout_secs)

    # Verify that the barrier function was called with the correct parameters
    mock_barrier.assert_called_once_with(
        store=mock_tcpstore_instance,
        world_size=master_rank_info.role_world_size,
        key_prefix="0",
        barrier_timeout=timeout_secs,
    )

    # All store traffic belongs to the barrier itself; no extra per-rank writes.
    mock_tcpstore_instance.set.assert_not_called()

    # Execute the barrier again to test sequence number increment
    barrier.execute_barrier(timeout_secs)

    # Verify that the barrier function was called with the incremented sequence number
    mock_barrier.assert_called_with(
        store=mock_tcpstore_instance,
        world_size=master_rank_info.role_world_size,
        key_prefix="1",
        barrier_timeout=timeout_secs,
    )


@mock.patch("torch.distributed.TCPStore")
@mock.patch("torch.distributed.elastic.utils.store.barrier")
def test_execute_barrier_with_key_prefix(mock_barrier, mock_tcpstore, master_rank_info):
    mock_tcpstore_instance = mock.MagicMock()
    mock_tcpstore.return_value = mock_tcpstore_instance
    barrier = TCPStoreBarrierConfig(
        timeout_barrier_init_sec=60,
        use_checkpoint_barrier_tcpstore_libuv=True,
        tcpstore_port=12345,
        master_address="localhost",
        key_prefix="tenant-a/",
    ).create_barrier(master_rank_info)

    barrier.execute_barrier(timeout_secs=30)

    mock_barrier.assert_called_once_with(
        store=mock_tcpstore_instance,
        world_size=master_rank_info.role_world_size,
        key_prefix="tenant-a/0",
        barrier_timeout=30,
    )


@mock.patch("torch.distributed.TCPStore")
@mock.patch("torch.distributed.elastic.utils.store.barrier")
def test_config_create_barrier_master_rank(
    mock_barrier, mock_tcpstore, master_rank_info
):
    """Test that config.create_barrier() works with the new architecture."""
    # Mock the TCPStore instance
    mock_tcpstore_instance = mock.MagicMock()
    mock_tcpstore.return_value = mock_tcpstore_instance

    # Create complete barrier config
    barrier_config = TCPStoreBarrierConfig(
        timeout_barrier_init_sec=30,
        use_checkpoint_barrier_tcpstore_libuv=False,
        tcpstore_port=12345,
        master_address="localhost",
    )
    barrier = barrier_config.create_barrier(master_rank_info)

    # Verify TCPStore was called with correct parameters
    mock_tcpstore.assert_called_once_with(
        "localhost",  # master_address from barrier_args
        12345,  # tcpstore_port from barrier_args
        world_size=4,  # world_size from rank_info
        is_master=True,
        timeout=mock.ANY,
        use_libuv=False,  # use_checkpoint_barrier_tcpstore_libuv from barrier_args
        wait_for_workers=False,
    )

    # Test barrier execution works
    barrier.execute_barrier(600)  # Use the timeout from barrier_args
    mock_barrier.assert_called_once_with(
        store=mock_tcpstore_instance,
        world_size=master_rank_info.role_world_size,
        key_prefix="0",
        barrier_timeout=600,
    )


@mock.patch("torch.distributed.TCPStore")
@mock.patch("torch.distributed.elastic.utils.store.barrier")
def test_config_create_barrier_non_master_rank(
    mock_barrier, mock_tcpstore, non_master_rank_info
):
    """Test that config.create_barrier() works with the new architecture."""
    # Mock the TCPStore instance
    mock_tcpstore_instance = mock.MagicMock()
    mock_tcpstore.return_value = mock_tcpstore_instance

    # Create complete barrier config
    barrier_config = TCPStoreBarrierConfig(
        timeout_barrier_init_sec=30,
        use_checkpoint_barrier_tcpstore_libuv=False,
        tcpstore_port=12345,
        master_address="localhost",
    )
    barrier = barrier_config.create_barrier(non_master_rank_info)

    # On non-master ranks, TCPStore should be lazily initialized
    assert mock_tcpstore.call_count == 0

    # Test barrier execution works
    barrier.execute_barrier(600)  # Use the timeout from barrier_args

    # Verify TCPStore was called with correct parameters
    mock_tcpstore.assert_called_once_with(
        "localhost",  # master_address from barrier_args
        12345,  # tcpstore_port from barrier_args
        world_size=4,  # world_size from rank_info
        is_master=False,
        timeout=mock.ANY,
        use_libuv=False,  # use_checkpoint_barrier_tcpstore_libuv from barrier_args
        wait_for_workers=False,
    )
    mock_barrier.assert_called_once_with(
        store=mock_tcpstore_instance,
        world_size=non_master_rank_info.role_world_size,
        key_prefix="0",
        barrier_timeout=600,
    )


@mock.patch("torch.distributed.TCPStore")
@mock.patch("torch.distributed.elastic.utils.store.barrier")
def test_execute_barrier_with_custom_timeout(
    mock_barrier, mock_tcpstore, master_rank_info
):
    """Test that execute_barrier properly uses custom timeout parameter."""
    # Mock the TCPStore instance
    mock_tcpstore_instance = mock.MagicMock()
    mock_tcpstore.return_value = mock_tcpstore_instance

    # Create the barrier using the new elegant config system
    barrier_config = TCPStoreBarrierConfig(
        # Complete configuration
        timeout_barrier_init_sec=60,
        # TCPStore-specific fields
        use_checkpoint_barrier_tcpstore_libuv=True,
        tcpstore_port=12345,
        master_address="localhost",
    )
    barrier = barrier_config.create_barrier(master_rank_info)

    # Execute barrier with custom timeout
    custom_timeout = 600
    barrier.execute_barrier(timeout_secs=custom_timeout)

    # Verify that the barrier function was called with the barrier_timeout parameter
    mock_barrier.assert_called_once_with(
        store=mock_tcpstore_instance,
        world_size=master_rank_info.role_world_size,
        key_prefix="0",
        barrier_timeout=custom_timeout,
    )

    # Execute barrier again without custom timeout (should use default)
    timeout_secs = 300
    barrier.execute_barrier(timeout_secs)

    # Verify that it still works with default timeout
    mock_barrier.assert_called_with(
        store=mock_tcpstore_instance,
        world_size=master_rank_info.role_world_size,
        key_prefix="1",
        barrier_timeout=timeout_secs,
    )


def _find_free_port():
    """Find an available port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_barrier_worker(rank, world_size, port, timeout=3, miss_barrier=False):
    """Create a TCPStoreBarrier and maybe execute it."""
    rank_info = RankInfo(
        global_rank=rank,
        global_world_size=world_size,
        role_rank=rank,
        role_world_size=world_size,
    )
    config = TCPStoreBarrierConfig(
        timeout_barrier_init_sec=timeout,
        use_checkpoint_barrier_tcpstore_libuv=True,
        tcpstore_port=port,
        master_address="localhost",
    )
    barrier = config.create_barrier(rank_info)
    if not miss_barrier:
        barrier.execute_barrier(timeout_secs=timeout)


# Multi-rank barrier tests spawn multiple processes; gate them on an
# accelerator so they only run in a multi-GPU environment.
@pytest.mark.gpus_needed_4
@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="requires an accelerator (multi-rank barrier coordination)",
)
def test_tcpstore_barrier_execute():
    """Test TCPStoreBarrier with a real TCPStore (no mocks)."""
    world_size = 4
    port = _find_free_port()
    with cf.ProcessPoolExecutor(max_workers=world_size) as pool:
        futs = [
            pool.submit(_run_barrier_worker, rank, world_size, port)
            for rank in range(world_size)
        ]
        cf.wait(futs, timeout=30)
        for rank, fut in enumerate(futs):
            assert fut.done(), f"Rank {rank} did not finish"
            assert fut.exception() is None, (
                f"Rank {rank} failed with {fut.exception()!r}"
            )


@pytest.mark.gpus_needed_4
@pytest.mark.skipif(
    not torch.accelerator.is_available(),
    reason="requires an accelerator (multi-rank barrier coordination)",
)
@pytest.mark.parametrize("missed_rank", [0, 2])
def test_tcpstore_barrier_timeout_on_missing_rank(missed_rank: int):
    """Test that barrier times out when one rank is too slow to get to the barrier."""
    world_size = 4
    port = _find_free_port()
    with cf.ProcessPoolExecutor(max_workers=world_size) as pool:
        futs = [
            pool.submit(
                _run_barrier_worker,
                rank,
                world_size,
                port,
                miss_barrier=(rank == missed_rank),
            )
            for rank in range(world_size)
        ]
        cf.wait(futs, timeout=30)
        for rank, fut in enumerate(futs):
            assert fut.done(), f"Rank {rank} did not finish"
            if rank != missed_rank:
                assert fut.exception() is not None, f"Rank {rank} did not fail"
