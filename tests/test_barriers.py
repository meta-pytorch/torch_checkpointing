# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import concurrent.futures as cf
import multiprocessing as mp
import os
import pickle
import socket
import unittest.mock as mock
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
    FakeStore,  # pyrefly: ignore[missing-module-attribute]
)
from torch_checkpointing.barriers import (
    _default_store_connection_info,
    _FileStoreConnectionInfo,
    _NullStoreConnectionInfo,
    _store_connection_info,
    _TCPStoreConnectionInfo,
    DEFAULT_BARRIER_INIT_TIMEOUT_SEC,
    DEFAULT_STORE_BARRIER_KEY_PREFIX,
    DefaultStoreBarrierConfig,
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


BARRIER_KEY_PREFIX = "ckpt-barrier"
BARRIER_TIMEOUT_SECS = 15
# The barrier itself is what should time out on a real failure; this only bounds
# how long a parent waits for a child that never got that far.
SUBPROCESS_TIMEOUT_SECS = 30


def _single_rank_info() -> RankInfo:
    return RankInfo(
        global_rank=0,
        global_world_size=1,
        role_rank=0,
        role_world_size=1,
    )


def _multi_rank_info(rank: int = 0, world_size: int = 2) -> RankInfo:
    return RankInfo(
        global_rank=rank,
        global_world_size=world_size,
        role_rank=rank,
        role_world_size=world_size,
    )


def _config_for(connection, **kwargs) -> DefaultStoreBarrierConfig:
    """A config already pointed at ``connection``, as its capture would leave it."""
    config = DefaultStoreBarrierConfig(**kwargs)
    config._store_connection = connection
    return config


def _execute_barrier_group(
    config: DefaultStoreBarrierConfig,
    *,
    world_size: int = 2,
    executions: int = 1,
    timeout_secs: int = 10,
) -> None:
    """Drive every rank of a barrier group, from a thread each.

    Only a group of more than one rank puts anything in the store, and one rank's
    ``execute_barrier`` blocks until the rest arrive -- so the store-facing
    behaviour can only be exercised by running the whole group. Each rank builds
    its own barrier, and therefore its own store handle, exactly as separate
    processes would.
    """

    def participate(rank: int) -> None:
        barrier = config.create_barrier(_multi_rank_info(rank, world_size))
        for _ in range(executions):
            barrier.execute_barrier(timeout_secs=timeout_secs)

    with cf.ThreadPoolExecutor(max_workers=world_size) as pool:
        futures = [pool.submit(participate, rank) for rank in range(world_size)]
        for future in futures:
            future.result(timeout=timeout_secs * 2)


@pytest.fixture
def tcpstore_server():
    """A TCPStore server, standing in for one that some other component owns."""
    port = _find_free_port()
    store = dist.TCPStore(
        "localhost",
        port,
        world_size=1,
        is_master=True,
        timeout=timedelta(seconds=10),
        wait_for_workers=False,
    )
    yield store, port


@pytest.fixture
def default_process_group(tcpstore_server):
    """A default process group backed by ``tcpstore_server``."""
    store, port = tcpstore_server
    dist.init_process_group(backend="gloo", store=store, rank=0, world_size=1)
    try:
        yield store, port
    finally:
        dist.destroy_process_group()


def test_store_connection_info_describes_a_tcpstore(tcpstore_server):
    store, port = tcpstore_server

    assert _store_connection_info(store) == _TCPStoreConnectionInfo(
        host="localhost", port=port
    )


def test_store_connection_info_unwraps_prefix_stores(tcpstore_server):
    store, port = tcpstore_server
    nested = dist.PrefixStore("outer", dist.PrefixStore("inner", store))

    assert _store_connection_info(nested) == _TCPStoreConnectionInfo(
        host="localhost", port=port
    )


def test_store_connection_info_describes_a_filestore(tmp_path):
    path = str(tmp_path / "filestore")

    assert _store_connection_info(dist.FileStore(path)) == _FileStoreConnectionInfo(
        path=path
    )


def test_store_connection_info_describes_a_fake_process_group_store():
    """A fake process group's store never leaves the process that made it."""
    assert isinstance(_store_connection_info(FakeStore()), _NullStoreConnectionInfo)


def test_store_connection_info_describes_a_hash_store():
    """A HashStore is a plain in-memory map"""
    assert isinstance(
        _store_connection_info(dist.HashStore()), _NullStoreConnectionInfo
    )


def test_an_unreachable_store_cannot_open_client():
    """Nothing may hand out a handle that would coordinate only itself."""
    with pytest.raises(ValueError, match="unusable as a distributed barrier"):
        _NullStoreConnectionInfo("test").create_client_store(timedelta(seconds=10))


def test_barrier_on_an_in_process_store_allows_only_one_rank():
    config = _config_for(_NullStoreConnectionInfo("test"))
    barrier = config.create_barrier(_single_rank_info())
    barrier.execute_barrier(timeout_secs=10)  # No throw, no opening client

    with pytest.raises(ValueError, match="unusable"):
        config.create_barrier(_multi_rank_info())


def test_default_store_connection_info_reads_the_default_store(default_process_group):
    _, port = default_process_group

    assert _default_store_connection_info() == _TCPStoreConnectionInfo(
        host="localhost", port=port
    )


def test_default_store_connection_info_without_a_process_group():
    """No process group is not an error here -- a lone rank has no use for a store."""
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        assert _default_store_connection_info() is None


def test_config_rejects_an_empty_key_prefix():
    with pytest.raises(ValueError, match="key_prefix"):
        DefaultStoreBarrierConfig(key_prefix="")


def test_default_store_barrier_keys_are_namespaced_and_sequenced(tcpstore_server):
    store, port = tcpstore_server

    _execute_barrier_group(
        _config_for(
            _TCPStoreConnectionInfo(host="localhost", port=port),
            key_prefix=BARRIER_KEY_PREFIX,
            generation="7",
        ),
        executions=2,
    )

    assert sorted(store.list_keys()) == [
        f"{BARRIER_KEY_PREFIX}/7/0/last_member",
        f"{BARRIER_KEY_PREFIX}/7/0/num_members",
        f"{BARRIER_KEY_PREFIX}/7/1/last_member",
        f"{BARRIER_KEY_PREFIX}/7/1/num_members",
    ]
    # Nothing outside our namespace, not even the init key a client writes when it
    # is given a world size: those keys count the store owner's own participants.
    assert store.num_keys() == 4


def test_successive_barriers_do_not_reuse_keys(tcpstore_server):
    """A rebuilt barrier restarts at sequence 0, and must not land on stale keys."""
    store, port = tcpstore_server

    def run_barrier_group() -> None:
        _execute_barrier_group(
            _config_for(
                _TCPStoreConnectionInfo(host="localhost", port=port),
                key_prefix=BARRIER_KEY_PREFIX,
            )
        )

    run_barrier_group()
    run_barrier_group()

    # Two barriers at sequence 0 and four keys: the second waited for its own ranks
    # instead of passing on the `last_member` the first one left behind.
    assert len(store.list_keys()) == 4


def test_generation_separates_worker_restarts(monkeypatch):
    """A restarted worker can be handed the store its predecessor used."""
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "0")
    before_restart = DefaultStoreBarrierConfig().generation

    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "1")
    after_restart = DefaultStoreBarrierConfig().generation

    # The counter alone repeats after a restart, since the process is new.
    assert before_restart.startswith("0.")
    assert after_restart.startswith("1.")


def test_generation_survives_serialization(tcpstore_server):
    """The child rebuilds the barrier the parent described, not a fresh generation."""
    _, port = tcpstore_server
    config = _config_for(_TCPStoreConnectionInfo(host="localhost", port=port))

    assert pickle.loads(pickle.dumps(config)).generation == config.generation


def test_default_store_barrier_leaves_the_shared_store_alone(tcpstore_server):
    store, port = tcpstore_server
    owner_timeout = store.timeout

    _execute_barrier_group(
        _config_for(_TCPStoreConnectionInfo(host="localhost", port=port)),
        timeout_secs=int(owner_timeout.total_seconds()) + 300,
    )

    # The barrier retunes the timeout of the handle it owns; the store it borrows
    # keeps the timeout its owner chose.
    assert store.timeout == owner_timeout


def test_default_store_barrier_over_a_filestore(tmp_path):
    path = str(tmp_path / "filestore")
    store = dist.FileStore(path)

    _execute_barrier_group(
        _config_for(
            _FileStoreConnectionInfo(path=path),
            key_prefix=BARRIER_KEY_PREFIX,
            generation="7",
        )
    )

    # A FileStore keeps reference-count bookkeeping of its own in the same file, so
    # look only at the namespace the barrier owns.
    assert sorted(
        key for key in store.list_keys() if key.startswith(BARRIER_KEY_PREFIX)
    ) == [
        f"{BARRIER_KEY_PREFIX}/7/0/last_member",
        f"{BARRIER_KEY_PREFIX}/7/0/num_members",
    ]


def test_default_store_barrier_opens_a_client_handle_under_its_own_prefix():
    """A client even on rank 0, and never a rendezvous participant of the store."""
    config = _config_for(
        _TCPStoreConnectionInfo(host="localhost", port=12345),
        generation="7",
    )

    with (
        mock.patch("torch.distributed.PrefixStore") as mock_prefix_store,
        mock.patch("torch.distributed.TCPStore") as mock_tcpstore,
    ):
        config.create_barrier(_multi_rank_info())

    mock_tcpstore.assert_called_once_with(
        "localhost",
        12345,
        world_size=None,
        is_master=False,
        timeout=timedelta(seconds=DEFAULT_BARRIER_INIT_TIMEOUT_SEC),
        wait_for_workers=False,
    )
    mock_prefix_store.assert_called_once_with(
        f"{DEFAULT_STORE_BARRIER_KEY_PREFIX}/7", mock_tcpstore.return_value
    )


def test_config_captures_the_store_connection_when_serialized(default_process_group):
    _, port = default_process_group
    config = DefaultStoreBarrierConfig()
    assert config._store_connection is None

    revived = pickle.loads(pickle.dumps(config))

    # A process this config is sent to cannot look up a store it never created, so
    # the address has to be captured on the way out.
    assert revived._store_connection == _TCPStoreConnectionInfo(
        host="localhost", port=port
    )


def test_a_single_rank_barrier_needs_no_process_group_and_no_store(tcpstore_server):
    """The single-rank case is what lets single-process training skip torch.distributed."""
    store, _ = tcpstore_server
    keys_before = store.num_keys()

    with mock.patch("torch.distributed.is_initialized", return_value=False):
        config = DefaultStoreBarrierConfig()
        barrier = pickle.loads(pickle.dumps(config)).create_barrier(_single_rank_info())
        barrier.execute_barrier(timeout_secs=10)

    assert store.num_keys() == keys_before


def test_a_multi_rank_barrier_without_a_process_group_fails_loudly():
    """Serializing is fine; there is just nowhere for several ranks to meet."""
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        revived = pickle.loads(pickle.dumps(DefaultStoreBarrierConfig()))
        assert revived._store_connection is None

        with pytest.raises(
            RuntimeError,
            match="Could not find information about the default process group",
        ):
            revived.create_barrier(_multi_rank_info())


def test_config_serialization_with_a_captured_connection(tcpstore_server):
    """A captured address makes serialization independent of the process group."""
    _, port = tcpstore_server
    config = _config_for(_TCPStoreConnectionInfo(host="localhost", port=port))
    assert pickle.loads(pickle.dumps(config))._store_connection == (
        config._store_connection
    )


def _join_barrier_from_serialized_config(
    pickled_config: bytes, rank: int, world_size: int, timeout_secs: int
) -> None:
    """Rebuild a barrier from bytes and join it, as the write subprocess does."""
    config = pickle.loads(pickled_config)
    assert config._store_connection is not None, "store address was not captured"
    barrier = config.create_barrier(
        RankInfo(
            global_rank=rank,
            global_world_size=world_size,
            role_rank=rank,
            role_world_size=world_size,
        )
    )
    barrier.execute_barrier(timeout_secs=timeout_secs)


def test_default_store_barrier_joins_from_another_process(default_process_group):
    """A process with no process group of its own can still join the barrier."""
    pickled_config = pickle.dumps(
        DefaultStoreBarrierConfig(
            timeout_barrier_init_sec=BARRIER_TIMEOUT_SECS,
            key_prefix=BARRIER_KEY_PREFIX,
        )
    )
    # The barrier group is this process plus the child, which is unrelated to the
    # size of the process group whose store they meet on.
    child = mp.get_context("spawn").Process(
        target=_join_barrier_from_serialized_config,
        args=(pickled_config, 1, 2, BARRIER_TIMEOUT_SECS),
    )
    child.start()
    try:
        _join_barrier_from_serialized_config(pickled_config, 0, 2, BARRIER_TIMEOUT_SECS)
    finally:
        child.join(timeout=SUBPROCESS_TIMEOUT_SECS)

    assert child.exitcode == 0


def _trainer_with_write_subprocess(rank: int, world_size: int, port: int) -> None:
    """One trainer rank: join a process group, then barrier from a child process.

    This is the production shape in miniature. The process group -- and therefore
    the store -- belongs to this process, while the barrier runs in a child that
    only ever sees a serialized config, so the child has to reach a store it never
    created and meet the *other* trainer's child there.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        backend="gloo", rank=rank, world_size=world_size, init_method="env://"
    )
    try:
        pickled_config = pickle.dumps(
            DefaultStoreBarrierConfig(
                timeout_barrier_init_sec=BARRIER_TIMEOUT_SECS,
                key_prefix=BARRIER_KEY_PREFIX,
                # Ranks build their configs independently here, so pin the value the
                # per-process counter would otherwise assign.
                generation="0",
            )
        )
        child = mp.get_context("spawn").Process(
            target=_join_barrier_from_serialized_config,
            args=(pickled_config, rank, world_size, BARRIER_TIMEOUT_SECS),
        )
        child.start()
        child.join(timeout=SUBPROCESS_TIMEOUT_SECS)
        assert child.exitcode == 0, (
            f"rank {rank}'s write subprocess exited {child.exitcode}"
        )
    finally:
        dist.destroy_process_group()


def test_default_store_barrier_across_ranks_and_subprocesses():
    """Two trainer ranks, each barriering from its own write subprocess.

    The barrier only passes if both children resolved the same store out of their
    parents' process group and agreed on the same keys, so a wrong address, a
    per-process key namespace or a missed capture all show up as a timeout.
    """
    world_size = 2
    torch.multiprocessing.spawn(
        _trainer_with_write_subprocess,
        args=(world_size, _find_free_port()),
        nprocs=world_size,
        join=True,
    )
