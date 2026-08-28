# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Barrier implementations for synchronizing distributed checkpoint operations.

This module provides abstract definition and concrete barrier implementations that are
useful to ensure that all ranks in a distributed training environment complete their
checkpoint operations before proceeding, which is essential for data consistency.
"""

from __future__ import annotations

import abc
import contextlib
import itertools
import logging
import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Generator

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util
from torch.distributed.distributed_c10d import _get_default_store

from .types import RankInfo

logger = logging.getLogger()

# Namespace for the keys DefaultStoreBarrier writes into the shared store.
DEFAULT_STORE_BARRIER_KEY_PREFIX = "torch_checkpointing_barrier"

# How long a barrier may spend setting itself up (for a store-backed barrier, this
# only bounds opening the handle -- waiting for the other ranks is bounded by the
# timeout passed to execute_barrier).
DEFAULT_BARRIER_INIT_TIMEOUT_SEC = 600


@dataclass
class BarrierConfig(abc.ABC):
    """Abstract base configuration class for all barrier implementations."""

    # Common fields shared by all barriers
    timeout_barrier_init_sec: int

    @abc.abstractmethod
    def create_barrier(self, rank_info: RankInfo) -> "Barrier":
        """Create a barrier instance from this configuration."""
        pass

    def use_in_subprocess(self) -> contextlib.AbstractContextManager[None]:
        """
        Init and teardown hooks to execute in the main process when the barrier is to
        be instantiated and used in a subprocess.

        By default it is a nullcontext.
        """
        return contextlib.nullcontext()


@dataclass
class TCPStoreBarrierConfig(BarrierConfig):
    """Configuration for TCPStoreBarrier.

    ``key_prefix`` isolates independent barriers that share a TCPStore endpoint.
    """

    use_checkpoint_barrier_tcpstore_libuv: bool
    tcpstore_port: int
    master_address: str
    key_prefix: str = ""

    def create_barrier(self, rank_info: RankInfo) -> "TCPStoreBarrier":
        """Create a TCPStoreBarrier instance from this configuration."""
        return TCPStoreBarrier(config=self, rank_info=rank_info)


@dataclass
class DefaultStoreBarrierConfig(BarrierConfig):
    """Configuration for DefaultStoreBarrier.

    ``key_prefix`` namespaces every key the barrier writes, keeping it clear of the
    keys the process group owns in the same store. Barriers that run independently
    of each other need distinct prefixes.

    ``generation`` makes each barrier independent of the ones built before it. This is
    to support multiple independent barriers using the same underlying store.
    """

    key_prefix: str = DEFAULT_STORE_BARRIER_KEY_PREFIX
    generation: str = field(
        # Deferred, so it can name a helper defined further down the file.
        default_factory=lambda: _next_barrier_generation()  # pyrefly: ignore[unbound-name]
    )
    timeout_barrier_init_sec: int = DEFAULT_BARRIER_INIT_TIMEOUT_SEC

    def __post_init__(self) -> None:
        if not self.key_prefix:
            raise ValueError(
                "key_prefix must be non-empty: this barrier shares a store with the "
                "process group, so its keys have to be namespaced."
            )
        # Deliberately not a field: the store is an implementation detail rather than
        # something to configure, and it is filled in once, wherever the default
        # process group happens to be reachable.
        self._store_connection: _StoreConnectionInfo | None = None

    def create_barrier(self, rank_info: RankInfo) -> "DefaultStoreBarrier":
        """Create a DefaultStoreBarrier instance from this configuration."""
        return DefaultStoreBarrier(config=self, rank_info=rank_info)

    @contextlib.contextmanager
    def use_in_subprocess(self) -> Generator[None]:
        """
        Hold a reference to the default process group's store so that it stays alive
        while a subprocess is using it.

        This is necessary for a TCPStore because the server gets destroyed when the
        final reference to it disappears. This may happen, for example, if the trainer
        calls torch.distributed.destroy_process_group() at the end of training but
        before the final async checkpoint has completed.
        """
        store = (
            _get_default_store()
            if dist.is_available() and dist.is_initialized()
            else None
        )
        if store is not None:
            self._store_connection = _store_connection_info(store)
        try:
            yield
        finally:
            del store

    def _resolve_store_connection(self) -> _StoreConnectionInfo | None:
        """The captured store coordinates, or the default process group's store.

        ``None`` when there is no default process group to read them from.
        """
        if self._store_connection is not None:
            return self._store_connection
        return _default_store_connection_info()

    def __getstate__(self) -> dict[str, Any]:
        # Serialization is the process boundary. The default store belongs to the
        # process that called init_process_group, so a process this config is sent
        # to cannot look it up -- capture the coordinates here, while they are still
        # reachable, and every path that ships this config gets that for free.
        return {**self.__dict__, "_store_connection": self._resolve_store_connection()}


class Barrier(abc.ABC):
    """
    Abstract base class for synchronization barriers.

    A barrier ensures that all ranks in a distributed environment reach a certain
    point in execution before any rank proceeds further, which is essential for
    coordinating operations like checkpointing across multiple processes.

    The barrier supports a flexible model where:
    1. Initialization sets up the barrier infrastructure with a base prefix
    2. Sub-worlds are created/registered during initialization
    3. Execution targets specific sub-worlds as needed

    This pattern matches how distributed checkpointing works in practice:
    - Set up barrier infrastructure once (e.g., with prefix "checkpoint_")
    - Create multiple sub-worlds (e.g., ["checkpoint_persistent_", "checkpoint_temp_"])
    - Execute barriers on specific sub-worlds as needed (e.g., "checkpoint_persistent_")
    """

    @abc.abstractmethod
    def __init__(self, config: BarrierConfig):
        """
        Initialize a barrier and set up sub-worlds.
        """

    @abc.abstractmethod
    def execute_barrier(self, timeout_secs: int) -> None:
        """
        Execute a synchronization barrier.

        Args:
            timeout_secs: Maximum time in seconds to wait for all ranks to reach
                         the barrier.

        Examples:
            # Execute barrier with custom timeout
            barrier.execute_barrier(timeout_secs=300)
        """


class TCPStoreBarrier(Barrier):
    """
    A barrier implementation using PyTorch's TCPStore for synchronization.

    This barrier uses TCP-based distributed key-value stores to coordinate
    synchronization across multiple processes. It uses separate TCPStore
    instances for each barrier prefix and maintaining global state for
    sequence tracking.

    TCPStore clients should be used independently as we expect undefined behavior
    (and potential deadlocks) if used by multiple threads. This implementation
    tracks one TCPStore per barrier prefix.
    """

    def __init__(
        self,
        config: TCPStoreBarrierConfig,
        rank_info: RankInfo,
    ):
        """
        Initialize a TCPStoreBarrier.

        Args:
            config: Configuration containing TCPStore-specific fields.
            rank_info: Information about the current rank in a distributed environment.
        """
        if config is None:
            raise ValueError("TCPStoreBarrierConfig is required for TCPStoreBarrier")

        # Store the config and rank info
        self._config = config
        self._rank_info = rank_info

        # Counter to track barrier sequence
        self._tcp_store_barrier_seq = 0

        # Create TCPStore instance for barrier synchronization
        self._is_master = rank_info.role_rank == 0
        self._tcp_store: dist.TCPStore | None = None
        if self._is_master:
            # Eagerly create TCPStore server on master node. Non-master nodes will
            # connect on first usage.
            self._init_store()

    def _init_store(self) -> None:
        """
        Initialize the TCPStore instance for barrier synchronization.
        """
        if self._tcp_store is not None:
            return
        logger.info(
            f"Initializing TCPStore master_address={self._config.master_address} tcpstore_port={self._config.tcpstore_port} rank={self._rank_info.role_rank} "
            f"world_size={self._rank_info.role_world_size} timeout_barrier_init_sec={self._config.timeout_barrier_init_sec} "
            f"use_checkpoint_barrier_tcpstore_libuv={self._config.use_checkpoint_barrier_tcpstore_libuv} is_master={self._is_master}"
        )
        self._tcp_store = dist.TCPStore(
            self._config.master_address,
            int(self._config.tcpstore_port),
            world_size=self._rank_info.role_world_size,
            is_master=self._is_master,
            timeout=timedelta(seconds=self._config.timeout_barrier_init_sec),
            use_libuv=self._config.use_checkpoint_barrier_tcpstore_libuv,
            wait_for_workers=False,  # Don't wait for non-master nodes to connect
        )

    def _namespace_key(self, key: str) -> str:
        return f"{self._config.key_prefix}{key}"

    def execute_barrier(self, timeout_secs: int) -> None:
        """
        Execute a synchronization barrier.

        Args:
            timeout_secs: Maximum time in seconds to wait for all ranks to reach
                         the barrier.

        The implementation follows:
        1. Uses TCPStore for synchronization
        2. Uses sequence numbers that increment per barrier execution
        """
        logger.info(f"Executing barrier timeout_secs={timeout_secs}")
        self._init_store()

        # Execute barrier for that sequence number
        store_util.barrier(
            store=self._tcp_store,
            world_size=self._rank_info.role_world_size,
            key_prefix=self._namespace_key(str(self._tcp_store_barrier_seq)),
            barrier_timeout=timeout_secs,
        )
        self._tcp_store_barrier_seq += 1


class DefaultStoreBarrier(Barrier):
    """
    A barrier implementation over the store that backs the default process group.

    Unlike TCPStoreBarrier, no store server is started: this opens a client handle
    to the store ``init_process_group`` already set up, so there is no second port
    to pick, agree on across ranks and keep free. The handle is always a client,
    even on rank 0 -- the server belongs to the process that created it, and this
    barrier neither owns nor outlives it.

    Every key lives under ``config.key_prefix`` and ``config.generation``, so nothing
    this barrier writes can collide with the keys the process group keeps in the same
    store, nor with those of a barrier built before it (see ``generation``).
    """

    def __init__(
        self,
        config: DefaultStoreBarrierConfig,
        rank_info: RankInfo,
    ):
        """
        Initialize a DefaultStoreBarrier.

        Args:
            config: Configuration naming the store and the key namespace to use.
            rank_info: Information about the current rank in a distributed environment.

        Raises:
            RuntimeError: if the group spans more than one rank and the store
                coordinates were neither captured earlier nor available from a
                default process group in this process.
            ValueError: if the group spans more than one rank but its store does
                not leave this process (a fake process group).
        """
        self._config = config
        self._rank_info = rank_info
        self._barrier_seq = 0

        # If role_world_size <= 1, then don't create a store; barrier is a no-op.
        self._store: dist.Store | None = None
        if rank_info.role_world_size <= 1:
            return

        connection = config._resolve_store_connection()
        if connection is None:
            raise RuntimeError(
                "Could not find information about the default process group. Call "
                "torch.distributed.init_process_group() before constructing "
                "CheckpointProcess or instantiating the DefaultStoreBarrier."
            )
        underlying_store = connection.create_client_store(
            timeout=timedelta(seconds=config.timeout_barrier_init_sec)
        )
        prefix = f"{config.key_prefix}/{config.generation}"
        self._store = dist.PrefixStore(prefix, underlying_store)
        logger.info(
            f"Connecting to the default process group's store {connection} as a client "
            f"rank={rank_info.role_rank} world_size={rank_info.role_world_size} "
            f"key_prefix={config.key_prefix} generation={config.generation} "
            f"timeout_barrier_init_sec={config.timeout_barrier_init_sec}"
        )

    def execute_barrier(self, timeout_secs: int) -> None:
        """
        Execute a synchronization barrier.

        Args:
            timeout_secs: Maximum time in seconds to wait for all ranks to reach
                         the barrier.
        """
        if self._store is None:
            return
        logger.info(f"Executing barrier timeout_secs={timeout_secs}")
        store_util.barrier(
            store=self._store,
            world_size=self._rank_info.role_world_size,
            key_prefix=str(self._barrier_seq),
            barrier_timeout=timeout_secs,
        )
        self._barrier_seq += 1


# ---------------------------------------------------------------------------
# Internals: describing a store well enough that another process can reach it,
# and naming a key space no earlier barrier can have used.
# ---------------------------------------------------------------------------


# Monotonic per-process counter behind the default barrier generation.
_barrier_sequence = itertools.count()


def _next_barrier_generation() -> str:
    """A namespace token that no barrier built before this one shares.

    Two things can hand a new barrier a store that still holds an older barrier's
    keys, and the token has to separate them both:

    - Another barrier built earlier in this process, covered by a counter.
    - A predecessor in an earlier *worker* process, when the store outlives the
      process. ``torch.distributed.elastic`` restarts workers in place against the
      agent's shared store, which the agent creates once and keeps for its whole
      lifetime, so a counter alone would repeat after a restart. Its restart count
      is per-worker-generation and identical across ranks, which is what the store
      itself no longer provides -- ``_create_c10d_store`` hands the shared store to
      a restarted worker unprefixed.
    """
    restart_count = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
    return f"{restart_count}.{next(_barrier_sequence)}"


class _StoreConnectionInfo(abc.ABC):
    """
    Serializable coordinates of a torch.distributed.Store that is already running, so
    that subprocesses can use the store.
    """

    @abc.abstractmethod
    def create_client_store(self, timeout: timedelta) -> dist.Store:
        """Open a new client handle to the store described by this object."""


@dataclass(frozen=True)
class _TCPStoreConnectionInfo(_StoreConnectionInfo):
    """Address of a running ``TCPStore`` server."""

    host: str
    port: int

    def create_client_store(self, timeout: timedelta) -> dist.Store:
        # No world size, and no waiting for workers: passing a world size makes a
        # client increment the store's init key, which counts the process group's
        # rendezvous participants and is none of our business.
        return dist.TCPStore(
            self.host,
            self.port,
            world_size=None,
            is_master=False,
            timeout=timeout,
            wait_for_workers=False,
        )


@dataclass(frozen=True)
class _FileStoreConnectionInfo(_StoreConnectionInfo):
    """Path of a running ``FileStore``."""

    path: str

    def create_client_store(self, timeout: timedelta) -> dist.Store:
        # No worker count, so this handle never removes the backing file when it is
        # closed: cleaning up belongs to whoever created the store.
        store = dist.FileStore(self.path)
        store.set_timeout(timeout)
        return store


@dataclass(frozen=True)
class _NullStoreConnectionInfo(_StoreConnectionInfo):
    """
    Stands in for a store that is unreachable by other ranks and thus unusable by
    DefaultStoreBarrier. Trying to build this in a distributed environment throws.
    """

    desc: str

    def create_client_store(self, timeout: timedelta) -> dist.Store:
        raise ValueError(
            f"{self.desc} is unusable as a distributed barrier. Call "
            "init_process_group with a TCPStore or a FileStore start method instead."
        )


def _store_connection_info(store: dist.Store) -> _StoreConnectionInfo:
    """Describe ``store`` so that another process can connect to it.

    ``PrefixStore`` layers are peeled off first -- ``init_process_group`` wraps the
    store it hands to each process group -- because only the store underneath owns
    the connection.
    """
    while isinstance(store, dist.PrefixStore):
        store = store.underlying_store

    if isinstance(store, dist.TCPStore):
        return _TCPStoreConnectionInfo(host=store.host, port=store.port)
    elif isinstance(store, dist.FileStore):
        return _FileStoreConnectionInfo(path=store.path)
    else:
        # HashStore (process-local and unusable) or FakeStore (for testing) or some
        # other kind of store that we don't know about. Will fail if we try to
        # instantiate it in a distributed training setup, but fine for single-rank.
        return _NullStoreConnectionInfo(desc=repr(store))


def _default_store_connection_info() -> _StoreConnectionInfo | None:
    """
    Describe the store that backs the default process group; none if not initialized.
    """
    if not dist.is_available() or not dist.is_initialized():
        return None
    return _store_connection_info(_get_default_store())
