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
import logging
from dataclasses import dataclass
from datetime import timedelta

import torch.distributed as dist
import torch.distributed.elastic.utils.store as store_util

from .types import RankInfo

logger = logging.getLogger()


@dataclass
class BarrierConfig(abc.ABC):
    """Abstract base configuration class for all barrier implementations."""

    # Common fields shared by all barriers
    timeout_barrier_init_sec: int

    @abc.abstractmethod
    def create_barrier(self, rank_info: RankInfo) -> "Barrier":
        """Create a barrier instance from this configuration."""
        pass


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
