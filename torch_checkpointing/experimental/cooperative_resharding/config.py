# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Configuration and result types for Python cooperative checkpoint loading."""

from dataclasses import dataclass, fields

_GIB = 1024**3
_MIB = 1024**2


@dataclass(frozen=True, slots=True)
class CooperativeLoadConfig:
    """Resource limits for the Python cooperative checkpoint loader."""

    # Upper bound applied per owner node; the loader clamps it to advertised
    # capacity before forming batches.
    batch_target_bytes: int = 256 * _GIB
    shared_memory_fraction: float = 0.40
    shared_memory_chunk_bytes: int = _GIB
    range_consolidation_gap_bytes: int = 8 * _MIB
    # Shared bound for source archive metadata inspection and storage reads.
    download_workers: int = 64
    fetch_workers: int = 32
    server_workers: int = 128
    # Combined with the batch target, this bounds the intended per-node shared
    # memory working set to 512 GiB by default.
    max_inflight_batches: int = 2
    retry_attempts: int = 4
    retry_backoff_seconds: float = 0.5
    progress_timeout_seconds: float = 900.0
    pinned_buffer_bytes: int = 100 * _MIB
    pinned_buffer_count: int = 100
    enable_fast_scatter: bool = True

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "batch_target_bytes",
            "shared_memory_chunk_bytes",
            "download_workers",
            "fetch_workers",
            "server_workers",
            "max_inflight_batches",
            "retry_attempts",
            "pinned_buffer_bytes",
            "pinned_buffer_count",
        )
        for name in positive_integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < self.shared_memory_fraction <= 1.0:
            raise ValueError("shared_memory_fraction must be in (0, 1]")
        if self.range_consolidation_gap_bytes < 0:
            raise ValueError("range_consolidation_gap_bytes must be non-negative")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if self.progress_timeout_seconds <= 0:
            raise ValueError("progress_timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class CooperativeLoadResult:
    """Aggregate measurements returned by a cooperative load."""

    unique_source_bytes: int = 0
    storage_bytes_read: int = 0
    network_bytes_received: int = 0
    target_count: int = 0
    batch_count: int = 0
    elapsed_seconds: float = 0.0
    slowest_rank_seconds: float = 0.0

    def __post_init__(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) < 0:
                raise ValueError(f"{field.name} must be non-negative")
