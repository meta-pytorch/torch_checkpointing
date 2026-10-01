# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import replace

import pytest
from torch_checkpointing.experimental.cooperative_resharding.config import (
    CooperativeLoadConfig,
    CooperativeLoadResult,
)


def test_default_config_has_bounded_positive_resources() -> None:
    config = CooperativeLoadConfig()

    assert config.batch_target_bytes == 256 * 1024**3
    assert config.shared_memory_fraction == 0.40
    assert config.shared_memory_chunk_bytes == 1024**3
    assert config.fetch_workers == 32
    assert config.server_workers == 128
    assert config.pinned_buffer_bytes == 100 * 1024**2
    assert config.pinned_buffer_count == 100


def test_default_config_bounds_inflight_batch_working_set() -> None:
    config = CooperativeLoadConfig()

    assert config.max_inflight_batches == 2
    assert config.batch_target_bytes * config.max_inflight_batches == 512 * 1024**3


def test_default_download_workers_bound_metadata_and_storage_concurrency() -> None:
    config = CooperativeLoadConfig()

    assert config.download_workers == 64


@pytest.mark.parametrize(
    "field_name",
    [
        "batch_target_bytes",
        "shared_memory_chunk_bytes",
        "download_workers",
        "fetch_workers",
        "server_workers",
        "max_inflight_batches",
        "retry_attempts",
        "pinned_buffer_bytes",
        "pinned_buffer_count",
    ],
)
def test_config_rejects_nonpositive_integer_limits(field_name: str) -> None:
    with pytest.raises(ValueError, match=f"{field_name} must be positive"):
        replace(CooperativeLoadConfig(), **{field_name: 0})


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.1])
def test_config_rejects_invalid_shared_memory_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="shared_memory_fraction"):
        replace(CooperativeLoadConfig(), shared_memory_fraction=fraction)


def test_config_rejects_negative_retry_and_range_values() -> None:
    with pytest.raises(ValueError, match="range_consolidation_gap_bytes"):
        replace(CooperativeLoadConfig(), range_consolidation_gap_bytes=-1)
    with pytest.raises(ValueError, match="retry_backoff_seconds"):
        replace(CooperativeLoadConfig(), retry_backoff_seconds=-0.1)
    with pytest.raises(ValueError, match="progress_timeout_seconds"):
        replace(CooperativeLoadConfig(), progress_timeout_seconds=0)


def test_result_rejects_negative_measurements() -> None:
    assert CooperativeLoadResult(storage_bytes_read=7).storage_bytes_read == 7
    with pytest.raises(ValueError, match="network_bytes_received"):
        CooperativeLoadResult(network_bytes_received=-1)
