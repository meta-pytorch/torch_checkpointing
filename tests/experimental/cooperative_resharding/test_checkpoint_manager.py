# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch_checkpointing.experimental.cooperative_resharding as cooperative_resharding
from torch_checkpointing.checkpoint_manager import (
    CheckpointManager as CoreCheckpointManager,
)
from torch_checkpointing.checkpoint_reader import (
    CheckpointReader as CoreCheckpointReader,
)
from torch_checkpointing.experimental.cooperative_resharding import (
    CheckpointLoader,
    CheckpointManager,
    CheckpointReader,
    DefaultResharder,
)
from torch_checkpointing.experimental.cooperative_resharding.config import (
    CooperativeLoadConfig,
    CooperativeLoadResult,
)


def test_package_exports_only_high_level_experimental_api() -> None:
    assert cooperative_resharding.__all__ == [
        "CheckpointLoader",
        "CheckpointManager",
        "CheckpointReader",
        "DefaultResharder",
    ]
    assert not hasattr(cooperative_resharding, "CooperativeLoadConfig")
    assert not hasattr(cooperative_resharding, "CooperativeLoadResult")
    assert CheckpointLoader.__module__.endswith(".checkpoint_loader")
    assert DefaultResharder.__module__.endswith(".default_resharder")
    assert CooperativeLoadConfig().max_inflight_batches > 0
    assert CooperativeLoadResult().target_count == 0


def test_config_builds_isolated_experimental_manager() -> None:
    manager = CheckpointManager.Config.with_sync_save().build()
    try:
        assert type(manager) is CheckpointManager
        assert type(manager._reader) is CheckpointReader
        assert type(manager) is not CoreCheckpointManager
        assert type(manager._reader) is not CoreCheckpointReader
    finally:
        manager.close()
