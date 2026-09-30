# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise checkpoint communication without launching a child process."""

from unittest import mock

from torch_checkpointing.checkpoint_process import (
    CheckpointProcess,
    CheckpointProcessConfig,
)
from torch_checkpointing.checkpoint_writer import (
    CheckpointWriterArgs,
    CheckpointWriterConfig,
)
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo


def make_detached_checkpoint_process() -> CheckpointProcess:
    rank = RankInfo(global_world_size=1, global_rank=0, role_rank=0, role_world_size=1)
    with mock.patch.object(CheckpointProcess, "_create_subprocess"):
        return CheckpointProcess(
            rank,
            CheckpointProcessConfig(),
            lambda: None,
            (),
            CheckpointWriterArgs(
                CheckpointWriterConfig(), rank, LocalFileSystemStorageConfig()
            ),
        )
