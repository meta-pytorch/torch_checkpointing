# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Configuration classes for checkpoint saver construction.
"""

from dataclasses import dataclass, field

from .checkpoint_process import CheckpointProcessConfig
from .checkpoint_writer import CheckpointWriterConfig
from .staging import CheckpointStagerConfig


class CheckpointSaverConfig:
    """Base class for checkpoint saver configurations."""

    pass


@dataclass
class SyncCheckpointSaverConfig(CheckpointSaverConfig):
    """
    Configuration for synchronous checkpoint saving.

    Attributes:
        writer_config: Configuration options for the checkpoint writer component.
    """

    writer_config: CheckpointWriterConfig = field(
        default_factory=CheckpointWriterConfig
    )


@dataclass
class AsyncCheckpointSaverConfig(CheckpointSaverConfig):
    """
    Configuration for asynchronous checkpoint saving.

    Attributes:
        writer_config: Configuration options for the checkpoint writer component.
        staging_config: Configuration options for the async staging component.
        process_config: Configuration options for the async checkpoint process component.
    """

    writer_config: CheckpointWriterConfig = field(
        default_factory=CheckpointWriterConfig
    )
    staging_config: CheckpointStagerConfig = field(
        default_factory=CheckpointStagerConfig
    )
    process_config: CheckpointProcessConfig = field(
        default_factory=CheckpointProcessConfig
    )
