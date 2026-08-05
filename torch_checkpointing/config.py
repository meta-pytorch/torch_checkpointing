# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Configuration classes for checkpoint saver and loader construction.
"""

from dataclasses import dataclass, field

from .checkpoint_process import CheckpointProcessConfig
from .checkpoint_writer import CheckpointWriterConfig
from .staging import CheckpointStagerConfig

DEFAULT_WAIT_TIMEOUT_SECS = 600


@dataclass(kw_only=True)
class CheckpointSaverConfig:
    """Base configuration shared by checkpoint savers.

    Attributes:
        writer_config: Configuration options for the checkpoint writer component.
        wait_timeout_secs: Maximum time ``CheckpointManager`` waits for staging
            or writing to finish. ``None`` disables the timeout.
    """

    writer_config: CheckpointWriterConfig = field(
        default_factory=CheckpointWriterConfig
    )
    wait_timeout_secs: int | None = DEFAULT_WAIT_TIMEOUT_SECS

    def __post_init__(self) -> None:
        if self.wait_timeout_secs is not None and self.wait_timeout_secs < 0:
            raise ValueError("wait_timeout_secs must be non-negative or None")


@dataclass(kw_only=True)
class CheckpointLoaderConfig:
    """Configuration for checkpoint loading."""

    use_mmap: bool = True


@dataclass
class SyncCheckpointSaverConfig(CheckpointSaverConfig):
    """
    Configuration for synchronous checkpoint saving.

    Attributes:
        writer_config: Configuration options for the checkpoint writer component.
    """

    pass


@dataclass
class AsyncCheckpointSaverConfig(CheckpointSaverConfig):
    """
    Configuration for asynchronous checkpoint saving.

    Attributes:
        writer_config: Configuration options for the checkpoint writer component.
        staging_config: Configuration options for the async staging component.
        process_config: Configuration options for the async checkpoint process component.
    """

    staging_config: CheckpointStagerConfig = field(
        default_factory=CheckpointStagerConfig
    )
    process_config: CheckpointProcessConfig = field(
        default_factory=CheckpointProcessConfig
    )
