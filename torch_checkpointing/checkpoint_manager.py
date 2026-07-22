# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
High-level checkpoint manager for save and load orchestration.

This module provides the trainer-facing manager API for checkpoint saves and
loads. It keeps the lower-level saver/loader configs composable while
centralizing default construction, waits, and staging prewarm behavior.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable

from .builder import (
    _get_default_rank_info,
    make_async_checkpoint_saver,
    make_sync_checkpoint_saver,
)
from .checkpoint_base import CheckpointBase
from .checkpoint_loader import CheckpointLoader
from .checkpoint_reader import CheckpointReader
from .checkpoint_saver import CheckpointSaver
from .config import (
    AsyncCheckpointSaverConfig,
    CheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from .logging_utils import EventLogger
from .metadata_manager import MetadataManager
from .staging import CheckpointStagerConfig
from .storage.base_storage import StorageConfig
from .storage.filesystem import LocalFileSystemStorageConfig
from .types import STATE_DICT
from .utils import ensure_future

DEFAULT_WAIT_TIMEOUT_SECS = 600


@dataclass(kw_only=True, slots=True)
class CheckpointSaveConfig:
    """
    Save-side configuration for CheckpointManager.
    """

    saver_config: CheckpointSaverConfig = field(
        default_factory=AsyncCheckpointSaverConfig
    )
    wait_timeout_secs: int | None = DEFAULT_WAIT_TIMEOUT_SECS

    def __post_init__(self) -> None:
        if not isinstance(self.saver_config, CheckpointSaverConfig):
            raise TypeError(
                "saver_config must be a SyncCheckpointSaverConfig or "
                "AsyncCheckpointSaverConfig"
            )
        if self.wait_timeout_secs is not None and self.wait_timeout_secs < 0:
            raise ValueError("wait_timeout_secs must be non-negative or None")


@dataclass(kw_only=True, slots=True)
class CheckpointLoadConfig:
    """
    Load-side configuration for CheckpointManager.
    """

    use_mmap: bool = True


class CheckpointManager:
    """
    Manager API for constructing and driving checkpoint saves and loads.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        save: CheckpointSaveConfig = field(default_factory=CheckpointSaveConfig)
        load: CheckpointLoadConfig = field(default_factory=CheckpointLoadConfig)
        storage_config: StorageConfig | None = None
        checkpoint_metadata_manager: MetadataManager | None = None
        subprocess_init_fn: Callable[..., None] | None = None
        subprocess_init_args: tuple[Any, ...] = ()
        pre_finalize_callback: Callable[[str, EventLogger], None] | None = None
        finalize_callback: Callable[[str, EventLogger], None] | None = None

        def build(self) -> "CheckpointManager":
            return CheckpointManager(config=self)

        @classmethod
        def sync_save(cls) -> "CheckpointManager.Config":
            return cls(
                save=CheckpointSaveConfig(
                    saver_config=SyncCheckpointSaverConfig(),
                )
            )

        @classmethod
        def async_save(
            cls,
            *,
            pinned_memory: bool = True,
        ) -> "CheckpointManager.Config":
            return cls(
                save=CheckpointSaveConfig(
                    saver_config=AsyncCheckpointSaverConfig(
                        staging_config=CheckpointStagerConfig(
                            use_pinned_memory=pinned_memory,
                            use_shared_memory=True,
                            use_async_staging=True,
                            use_non_blocking_copy=pinned_memory,
                        )
                    )
                )
            )

    def __init__(self, config: "CheckpointManager.Config" | None = None) -> None:
        self._config = config or CheckpointManager.Config()
        self._metadata_manager: MetadataManager | None = (
            self._config.checkpoint_metadata_manager
        )
        self._prewarm_staging_future: Future[Any] | None = None
        self._staging_future: Future[Any] | None = None
        self._write_future: Future[Any] | None = None
        self._closed = False

        rank_info = _get_default_rank_info()
        storage_config = self._config.storage_config or LocalFileSystemStorageConfig()
        saver_config = self._config.save.saver_config
        if isinstance(saver_config, SyncCheckpointSaverConfig):
            self._saver: CheckpointSaver = make_sync_checkpoint_saver(
                config=saver_config,
                rank_info=rank_info,
                pre_finalize_callback=self._config.pre_finalize_callback,
                finalize_callback=self._config.finalize_callback,
                storage_config=storage_config,
                checkpoint_metadata_manager=self._metadata_manager,
            )
        elif isinstance(saver_config, AsyncCheckpointSaverConfig):
            self._saver = make_async_checkpoint_saver(
                config=saver_config,
                rank_info=rank_info,
                pre_finalize_callback=self._config.pre_finalize_callback,
                finalize_callback=self._config.finalize_callback,
                subprocess_init_fn=self._config.subprocess_init_fn,
                subprocess_init_args=self._config.subprocess_init_args,
                storage_config=storage_config,
                checkpoint_metadata_manager=self._metadata_manager,
            )
        else:
            raise TypeError(
                f"Unsupported saver_config type: {saver_config.__class__.__name__}"
            )

        self._reader = CheckpointReader(
            rank_info=rank_info,
            storage_config=storage_config,
            disable_use_mmap_backed_storage_on_load=not self._config.load.use_mmap,
        )
        self._loader = CheckpointLoader(
            reader=self._reader,
            metadata_manager=self._metadata_manager,
        )

    def save(
        self,
        checkpoint_id: str,
        checkpoint: CheckpointBase,
    ) -> tuple[Future[Any], Future[Any]] | None:
        if self._closed:
            raise RuntimeError("Cannot save with a closed CheckpointManager")
        if self._prewarm_staging_future is not None:
            self._prewarm_staging_future.result(
                timeout=self._config.save.wait_timeout_secs
            )
            self._prewarm_staging_future = None

        save_result = self._saver.save(checkpoint_id, checkpoint)
        if save_result is None:
            self._staging_future = None
            self._write_future = None
            return None

        self._staging_future, self._write_future = save_result
        return save_result

    def load(
        self,
        checkpoint_id: str,
        checkpoint: CheckpointBase,
        *,
        map_location: Any = None,
        strict: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot load with a closed CheckpointManager")
        self._loader.load(
            checkpoint_id,
            checkpoint,
            default_map_location=map_location,
            strict=strict,
        )

    def prewarm_staging(self, checkpoint: CheckpointBase) -> None:
        if self._closed:
            raise RuntimeError("Cannot prewarm staging with a closed CheckpointManager")

        stager = self._saver.stager
        if stager is None:
            return

        # The stager may reuse pinned/shared buffers, so do not prewarm while a
        # previous checkpoint write may still be reading from those buffers.
        if self._write_future is not None:
            if not self._write_future.done():
                raise RuntimeError("Cannot prewarm staging while a write is active")
            self._write_future.result()
            self._write_future = None
            self._staging_future = None
        if self._staging_future is not None:
            if not self._staging_future.done():
                raise RuntimeError(
                    "Cannot prewarm staging while checkpoint staging is active"
                )
            self._staging_future.result()
            self._staging_future = None
        if self._prewarm_staging_future is not None:
            if not self._prewarm_staging_future.done():
                raise RuntimeError("Prewarm staging is already active")
            self._prewarm_staging_future.result()
            self._prewarm_staging_future = None

        state_dict_to_stage: STATE_DICT = {
            key: item.value
            for key, item in checkpoint.get_items().items()
            if item.requires_copy
        }
        if not state_dict_to_stage:
            return

        self._prewarm_staging_future = ensure_future(stager.stage(state_dict_to_stage))

    def close(self) -> None:
        if self._closed:
            return

        try:
            if self._prewarm_staging_future is not None:
                self._prewarm_staging_future.result(
                    timeout=self._config.save.wait_timeout_secs
                )
            if self._write_future is not None:
                self._write_future.result(timeout=self._config.save.wait_timeout_secs)
        finally:
            self._closed = True
            self._saver.close()
            self._loader.close()
            if self._metadata_manager is not None:
                self._metadata_manager.close()
