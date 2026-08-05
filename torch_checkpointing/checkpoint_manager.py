# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
High-level checkpoint manager for save and load orchestration.

``CheckpointManager`` is the single entry point trainers use to checkpoint. It
binds a declarative schema (per-item :class:`ItemSpec`: ``requires_copy``,
on-disk ``layout``, ``resharder``, ``required``) once in its ``Config``, then
accepts plain ``Mapping[ItemKey, Any]`` payloads::

    manager = CheckpointManager(CheckpointManager.Config(
        items={"model": ItemSpec(resharder=DTensorResharder())},
    ))
    manager.save("/ckpt/step_1000", {"model": model.state_dict(), "step": 1000})
    manager.load("/ckpt/step_1000", into={"model": model.state_dict()})

It hides the saver, stager, subprocess, reader, and metadata manager. The
metadata manager is auto-wired on both save and load whenever any item declares
a resharder, so a resharding load cannot silently take the layout-unchanged
fast path.
"""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager

from .builder import (
    _get_default_rank_info,
    make_async_checkpoint_saver,
    make_sync_checkpoint_saver,
)
from .checkpoint_base import CheckpointBase, CheckpointItem
from .checkpoint_loader import CheckpointLoader
from .checkpoint_reader import CheckpointReader
from .checkpoint_saver import CheckpointSaver
from .config import (
    AsyncCheckpointSaverConfig,
    CheckpointLoaderConfig,
    CheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from .logging_utils import EventLogger
from .metadata_manager import DefaultMetadataManager, MetadataManager
from .schema import _CheckpointSchema, ItemSpec
from .storage.base_storage import StorageConfig
from .storage.filesystem import LocalFileSystemStorageConfig
from .types import ItemKey, STATE_DICT
from .utils import ensure_future


class _DictCheckpoint(CheckpointBase):
    """Adapter presenting a reconciled ``{key: CheckpointItem}`` mapping to the
    ``CheckpointBase``-based saver/loader, and capturing the loaded state_dict.

    On save, only :meth:`get_items` is used. On load, :meth:`get_items` supplies
    the per-item templates (the live values from ``into=``) that the reader
    reshards/copies into, and :meth:`load_state_dict` captures the loaded mapping
    so :meth:`CheckpointManager.load` can return it.
    """

    def __init__(self, items: dict[str, CheckpointItem]) -> None:
        self._items = items
        self.result: dict[str, Any] = {}

    def get_items(self) -> dict[str, CheckpointItem]:
        return self._items

    def load_state_dict(self, state_dict: STATE_DICT) -> None:
        self.result = dict(state_dict)


class CheckpointManager:
    """
    Manager API for constructing and driving checkpoint saves and loads.
    """

    @dataclass(kw_only=True, slots=True)
    class Config:
        # Per-item "how" (requires_copy / layout / resharder / required), bound
        # once. Un-named payload keys fall back to ``default``; ``default=None``
        # makes the schema strict (an un-named key raises).
        items: Mapping[ItemKey, ItemSpec] = field(default_factory=dict)
        default: ItemSpec | None = field(default_factory=ItemSpec)
        save: CheckpointSaverConfig = field(default_factory=AsyncCheckpointSaverConfig)
        load: CheckpointLoaderConfig = field(default_factory=CheckpointLoaderConfig)
        storage_config: StorageConfig | None = None
        subprocess_init_fn: Callable[..., None] | None = None
        subprocess_init_args: tuple[Any, ...] = ()
        pre_finalize_callback: Callable[[str, EventLogger], None] | None = None
        finalize_callback: Callable[[str, EventLogger], None] | None = None

        def build(self) -> "CheckpointManager":
            return CheckpointManager(config=self)

        @classmethod
        def with_sync_save(cls) -> "CheckpointManager.Config":
            return cls(save=SyncCheckpointSaverConfig())

        @classmethod
        def with_async_save(cls) -> "CheckpointManager.Config":
            return cls(save=AsyncCheckpointSaverConfig())

    def __init__(self, config: "CheckpointManager.Config" | None = None) -> None:
        self._config = config or CheckpointManager.Config()
        self._schema = _CheckpointSchema(
            items=self._config.items, default=self._config.default
        )
        self._prewarm_staging_future: Future[Any] | None = None
        self._staging_future: Future[Any] | None = None
        self._write_future: Future[Any] | None = None
        self._closed = False

        self._storage_config = (
            self._config.storage_config or LocalFileSystemStorageConfig()
        )
        rank_info = _get_default_rank_info()
        self._rank = rank_info.global_rank

        # Auto-wire a metadata manager when any item declares a resharder, so a
        # resharding load cannot silently fall back to the layout-unchanged fast
        # path. One instance is shared between the saver and the loader: metadata
        # computed and serialized on a load is cached and reused by a later save
        # (same layout), rather than being recomputed.
        if self._schema.has_resharder():
            self._metadata_manager: MetadataManager | None = DefaultMetadataManager(
                rank_info=rank_info
            )
        else:
            self._metadata_manager = None

        saver_config = self._config.save
        if isinstance(saver_config, SyncCheckpointSaverConfig):
            self._saver: CheckpointSaver = make_sync_checkpoint_saver(
                config=saver_config,
                rank_info=rank_info,
                pre_finalize_callback=self._config.pre_finalize_callback,
                finalize_callback=self._config.finalize_callback,
                storage_config=self._storage_config,
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
                storage_config=self._storage_config,
                checkpoint_metadata_manager=self._metadata_manager,
            )
        else:
            raise TypeError(
                f"Unsupported saver_config type: {saver_config.__class__.__name__}"
            )

        self._reader = CheckpointReader(
            rank_info=rank_info,
            storage_config=self._storage_config,
            disable_use_mmap_backed_storage_on_load=not self._config.load.use_mmap,
        )
        self._loader = CheckpointLoader(
            reader=self._reader,
            metadata_manager=self._metadata_manager,
        )

    def save(
        self,
        checkpoint_id: str,
        checkpoint: Mapping[ItemKey, Any],
    ) -> Future[Any] | None:
        """Save a payload mapping.

        Returns the write ``Future`` for async savers (await it to block until the
        checkpoint is durable), or ``None`` for synchronous savers.
        """
        if self._closed:
            raise RuntimeError("Cannot save with a closed CheckpointManager")
        if self._prewarm_staging_future is not None:
            self._prewarm_staging_future.result(
                timeout=self._config.save.wait_timeout_secs
            )
            self._prewarm_staging_future = None

        items = self._schema.build_items(checkpoint, rank=self._rank)
        save_result = self._saver.save(checkpoint_id, _DictCheckpoint(items))
        if save_result is None:
            self._staging_future = None
            self._write_future = None
            return None

        self._staging_future, self._write_future = save_result
        return self._write_future

    def load(
        self,
        checkpoint_id: str,
        into: Mapping[ItemKey, Any] | None = None,
        *,
        map_location: Any = None,
        strict: bool = False,
    ) -> Mapping[ItemKey, Any]:
        """Load a checkpoint and return the loaded mapping.

        ``into`` supplies the live per-item templates the reader reshards/copies
        into in place (preserving tensor/DTensor identity); the same values come
        back in the returned mapping. With ``into=None`` the schema's declared
        items are read as-is -- valid only for leaves, since resharding needs a
        live target.
        """
        if self._closed:
            raise RuntimeError("Cannot load with a closed CheckpointManager")

        if into is None:
            into = dict.fromkeys(self._schema.items)
            if not into:
                raise ValueError(
                    "load requires into= (or a non-empty Config.items) to know "
                    "which items to read"
                )

        items = self._schema.build_items(into, rank=self._rank)
        needs_target = sorted(
            key
            for key, item in items.items()
            if item.resharder is not None and item.value is None
        )
        if needs_target:
            raise ValueError(
                f"Resharding load requires a live target in into= for: {needs_target}"
            )

        adapter = _DictCheckpoint(items)
        self._loader.load(
            checkpoint_id,
            adapter,
            default_map_location=map_location,
            strict=strict,
        )
        return adapter.result

    def prewarm_staging(self, checkpoint: Mapping[ItemKey, Any]) -> None:
        """Pre-allocate the staging pool for the ``requires_copy`` items of a payload."""
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

        # Prewarm only the items real saves will pin (requires_copy). The stager's
        # stage() takes a plain state_dict of values, so build one from the
        # reconciled requires_copy items.
        items = self._schema.build_items(checkpoint, rank=self._rank)
        state_dict_to_stage: STATE_DICT = {
            key: item.value for key, item in items.items() if item.requires_copy
        }
        if not state_dict_to_stage:
            return

        self._prewarm_staging_future = ensure_future(stager.stage(state_dict_to_stage))

    def lock(self) -> ContextManager:
        """Guard an in-place update of checkpointed state against a concurrent save.

        Wrap an in-place update of checkpointed state -- typically
        ``optimizer.step()`` -- as ``with manager.lock(): optimizer.step()`` so the
        update cannot mutate params while an async save is still capturing them.
        Staging runs asynchronously so the rest of the step overlaps it; the guarded
        block waits only while a capture is actually in progress. A no-op for
        synchronous savers, which have no async staging.

        Returns:
            ContextManager: a context manager to wrap the in-place update.
        """
        staging_lock = getattr(self._saver, "staging_lock", None)
        if staging_lock is None:
            return nullcontext()
        return staging_lock.read

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
