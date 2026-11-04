# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint reader functionality for machine learning models.

This module provides classes for reading checkpoints from storage, including
determining checkpoint layout and configuring the reader.
"""

import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import torch

from .checkpoint_base import (
    CheckpointInfo,
    CheckpointItem,
    CheckpointReadInfo,
)
from .checkpoint_layout import (
    default_torch_layout_info,
    JsonSerialization,
    LayoutInfo,
    RawSerialization,
    SafetensorsSerialization,
    TorchSerialization,
)
from .default_load_plan_executor import DefaultLoadPlanExecutor
from .distributed_metadata import DistributedItemMetadata, LocalMetadata
from .load_plan_executor import LoadExecutionContext, LoadPlanExecutor
from .logging_utils import EventLogger, EventType
from .metadata_serialization import (
    DistributedMetadataFormat,
    load_distributed_metadata,
    RankAddressableDistributedMetadataFormat,
    TorchDistributedMetadataFormat,
)
from .resharding import ReshardingInfo
from .storage.base_storage import Storage, StorageConfig
from .storage.torch_serialization import MmapFill
from .types import CheckpointPath, ItemKey, RankInfo, STATE_DICT
from .utils import from_dict
from .walk_utils import walk_checkpoint_structure

logger = logging.getLogger(__name__)


class CheckpointReader:
    """
    Handles reading state dictionaries from storage.

    This class is responsible for reading model state dictionaries from storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    # Formats tried, in order, to load a checkpoint's distributed metadata when
    # an item may need resharding and the caller names no metadata_format.
    _METADATA_FORMATS: ClassVar[tuple[type[DistributedMetadataFormat], ...]] = (
        TorchDistributedMetadataFormat,
    )

    # Readers whose checkpoints always carry distributed metadata reject a
    # directory with none. Without this, a missing or renamed metadata artifact
    # leaves every item without source metadata, so nothing reshards and each
    # rank silently reads its own conventional file instead.
    _REQUIRE_METADATA: ClassVar[bool] = False

    def __init__(
        self,
        rank_info: RankInfo,
        storage_config: StorageConfig,
        disable_use_mmap_backed_storage_on_load: bool = False,
        mmap_fill_factory: Callable[[Storage], MmapFill | None] | None = None,
        storage: Storage | None = None,
        load_plan_executor: LoadPlanExecutor | None = None,
    ):
        """
        Initialize a CheckpointReader.

        Args:
            rank_info: Information about the current global/local rank.
            storage_config: Configuration for the storage backend.
            disable_use_mmap_backed_storage_on_load: If True, fall back to the
                BytesIO-based torch.load path for torch-serialized files. The
                default (False) routes loads through a single mmap-backed
                overall storage to reduce allocator fragmentation after load
                cleanup.
            mmap_fill_factory: Optional factory for a storage-specific mmap
                fill callback. It is resolved only for mmap-backed torch loads.
            storage: Optional operation-scoped storage instance. When supplied,
                it is reused instead of constructing another backend client.
            load_plan_executor: Reads the data that resharders plan, for every item
                that reshards. Defaults to a ``DefaultLoadPlanExecutor``.
        """

        self._rank_info = rank_info
        self._storage = (
            storage if storage is not None else storage_config.create_storage()
        )
        self._disable_use_mmap_backed_storage_on_load = (
            disable_use_mmap_backed_storage_on_load
        )
        self._mmap_fill_factory = mmap_fill_factory
        self._load_plan_executor = (
            load_plan_executor
            if load_plan_executor is not None
            else DefaultLoadPlanExecutor()
        )

    def read(
        self,
        path: str | os.PathLike[str],
        checkpoint_info: CheckpointReadInfo,
        map_location: Any = None,
        *,
        metadata_format: type[DistributedMetadataFormat] | None = None,
    ) -> tuple[STATE_DICT, list[str]]:
        """
        Reads a state dictionary from storage.

        Only keys defined in checkpoint_info will be loaded. Each file is loaded in full.

        An item read without resharding uses its configured layout, else the
        layout recorded in source metadata, else the default
        ``<item_key>_<rank>.pt`` layout.
        Source metadata is loaded only when an item may need resharding.

        In-place modification behavior:
            When checkpoint_info contains values (not None), loaded data is merged with
            those values. The following are modified IN-PLACE:
            - Mutable containers (dict, list, deque): updated in the existing objects
            - Tensors: data is copied via copy_() into the target tensors, preserving
              the target tensor's identity (same object, updated data)

            The following are NOT modified in-place:
            - Immutable containers (tuple): new containers are created
            - Non-tensor leaf values: source value replaces target value

            When checkpoint_info values are None, new objects are created from the
            loaded checkpoint data.

        Args:
            path: The checkpoint directory to read.
            checkpoint_info (CheckpointReadInfo): Encapsulates state_dict, layout_info_mappings,
                and optional target_local_metadata for resharding.
                Each item in checkpoint_info.checkpoint_items may have a resharder for resharding.
                target_local_metadata is this rank's sharding metadata for those items.
            map_location (Any): Device mapping function or device name for relocating tensors.
            metadata_format: The only metadata format to load, which must be
                present whenever the read needs metadata. A format that is not
                rank addressable cannot be read one file per rank, so its items
                require a resharder. When unset, this reader tries its own
                formats.

        Returns:
            STATE_DICT: The loaded state dictionary.
            list[str]: List of missing keys.
        """
        event_logger = EventLogger()
        logger.debug(
            f"Reading checkpoint from {path} for rank {self._rank_info.global_rank}"
        )

        checkpoint_dir = Path(path)
        if not self._storage.exists(checkpoint_dir):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")

        no_item_reshards = all(
            item.resharder is None or item.resharder.skip_resharding
            for item in checkpoint_info.checkpoint_items.values()
        )

        rank_addressable = metadata_format is None or issubclass(
            metadata_format, RankAddressableDistributedMetadataFormat
        )

        # Fast path: if no items have resharders OR all resharders have
        # skip_resharding=True, use direct full-file reads without loading
        # metadata. Only a rank-addressable format guarantees that each rank's
        # file holds whole items; any other format takes the normal path, which
        # rejects direct reads of the items it describes.
        if no_item_reshards and rank_addressable:
            source_layouts = self._resolve_layouts_without_resharding(
                checkpoint_info,
                {},
                requested_rank=self._rank_info.global_rank,
            )
            logger.info("No resharding required: using direct file reads")
            # _read_without_resharding loads full files and filters to requested keys
            result, missing_paths = self._read_without_resharding(
                str(checkpoint_dir),
                checkpoint_info,
                source_layouts,
                map_location=map_location,
            )
            missing_keys = [str(checkpoint_path) for checkpoint_path in missing_paths]
            logger.info(
                f"Successfully read checkpoint file from {path}",
                extra=event_logger(EventType.LOG_METRIC, end_to_end=True),
            )
            return result, missing_keys

        # Normal path with resharding support
        source_metadata = self._load_source_metadata(checkpoint_dir, metadata_format)
        target_local_metadata = checkpoint_info.target_local_metadata or {}
        # Split items into two lists based on whether they need resharding
        items_needing_reshard: dict[str, CheckpointItem] = {}
        items_not_needing_reshard: dict[str, CheckpointItem] = {}
        for key, item in checkpoint_info.checkpoint_items.items():
            source_item_metadata = source_metadata.get(key)
            target_item_metadata = target_local_metadata.get(key)

            # A skipping resharder has promised its layout is unchanged, so it is
            # read directly even when another item forces this path.
            if (
                item.resharder is not None
                and not item.resharder.skip_resharding
                and item.resharder.should_reshard(
                    source_item_metadata, target_item_metadata
                )
            ):
                assert source_item_metadata is not None, (
                    f"Missing source metadata for checkpoint item {key!r}"
                )
                items_needing_reshard[key] = item
            else:
                items_not_needing_reshard[key] = item

        if not rank_addressable:
            assert metadata_format is not None
            unaddressable_keys = sorted(
                items_not_needing_reshard.keys() & source_metadata.keys()
            )
            if unaddressable_keys:
                raise ValueError(
                    f"{metadata_format.__name__} is not rank addressable, so "
                    f"reading one file per rank would load only part of "
                    f"{unaddressable_keys}. Each of these items requires a resharder."
                )

        result_dict: dict[str, Any] = {}
        missing_paths: list[CheckpointPath] = []

        # Load items that don't need resharding
        if items_not_needing_reshard:
            checkpoint_info_no_reshard = CheckpointInfo(
                checkpoint_items=items_not_needing_reshard
            )
            source_layouts = self._resolve_layouts_without_resharding(
                checkpoint_info_no_reshard,
                source_metadata,
                requested_rank=self._rank_info.global_rank,
            )
            non_reshard_result, non_reshard_missing = self._read_without_resharding(
                str(checkpoint_dir),
                checkpoint_info_no_reshard,
                source_layouts,
                map_location=map_location,
            )
            result_dict.update(non_reshard_result)
            missing_paths.extend(non_reshard_missing)

        # Load items that need resharding
        if items_needing_reshard:
            checkpoint_info_reshard = CheckpointInfo(
                checkpoint_items=items_needing_reshard
            )
            reshard_result, reshard_missing = self._read_with_resharding(
                str(checkpoint_dir),
                checkpoint_info_reshard,
                target_local_metadata,
                source_metadata,
                map_location=map_location,
            )
            result_dict.update(reshard_result)
            missing_paths.extend(reshard_missing)

        missing_keys = [str(checkpoint_path) for checkpoint_path in missing_paths]
        if missing_keys:
            if len(missing_keys) > 10:
                logger.warning(
                    f"Missing {len(missing_keys)} keys from checkpoint: {missing_keys[:10]}... (and {len(missing_keys) - 10} more)"
                )
            else:
                logger.warning(
                    f"Missing {len(missing_keys)} keys from checkpoint: {missing_keys}"
                )
        logger.info(
            f"Successfully read checkpoint file from {path}",
            extra=event_logger(EventType.LOG_METRIC),
        )
        return result_dict, missing_keys

    def _load_source_metadata(
        self,
        checkpoint_dir: Path,
        metadata_format: type[DistributedMetadataFormat] | None,
    ) -> Mapping[str, DistributedItemMetadata]:
        event_logger = EventLogger()
        formats = (
            (metadata_format,)
            if metadata_format is not None
            else self._METADATA_FORMATS
        )
        source_distributed_metadata = load_distributed_metadata(
            checkpoint_dir,
            self._storage,
            formats=formats,
        )
        if source_distributed_metadata is None and (
            metadata_format is not None or self._REQUIRE_METADATA
        ):
            raise FileNotFoundError(
                f"No distributed metadata found in {checkpoint_dir}; tried "
                f"{[fmt.__name__ for fmt in formats]}"
            )
        logger.info(
            "Finished reading checkpoint metadata",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.execute.filesystem.metadata.read.latency_ms",
            ),
        )
        return (
            source_distributed_metadata.metadata
            if source_distributed_metadata is not None
            else {}
        )

    def _resolve_layouts_without_resharding(
        self,
        checkpoint_info: CheckpointInfo,
        source_metadata: Mapping[str, DistributedItemMetadata],
        *,
        requested_rank: int,
    ) -> dict[str, LayoutInfo]:
        """Resolve the checkpoint-relative layout of each item read without resharding.

        Sources for each item, highest priority first:

        1. The layout recorded for this rank in source metadata, which is
           loaded only when another item in the same read reshards.
        2. The layout configured on the item.
        3. The default ``<item_key>_<rank>.pt`` layout.

        Nothing is probed: if a resolved file is absent, the read fails naming
        the item instead of falling back to another file.
        """
        layouts: dict[str, LayoutInfo] = {}
        for item_key, item in checkpoint_info.checkpoint_items.items():
            source_item_metadata = source_metadata.get(item_key)
            # Not get_layout_info: its default would shadow a configured layout.
            recorded = (
                source_item_metadata.rank_to_layout_info.get(requested_rank)
                if source_item_metadata is not None
                else None
            )
            if recorded is not None:
                layouts[item_key] = recorded
            elif item.layout is not None:
                layouts[item_key] = item.layout
            else:
                layouts[item_key] = default_torch_layout_info(item_key, requested_rank)
        return layouts

    def _read_without_resharding(
        self,
        path: str,
        checkpoint_info: CheckpointInfo,
        source_layouts_by_item_key: Mapping[str, LayoutInfo],
        *,
        map_location: Any = None,
    ) -> tuple[STATE_DICT, list[CheckpointPath]]:
        """
        Load checkpoint without resharding support.

        This method loads checkpoint data using the standard layout without any
        resharding logic. It reads from the local rank's checkpoint files only.
        Each file is loaded in full.

        Args:
            path: Path to the checkpoint directory.
            checkpoint_info: Encapsulates state_dict and layout_info_mappings.
            source_layouts_by_item_key: Checkpoint-relative source layout for
                each item.
            map_location: Device mapping for tensor relocation.

        Returns:
            Tuple of (loaded_state_dict, missing_paths).
        """
        event_logger = EventLogger()
        logger.debug(
            f"Reading checkpoint from {path} for rank {self._rank_info.global_rank}"
        )

        result_dict: dict[str, Any] = {}
        missing_paths: list[CheckpointPath] = []

        for key in checkpoint_info.keys:
            layout_info = source_layouts_by_item_key[key]
            file_path = Path(path) / layout_info.file_path
            if not self._storage.exists(file_path):
                raise FileNotFoundError(
                    f"Missing file {file_path} for checkpoint item {key!r}."
                )
            requested_value = checkpoint_info.checkpoint_items[key].value

            loaded_data = self._load_full_file(
                file_path, layout_info, map_location=map_location
            )
            # Filter loaded data to only include keys present in the requested structure
            # Also track any missing keys within the nested structure
            # safetensors only supports flat dict[str, Tensor], so the writer flattens
            # nested inputs with '.' separators. Re-nest here to match the target's shape
            # — otherwise walk_checkpoint_structure (which descends source + target in
            # parallel) would misalign and silently drop everything.
            if (
                isinstance(layout_info.serialization_format, SafetensorsSerialization)
                and requested_value is not None
                and isinstance(loaded_data, dict)
            ):
                loaded_data = SafetensorsSerialization.unflatten_to_target(
                    loaded_data, requested_value
                )
            result_dict[key], item_missing_paths = walk_checkpoint_structure(
                item_key=key,
                source=loaded_data,
                target=requested_value,
            )
            missing_paths.extend(item_missing_paths)
            logger.info(
                f"Done Loading {key} checkpoint from {file_path} without resharder.",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"train.checkpoint_read.execute.filesystem.{key}.read.latency_ms",
                ),
            )
        logger.info(
            "Successfully read checkpoint without resharding",
            extra=event_logger(EventType.LOG_METRIC, end_to_end=True),
        )
        return result_dict, missing_paths

    def _read_with_resharding(
        self,
        path: str,
        checkpoint_info: CheckpointInfo,
        target_local_metadata: LocalMetadata,
        source_metadata: Mapping[str, DistributedItemMetadata],
        *,
        map_location: Any = None,
    ) -> tuple[STATE_DICT, list[CheckpointPath]]:
        """
        Load checkpoint with resharding support.

        This method handles loading checkpoint data when the distributed configuration
        differs between save time and load time. Each item's resharder plans which
        source ranks to read from and where their data goes in the current layout.
        Once every item is planned, the load executor reads all of them in one call.

        All items in checkpoint_info are expected to need resharding with valid resharders.

        Args:
            path: Path to the checkpoint directory.
            checkpoint_info: Encapsulates state_dict and layout_info_mappings.
                All items must have non-None resharders.
            target_local_metadata: This rank's sharding metadata per item, used
                to generate load plans.
            source_metadata: Source distributed metadata and file layouts.
            map_location: Device mapping for tensor relocation.

        Returns:
            Tuple of (loaded_state_dict, unhandled_paths) where unhandled_paths contains
            CheckpointPaths for keys that could not be resharded.
        """
        event_logger = EventLogger()
        logger.debug(
            f"Reading checkpoint from {path} for rank {self._rank_info.global_rank}"
        )

        result_dict: dict[str, Any] = checkpoint_info.state_dict  # type: ignore
        unhandled_paths: list[CheckpointPath] = []
        planned_items: dict[ItemKey, ReshardingInfo] = {}

        for key, item in checkpoint_info.checkpoint_items.items():
            resharder = item.resharder
            assert resharder is not None  # API contract guarantees this

            target_item_metadata = target_local_metadata.get(key)
            source_item_metadata = source_metadata.get(key)

            if not target_item_metadata or source_item_metadata is None:
                logger.warning(f"Missing metadata for item {key}, skipping resharding")
                continue

            resharding_info = resharder.generate_load_plans(
                item_key=key,
                target=result_dict[key],
                target_metadata=target_item_metadata,
                source_metadata=source_item_metadata,
            )
            planned_items[key] = resharding_info
            unhandled_paths.extend(
                CheckpointPath(key, nested_path)
                for nested_path in resharding_info.non_reshardable_paths
            )

        logger.info(
            f"Planned resharded loads of {list(planned_items)}.",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.execute.plan.latency_ms",
            ),
        )
        self._load_plan_executor.execute(
            planned_items,
            LoadExecutionContext(
                checkpoint_path=Path(path),
                storage=self._storage,
                source_metadata={key: source_metadata[key] for key in planned_items},
            ),
        )
        logger.info(
            f"Read resharded loads of {list(planned_items)}.",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.execute.load_plan_executor.latency_ms",
            ),
        )
        return result_dict, unhandled_paths

    def _load_full_file(
        self,
        file_path: Path,
        layout_info: Any,
        *,
        map_location: Any = None,
    ) -> Any:
        """
        Load an entire file based on its serialization format.

        Supports TorchSerialization, JsonSerialization, and RawSerialization formats.

        Args:
            file_path: Path to the file to load.
            layout_info: LayoutInfo containing serialization format information.
            map_location: Device mapping for tensor relocation (torch files only).

        Returns:
            The deserialized content of the file.

        Raises:
            ValueError: If the serialization format is not supported.
        """
        if isinstance(layout_info.serialization_format, TorchSerialization):
            if not self._disable_use_mmap_backed_storage_on_load:
                from .storage.torch_serialization import (
                    load_torch_serialized_from_storage,
                )

                return load_torch_serialized_from_storage(
                    file_path,
                    self._storage,
                    map_location=map_location,
                    mmap_fill=(
                        self._mmap_fill_factory(self._storage)
                        if self._mmap_fill_factory is not None
                        else None
                    ),
                )

            with self._storage.stream_read(file_path) as f:
                state_dict = torch.load(
                    f,  # type: ignore[arg-type]
                    map_location=map_location,
                    weights_only=False,
                )
            return state_dict
        elif isinstance(layout_info.serialization_format, JsonSerialization):
            data = self._storage.read(file_path)
            json_data = json.loads(data.decode("utf-8"))
            if layout_info.serialization_format.cls is None:
                return json_data
            else:
                return from_dict(layout_info.serialization_format.cls, json_data)
        elif isinstance(layout_info.serialization_format, RawSerialization):
            return self._storage.read(file_path)
        elif isinstance(layout_info.serialization_format, SafetensorsSerialization):
            from safetensors.torch import load as safetensors_load

            data = self._storage.read(file_path)
            loaded = safetensors_load(data)
            if map_location is not None:
                # safetensors has no callable/dict remapper concept like torch.load —
                # only a concrete destination device is meaningful here. Reject other
                # forms loudly so callers don't get a confusing `Tensor.to(<function ...>)`.
                if not isinstance(map_location, (str, torch.device)):
                    raise ValueError(
                        f"SafetensorsSerialization map_location must be a str or "
                        f"torch.device, got {type(map_location).__name__}. Callables "
                        f"and dict remappings (accepted by torch.load) are not supported."
                    )
                loaded = {k: v.to(map_location) for k, v in loaded.items()}
            return loaded
        else:
            raise ValueError(
                f"Unsupported serialization format: {layout_info.serialization_format}"
            )
