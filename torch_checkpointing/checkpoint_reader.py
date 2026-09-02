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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from .checkpoint_base import (
    CheckpointInfo,
    CheckpointItem,
    CheckpointReadInfo,
)
from .checkpoint_layout import (
    default_layout_info,
    JsonSerialization,
    LayoutInfo,
    RawSerialization,
    SafetensorsSerialization,
    TorchSerialization,
)
from .distributed_metadata import (
    CheckpointMetadata,
    DistributedItemMetadata,
    DistributedMetadata,
    load_distributed_metadata,
    ShardingMetadata,
)
from .logging_utils import EventLogger, EventType
from .storage.base_storage import Storage, StorageConfig
from .storage.torch_serialization import MmapFill
from .types import CheckpointPath, NestedPath, RankInfo, STATE_DICT
from .utils import from_dict
from .walk_utils import walk_checkpoint_structure

logger = logging.getLogger(__name__)


def _build_src_to_layout_info_mappings(
    distributed_metadata: "DistributedMetadata",
) -> dict[int, dict[str, LayoutInfo | None]]:
    """Build a mapping from source ranks to their per-item layout info.

    Pivots the per-item rank_to_layout_info into a per-rank item_to_layout_info
    structure needed by the checkpoint reader for file path resolution.
    """
    result: dict[int, dict[str, LayoutInfo | None]] = {}
    for item_key, item_metadata in distributed_metadata.metadata.items():
        for rank, layout_info in item_metadata.rank_to_layout_info.items():
            if rank not in result:
                result[rank] = {}
            result[rank][item_key] = layout_info
    return result


class CheckpointReader:
    """
    Handles reading state dictionaries from storage.

    This class is responsible for reading model state dictionaries from storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    def __init__(
        self,
        rank_info: RankInfo,
        storage_config: StorageConfig,
        disable_use_mmap_backed_storage_on_load: bool = False,
        mmap_fill_factory: Callable[[Storage], MmapFill | None] | None = None,
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
        """

        self._rank_info = rank_info
        self._storage: Storage = storage_config.create_storage()
        self._disable_use_mmap_backed_storage_on_load = (
            disable_use_mmap_backed_storage_on_load
        )
        self._mmap_fill_factory = mmap_fill_factory

    def read(
        self,
        path: str,
        checkpoint_info: CheckpointReadInfo,
        map_location: Any = None,
    ) -> tuple[STATE_DICT, list[str]]:
        """
        Reads a state dictionary from storage.

        Only keys defined in checkpoint_info will be loaded. Each file is loaded in full.

        File names are discovered by looking at the layout_info_mappings in checkpoint_info.

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
            path (str): The path from which to read the checkpoint.
            checkpoint_info (CheckpointReadInfo): Encapsulates state_dict, layout_info_mappings,
                and optional checkpoint_metadata for resharding.
                Each item in checkpoint_info.checkpoint_items may have a resharder for resharding.
                checkpoint_metadata is used for resharding.
            map_location (Any): Device mapping function or device name for relocating tensors.

        Returns:
            STATE_DICT: The loaded state dictionary.
            list[str]: List of missing keys.
        """
        event_logger = EventLogger()
        logger.debug(
            f"Reading checkpoint from {path} for rank {self._rank_info.global_rank}"
        )

        if not self._storage.exists(Path(path)):
            raise FileNotFoundError(f"Checkpoint path {path} does not exist.")

        # Check if any items have resharders configured
        has_any_resharder = any(
            item.resharder is not None
            for item in checkpoint_info.checkpoint_items.values()
        )

        # Check if all resharders have skip_resharding=True
        all_resharders_skip = all(
            item.resharder.skip_resharding
            for item in checkpoint_info.checkpoint_items.values()
            if item.resharder is not None
        )

        # Fast path: if no items have resharders OR all resharders have skip_resharding=True,
        # skip metadata loading entirely and use direct file reads
        if not has_any_resharder or all_resharders_skip:
            if all_resharders_skip:
                logger.info(
                    "Resharder skip_resharding=True: skipping metadata loading and using direct file reads"
                )
            else:
                logger.info(
                    "No resharders configured: skipping metadata loading and using direct file reads"
                )
            # _read_without_resharding loads full files and filters to requested keys
            result, missing_paths = self._read_without_resharding(
                path,
                checkpoint_info,
                map_location=map_location,
            )
            missing_keys = [str(checkpoint_path) for checkpoint_path in missing_paths]
            logger.info(
                f"Successfully read checkpoint file from {path}",
                extra=event_logger(EventType.LOG_METRIC, end_to_end=True),
            )
            return result, missing_keys

        # Normal path with resharding support
        checkpoint_metadata = checkpoint_info.checkpoint_metadata
        source_distributed_metadata = self._load_metadata(path)
        logger.info(
            "Finished reading checkpoint metadata",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.execute.filesystem.metadata.read.latency_ms",
            ),
        )

        # If resharding is needed, use metadata to determine source ranks for loading.
        # This mapping uses source ranks as keys because the world size or device mesh
        # may differ between when the checkpoint was saved and when it is loaded.
        src_to_layout_info_mappings = (
            _build_src_to_layout_info_mappings(source_distributed_metadata)
            if source_distributed_metadata
            else None
        )

        # Split items into two lists based on whether they need resharding
        items_needing_reshard: dict[str, CheckpointItem] = {}
        items_not_needing_reshard: dict[str, CheckpointItem] = {}

        for key, item in checkpoint_info.checkpoint_items.items():
            # Get item-level metadata for should_reshard check
            source_item_metadata: DistributedItemMetadata | None = (
                source_distributed_metadata.metadata.get(key)
                if source_distributed_metadata
                else None
            )

            # Extract target metadata for this item from local_metadata
            target_metadata: dict[NestedPath, ShardingMetadata] | None = None
            if checkpoint_metadata and checkpoint_metadata.local_metadata:
                # Get metadata directly by item_key
                target_metadata = checkpoint_metadata.local_metadata.get(key)

            if item.resharder is not None and item.resharder.should_reshard(
                source_item_metadata, target_metadata
            ):
                items_needing_reshard[key] = item
            else:
                items_not_needing_reshard[key] = item

        result_dict: dict[str, Any] = {}
        missing_paths: list[CheckpointPath] = []

        # Load items that don't need resharding
        if items_not_needing_reshard:
            checkpoint_info_no_reshard = CheckpointInfo(
                checkpoint_items=items_not_needing_reshard
            )
            non_reshard_result, non_reshard_missing = self._read_without_resharding(
                path,
                checkpoint_info_no_reshard,
                map_location=map_location,
            )
            result_dict.update(non_reshard_result)
            missing_paths.extend(non_reshard_missing)

        # Load items that need resharding
        if items_needing_reshard:
            assert checkpoint_metadata is not None
            assert source_distributed_metadata is not None
            assert src_to_layout_info_mappings is not None
            checkpoint_info_reshard = CheckpointInfo(
                checkpoint_items=items_needing_reshard
            )
            reshard_result, reshard_missing = self._read_with_resharding(
                path,
                checkpoint_info_reshard,
                checkpoint_metadata,
                src_to_layout_info_mappings,
                source_distributed_metadata,
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

    def _load_metadata(
        self,
        checkpoint_dir: str | Path,
    ) -> DistributedMetadata | None:
        """
        Load distributed metadata from the checkpoint directory.

        Args:
            checkpoint_dir: Path to the checkpoint directory.

        Returns:
            DistributedMetadata if the checkpoint carries one, None otherwise.
        """
        return load_distributed_metadata(checkpoint_dir, self._storage)

    def _read_without_resharding(
        self,
        path: str,
        checkpoint_info: CheckpointInfo,
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
            if key not in checkpoint_info.layout_info_mappings:
                logger.warning(
                    f"Item {key=} not found in layout_info_mappings. Skipping."
                )

                # Add all leaf keys to missing_keys
                def collect_all_paths(
                    checkpoint_path: CheckpointPath, src: Any, tgt: Any
                ) -> Any:
                    missing_paths.append(checkpoint_path)
                    return src

                walk_checkpoint_structure(
                    item_key=key,
                    source=checkpoint_info.checkpoint_items[key].value,
                    target=None,
                    leaf_fn=collect_all_paths,
                )
                continue

            layout_info = checkpoint_info.layout_info_mappings[key]
            if layout_info is None:
                layout_info = default_layout_info(key, self._rank_info.global_rank)

            file_path = Path(path) / layout_info.file_path
            if not self._storage.exists(file_path):
                raise RuntimeError(f"Missing file {file_path} for key {key}.")

            loaded_data = self._load_full_file(
                file_path, layout_info, map_location=map_location
            )
            # Filter loaded data to only include keys present in the requested structure
            # Also track any missing keys within the nested structure
            requested_value = checkpoint_info.checkpoint_items[key].value
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
            f"Successfully read checkpoint file from {path} without resharding",
            extra=event_logger(EventType.LOG_METRIC, end_to_end=True),
        )
        return result_dict, missing_paths

    def _read_with_resharding(
        self,
        path: str,
        checkpoint_info: CheckpointInfo,
        checkpoint_metadata: CheckpointMetadata,
        src_to_layout_info_mappings: dict[int, dict[str, LayoutInfo | None]],
        distributed_metadata: DistributedMetadata,
        *,
        map_location: Any = None,
    ) -> tuple[STATE_DICT, list[CheckpointPath]]:
        """
        Load checkpoint with resharding support.

        This method handles loading checkpoint data when the distributed configuration
        differs between save time and load time. It uses resharders to generate load
        plans that determine which source ranks to read from and how to redistribute
        the data to match the current distributed layout.

        All items in checkpoint_info are expected to need resharding with valid resharders.

        Args:
            path: Path to the checkpoint directory.
            checkpoint_info: Encapsulates state_dict and layout_info_mappings.
                All items must have non-None resharders.
            checkpoint_metadata: Metadata for the target checkpoint, containing
                local metadata for generating load plans.
            src_to_layout_info_mappings: Mapping from source ranks to their layout info,
                used to locate checkpoint files from different source ranks.
            distributed_metadata: Source distributed metadata from the saved checkpoint,
                used for resharding decisions.
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

        for key, item in checkpoint_info.checkpoint_items.items():
            logger.info(
                f"Loading {key} checkpoint with resharder.",
                extra=event_logger(EventType.LOG_METRIC),
            )
            resharder = item.resharder
            assert resharder is not None  # API contract guarantees this

            # Get target metadata for this item (direct access by item_key)
            target_metadata: dict[NestedPath, ShardingMetadata] | None = (
                checkpoint_metadata.local_metadata.get(key)
            )

            # Get item-level source metadata (direct access by item_key)
            source_item_metadata = distributed_metadata.metadata.get(key)

            if not target_metadata or source_item_metadata is None:
                logger.warning(f"Missing metadata for item {key}, skipping resharding")
                continue

            # Load and reshard checkpoint data using per-path API
            unhandled_nested_paths = resharder.load(
                source_path=Path(path),
                item_key=key,
                target_metadata=target_metadata,
                source_metadata=source_item_metadata,
                target=result_dict[key],
                storage=self._storage,
            )

            # Convert NestedPath to CheckpointPath for reporting
            for nested_path in unhandled_nested_paths:
                unhandled_paths.append(CheckpointPath(key, nested_path))

            logger.info(
                f"Done Loading {key} checkpoint with resharder.",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"train.checkpoint_read.execute.filesystem.{key}.read.latency_ms",
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
