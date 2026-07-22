# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module provides the interface for composable metadata management systems
that can extract, process, and coordinate metadata from state dictionaries
for distributed tensor checkpointing.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .checkpoint_base import CheckpointInfo
from .checkpoint_layout import LayoutInfo
from .distributed_metadata import (
    CheckpointMetadata,
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
    ShardingMetadata,
)
from .logging_utils import EventLogger, EventType
from .types import NestedPath, RankInfo

logger = logging.getLogger(__name__)


class MetadataManager(ABC):
    """
    Abstract base class for metadata managers in distributed checkpointing.

    A metadata manager is responsible for:
    1. Extracting metadata from state dictionaries
    2. Providing utilities for metadata aggregation, validation and consumption.

    This ABC enables composable metadata management that can be plugged
    into different checkpointer implementations.
    """

    @abstractmethod
    def compute_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> CheckpointMetadata | None:
        """
        Compute complete distributed metadata from checkpoint info.

        This is the main method that orchestrates metadata extraction,
        distributed coordination, and metadata construction.

        Args:
            checkpoint_info: CheckpointInfo containing state_dict and layout_info_mappings

        Returns:
            Complete DistributedMetadata structure if distributed metadata isn't computed yet.
            None if distributed metadata is already computed and local metadata cached.
        """

    @abstractmethod
    def extract_object_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> dict[str, dict[NestedPath, ShardingMetadata]]:
        """
        Extract per-path sharding metadata from checkpoint info, grouped by item_key.

        This method extracts metadata for all CheckpointItems with resharders,
        delegating to each resharder's extract_sharding_metadata method.

        Args:
            checkpoint_info: CheckpointInfo containing checkpoint items

        Returns:
            Dict mapping item_key to dict of NestedPath -> ShardingMetadata
            for each sharded object.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources used by the metadata manager."""


class DefaultMetadataManager(MetadataManager):
    """
    Provides a basic implementation suitable for most use cases that aggregates
    metadata from all ranks and validates consistency with distributed coordination using standard PyTorch distributed primitives.
    """

    def __init__(
        self,
        rank_info: RankInfo,
        process_group: ProcessGroup | None = None,
        should_cache_metadata: bool = True,
        enable_serialization: bool = True,
    ):
        """
        Initialize the default metadata manager.

        Args:
            rank_info: Information about current rank and world size
            process_group: The process group to use for distributed coordination. If None, uses the default process group.
            should_cache_metadata: Whether to cache computed metadata for reuse across checkpoints.
                Note: Caching metadata assumes that the state_dict does not change between
                intermediate checkpoints. If the state_dict is modified, cached metadata may become invalid
            enable_serialization: Whether to kick off async serialization of metadata after
                first compute. Set to False for load-only scenarios where no save will happen,
                avoiding unnecessary serialization overhead.
        """
        self._rank_info = rank_info
        self._process_group = process_group
        self._should_cache_metadata = should_cache_metadata
        self._enable_serialization = enable_serialization
        self._cached_local_metadata: (
            dict[str, dict[NestedPath, ShardingMetadata]] | None
        ) = None

        # Async metadata serialization state (shared between loader and saver)
        self._metadata_executor = ThreadPoolExecutor(max_workers=1)
        self._serialization_future: Future[bytes] | None = None
        self._cached_serialized_metadata: bytes | None = None

    def _async_serialize_distributed_metadata(
        self,
        distributed_metadata: DistributedMetadata,
    ) -> None:
        """
        Trigger async serialization of distributed metadata.

        Only serializes once. Starts in background thread to overlap with other work.
        This is called automatically after the first compute_metadata() call.
        """
        if (
            not self._enable_serialization
            or self._serialization_future is not None
            or self._cached_serialized_metadata is not None
        ):
            return  # Serialization disabled, already serializing, or already serialized

        def serialize() -> bytes:
            event_logger = EventLogger()
            serialized_dist_metadata = pickle.dumps(distributed_metadata.to_dict())
            logger.info(
                "Serialized distributed metadata",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name="train.checkpoint_metadata.serialize.latency_ms",
                ),
            )
            return serialized_dist_metadata

        self._serialization_future = self._metadata_executor.submit(serialize)

    def get_serialized_metadata(self) -> bytes | None:
        """
        Get serialized metadata bytes, waiting for serialization to complete if needed.

        Returns:
            Serialized metadata bytes if available, None if no metadata has been computed.
        """
        # Wait for serialization to complete if in progress
        if self._serialization_future is not None:
            self._cached_serialized_metadata = self._serialization_future.result()
            self._serialization_future = None

        return self._cached_serialized_metadata

    def close(self) -> None:
        """Release resources used by the metadata manager."""
        self._metadata_executor.shutdown(wait=False, cancel_futures=False)

    def compute_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> CheckpointMetadata | None:
        """Compute complete distributed metadata from checkpoint info. Return None if cache is valid."""

        event_logger = EventLogger()

        local_metadata = self.extract_object_metadata(checkpoint_info)

        logger.info(
            f"MetadataManager: extracted metadata for {len(local_metadata)} paths",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.extract_local.latency_ms",
            ),
        )

        # Return None if cache is valid
        if self._should_cache_metadata and self._cached_local_metadata is not None:
            if local_metadata != self._cached_local_metadata:
                raise RuntimeError(
                    "State dictionary has changed since last checkpoint. Cached metadata is no longer valid.",
                )
            return None

        # Compute fresh metadata
        distributed_metadata = self._aggregate_metadata(
            local_metadata, checkpoint_info.layout_info_mappings
        )

        if self._should_cache_metadata:
            self._cached_local_metadata = local_metadata

        logger.info(
            f"MetadataManager: computed metadata for {len(distributed_metadata.metadata)} paths",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.compute_global.e2e.latency_ms",
                end_to_end=True,
            ),
        )

        result = CheckpointMetadata(
            distributed_metadata=distributed_metadata,
            local_metadata=local_metadata,
        )

        # Kick off async serialization immediately after first compute
        self._async_serialize_distributed_metadata(distributed_metadata)

        return result

    def extract_object_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> dict[str, dict[NestedPath, ShardingMetadata]]:
        """
        Extract per-path sharding metadata from checkpoint info, grouped by item_key.

        This calls each resharder's extract_sharding_metadata method
        and aggregates the results.

        Args:
            checkpoint_info: CheckpointInfo containing state_dict and layout_info_mappings

        Returns:
            Dict mapping item_key to dict of NestedPath -> ShardingMetadata
            for each sharded object.

        Note:
            Only extracts metadata for CheckpointItems that have a resharder
            configured. Items without resharders are skipped entirely.
        """
        result: dict[str, dict[NestedPath, ShardingMetadata]] = {}

        for item_key, checkpoint_item in checkpoint_info.checkpoint_items.items():
            if checkpoint_item.resharder is None:
                continue

            nested_path_to_metadata = (
                checkpoint_item.resharder.extract_sharding_metadata(
                    item_key, checkpoint_item.value
                )
            )
            if nested_path_to_metadata:
                result[item_key] = nested_path_to_metadata

        return result

    def _compact(
        self,
        item_to_metadata: dict[str, dict[NestedPath, ShardingMetadata]],
    ) -> dict[str, dict[NestedPath, ShardingMetadata]]:
        """
        Compact metadata by only keeping entries where this rank is the representative.

        For sharding metadata that supports compaction (e.g., DTensors), only the
        representative rank (min of equivalent_ranks) needs to send metadata during
        all_gather since all ranks in the mesh have identical metadata.
        This significantly reduces collective payload at large-scale.

        Metadata without equivalent_ranks (returns None) is kept unchanged.

        Args:
            item_to_metadata: Original grouped metadata dict from this rank

        Returns:
            Compacted metadata dict with non-representative entries removed
        """
        compacted: dict[str, dict[NestedPath, ShardingMetadata]] = {}
        current_rank = self._rank_info.global_rank

        for item_key, nested_path_to_metadata in item_to_metadata.items():
            item_compacted: dict[NestedPath, ShardingMetadata] = {}
            for nested_path, sharding_metadata in nested_path_to_metadata.items():
                # Skip entries where this rank is not the representative
                equiv_ranks = sharding_metadata.equivalent_ranks
                if equiv_ranks is not None:
                    representative = min(equiv_ranks)
                    if current_rank != representative:
                        continue
                item_compacted[nested_path] = sharding_metadata

            if item_compacted:
                compacted[item_key] = item_compacted

        return compacted

    def _aggregate_metadata(
        self,
        item_to_metadata: dict[str, dict[NestedPath, ShardingMetadata]],
        layout_info_mappings: dict[str, LayoutInfo | None],
    ) -> DistributedMetadata:
        """
        Aggregate metadata across ranks and return final DistributedMetadata.

        Args:
            item_to_metadata: Dict mapping item_key to dict of NestedPath -> ShardingMetadata
            layout_info_mappings: Mapping from state_dict keys to LayoutInfo

        Returns:
            Final DistributedMetadata after aggregation across all ranks
        """
        rank_info = self._rank_info
        if rank_info.global_world_size == 1 or not dist.is_initialized():
            # For single rank, create metadata directly
            metadata: dict[str, DistributedItemMetadata] = {}
            for item_key, nested_path_to_metadata in item_to_metadata.items():
                nested_to_groups: dict[NestedPath, list[GlobalObjectMetadata]] = {}
                for nested_path, sharding_metadata in nested_path_to_metadata.items():
                    nested_to_groups[nested_path] = [
                        GlobalObjectMetadata(
                            sharding_metadata=sharding_metadata,
                            ranks=(rank_info.global_rank,),
                        )
                    ]
                metadata[item_key] = DistributedItemMetadata(
                    nested_path_to_metadata=nested_to_groups,
                    rank_to_layout_info={
                        rank_info.global_rank: layout_info_mappings.get(item_key)
                    },
                )

            return DistributedMetadata(
                metadata=metadata,
                world_size=rank_info.global_world_size,
            )

        # Multi-rank aggregation
        aggregated_item_metadata = self._aggregate_metadata_across_ranks(
            item_to_metadata, self._process_group
        )

        # Aggregate layout info across ranks
        rank_to_layout_info = self._aggregate_layout_info_across_ranks(
            layout_info_mappings, self._process_group
        )

        # Build final DistributedItemMetadata for each item
        metadata: dict[str, DistributedItemMetadata] = {}
        for item_key, nested_path_to_groups in aggregated_item_metadata.items():
            # Build per-item rank_to_layout_info
            item_rank_to_layout: dict[int, LayoutInfo | None] = {
                rank: layout_mapping.get(item_key)
                for rank, layout_mapping in rank_to_layout_info.items()
            }
            metadata[item_key] = DistributedItemMetadata(
                nested_path_to_metadata=nested_path_to_groups,
                rank_to_layout_info=item_rank_to_layout,
            )

        return DistributedMetadata(
            metadata=metadata,
            world_size=rank_info.global_world_size,
        )

    def _all_gather_pickled_data(
        self,
        data: Any,
        process_group: ProcessGroup | None = None,
    ) -> dict[int, Any]:
        """
        All-gather pickled data across all ranks.

        This helper method handles the common pattern of:
        1. Serializing data with pickle
        2. Prepending size information
        3. Finding max size across ranks
        4. Padding and converting to tensor
        5. All-gathering tensors
        6. Deserializing from all ranks

        Args:
            data: Data to serialize and gather (must be picklable)
            process_group: Process group for distributed coordination

        Returns:
            Dict mapping rank to deserialized data from that rank
        """
        world_size = dist.get_world_size(process_group)

        # Step 1: Serialize local data using pickle
        data_bytes = pickle.dumps(data)
        data_size = len(data_bytes)

        # Prepend data with its size (8 bytes for int64)
        size_bytes = data_size.to_bytes(8, byteorder="little")
        prefixed_data = size_bytes + data_bytes

        # Use all_reduce with max operation to find the maximum total size across all ranks
        total_size = len(prefixed_data)
        max_size_tensor = torch.tensor([total_size], dtype=torch.int64, device="cuda")
        dist.all_reduce(max_size_tensor, op=dist.ReduceOp.MAX, group=process_group)
        max_size = int(max_size_tensor.item())

        # Pad prefixed data and convert to tensor
        padded_bytes = prefixed_data + b"\0" * (max_size - len(prefixed_data))
        data_tensor = torch.frombuffer(padded_bytes, dtype=torch.uint8).cuda()

        # All-gather data tensors
        all_data_tensors = [torch.zeros_like(data_tensor) for _ in range(world_size)]
        dist.all_gather(all_data_tensors, data_tensor, group=process_group)

        # Step 2: Deserialize data from all ranks using pickle
        all_ranks_data = {}
        for rank, tensor in enumerate(all_data_tensors):
            tensor_bytes = tensor.cpu().numpy().tobytes()

            # Extract size from first 8 bytes
            actual_size = int.from_bytes(tensor_bytes[:8], byteorder="little")

            # Extract actual data (skip size prefix)
            data_bytes = tensor_bytes[8 : 8 + actual_size]

            # Deserialize using pickle
            all_ranks_data[rank] = pickle.loads(data_bytes)

        return all_ranks_data

    def _aggregate_metadata_across_ranks(
        self,
        item_to_metadata: dict[str, dict[NestedPath, ShardingMetadata]],
        process_group: ProcessGroup | None = None,
    ) -> dict[str, dict[NestedPath, list[GlobalObjectMetadata]]]:
        """
        Perform single-level metadata aggregation using specified process group.

        Returns:
            metadata: dict mapping item_key to dict of NestedPath to list of GlobalObjectMetadata
                      with ranks field set for deduplication.
        """
        event_logger = EventLogger()

        # Count original paths
        original_path_count = sum(
            len(nested_paths) for nested_paths in item_to_metadata.values()
        )

        # Compact metadata - only representative ranks send entries with equivalent_ranks
        compacted_metadata = self._compact(item_to_metadata)

        compacted_path_count = sum(
            len(nested_paths) for nested_paths in compacted_metadata.values()
        )

        logger.info(
            f"Metadata aggregation: original_paths={original_path_count}, "
            f"compacted_paths={compacted_path_count}",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.compact.latency_ms",
            ),
        )

        # Gather metadata from all ranks (now with compacted payload)
        all_ranks_metadata = self._all_gather_pickled_data(
            compacted_metadata, process_group
        )

        logger.info(
            f"Metadata aggregation: gathered_ranks={len(all_ranks_metadata)}",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.all_gather.latency_ms",
            ),
        )

        # Merge and deduplicate to create global view
        merged_metadata = self._merge_object_metadata_from_ranks(all_ranks_metadata)

        merged_path_count = sum(
            len(nested_paths) for nested_paths in merged_metadata.values()
        )
        logger.info(
            f"Metadata aggregation: merged_items={len(merged_metadata)}, merged_paths={merged_path_count}",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.merge.latency_ms",
            ),
        )

        return merged_metadata

    def _merge_object_metadata_from_ranks(
        self,
        all_ranks_metadata: dict[int, dict[str, dict[NestedPath, ShardingMetadata]]],
    ) -> dict[str, dict[NestedPath, list[GlobalObjectMetadata]]]:
        """
        Merge per-path metadata from all ranks, grouping by ShardingMetadata hash.

        Optimized for large-scale scenarios (300k+ ranks) by using ShardingMetadata's
        cached hash for efficient deduplication. This is more efficient than
        per-item grouping because:
        1. ShardingMetadata subclasses can implement __hash__ with caching
        2. Direct hash comparison avoids serialization overhead
        3. Per-path granularity is simpler and easier to reason about

        Args:
            all_ranks_metadata: Dict mapping rank_id to dict[item_key, dict[NestedPath, ShardingMetadata]]

        Returns:
            Dict mapping item_key to dict of NestedPath to list of GlobalObjectMetadata.
            In the common case (FSDP/DP), each path has a single entry with all ranks.
        """
        event_logger = EventLogger()

        # Collect all item_keys and their nested paths
        all_items: dict[str, set[NestedPath]] = {}
        for rank_metadata in all_ranks_metadata.values():
            for item_key, nested_paths in rank_metadata.items():
                if item_key not in all_items:
                    all_items[item_key] = set()
                all_items[item_key].update(nested_paths.keys())

        # For each item and nested path, group ranks by their ShardingMetadata
        result: dict[str, dict[NestedPath, list[GlobalObjectMetadata]]] = {}

        for item_key, nested_paths in all_items.items():
            result[item_key] = {}

            for nested_path in nested_paths:
                # Group ranks by their ShardingMetadata for this path
                # Use ShardingMetadata._pack() string as key since ShardingMetadata may not be hashable
                metadata_to_ranks: dict[str, tuple[ShardingMetadata, list[int]]] = {}

                for rank, rank_metadata in all_ranks_metadata.items():
                    if item_key not in rank_metadata:
                        continue
                    item_metadata = rank_metadata[item_key]
                    if nested_path not in item_metadata:
                        continue

                    sharding_meta = item_metadata[nested_path]

                    # Create hashable key from ShardingMetadata
                    meta_key = str(sharding_meta._pack())

                    if meta_key in metadata_to_ranks:
                        metadata_to_ranks[meta_key][1].append(rank)
                    else:
                        metadata_to_ranks[meta_key] = (sharding_meta, [rank])

                # Build GlobalObjectMetadata list for this path
                result[item_key][nested_path] = []
                for _meta_key, (sharding_meta, ranks_list) in metadata_to_ranks.items():
                    # Determine which ranks hold this data.
                    # If equivalent_ranks is set (e.g. DTensor), compaction means
                    # only the representative rank sent metadata, so ranks_list
                    # is incomplete — use equivalent_ranks for the full set.
                    # If None, every rank sent its own metadata, so
                    # ranks_list is already complete.
                    equiv_ranks = sharding_meta.equivalent_ranks
                    final_ranks = (
                        tuple(sorted(equiv_ranks))
                        if equiv_ranks is not None
                        else tuple(ranks_list)
                    )

                    result[item_key][nested_path].append(
                        GlobalObjectMetadata(
                            sharding_metadata=sharding_meta,
                            ranks=final_ranks,
                        )
                    )

        total_paths = sum(len(nested_paths) for nested_paths in result.values())
        logger.info(
            f"Metadata aggregation: merged_items={len(result)}, merged_paths={total_paths}",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.merge.group.latency_ms",
            ),
        )

        return result

    def _aggregate_layout_info_across_ranks(
        self,
        layout_info_mappings: dict[str, LayoutInfo | None],
        process_group: ProcessGroup | None = None,
    ) -> dict[int, dict[str, LayoutInfo | None]]:
        """
        Aggregate layout info across ranks using all_gather.

        Args:
            layout_info_mappings: Local layout info mappings for this rank
            process_group: Process group for distributed coordination

        Returns:
            Dict mapping rank to layout info mappings (with LayoutInfo objects)
        """
        # Serialize local layout info for transmission
        serialized_layout = {
            key: layout_info.to_dict() if layout_info is not None else None
            for key, layout_info in layout_info_mappings.items()
        }

        # Gather serialized layout info from all ranks
        all_ranks_serialized = self._all_gather_pickled_data(
            serialized_layout, process_group
        )

        # Convert serialized dicts back to LayoutInfo objects
        rank_to_layout_info = {}
        for rank, serialized_dict in all_ranks_serialized.items():
            rank_to_layout_info[rank] = {
                key: (
                    LayoutInfo.from_dict(layout_dict)
                    if layout_dict is not None
                    else None
                )
                for key, layout_dict in serialized_dict.items()
            }

        return rank_to_layout_info
