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

import functools
import logging
import pickle
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_object_coll_device

from .checkpoint_base import CheckpointInfo
from .checkpoint_layout import LayoutInfo
from .distributed_metadata import (
    CheckpointMetadata,
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
    LocalMetadata,
    ShardingMetadata,
)
from .logging_utils import EventLogger, EventType
from .types import NestedPath, RankInfo

logger = logging.getLogger(__name__)


def _collective_device(process_group: ProcessGroup | None) -> torch.device:
    """
    Device for the tensors of a collective on ``process_group``.

    Uses the rule of torch's object collectives, via the private
    ``torch.distributed.distributed_c10d._get_object_coll_device``: CPU if the
    group has a CPU backend (e.g. Gloo, alone or next to NCCL), otherwise its
    accelerator.
    """
    return torch.device(_get_object_coll_device(process_group))


def _resharded_item_layouts(
    checkpoint_info: CheckpointInfo,
) -> dict[str, LayoutInfo | None]:
    """This rank's layouts of the items that the distributed metadata describes."""
    return {
        key: item.layout
        for key, item in checkpoint_info.checkpoint_items.items()
        if item.resharder is not None
    }


class MetadataManager(ABC):
    """
    Abstract base class for metadata managers in distributed checkpointing.

    A metadata manager extracts sharding metadata from checkpoint items and
    aggregates it across ranks into the global view that a checkpoint records.

    Only ``compute_distributed_metadata`` issues collectives, and every rank must
    call it. It caches its result, which the ``get_*`` methods return on any
    single rank without communicating or checking it against later state: after
    changing the structure or sharding of a resharded item, call
    ``invalidate_computed_metadata`` on every rank.

    This ABC enables composable metadata management that can be plugged
    into different checkpointer implementations.
    """

    @abstractmethod
    def extract_object_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> LocalMetadata:
        """
        Extract per-path sharding metadata from checkpoint info, grouped by item_key.

        This method extracts metadata for all CheckpointItems with resharders,
        delegating to each resharder's extract_sharding_metadata method.

        Loads call this directly to get their target metadata, so it must stay
        purely local: no collectives, and no reads or writes of any cache.

        Args:
            checkpoint_info: CheckpointInfo containing checkpoint items

        Returns:
            Dict mapping item_key to dict of NestedPath -> ShardingMetadata
            for each sharded object.
        """

    @abstractmethod
    def compute_distributed_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> None:
        """
        Compute the global metadata for ``checkpoint_info`` and cache it.

        Issues collectives, so every rank must call it, each with its own items.
        It does not assume the structure is unchanged since the previous call: it
        keeps the cache only if every rank's metadata and layouts still match it,
        and otherwise recomputes and replaces it.

        Args:
            checkpoint_info: CheckpointInfo containing checkpoint items
        """

    @abstractmethod
    def has_computed_metadata(self) -> bool:
        """Whether computed metadata is cached. No collectives."""

    @abstractmethod
    def computed_metadata_matches(self, checkpoint_info: CheckpointInfo) -> bool:
        """
        Whether this rank's part of ``checkpoint_info`` still matches the cached
        metadata. False if nothing is cached.

        No collectives, so it says nothing about other ranks.
        """

    @abstractmethod
    def get_computed_metadata(self) -> CheckpointMetadata:
        """
        Return the metadata cached by ``compute_distributed_metadata``.

        No collectives. The metadata is not checked against the current state.

        Raises:
            RuntimeError: If nothing has been computed since construction or the
                last ``invalidate_computed_metadata``.
        """

    @abstractmethod
    def get_serialized_computed_metadata(self) -> bytes:
        """
        Return the global part of ``get_computed_metadata()`` serialized as the
        checkpoint's metadata file. No collectives.

        Raises:
            RuntimeError: If ``get_computed_metadata`` would raise.
        """

    @abstractmethod
    def invalidate_computed_metadata(self) -> None:
        """
        Drop the cached metadata, if any.

        Issues no collectives, but call it on every rank: savers call
        ``compute_distributed_metadata`` when ``has_computed_metadata()`` is
        False, and a rank that skips that collective hangs the others.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources used by the metadata manager."""


class DefaultMetadataManager(MetadataManager):
    """
    Provides a basic implementation suitable for most use cases that aggregates
    metadata from all ranks and validates consistency with distributed coordination using standard PyTorch distributed primitives.

    Not thread-safe: call it from one thread.
    """

    def __init__(
        self,
        rank_info: RankInfo,
        process_group: ProcessGroup | None = None,
    ):
        """
        Initialize the default metadata manager.

        Args:
            rank_info: Information about current rank and world size
            process_group: The process group to use for distributed coordination. If None, uses the default process group.
        """
        self._rank_info = rank_info
        self._process_group = process_group
        self._computed_metadata: CheckpointMetadata | None = None
        # This rank's layouts of the resharded items the computed metadata was
        # built from; with the local metadata, they decide whether it still holds.
        self._computed_layout_info: dict[str, LayoutInfo | None] | None = None
        self._serialized_metadata: bytes | None = None

    def close(self) -> None:
        """Release resources used by the metadata manager."""

    def compute_distributed_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> None:
        event_logger = EventLogger()

        local_metadata = self.extract_object_metadata(checkpoint_info)

        logger.info(
            f"MetadataManager: extracted metadata for {len(local_metadata)} paths",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.extract_local.latency_ms",
            ),
        )

        layout_info = _resharded_item_layouts(checkpoint_info)
        if self._all_ranks_agree(self._cache_matches(local_metadata, layout_info)):
            logger.info("MetadataManager: computed metadata is unchanged; keeping it")
            return

        # A failed aggregation must not leave the old metadata in place.
        self.invalidate_computed_metadata()
        distributed_metadata = self._aggregate_metadata(
            local_metadata, checkpoint_info.layout_info_mappings
        )
        self._computed_metadata = CheckpointMetadata(
            distributed_metadata=distributed_metadata,
            local_metadata=local_metadata,
        )
        self._computed_layout_info = layout_info

        logger.info(
            f"MetadataManager: computed metadata for {len(distributed_metadata.metadata)} paths",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_metadata.compute_global.e2e.latency_ms",
                end_to_end=True,
            ),
        )

    def has_computed_metadata(self) -> bool:
        return self._computed_metadata is not None

    def computed_metadata_matches(self, checkpoint_info: CheckpointInfo) -> bool:
        return self._cache_matches(
            self.extract_object_metadata(checkpoint_info),
            _resharded_item_layouts(checkpoint_info),
        )

    def _cache_matches(
        self,
        local_metadata: LocalMetadata,
        layout_info: dict[str, LayoutInfo | None],
    ) -> bool:
        return (
            self._computed_metadata is not None
            and self._computed_metadata.local_metadata == local_metadata
            and self._computed_layout_info == layout_info
        )

    def get_computed_metadata(self) -> CheckpointMetadata:
        if self._computed_metadata is None:
            raise RuntimeError(
                "No computed metadata: call compute_distributed_metadata() on "
                "every rank first."
            )
        return self._computed_metadata

    def get_serialized_computed_metadata(self) -> bytes:
        if self._serialized_metadata is None:
            event_logger = EventLogger()
            self._serialized_metadata = pickle.dumps(
                self.get_computed_metadata().distributed_metadata.to_dict()
            )
            logger.info(
                "Serialized distributed metadata",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name="train.checkpoint_metadata.serialize.latency_ms",
                ),
            )
        return self._serialized_metadata

    def invalidate_computed_metadata(self) -> None:
        self._computed_metadata = None
        self._computed_layout_info = None
        self._serialized_metadata = None

    def extract_object_metadata(
        self,
        checkpoint_info: CheckpointInfo,
    ) -> LocalMetadata:
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
        result: LocalMetadata = {}

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

    def _is_distributed(self) -> bool:
        return self._rank_info.global_world_size > 1 and dist.is_initialized()

    def _all_ranks_agree(self, value: bool) -> bool:
        """Return True on every rank iff ``value`` is True on every rank."""
        if not self._is_distributed():
            return value
        flag = torch.tensor(
            [int(value)],
            dtype=torch.int32,
            device=_collective_device(self._process_group),
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self._process_group)
        return bool(flag.item())

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
        if not self._is_distributed():
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
        device = _collective_device(process_group)

        # Step 1: Serialize local data using pickle
        data_bytes = pickle.dumps(data)
        data_size = len(data_bytes)

        # Prepend data with its size (8 bytes for int64)
        size_bytes = data_size.to_bytes(8, byteorder="little")
        prefixed_data = size_bytes + data_bytes

        # Use all_reduce with max operation to find the maximum total size across all ranks
        total_size = len(prefixed_data)
        max_size_tensor = torch.tensor([total_size], dtype=torch.int64, device=device)
        dist.all_reduce(max_size_tensor, op=dist.ReduceOp.MAX, group=process_group)
        max_size = int(max_size_tensor.item())

        # Pad prefixed data and convert to tensor
        padded_bytes = prefixed_data + b"\0" * (max_size - len(prefixed_data))
        data_tensor = torch.frombuffer(padded_bytes, dtype=torch.uint8).to(device)

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

        # Share equal rank tuples so pickle memoizes them across tensors instead
        # of serializing the same rank list for every tensor.
        @functools.cache
        def canonical_ranks(x: tuple[int, ...]) -> tuple[int, ...]:
            return x

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
                    final_ranks = canonical_ranks(
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
