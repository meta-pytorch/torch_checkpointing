# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DTensor resharding for checkpoint loading across different distributed configurations.

This module provides a Resharder implementation that uses DTensor's native placement APIs
to compute shard geometry and perform resharding.

The core algorithm:
1. For each target nested path, compute the target rank's local shape and global offset
   using `_compute_local_shape_and_global_offset` from DTensor internals.
2. For each source rank, compute its local shape and global offset similarly.
3. Calculate the intersection of source and target global slices.
4. If an intersection exists, create a LoadPlan mapping source data to target locations.
5. Deduplicate across replicated source ranks and optimize source rank selection.
"""

import functools
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    _StridedShard as DTensorStridedShard,
    Placement as DTensorPlacement,
    Replicate as DTensorReplicate,
    Shard as DTensorShard,
)
from typing_extensions import override

from .checkpoint_layout import SafetensorsSerialization
from .default_load_plan_executor import (
    DefaultLoadPlanExecutor,
    ReshardingReadStrategy,
)
from .distributed_metadata import (
    DistributedItemMetadata,
    ShardingMetadata,
)
from .dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
    StridedShardSpec,
)
from .load_plan_executor import LoadExecutionContext
from .resharding import (
    LoadPlan,
    Resharder,
    ReshardingInfo,
)
from .resharding_utils import (
    convert_nested_path_dict_to_fqn,
    deduplicate_source_chunks,
    get_fqn_from_nested_path,
)
from .storage.base_storage import Storage
from .types import CheckpointPath, NestedPath
from .walk_utils import walk_checkpoint_structure

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["DefaultResharder", "ReshardingReadStrategy"]


@functools.lru_cache(maxsize=None)
def _world_mesh_spec(device_type: str, world_size: int) -> DeviceMeshSpec:
    """A 1-D mesh of every rank, built once per device type and world size."""
    return get_device_mesh_spec(
        device_type=device_type,
        mesh_shape=(world_size,),
        mesh_data=tuple(range(world_size)),
    )


def _replicated_tensor_metadata(tensor: torch.Tensor) -> DTensorShardingMetadata:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    return DTensorShardingMetadata(
        global_shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        stride=tuple(tensor.stride()),
        mesh_spec=_world_mesh_spec(tensor.device.type, world_size),
        placements=(ReplicateSpec(),),
    )


def _to_dtensor_placements(
    metadata: DTensorShardingMetadata,
) -> list[DTensorPlacement]:
    """Convert DTensorShardingMetadata placements to DTensor placement types.

    Args:
        metadata: DTensorShardingMetadata containing ShardSpec/ReplicateSpec placements.

    Returns:
        List of DTensor placement objects (Shard/StridedShard/Replicate).

    Raises:
        ValueError: If an unsupported placement type is encountered.
    """
    placements: list[DTensorPlacement] = []
    for p in metadata.placements:
        if isinstance(p, StridedShardSpec):
            placements.append(DTensorStridedShard(p.dim, split_factor=p.split_factor))
        elif isinstance(p, ShardSpec):
            placements.append(DTensorShard(p.dim))
        elif isinstance(p, ReplicateSpec):
            placements.append(DTensorReplicate())
        else:
            raise ValueError(f"Unsupported placement type: {type(p)}")
    return placements


def compute_local_shard_info(
    metadata: DTensorShardingMetadata,
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute local shape and global offset for a rank given DTensor metadata.

    Uses DTensor's `_compute_local_shape_and_global_offset` internally,
    converting our ShardSpec/ReplicateSpec to DTensor Shard/Replicate placements.

    Args:
        metadata: DTensor sharding metadata describing the distribution.
        rank: The rank to compute shard info for.

    Returns:
        Tuple of (local_shape, global_offset) as tuples of ints.
    """
    coordinate = metadata.mesh_spec.get_coordinate(rank)
    if coordinate is None:
        raise ValueError(
            f"Rank {rank} not found in mesh with data {metadata.mesh_spec.mesh_data}"
        )

    placements = _to_dtensor_placements(metadata)

    local_shape, global_offset = _compute_local_shape_and_global_offset(
        torch.Size(metadata.global_shape),
        metadata.mesh_spec.mesh_shape,
        list(coordinate),
        placements,
    )

    return local_shape, global_offset


def _intersect_slices(
    s1: tuple[slice, ...],
    s2: tuple[slice, ...],
) -> tuple[slice, ...] | None:
    """Calculate intersection of two multi-dimensional slices.

    Each slice must have non-None start and stop values.

    Args:
        s1: First tuple of slices (one per dimension).
        s2: Second tuple of slices (one per dimension).

    Returns:
        Tuple of intersection slices, or None if no intersection exists.
    """
    if len(s1) != len(s2):
        return None

    intersection = []
    for a, b in zip(s1, s2):
        start = max(a.start, b.start)
        stop = min(a.stop, b.stop)
        if start >= stop:
            return None
        intersection.append(slice(start, stop))

    return tuple(intersection)


def _global_to_local_slice(
    global_slice: tuple[slice, ...],
    global_offset: tuple[int, ...],
) -> tuple[slice, ...]:
    """Convert global slice to local slice by subtracting global offset.

    Args:
        global_slice: Tuple of slices in global coordinates.
        global_offset: Global offset to subtract (one per dimension).

    Returns:
        Tuple of slices in local coordinates.
    """
    return tuple(
        slice(s.start - offset, s.stop - offset)
        for s, offset in zip(global_slice, global_offset)
    )


class DefaultResharder(Resharder):
    """Resharder for DTensor checkpoints.

    Uses DTensor's native placement APIs to compute shard geometry and perform
    resharding during checkpoint loading. Supports transitions between different
    Shard/Replicate placements and device mesh configurations.

    Args:
        read_strategy: Passed to the ``DefaultLoadPlanExecutor`` that ``load``
            executes plans with.
        file_read_workers: Passed to the ``DefaultLoadPlanExecutor`` that
            ``load`` executes plans with.
    """

    def __init__(
        self,
        *,
        read_strategy: ReshardingReadStrategy = ReshardingReadStrategy.AUTO,
        file_read_workers: int | None = None,
    ) -> None:
        self._executor = DefaultLoadPlanExecutor(
            read_strategy=read_strategy,
            file_read_workers=file_read_workers,
        )

    @override
    def extract_sharding_metadata(
        self,
        item_key: str,
        item_value: Any,
    ) -> dict[NestedPath, ShardingMetadata]:
        """Extract DTensorShardingMetadata for all tensor leaves in the item.

        Plain tensors are represented as replicated over the default process group.

        Args:
            item_key: The checkpoint item key (e.g., "model", "optimizer").
            item_value: The item's value (e.g., state_dict).

        Returns:
            Dictionary mapping NestedPath to ShardingMetadata for each tensor.
        """
        result: dict[NestedPath, ShardingMetadata] = {}
        plain_tensor_paths: list[NestedPath] = []

        def _collect(path: CheckpointPath, obj: Any, _: Any) -> None:
            if isinstance(obj, DTensor):
                result[path.nested_path] = DTensorShardingMetadata.from_dtensor(obj)
            elif isinstance(obj, torch.Tensor):
                result[path.nested_path] = _replicated_tensor_metadata(obj)
                plain_tensor_paths.append(path.nested_path)

        walk_checkpoint_structure(
            item_key=item_key,
            source=item_value,
            target=None,
            leaf_fn=_collect,
        )
        if plain_tensor_paths:
            logger.warning(
                "Found %s plain tensors in checkpoint item %r; treating them as "
                "replicated tensors. sample_paths=%s",
                len(plain_tensor_paths),
                item_key,
                plain_tensor_paths[:10],
            )
        return result

    @override
    def should_reshard(
        self,
        source_metadata: DistributedItemMetadata | None,
        target_metadata: dict[NestedPath, ShardingMetadata] | None,
    ) -> bool:
        """Reshard every safetensors source, even one whose sharding matches."""
        # Safetensors stores bare tensors, so saving a DTensor drops its wrapper.
        # A direct read cannot tell the resulting local shard from a whole tensor;
        # only the resharder, which takes sharding from metadata, can place it.
        if (
            source_metadata is not None
            and target_metadata is not None
            and any(
                layout is not None
                and isinstance(layout.serialization_format, SafetensorsSerialization)
                for layout in source_metadata.rank_to_layout_info.values()
            )
        ):
            return True
        return super().should_reshard(source_metadata, target_metadata)

    @override
    def load(
        self,
        source_path: Path,
        item_key: str,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
        target: Any,
        storage: Storage,
    ) -> list[NestedPath]:
        """Load and reshard checkpoint data into target.

        Orchestrates the full resharding pipeline:
        1. Generate load plans, bound to the target tensors they fill.
        2. Execute load plans to read source files and copy data into target.

        Args:
            source_path: Base path to the source checkpoint directory.
            item_key: The checkpoint item key being loaded (e.g., "model").
            target_metadata: This rank's target sharding from extract_sharding_metadata.
            source_metadata: Source checkpoint's distributed metadata for this item.
            target: Target object to load data into (modified in-place).
            storage: Storage backend for reading checkpoint files.

        Returns:
            List of NestedPaths that could not be resharded.
        """
        resharding_info = self.generate_load_plans(
            item_key, target, target_metadata, source_metadata
        )
        self._executor.execute(
            {item_key: resharding_info},
            LoadExecutionContext(
                checkpoint_path=source_path,
                storage=storage,
                source_metadata={item_key: source_metadata},
            ),
        )
        return resharding_info.non_reshardable_paths

    def generate_load_plans(
        self,
        item_key: str,
        target: Any,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
    ) -> ReshardingInfo:
        """Plan which source chunks fill each target tensor, without reading data.

        Args:
            item_key: The checkpoint item key being loaded (e.g., "model").
            target: The item's live value, which the planned chunks are written into.
            target_metadata: This rank's target sharding from extract_sharding_metadata.
            source_metadata: Source checkpoint's distributed metadata for this item.

        Returns:
            ReshardingInfo binding each planned path to its target tensor.
        """
        load_plans, non_reshardable_paths = self._plan_from_metadata(
            target_metadata, source_metadata
        )
        return ReshardingInfo(
            nested_path_to_load_plans=load_plans,
            non_reshardable_paths=non_reshardable_paths,
            nested_path_to_target_tensors=self.resolve_targets(item_key, target),
        )

    def _plan_from_metadata(
        self,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
    ) -> tuple[dict[NestedPath, list[LoadPlan]], list[NestedPath]]:
        """Generate load plans by computing chunk mappings between source and target.

        For each NestedPath in target_metadata:
        1. Compute target local shape + global offset for current rank.
        2. For each source rank, compute source local shape + global offset.
        3. Calculate intersection of source and target global slices.
        4. If intersection exists, create a LoadPlan.
        5. Deduplicate across replicated source ranks.

        Args:
            target_metadata: Target sharding metadata for this rank.
            source_metadata: Source distributed metadata with rank groups.

        Returns:
            The load plans for each target path, and the paths that could not
            be planned.
        """
        current_rank = dist.get_rank() if dist.is_initialized() else 0

        result: dict[NestedPath, list[LoadPlan]] = {}
        non_reshardable_paths: list[NestedPath] = []

        for nested_path, target_sharding in target_metadata.items():
            # Both source and target must be DTensorShardingMetadata
            if not isinstance(target_sharding, DTensorShardingMetadata):
                non_reshardable_paths.append(nested_path)
                continue

            source_groups = source_metadata.nested_path_to_metadata.get(nested_path)
            if source_groups is None:
                logger.warning(f"Missing source metadata for path: {nested_path}")
                non_reshardable_paths.append(nested_path)
                continue

            # Compute target local shape and global offset for current rank
            target_local_shape, target_global_offset = compute_local_shard_info(
                target_sharding, current_rank
            )

            # Compute target global slice
            target_global_slice = tuple(
                slice(offset, offset + size)
                for offset, size in zip(target_global_offset, target_local_shape)
            )

            fqn = get_fqn_from_nested_path(nested_path)
            param_load_plans: list[LoadPlan] = []
            path_is_reshardable = True

            # Track seen source global slices to deduplicate replicated ranks
            seen_source_slices: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()

            for group in source_groups:
                src_sharding = group.sharding_metadata
                if not isinstance(src_sharding, DTensorShardingMetadata):
                    non_reshardable_paths.append(nested_path)
                    path_is_reshardable = False
                    param_load_plans = []
                    break

                for src_rank in group.ranks:
                    src_local_shape, src_global_offset = compute_local_shard_info(
                        src_sharding, src_rank
                    )

                    # Deduplicate: skip if we've already seen this global slice
                    slice_key = (src_global_offset, src_local_shape)
                    if slice_key in seen_source_slices:
                        continue
                    seen_source_slices.add(slice_key)

                    # Compute source global slice
                    source_global_slice = tuple(
                        slice(offset, offset + size)
                        for offset, size in zip(src_global_offset, src_local_shape)
                    )

                    # Calculate intersection
                    intersection = _intersect_slices(
                        source_global_slice, target_global_slice
                    )

                    if intersection is not None:
                        # Convert global intersection to local slices
                        src_local_slice = _global_to_local_slice(
                            intersection, src_global_offset
                        )
                        tgt_local_slice = _global_to_local_slice(
                            intersection, target_global_offset
                        )

                        # Compute sizes from intersection
                        sizes = tuple(s.stop - s.start for s in intersection)

                        param_load_plans.append(
                            LoadPlan(
                                offsets=tuple(s.start for s in tgt_local_slice),
                                sizes=sizes,
                                src_rank=src_rank,
                                src_fqn=fqn,
                                src_offsets=tuple(s.start for s in src_local_slice),
                                src_sizes=sizes,
                                transpose_dims=(),
                            )
                        )

            if path_is_reshardable:
                if param_load_plans:
                    result[nested_path] = param_load_plans
                elif all(size > 0 for size in target_local_shape):
                    logger.warning(
                        f"No source DTensor shard intersects target shard for path: {nested_path}"
                    )
                    non_reshardable_paths.append(nested_path)

        # Apply deduplicate_source_chunks to minimize source ranks
        if result:
            fqn_keyed_result, fqn_to_path = convert_nested_path_dict_to_fqn(result)
            optimized_str_result, _selected_ranks = deduplicate_source_chunks(
                fqn_keyed_result
            )
            result = {
                fqn_to_path[fqn]: plans for fqn, plans in optimized_str_result.items()
            }

        return result, non_reshardable_paths
