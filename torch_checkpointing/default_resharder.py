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

from .distributed_metadata import (
    DistributedItemMetadata,
    ShardingMetadata,
)
from .dtensor_metadata import (
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
    StridedShardSpec,
)
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

__all__ = ["DefaultResharder"]


def _replicated_tensor_metadata(tensor: torch.Tensor) -> DTensorShardingMetadata:
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    mesh_spec = get_device_mesh_spec(
        device_type=tensor.device.type,
        mesh_shape=(world_size,),
        mesh_data=tuple(range(world_size)),
    )
    return DTensorShardingMetadata(
        global_shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        stride=tuple(tensor.stride()),
        mesh_spec=mesh_spec,
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
    mesh_tensor = metadata.mesh_spec.mesh
    coordinate = metadata.mesh_spec.get_coordinate(rank)
    if coordinate is None:
        raise ValueError(
            f"Rank {rank} not found in mesh with data {metadata.mesh_spec.mesh_data}"
        )

    placements = _to_dtensor_placements(metadata)

    local_shape, global_offset = _compute_local_shape_and_global_offset(
        torch.Size(metadata.global_shape),
        mesh_tensor.shape,
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


def _collect_leaf_values(item_key: str, value: Any) -> dict[NestedPath, Any]:
    """Collect checkpoint leaves keyed by their NestedPath.

    Args:
        item_key: The checkpoint item key.
        value: Nested checkpoint value to walk.

    Returns:
        Dictionary mapping NestedPath to leaf values.
    """
    result: dict[NestedPath, Any] = {}

    def _collect(path: CheckpointPath, obj: Any, _: Any) -> Any:
        result[path.nested_path] = obj
        return obj

    walk_checkpoint_structure(
        item_key=item_key,
        source=value,
        target=None,
        leaf_fn=_collect,
    )
    return result


def _flatten_state_dict(item_key: str, value: Any) -> dict[str, Any]:
    """Flatten a checkpoint value to dot-separated keys.

    Args:
        item_key: The checkpoint item key.
        value: Nested checkpoint value to flatten.

    Returns:
        Flat dictionary with dot-separated keys.
    """
    path_to_value = _collect_leaf_values(item_key, value)
    flattened, _ = convert_nested_path_dict_to_fqn(path_to_value)
    return flattened


class DefaultResharder(Resharder):
    """Resharder for DTensor checkpoints.

    Uses DTensor's native placement APIs to compute shard geometry and perform
    resharding during checkpoint loading. Supports transitions between different
    Shard/Replicate placements and device mesh configurations.
    """

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
        1. Generate load plans computing chunk mappings between source and target.
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
        resharding_info = self._generate_load_plans(target_metadata, source_metadata)

        if resharding_info.nested_path_to_load_plans:
            # Create src_path_fn using DistributedItemMetadata
            def src_path_fn(source_rank: int) -> Path:
                return source_metadata.get_file_path(source_rank, source_path, item_key)

            self._execute_load_plans(
                src_path_fn,
                item_key,
                resharding_info.nested_path_to_load_plans,
                target,
                storage,
            )
        else:
            logger.warning(
                f"DefaultResharder.load: no load plans generated for item '{item_key}'."
            )

        return resharding_info.non_reshardable_paths

    def _generate_load_plans(
        self,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
    ) -> ReshardingInfo:
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
            ReshardingInfo with load plans and non-reshardable paths.
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

        return ReshardingInfo(
            nested_path_to_load_plans=result,
            non_reshardable_paths=non_reshardable_paths,
        )

    def _execute_load_plans(
        self,
        src_path_fn: Any,
        item_key: str,
        nested_path_to_load_plans: dict[NestedPath, list[LoadPlan]],
        target: Any,
        storage: Storage,
    ) -> None:
        """Execute load plans by reading source files and copying data into target.

        Groups load plans by source rank to minimize file reads, then for each
        source rank:
        1. Read the full checkpoint file.
        2. Flatten the loaded state dict.
        3. For each load plan, extract source data and copy into target tensor.

        Args:
            src_path_fn: Callable that returns the file path for a given source rank.
            item_key: The checkpoint item key being loaded.
            nested_path_to_load_plans: Mapping from NestedPath to LoadPlans.
            target: Target dict-like structure to load data into.
            storage: Storage backend for reading checkpoint files.
        """
        target_by_path = _collect_leaf_values(item_key, target)

        # Group load plans by source rank
        plans_by_rank: dict[int, list[tuple[NestedPath, LoadPlan]]] = {}
        for nested_path, load_plans in nested_path_to_load_plans.items():
            for lp in load_plans:
                if lp.src_rank not in plans_by_rank:
                    plans_by_rank[lp.src_rank] = []
                plans_by_rank[lp.src_rank].append((nested_path, lp))

        # Process each source rank
        for src_rank, rank_plans in plans_by_rank.items():
            file_path = src_path_fn(src_rank)

            # TODO: Use offset reads for tensor chunks.
            with storage.stream_read(file_path) as f:
                loaded_data = torch.load(
                    f,  # type: ignore[arg-type]
                    map_location="cpu",
                    weights_only=False,
                )

            # Flatten the loaded state dict
            flattened = _flatten_state_dict(item_key, loaded_data)

            # Execute each load plan for this rank
            for nested_path, lp in rank_plans:
                # Get source tensor from flattened dict
                src_tensor = flattened[lp.src_fqn]

                # Unwrap DTensor if needed
                if isinstance(src_tensor, DTensor):
                    src_tensor = src_tensor._local_tensor

                # Extract source slice
                src_slice = tuple(
                    slice(o, o + s) for o, s in zip(lp.src_offsets, lp.src_sizes)
                )
                src_data = src_tensor[src_slice]

                # Get target tensor from the walked target structure
                target_obj = target_by_path[nested_path]

                # Unwrap DTensor if needed
                if isinstance(target_obj, DTensor):
                    target_tensor = target_obj._local_tensor
                else:
                    target_tensor = target_obj

                # Copy into target
                tgt_slice = tuple(slice(o, o + s) for o, s in zip(lp.offsets, lp.sizes))
                target_tensor[tgt_slice].copy_(src_data)
