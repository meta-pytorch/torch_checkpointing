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
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import prod
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    _StridedShard as DTensorStridedShard,
    Placement as DTensorPlacement,
    Replicate as DTensorReplicate,
    Shard as DTensorShard,
)
from typing_extensions import override

from ...distributed_metadata import (
    DistributedItemMetadata,
    GlobalObjectMetadata,
    ShardingMetadata,
)
from ...dtensor_metadata import (
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
    StridedShardSpec,
)
from ...resharding import (
    LoadPlan,
    Resharder,
    ReshardingInfo,
)
from ...resharding_utils import (
    convert_nested_path_dict_to_fqn,
    deduplicate_source_chunks,
    get_fqn_from_nested_path,
)
from ...storage.base_storage import ReadArgs, Storage
from ...types import CheckpointPath, NestedPath
from ...walk_utils import walk_checkpoint_structure

logger: logging.Logger = logging.getLogger(__name__)

__all__ = ["DefaultResharder"]

_MAX_PLACEMENT_CACHE_ENTRIES = 128
_MAX_SHARD_GEOMETRY_CACHE_ENTRIES = 8192
_MAX_LOAD_PLAN_TEMPLATE_CACHE_ENTRIES = 128

_ShardGeometryKey = tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[object, ...],
]
_LoadPlanTemplateCacheKey = tuple[
    int,
    _ShardGeometryKey,
    tuple[tuple[_ShardGeometryKey, tuple[int, ...]], ...],
]


@lru_cache(maxsize=_MAX_PLACEMENT_CACHE_ENTRIES)
def _rank_to_coordinate(
    mesh_shape: tuple[int, ...],
    mesh_data: tuple[int, ...],
) -> dict[int, tuple[int, ...]]:
    if prod(mesh_shape) != len(mesh_data):
        raise ValueError(
            "Device mesh shape does not match the number of mesh ranks: "
            f"shape={mesh_shape}, ranks={len(mesh_data)}"
        )

    result: dict[int, tuple[int, ...]] = {}
    for rank, coordinate in zip(
        mesh_data,
        product(*(range(size) for size in mesh_shape)),
    ):
        result.setdefault(rank, coordinate)
    return result


def _get_coordinate(
    metadata: DTensorShardingMetadata,
    rank: int,
) -> tuple[int, ...] | None:
    mesh_spec = metadata.mesh_spec
    return _rank_to_coordinate(mesh_spec.mesh_shape, mesh_spec.mesh_data).get(rank)


@dataclass
class _PlanningCacheMetrics:
    geometry_hits: int = 0
    geometry_misses: int = 0
    geometry_evictions: int = 0
    geometry_fast_paths: int = 0
    geometry_fallbacks: int = 0
    placement_hits: int = 0
    placement_misses: int = 0
    placement_evictions: int = 0
    plan_template_hits: int = 0
    plan_template_misses: int = 0
    plan_template_evictions: int = 0


@dataclass(frozen=True, slots=True)
class _LoadPlanTemplate:
    offsets: tuple[int, ...]
    sizes: tuple[int, ...]
    src_rank: int
    src_offsets: tuple[int, ...]
    src_sizes: tuple[int, ...]
    transpose_dims: tuple[int, ...]
    src_elem_size: int
    src_dtype: str

    def bind(self, src_fqn: str) -> LoadPlan:
        return LoadPlan(
            offsets=self.offsets,
            sizes=self.sizes,
            src_rank=self.src_rank,
            src_fqn=src_fqn,
            src_offsets=self.src_offsets,
            src_sizes=self.src_sizes,
            transpose_dims=self.transpose_dims,
            src_elem_size=self.src_elem_size,
            src_dtype=self.src_dtype,
        )


@dataclass(frozen=True, slots=True)
class _LoadPlanTemplateEntry:
    plans: tuple[_LoadPlanTemplate, ...]
    source_group_count: int
    source_rank_candidate_count: int
    duplicate_source_slice_count: int
    target_has_elements: bool


@dataclass(frozen=True, slots=True)
class _LoadPlanTemplateBuildResult:
    template: _LoadPlanTemplateEntry | None
    source_group_count: int
    source_rank_candidate_count: int
    duplicate_source_slice_count: int
    source_iteration_seconds: float


@dataclass(frozen=True, slots=True)
class _PreparedCooperativeLoad:
    local_load_plan: dict[str, tuple[LoadPlan, ...]]
    target_state_dict: dict[str, torch.Tensor]
    non_reshardable_paths: tuple[NestedPath, ...]


class _ShardGeometryCache:
    """Operation-local bounded cache for DTensor shard calculations."""

    def __init__(
        self,
        *,
        geometry_max_entries: int = _MAX_SHARD_GEOMETRY_CACHE_ENTRIES,
        placement_max_entries: int = _MAX_PLACEMENT_CACHE_ENTRIES,
    ) -> None:
        if geometry_max_entries <= 0 or placement_max_entries <= 0:
            raise ValueError("Shard geometry cache limits must be positive")
        self._geometry_max_entries = geometry_max_entries
        self._placement_max_entries = placement_max_entries
        self._geometry: dict[
            tuple[DTensorShardingMetadata, int],
            tuple[tuple[int, ...], tuple[int, ...]],
        ] = {}
        self._placements: dict[tuple[object, ...], tuple[DTensorPlacement, ...]] = {}
        self.metrics = _PlanningCacheMetrics()

    @property
    def geometry_entry_count(self) -> int:
        return len(self._geometry)

    @property
    def placement_entry_count(self) -> int:
        return len(self._placements)

    def _get_placements(
        self, metadata: DTensorShardingMetadata
    ) -> tuple[DTensorPlacement, ...]:
        key: tuple[object, ...] = metadata.placements
        try:
            placements = self._placements.pop(key)
        except KeyError:
            self.metrics.placement_misses += 1
            placements = tuple(_to_dtensor_placements(metadata))
            if len(self._placements) >= self._placement_max_entries:
                del self._placements[next(iter(self._placements))]
                self.metrics.placement_evictions += 1
        else:
            self.metrics.placement_hits += 1
        self._placements[key] = placements
        return placements

    def get_local_shard_info(
        self,
        metadata: DTensorShardingMetadata,
        rank: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        key = (metadata, rank)
        try:
            shard_info = self._geometry.pop(key)
        except KeyError:
            self.metrics.geometry_misses += 1
            shard_info = _try_compute_local_shard_info(metadata, rank)
            if shard_info is None:
                self.metrics.geometry_fallbacks += 1
                shard_info = _compute_local_shard_info_from_placements(
                    metadata,
                    rank,
                    self._get_placements(metadata),
                )
            else:
                self.metrics.geometry_fast_paths += 1
            if len(self._geometry) >= self._geometry_max_entries:
                del self._geometry[next(iter(self._geometry))]
                self.metrics.geometry_evictions += 1
        else:
            self.metrics.geometry_hits += 1
        self._geometry[key] = shard_info
        return shard_info


class _LoadPlanTemplateCache:
    """Operation-local LRU cache for path-independent load-plan geometry."""

    def __init__(
        self,
        *,
        max_entries: int = _MAX_LOAD_PLAN_TEMPLATE_CACHE_ENTRIES,
        metrics: _PlanningCacheMetrics | None = None,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("Load-plan template cache limit must be positive")
        self._max_entries = max_entries
        self._entries: dict[_LoadPlanTemplateCacheKey, _LoadPlanTemplateEntry] = {}
        self.metrics = metrics if metrics is not None else _PlanningCacheMetrics()

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def get(
        self,
        key: _LoadPlanTemplateCacheKey,
    ) -> _LoadPlanTemplateEntry | None:
        try:
            entry = self._entries.pop(key)
        except KeyError:
            self.metrics.plan_template_misses += 1
            return None
        self.metrics.plan_template_hits += 1
        self._entries[key] = entry
        return entry

    def put(
        self,
        key: _LoadPlanTemplateCacheKey,
        entry: _LoadPlanTemplateEntry,
    ) -> None:
        if key in self._entries:
            self._entries.pop(key)
        elif len(self._entries) >= self._max_entries:
            del self._entries[next(iter(self._entries))]
            self.metrics.plan_template_evictions += 1
        self._entries[key] = entry


def _shard_geometry_key(metadata: DTensorShardingMetadata) -> _ShardGeometryKey:
    """Return only fields that affect shard shapes and offsets."""
    return (
        metadata.global_shape,
        metadata.mesh_spec.mesh_shape,
        metadata.mesh_spec.mesh_data,
        tuple(metadata.placements),
    )


def _load_plan_template_cache_key(
    current_rank: int,
    target_sharding: DTensorShardingMetadata,
    source_groups: list[GlobalObjectMetadata],
) -> _LoadPlanTemplateCacheKey | None:
    source_geometry = []
    for group in source_groups:
        source_sharding = group.sharding_metadata
        if not isinstance(source_sharding, DTensorShardingMetadata):
            return None
        source_geometry.append(
            (_shard_geometry_key(source_sharding), tuple(group.ranks))
        )
    return (
        current_rank,
        _shard_geometry_key(target_sharding),
        tuple(source_geometry),
    )


def _make_load_plan_template(
    source_local_shape: tuple[int, ...],
    source_global_offset: tuple[int, ...],
    target_global_slice: tuple[slice, ...],
    target_global_offset: tuple[int, ...],
    source_rank: int,
) -> _LoadPlanTemplate | None:
    source_global_slice = tuple(
        slice(offset, offset + size)
        for offset, size in zip(source_global_offset, source_local_shape)
    )
    intersection = _intersect_slices(source_global_slice, target_global_slice)
    if intersection is None:
        return None
    source_local_slice = _global_to_local_slice(
        intersection,
        source_global_offset,
    )
    target_local_slice = _global_to_local_slice(
        intersection,
        target_global_offset,
    )
    sizes = tuple(item.stop - item.start for item in intersection)
    return _LoadPlanTemplate(
        offsets=tuple(item.start for item in target_local_slice),
        sizes=sizes,
        src_rank=source_rank,
        src_offsets=tuple(item.start for item in source_local_slice),
        src_sizes=sizes,
        transpose_dims=(),
        src_elem_size=0,
        src_dtype="",
    )


def _build_load_plan_template(
    target_sharding: DTensorShardingMetadata,
    source_groups: list[GlobalObjectMetadata],
    current_rank: int,
    geometry_cache: _ShardGeometryCache,
) -> _LoadPlanTemplateBuildResult:
    target_local_shape, target_global_offset = geometry_cache.get_local_shard_info(
        target_sharding,
        current_rank,
    )
    target_global_slice = tuple(
        slice(offset, offset + size)
        for offset, size in zip(target_global_offset, target_local_shape)
    )
    plans: list[_LoadPlanTemplate] = []
    seen_source_slices: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    group_count = candidate_count = duplicate_count = 0
    iteration_started = perf_counter()
    for group in source_groups:
        group_count += 1
        source_sharding = group.sharding_metadata
        if not isinstance(source_sharding, DTensorShardingMetadata):
            return _LoadPlanTemplateBuildResult(
                None,
                group_count,
                candidate_count,
                duplicate_count,
                perf_counter() - iteration_started,
            )
        for source_rank in group.ranks:
            candidate_count += 1
            source_local_shape, source_global_offset = (
                geometry_cache.get_local_shard_info(source_sharding, source_rank)
            )
            slice_key = (source_global_offset, source_local_shape)
            if slice_key in seen_source_slices:
                duplicate_count += 1
                continue
            seen_source_slices.add(slice_key)
            plan = _make_load_plan_template(
                source_local_shape,
                source_global_offset,
                target_global_slice,
                target_global_offset,
                source_rank,
            )
            if plan is not None:
                plans.append(plan)
    entry = _LoadPlanTemplateEntry(
        plans=tuple(plans),
        source_group_count=group_count,
        source_rank_candidate_count=candidate_count,
        duplicate_source_slice_count=duplicate_count,
        target_has_elements=all(size > 0 for size in target_local_shape),
    )
    return _LoadPlanTemplateBuildResult(
        entry,
        group_count,
        candidate_count,
        duplicate_count,
        perf_counter() - iteration_started,
    )


def _read_exact(stream: Any, offset: int, buffer: memoryview) -> None:
    stream.seek(offset)
    if stream.tell() != offset:
        raise OSError(f"Failed to seek to checkpoint offset {offset}")

    bytes_read = 0
    while bytes_read < len(buffer):
        count = stream.readinto(buffer[bytes_read:])
        if not count:
            raise EOFError(
                f"Expected {len(buffer)} bytes at checkpoint offset {offset}, "
                f"but read {bytes_read}"
            )
        bytes_read += count


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


def _compute_local_shard_info_from_placements(
    metadata: DTensorShardingMetadata,
    rank: int,
    placements: tuple[DTensorPlacement, ...] | list[DTensorPlacement],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    coordinate = _get_coordinate(metadata, rank)
    if coordinate is None:
        raise ValueError(
            f"Rank {rank} not found in mesh with data {metadata.mesh_spec.mesh_data}"
        )

    return _compute_local_shape_and_global_offset(
        torch.Size(metadata.global_shape),
        torch.Size(metadata.mesh_spec.mesh_shape),
        list(coordinate),
        list(placements),
    )


def _chunk_size_and_offset(
    size: int,
    chunk_count: int,
    chunk_index: int,
) -> tuple[int, int]:
    chunk_size = (size + chunk_count - 1) // chunk_count
    offset = min(size, chunk_size * chunk_index)
    return min(chunk_size, size - offset), offset


def _strided_shard_size_and_offset(
    size: int,
    chunk_count: int,
    chunk_index: int,
    split_factor: int,
) -> tuple[int, int]:
    if size == 0:
        return 0, 0

    split_size = (size + split_factor - 1) // split_factor
    full_split_count, remainder = divmod(size, split_size)
    full_size, full_offset = _chunk_size_and_offset(
        split_size,
        chunk_count,
        chunk_index,
    )
    remainder_size, remainder_offset = _chunk_size_and_offset(
        remainder,
        chunk_count,
        chunk_index,
    )
    local_size = full_split_count * full_size + remainder_size
    if full_size > 0:
        return local_size, full_offset
    if remainder_size > 0:
        return local_size, full_split_count * split_size + remainder_offset
    return 0, size


def _supports_integer_shard_geometry(
    metadata: DTensorShardingMetadata,
    coordinate: tuple[int, ...],
) -> bool:
    global_shape = metadata.global_shape
    mesh_shape = metadata.mesh_spec.mesh_shape
    if len(mesh_shape) != len(metadata.placements) or len(coordinate) != len(
        mesh_shape
    ):
        return False
    if any(type(size) is not int or size < 0 for size in global_shape):
        return False
    if any(type(size) is not int or size <= 0 for size in mesh_shape):
        return False

    sharding_by_dim: dict[int, tuple[int, ShardSpec | StridedShardSpec]] = {}
    for mesh_dim, placement in enumerate(metadata.placements):
        if isinstance(placement, ReplicateSpec):
            continue
        if not isinstance(placement, (ShardSpec, StridedShardSpec)):
            return False
        shard_dim = placement.dim
        if (
            type(shard_dim) is not int
            or shard_dim < 0
            or shard_dim >= len(global_shape)
        ):
            return False
        if isinstance(placement, StridedShardSpec) and (
            type(placement.split_factor) is not int or placement.split_factor <= 0
        ):
            return False
        previous = sharding_by_dim.get(shard_dim)
        if previous is not None:
            previous_mesh_dim, previous_placement = previous
            repeated_shard_count = mesh_shape[previous_mesh_dim] * mesh_shape[mesh_dim]
            if not (
                isinstance(previous_placement, StridedShardSpec)
                and isinstance(placement, ShardSpec)
                and previous_placement.split_factor == mesh_shape[mesh_dim]
                and global_shape[shard_dim] % repeated_shard_count == 0
            ):
                return False
        sharding_by_dim[shard_dim] = (mesh_dim, placement)
    return True


def _apply_shard_after_strided_shard(
    global_size: int,
    chunk_count: int,
    chunk_index: int,
    strided_shard: tuple[int, int, int],
) -> tuple[int, int]:
    first_chunk_count, first_chunk_index, split_factor = strided_shard
    if split_factor != chunk_count:
        raise AssertionError("Strided shard split factor does not match shard count")
    shard_size = global_size // (first_chunk_count * chunk_count)
    return (
        shard_size,
        shard_size * (chunk_index * first_chunk_count + first_chunk_index),
    )


def _try_compute_local_shard_info(
    metadata: DTensorShardingMetadata,
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    coordinate = _get_coordinate(metadata, rank)
    if coordinate is None:
        raise ValueError(
            f"Rank {rank} not found in mesh with data {metadata.mesh_spec.mesh_data}"
        )

    if not _supports_integer_shard_geometry(metadata, coordinate):
        return None

    global_shape = metadata.global_shape
    mesh_shape = metadata.mesh_spec.mesh_shape
    local_shape = list(global_shape)
    global_offset = [0] * len(global_shape)
    strided_shards: dict[int, tuple[int, int, int]] = {}
    for mesh_dim, placement in enumerate(metadata.placements):
        if isinstance(placement, ReplicateSpec):
            continue
        assert isinstance(placement, (ShardSpec, StridedShardSpec))
        shard_dim = placement.dim
        shard_args = (
            local_shape[shard_dim],
            mesh_shape[mesh_dim],
            coordinate[mesh_dim],
        )
        strided_shard = strided_shards.pop(shard_dim, None)
        if strided_shard is not None:
            shard_size, shard_offset = _apply_shard_after_strided_shard(
                global_shape[shard_dim],
                mesh_shape[mesh_dim],
                coordinate[mesh_dim],
                strided_shard,
            )
        elif isinstance(placement, StridedShardSpec):
            strided_shards[shard_dim] = (
                mesh_shape[mesh_dim],
                coordinate[mesh_dim],
                placement.split_factor,
            )
            shard_size, shard_offset = _strided_shard_size_and_offset(
                *shard_args,
                placement.split_factor,
            )
        else:
            shard_size, shard_offset = _chunk_size_and_offset(*shard_args)
        local_shape[shard_dim] = shard_size
        global_offset[shard_dim] = shard_offset
    return tuple(local_shape), tuple(global_offset)


def compute_local_shard_info(
    metadata: DTensorShardingMetadata,
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Compute local shape and global offset for a rank given DTensor metadata.

    Uses integer shard arithmetic for simple placements and balanced
    strided-then-contiguous sharding. Other layouts use DTensor's geometry helper.

    Args:
        metadata: DTensor sharding metadata describing the distribution.
        rank: The rank to compute shard info for.

    Returns:
        Tuple of (local_shape, global_offset) as tuples of ints.
    """
    shard_info = _try_compute_local_shard_info(metadata, rank)
    if shard_info is not None:
        return shard_info
    return _compute_local_shard_info_from_placements(
        metadata,
        rank,
        _to_dtensor_placements(metadata),
    )


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


def _unwrap_dtensor(value: Any) -> Any:
    return value._local_tensor if isinstance(value, DTensor) else value


def _slice_source_tensor(source: torch.Tensor, load_plan: LoadPlan) -> torch.Tensor:
    source_slice = tuple(
        slice(offset, offset + size)
        for offset, size in zip(load_plan.src_offsets, load_plan.src_sizes)
    )
    return source[source_slice]


def _read_source_tensor_slice(
    stream: Any,
    source: Any,
    load_plan: LoadPlan,
) -> torch.Tensor:
    source = _unwrap_dtensor(source)
    if not isinstance(source, FakeTensor):
        raise NotImplementedError(f"Source {load_plan.src_fqn!r} is not a plain tensor")
    if source.layout is not torch.strided or source.is_quantized:
        raise NotImplementedError(
            f"Source {load_plan.src_fqn!r} does not use a strided storage"
        )
    assert len(load_plan.src_offsets) == len(load_plan.src_sizes) == source.ndim, (
        f"Load plan rank does not match source tensor {load_plan.src_fqn!r}"
    )
    strides = tuple(source.stride())

    if any(size == 0 for size in load_plan.src_sizes):
        result = torch.empty(load_plan.src_sizes, dtype=source.dtype)
    else:
        checkpoint_offset = getattr(
            source.untyped_storage(), "_checkpoint_offset", None
        )
        if checkpoint_offset is None:
            raise NotImplementedError(
                f"Source {load_plan.src_fqn!r} has no checkpoint offset"
            )

        element_size = source.element_size()
        first_element = source.storage_offset() + sum(
            offset * stride for offset, stride in zip(load_plan.src_offsets, strides)
        )
        # Span from the first wanted element through the last, read in one go.
        # For a contiguous source this never exceeds the tensor itself, so the
        # worst case matches reading the whole tensor.
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(load_plan.src_sizes, strides)
        )
        packed = bytearray(span * element_size)
        _read_exact(
            stream,
            checkpoint_offset + first_element * element_size,
            memoryview(packed),
        )
        result = torch.frombuffer(packed, dtype=source.dtype).as_strided(
            load_plan.src_sizes, strides
        )

    return result


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
        load_started = perf_counter()
        planning_started = perf_counter()
        resharding_info = self._generate_load_plans(target_metadata, source_metadata)
        planning_seconds = perf_counter() - planning_started
        execution_seconds = 0.0

        if resharding_info.nested_path_to_load_plans:
            # Create src_path_fn using DistributedItemMetadata
            def src_path_fn(source_rank: int) -> Path:
                return source_metadata.get_file_path(source_rank, source_path, item_key)

            execution_started = perf_counter()
            self._execute_load_plans(
                src_path_fn,
                item_key,
                resharding_info.nested_path_to_load_plans,
                target,
                storage,
            )
            execution_seconds = perf_counter() - execution_started
        else:
            logger.warning(
                f"DefaultResharder.load: no load plans generated for item '{item_key}'."
            )

        load_plan_count = sum(
            len(plans) for plans in resharding_info.nested_path_to_load_plans.values()
        )
        source_ranks = {
            plan.src_rank
            for plans in resharding_info.nested_path_to_load_plans.values()
            for plan in plans
        }
        logger.info(
            "DefaultResharder load metrics item=%r target_paths=%d load_plans=%d "
            "source_files=%d non_reshardable_paths=%d planning_seconds=%.6f "
            "execution_seconds=%.6f total_seconds=%.6f",
            item_key,
            len(target_metadata),
            load_plan_count,
            len(source_ranks),
            len(resharding_info.non_reshardable_paths),
            planning_seconds,
            execution_seconds,
            perf_counter() - load_started,
        )
        return resharding_info.non_reshardable_paths

    def _prepare_cooperative_load(
        self,
        item_key: str,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
        target: Any,
    ) -> _PreparedCooperativeLoad:
        """Prepare archive-neutral plans and tensor targets without writing them."""

        resharding_info = self._generate_load_plans(
            target_metadata,
            source_metadata,
        )
        flattened_plans, fqn_to_path = convert_nested_path_dict_to_fqn(
            resharding_info.nested_path_to_load_plans
        )
        target_by_path = _collect_leaf_values(item_key, target)
        target_state_dict: dict[str, torch.Tensor] = {}
        for fqn, nested_path in fqn_to_path.items():
            value = _unwrap_dtensor(target_by_path[nested_path])
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Target {fqn!r} is not a tensor")
            target_state_dict[fqn] = value
        return _PreparedCooperativeLoad(
            local_load_plan={
                fqn: tuple(plans) for fqn, plans in flattened_plans.items()
            },
            target_state_dict=target_state_dict,
            non_reshardable_paths=tuple(resharding_info.non_reshardable_paths),
        )

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
        planning_started = perf_counter()
        current_rank = dist.get_rank() if dist.is_initialized() else 0
        geometry_cache = _ShardGeometryCache()
        plan_template_cache = _LoadPlanTemplateCache(
            metrics=geometry_cache.metrics,
        )

        result: dict[NestedPath, list[LoadPlan]] = {}
        non_reshardable_paths: list[NestedPath] = []
        source_group_count = 0
        source_rank_candidate_count = 0
        source_rank_scan_count = 0
        duplicate_source_slice_count = 0
        source_iteration_seconds = 0.0

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

            fqn = get_fqn_from_nested_path(nested_path)
            template_key = _load_plan_template_cache_key(
                current_rank,
                target_sharding,
                source_groups,
            )
            cached_template = (
                plan_template_cache.get(template_key)
                if template_key is not None
                else None
            )
            if cached_template is None:
                build_result = _build_load_plan_template(
                    target_sharding,
                    source_groups,
                    current_rank,
                    geometry_cache,
                )
                source_group_count += build_result.source_group_count
                source_rank_candidate_count += build_result.source_rank_candidate_count
                source_rank_scan_count += build_result.source_rank_candidate_count
                duplicate_source_slice_count += (
                    build_result.duplicate_source_slice_count
                )
                source_iteration_seconds += build_result.source_iteration_seconds
                cached_template = build_result.template
                if cached_template is None:
                    non_reshardable_paths.append(nested_path)
                    continue
                if template_key is not None:
                    plan_template_cache.put(template_key, cached_template)
            else:
                source_group_count += cached_template.source_group_count
                source_rank_candidate_count += (
                    cached_template.source_rank_candidate_count
                )
                duplicate_source_slice_count += (
                    cached_template.duplicate_source_slice_count
                )
            param_load_plans = [plan.bind(fqn) for plan in cached_template.plans]
            if param_load_plans:
                result[nested_path] = param_load_plans
            elif cached_template.target_has_elements:
                logger.warning(
                    f"No source DTensor shard intersects target shard for path: {nested_path}"
                )
                non_reshardable_paths.append(nested_path)

        # Apply deduplicate_source_chunks to minimize source ranks
        deduplication_started = perf_counter()
        if result:
            fqn_keyed_result, fqn_to_path = convert_nested_path_dict_to_fqn(result)
            optimized_str_result, _selected_ranks = deduplicate_source_chunks(
                fqn_keyed_result
            )
            result = {
                fqn_to_path[fqn]: plans for fqn, plans in optimized_str_result.items()
            }
        deduplication_seconds = perf_counter() - deduplication_started

        cache_metrics = geometry_cache.metrics
        logger.info(
            "DefaultResharder plan metrics target_paths=%d source_groups=%d "
            "source_rank_candidates=%d source_rank_scans=%d "
            "duplicate_source_slices=%d load_plans=%d "
            "geometry_cache_hits=%d geometry_cache_misses=%d "
            "geometry_cache_evictions=%d geometry_fast_paths=%d "
            "geometry_fallbacks=%d placement_cache_hits=%d "
            "placement_cache_misses=%d placement_cache_evictions=%d "
            "plan_template_cache_hits=%d plan_template_cache_misses=%d "
            "plan_template_cache_evictions=%d plan_template_cache_entries=%d "
            "geometry_cache_entries=%d placement_cache_entries=%d "
            "source_iteration_seconds=%.6f deduplication_seconds=%.6f "
            "total_seconds=%.6f",
            len(target_metadata),
            source_group_count,
            source_rank_candidate_count,
            source_rank_scan_count,
            duplicate_source_slice_count,
            sum(len(plans) for plans in result.values()),
            cache_metrics.geometry_hits,
            cache_metrics.geometry_misses,
            cache_metrics.geometry_evictions,
            cache_metrics.geometry_fast_paths,
            cache_metrics.geometry_fallbacks,
            cache_metrics.placement_hits,
            cache_metrics.placement_misses,
            cache_metrics.placement_evictions,
            cache_metrics.plan_template_hits,
            cache_metrics.plan_template_misses,
            cache_metrics.plan_template_evictions,
            plan_template_cache.entry_count,
            geometry_cache.geometry_entry_count,
            geometry_cache.placement_entry_count,
            source_iteration_seconds,
            deduplication_seconds,
            perf_counter() - planning_started,
        )

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
        1. Read checkpoint metadata and the storage span needed by each load plan.
        2. Fall back to a full checkpoint load for unsupported archive features.
        3. Copy the staged source data into target tensors.

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
            staged = self._read_source_slices(
                file_path,
                item_key,
                rank_plans,
                storage,
            )

            for nested_path, lp, src_data in staged:
                target_tensor = _unwrap_dtensor(target_by_path[nested_path])
                tgt_slice = tuple(slice(o, o + s) for o, s in zip(lp.offsets, lp.sizes))
                target_tensor[tgt_slice].copy_(src_data)

    def _read_source_slices(
        self,
        file_path: Path,
        item_key: str,
        rank_plans: list[tuple[NestedPath, LoadPlan]],
        storage: Storage,
    ) -> list[tuple[NestedPath, LoadPlan, torch.Tensor]]:
        """Read the source data every plan needs from one checkpoint file."""
        try:
            with storage.stream_read(
                file_path,
                ReadArgs(pre_read_full_file=False),
            ) as stream:
                with FakeTensorMode():
                    metadata = torch.load(
                        stream,  # type: ignore[arg-type]
                        map_location="cpu",
                        weights_only=False,
                    )
                flattened = _flatten_state_dict(item_key, metadata)
                return [
                    (
                        path,
                        plan,
                        _read_source_tensor_slice(
                            stream, flattened[plan.src_fqn], plan
                        ),
                    )
                    for path, plan in rank_plans
                ]
        except NotImplementedError as error:
            logger.warning(
                "Offset reads unavailable for %s, reading it in full: %s",
                file_path,
                error,
            )

        with storage.stream_read(file_path) as stream:
            loaded_data = torch.load(
                stream,  # type: ignore[arg-type]
                map_location="cpu",
                weights_only=False,
            )
        flattened = _flatten_state_dict(item_key, loaded_data)
        return [
            (
                path,
                plan,
                _slice_source_tensor(_unwrap_dtensor(flattened[plan.src_fqn]), plan),
            )
            for path, plan in rank_plans
        ]
