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

import io
import logging
import os
import zipfile
from collections.abc import Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from enum import auto, Enum
from pathlib import Path
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

from .checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
    TorchSerialization,
)
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
    dedupe_aliased_targets,
    deduplicate_source_chunks,
    get_fqn_from_nested_path,
)
from .safetensors_metadata import SafetensorsFileMetadata
from .storage.base_storage import ReadArgs, Storage
from .storage.torch_serialization import load_torch_serialized_from_storage
from .types import CheckpointPath, NestedPath
from .walk_utils import walk_checkpoint_structure

logger: logging.Logger = logging.getLogger(__name__)

# Source files read concurrently per host when the resharder does not set
# ``file_read_workers``. Ranks on a host share its storage client and network
# link, so each local rank reads an equal share. Too few reads in flight leave
# the link idle while each waits on latency: filling it takes about host read
# bandwidth / one read's throughput. Clients that read ahead into a fixed cache
# start evicting each other past cache size / read-ahead window.
DEFAULT_READS_IN_FLIGHT_PER_HOST: int = 24
# Source files each rank reads concurrently when ``LOCAL_WORLD_SIZE`` is unset or
# not a positive integer, so the number of ranks sharing the host is unknown.
DEFAULT_FILE_READ_WORKERS: int = 8

__all__ = ["DefaultResharder", "ReshardingReadStrategy"]


def _default_file_read_workers() -> int:
    # Assumes the host runs only this job and every local rank is a trainer that
    # loads at the same time; set ``file_read_workers`` on the resharder otherwise.
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE", "")
    if not local_world_size.isdigit() or int(local_world_size) < 1:
        return DEFAULT_FILE_READ_WORKERS
    return max(1, DEFAULT_READS_IN_FLIGHT_PER_HOST // int(local_world_size))


class ReshardingReadStrategy(Enum):
    """How DefaultResharder reads source checkpoint files."""

    AUTO = auto()
    OFFSET = auto()
    FULL_FILE = auto()


def _read_exact(stream: io.RawIOBase, offset: int, buffer: memoryview) -> None:
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


def _validate_offset_read_archive(stream: io.RawIOBase) -> None:
    """Require a seekable torch archive whose records are stored uncompressed."""
    try:
        if not stream.seekable():
            raise NotImplementedError("Checkpoint stream is not seekable")
        with zipfile.ZipFile(stream) as archive:
            if any(
                member.compress_type != zipfile.ZIP_STORED
                for member in archive.infolist()
            ):
                raise NotImplementedError("Checkpoint archive contains compressed data")
        stream.seek(0)
    except (io.UnsupportedOperation, zipfile.BadZipFile) as error:
        raise NotImplementedError(
            "Checkpoint is not a seekable, uncompressed torch archive"
        ) from error


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


def _unwrap_dtensor(value: torch.Tensor) -> torch.Tensor:
    return value._local_tensor if isinstance(value, DTensor) else value


def _slice_source_tensor(source: torch.Tensor, load_plan: LoadPlan) -> torch.Tensor:
    # The full-file fallback does not pass through the offset-read validation.
    # Validate here because Python slicing silently clamps out-of-bounds ranges.
    _validate_source_slice_bounds(tuple(source.shape), load_plan)
    source_slice = tuple(
        slice(offset, offset + size)
        for offset, size in zip(load_plan.src_offsets, load_plan.src_sizes)
    )
    return source[source_slice]


def _validate_source_slice_bounds(
    source_shape: tuple[int, ...],
    load_plan: LoadPlan,
) -> None:
    if not (
        len(load_plan.src_offsets) == len(load_plan.src_sizes) == len(source_shape)
    ):
        raise ValueError(
            f"Load plan rank does not match source tensor {load_plan.src_fqn!r}"
        )

    for dimension, (source_size, offset, size) in enumerate(
        zip(source_shape, load_plan.src_offsets, load_plan.src_sizes)
    ):
        if source_size < 0 or offset < 0 or size < 0 or offset + size > source_size:
            raise ValueError(
                f"Source slice for {load_plan.src_fqn!r} is out of bounds in "
                f"dimension {dimension}: offset={offset}, size={size}, "
                f"source_size={source_size}"
            )


def _read_source_tensor_slice(
    stream: io.RawIOBase,
    source: torch.Tensor,
    load_plan: LoadPlan,
) -> torch.Tensor:
    source = _unwrap_dtensor(source)
    if not isinstance(source, FakeTensor):
        raise NotImplementedError(f"Source {load_plan.src_fqn!r} is not a plain tensor")
    if source.layout is not torch.strided:
        raise NotImplementedError(
            f"Source {load_plan.src_fqn!r} does not use a strided storage"
        )
    if source.is_quantized:
        raise NotImplementedError(f"Source {load_plan.src_fqn!r} is quantized")
    _validate_source_slice_bounds(tuple(source.shape), load_plan)
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
        # Uninitialized, unlike a bytearray, which would zero-fill memory that
        # the read overwrites anyway.
        packed = torch.empty(span * element_size, dtype=torch.uint8)
        _read_exact(
            stream,
            checkpoint_offset + first_element * element_size,
            memoryview(packed.numpy()),
        )
        result = packed.view(source.dtype).as_strided(load_plan.src_sizes, strides)

    if source.is_conj():
        result = result.conj()
    if source.is_neg():
        result = result._neg_view()

    return result


class DefaultResharder(Resharder):
    """Resharder for DTensor checkpoints.

    Uses DTensor's native placement APIs to compute shard geometry and perform
    resharding during checkpoint loading. Supports transitions between different
    Shard/Replicate placements and device mesh configurations.

    Args:
        read_strategy: Whether to use offset reads, full-file reads, or try
            offset reads and fall back to full-file reads when unsupported.
        file_read_workers: Number of source files read concurrently. None
            splits a per-host default across ``LOCAL_WORLD_SIZE`` ranks, or uses
            ``DEFAULT_FILE_READ_WORKERS`` if it is unset or invalid. Set 1 to read one file
            at a time.
    """

    def __init__(
        self,
        *,
        read_strategy: ReshardingReadStrategy = ReshardingReadStrategy.AUTO,
        file_read_workers: int | None = None,
    ) -> None:
        if file_read_workers is not None and file_read_workers < 1:
            raise ValueError(
                f"file_read_workers must be at least 1, got {file_read_workers}"
            )
        self._read_strategy = read_strategy
        self._file_read_workers = (
            _default_file_read_workers()
            if file_read_workers is None
            else file_read_workers
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
            self._execute_load_plans(
                source_path,
                source_metadata,
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
        source_path: Path,
        source_metadata: DistributedItemMetadata,
        item_key: str,
        nested_path_to_load_plans: dict[NestedPath, list[LoadPlan]],
        target: Any,
        storage: Storage,
    ) -> None:
        """Execute load plans by reading source files and copying data into target.

        Groups load plans by source rank, then reads and copies the planned
        source data into target tensors.

        Args:
            source_path: Base path to the source checkpoint directory.
            source_metadata: Source checkpoint metadata for this item.
            item_key: The checkpoint item key being loaded.
            nested_path_to_load_plans: Mapping from NestedPath to LoadPlans.
            target: Target dict-like structure to load data into.
            storage: Storage backend for reading checkpoint files.
        """
        target_by_path = _collect_leaf_values(item_key, target)
        _, nested_path_to_load_plans, aliased_paths = dedupe_aliased_targets(
            {
                path: _unwrap_dtensor(target_by_path[path])
                for path in nested_path_to_load_plans
            },
            nested_path_to_load_plans,
        )
        # Aliased targets share memory, so concurrent reads could race on it.
        # Read one file at a time whenever any are found.
        file_read_workers = 1 if aliased_paths else self._file_read_workers
        if aliased_paths:
            logger.warning(
                "Loading %d aliased target(s) once, from the last path that "
                "shares each tensor, reading one file at a time: %s",
                len(aliased_paths),
                ", ".join(
                    f"{get_fqn_from_nested_path(dropped)} -> "
                    f"{get_fqn_from_nested_path(kept)}"
                    for dropped, kept in aliased_paths
                ),
            )

        # Group load plans by source rank
        plans_by_rank: dict[int, list[tuple[NestedPath, LoadPlan]]] = {}
        for nested_path, load_plans in nested_path_to_load_plans.items():
            for lp in load_plans:
                if lp.src_rank not in plans_by_rank:
                    plans_by_rank[lp.src_rank] = []
                plans_by_rank[lp.src_rank].append((nested_path, lp))

        source_layouts_by_rank = {
            src_rank: source_metadata.get_layout_info(src_rank, item_key)
            for src_rank in plans_by_rank
        }
        self._execute_load_plans_with_read_strategy(
            self._read_strategy,
            source_path,
            source_layouts_by_rank,
            item_key,
            plans_by_rank,
            target_by_path,
            storage,
            file_read_workers,
        )

    def _execute_load_plans_with_read_strategy(
        self,
        read_strategy: ReshardingReadStrategy,
        source_path: Path,
        source_layouts_by_rank: dict[int, LayoutInfo],
        item_key: str,
        plans_by_rank: dict[int, list[tuple[NestedPath, LoadPlan]]],
        target_by_path: dict[NestedPath, Any],
        storage: Storage,
        file_read_workers: int,
    ) -> None:
        # Worker threads start on the default stream. Copy on the caller's
        # current stream so the writes are ordered after its pending work on the
        # targets. Assumes accelerator targets are on the current device.
        on_accelerator = any(
            _unwrap_dtensor(target_by_path[path]).device.type != "cpu"
            for rank_plans in plans_by_rank.values()
            for path, _ in rank_plans
        )
        caller_stream = torch.accelerator.current_stream() if on_accelerator else None

        def copy(staged: Iterable[tuple[NestedPath, LoadPlan, torch.Tensor]]) -> None:
            if caller_stream is not None:
                torch.accelerator.set_stream(caller_stream)
            for nested_path, load_plan, src_data in staged:
                target_tensor = _unwrap_dtensor(target_by_path[nested_path])
                tgt_slice = tuple(
                    slice(o, o + s) for o, s in zip(load_plan.offsets, load_plan.sizes)
                )
                target_tensor[tgt_slice].copy_(src_data)

        def read_with_offsets(src_rank: int) -> bool:
            """Return False if the file must be read in full instead."""
            layout_info = source_layouts_by_rank[src_rank]
            file_path = source_path / layout_info.file_path
            try:
                # Close the generator even if copy() raises, so its stream is
                # closed then rather than whenever the generator is collected.
                with closing(
                    self._read_source_slices_with_offset_reads(
                        file_path,
                        layout_info,
                        src_rank,
                        item_key,
                        plans_by_rank[src_rank],
                        storage,
                    )
                ) as staged:
                    copy(staged)
                return True
            except NotImplementedError as error:
                if read_strategy is ReshardingReadStrategy.OFFSET:
                    raise
                logger.warning(
                    "Offset reads unavailable for %s; reading it in full: %s",
                    file_path,
                    error,
                )
                return False
            except Exception:
                # Only the first failure is raised to the caller; log every one.
                logger.exception("Reading %s failed", file_path)
                raise

        full_file_ranks = list(plans_by_rank)
        if read_strategy is not ReshardingReadStrategy.FULL_FILE:
            with ThreadPoolExecutor(
                max_workers=file_read_workers,
                thread_name_prefix="ckpt-read",
            ) as pool:
                try:
                    read = list(pool.map(read_with_offsets, plans_by_rank))
                except BaseException:
                    # The load has failed, so skip the files not yet started.
                    pool.shutdown(cancel_futures=True)
                    raise
            full_file_ranks = [
                src_rank for src_rank, done in zip(plans_by_rank, read) if not done
            ]

        # A full-file read holds the whole file in memory, so read these one at a
        # time to bound memory regardless of file_read_workers.
        for src_rank in full_file_ranks:
            layout_info = source_layouts_by_rank[src_rank]
            copy(
                self._read_source_slices_with_full_file_read(
                    source_path / layout_info.file_path,
                    layout_info,
                    item_key,
                    plans_by_rank[src_rank],
                    storage,
                )
            )

    def _read_source_slices_with_offset_reads(
        self,
        file_path: Path,
        layout_info: LayoutInfo,
        source_rank: int,
        item_key: str,
        rank_plans: list[tuple[NestedPath, LoadPlan]],
        storage: Storage,
    ) -> Generator[tuple[NestedPath, LoadPlan, torch.Tensor], None, None]:
        """Yield the source data each plan needs, one slice at a time.

        Each slice's buffer is released after the caller copies it, rather
        than every slice in the file being held until the last one is read.
        """
        # O_DIRECT bypasses the page cache. On FUSE mounts, page-cache reads
        # share a small per-mount limit on in-flight requests, which caps how
        # fast concurrent readers on one host can go.
        with storage.stream_read(
            file_path,
            ReadArgs(pre_read_full_file=False, direct_io=True),
        ) as stream:
            serialization_format = layout_info.serialization_format
            source_fqns = {load_plan.src_fqn for _, load_plan in rank_plans}
            if isinstance(serialization_format, TorchSerialization):
                _validate_offset_read_archive(stream)
                with FakeTensorMode():
                    metadata = torch.load(
                        stream,  # type: ignore[arg-type]
                        map_location="cpu",
                        weights_only=False,
                    )
                flattened = _flatten_state_dict(item_key, metadata)
                source_tensors = {
                    source_fqn: _unwrap_dtensor(flattened[source_fqn])
                    for source_fqn in source_fqns
                }
            elif isinstance(serialization_format, SafetensorsSerialization):
                source_tensors = SafetensorsFileMetadata.from_stream(
                    stream,
                    layout_info.file_path,
                    source_rank,
                ).as_fake_tensors(source_fqns)
            else:
                raise ValueError(
                    "Unsupported serialization format "
                    f"{type(serialization_format).__name__}"
                )
            for path, plan in rank_plans:
                yield (
                    path,
                    plan,
                    _read_source_tensor_slice(
                        stream,
                        source_tensors[plan.src_fqn],
                        plan,
                    ),
                )

    def _read_source_slices_with_full_file_read(
        self,
        file_path: Path,
        layout_info: LayoutInfo,
        item_key: str,
        rank_plans: list[tuple[NestedPath, LoadPlan]],
        storage: Storage,
    ) -> list[tuple[NestedPath, LoadPlan, torch.Tensor]]:
        """Read the source data every plan needs by loading the full file."""
        serialization_format = layout_info.serialization_format
        if isinstance(serialization_format, TorchSerialization):
            loaded_data = load_torch_serialized_from_storage(
                file_path,
                storage,
                map_location="cpu",
            )
        elif isinstance(serialization_format, SafetensorsSerialization):
            from safetensors.torch import load as safetensors_load

            loaded_data = safetensors_load(
                storage.read(file_path, ReadArgs(pre_read_full_file=False))
            )
        else:
            raise ValueError(
                "Unsupported serialization format "
                f"{type(serialization_format).__name__}"
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
