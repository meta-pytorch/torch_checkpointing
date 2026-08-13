# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for checkpoint consolidation.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import struct
import threading
import time
import uuid
from collections import deque
from collections.abc import (
    Callable,
    Container,
    Generator,
    Iterable,
    Iterator,
    Mapping,
)
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

import safetensors.torch as safetensors_torch
import torch
from torch import distributed as dist
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    _write_sub_tensor_to_file_optimized,
)
from torch.distributed.checkpoint._hf_utils import (
    _get_safetensors_file_metadata,
    DATA_OFFSETS_KEY,
    DTYPE_KEY,
    SHAPE_KEY,
)

from ..checkpoint_layout import LayoutInfo, SafetensorsSerialization
from ..distributed_metadata import (
    DistributedItemMetadata,
    GlobalObjectMetadata,
    load_distributed_metadata,
)
from ..dtensor_metadata import DTensorShardingMetadata
from ..dtensor_resharder import compute_local_shard_info
from ..resharding_utils import get_fqn_from_nested_path
from ..storage.base_storage import ReadArgs, Storage, StorageConfig
from ..storage.filesystem import LocalFileSystemStorageConfig
from ..types import NestedPath

logger: logging.Logger = logging.getLogger(__name__)

# Cap on concurrent readers. Reads block, so extra threads mainly hide
# per-request latency rather than add bandwidth. Memory does not scale with this
# value -- the budget below bounds it independently -- so the cap only limits
# open file handles and scheduling overhead.
_MAX_READER_THREADS = 16

# Ceiling on bytes of consolidated tensors held for read-ahead at once. Fixed
# rather than derived from the host: available-memory probes report the host and
# not the container, and the local world size counts training processes rather
# than the ranks doing consolidation work.
_READER_MEMORY_BUDGET_BYTES = int(
    os.environ.get("HF_CONSOLIDATION_MEMORY_BUDGET_BYTES", 8 * 1024 * 1024 * 1024)
)
_Item = TypeVar("_Item", str, int)


@contextmanager
def _atomic_stream_write(storage: Storage, path: Path) -> Iterator[Any]:
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with storage.stream_write(temporary_path) as stream:
            yield stream
        storage.rename(temporary_path, path)
    except BaseException:
        try:
            if storage.exists(temporary_path):
                storage.delete(temporary_path)
        except Exception:
            logger.exception("Failed to clean up temporary file %s", temporary_path)
        raise


@dataclass(frozen=True)
class HFSafetensorByteAddress:
    file_path: str
    start_byte_offset: int
    end_byte_offset: int

    @property
    def num_bytes(self) -> int:
        return self.end_byte_offset - self.start_byte_offset


@dataclass(frozen=True)
class TensorSlice:
    source_rank: int
    local_shape: tuple[int, ...]
    torch_dtype: torch.dtype
    byte_address: HFSafetensorByteAddress
    global_shape: tuple[int, ...] | None = None
    global_offsets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        assert (self.global_shape is None) == (self.global_offsets is None), (
            "TensorSlice global_shape and global_offsets must either both be set "
            "or both be unset"
        )
        if self.global_shape is None or self.global_offsets is None:
            return
        assert (
            len(self.global_shape) == len(self.global_offsets) == len(self.local_shape)
        ), (
            f"Invalid tensor slice: {self.global_shape=}, {self.global_offsets=}, "
            f"{self.local_shape=}"
        )
        for dimension, (offset, local_size, global_size) in enumerate(
            zip(self.global_offsets, self.local_shape, self.global_shape)
        ):
            assert 0 <= offset <= offset + local_size <= global_size, (
                f"Tensor slice does not fit in dimension {dimension}: "
                f"{offset=}, {local_size=}, {global_size=}"
            )

    def with_global_layout(
        self,
        global_shape: tuple[int, ...],
        global_offsets: tuple[int, ...],
        *,
        local_shape: tuple[int, ...] | None = None,
        torch_dtype: torch.dtype | None = None,
    ) -> "TensorSlice":
        return replace(
            self,
            local_shape=self.local_shape if local_shape is None else local_shape,
            torch_dtype=self.torch_dtype if torch_dtype is None else torch_dtype,
            global_shape=global_shape,
            global_offsets=global_offsets,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        assert self.global_shape is not None, "TensorSlice global layout is unresolved"
        return self.global_shape

    @property
    def offsets(self) -> tuple[int, ...]:
        assert self.global_offsets is not None, (
            "TensorSlice global layout is unresolved"
        )
        return self.global_offsets

    @property
    def sizes(self) -> tuple[int, ...]:
        return self.local_shape

    @property
    def slices(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (offset, offset + size)
            for offset, size in zip(self.offsets, self.local_shape)
        )

    @property
    def dtype_size(self) -> int:
        return torch._utils._element_size(self.torch_dtype)

    @property
    def contiguous_byte_range(self) -> tuple[int, int] | None:
        num_elements = math.prod(self.sizes)
        if num_elements == 0:
            return (0, 0)

        strides = [
            math.prod(self.shape[index + 1 :]) for index in range(len(self.shape))
        ]
        start_element = sum(
            offset * stride for offset, stride in zip(self.offsets, strides)
        )
        last_element = sum(
            (offset + size - 1) * stride
            for offset, size, stride in zip(self.offsets, self.sizes, strides)
        )
        if last_element - start_element + 1 != num_elements:
            return None

        start_byte = start_element * self.dtype_size
        return start_byte, start_byte + num_elements * self.dtype_size


@dataclass(frozen=True)
class SafetensorsFileMetadata:
    file_path: str
    tensors: dict[str, TensorSlice]

    @classmethod
    def from_file(
        cls,
        storage: Storage,
        file_path: str,
        source_rank: int,
    ) -> "SafetensorsFileMetadata":
        with storage.stream_read(
            Path(file_path),
            ReadArgs(pre_read_full_file=False, direct_io=False),
        ) as f:
            metadata, file_start_byte_offset = _get_safetensors_file_metadata(f)

        tensors: dict[str, TensorSlice] = {}
        for fqn, tensor_metadata in metadata.items():
            if fqn == "__metadata__":
                continue
            start, end = tensor_metadata[DATA_OFFSETS_KEY]
            dtype = tensor_metadata[DTYPE_KEY]
            try:
                torch_dtype = safetensors_torch._TYPES[dtype]
            except KeyError as e:
                raise ValueError(
                    f"Safetensors file {file_path!r} has unsupported dtype "
                    f"{dtype!r} for {fqn!r}"
                ) from e
            tensors[fqn] = TensorSlice(
                source_rank=source_rank,
                local_shape=tuple(tensor_metadata[SHAPE_KEY]),
                torch_dtype=torch_dtype,
                byte_address=HFSafetensorByteAddress(
                    file_path=file_path,
                    start_byte_offset=file_start_byte_offset + start,
                    end_byte_offset=file_start_byte_offset + end,
                ),
            )

        return cls(file_path=file_path, tensors=tensors)


def _distributed_rank_and_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _load_item_metadata(
    input_checkpoint_dir: str,
    item_key: str,
    storage: Storage,
) -> DistributedItemMetadata:
    distributed_metadata = load_distributed_metadata(input_checkpoint_dir, storage)
    if distributed_metadata is None:
        raise FileNotFoundError(
            f"No distributed metadata found in {input_checkpoint_dir}"
        )
    item_metadata = distributed_metadata.metadata.get(item_key)
    if item_metadata is None:
        raise ValueError(
            f"No checkpoint metadata found for item {item_key!r} in "
            f"{input_checkpoint_dir}"
        )
    if not any(
        layout_info is not None
        for layout_info in item_metadata.rank_to_layout_info.values()
    ):
        raise ValueError(
            f"No rank-local safetensors files for checkpoint item "
            f"{item_key!r} in {input_checkpoint_dir}"
        )
    return item_metadata


def _fqn_to_num_bytes_from_metadata(
    nested_path_to_metadata: Mapping[NestedPath, list[GlobalObjectMetadata]],
) -> dict[str, int]:
    """Consolidated tensor sizes, taken from ``metadata.pkl`` alone.

    Sizing without the safetensors headers is what lets output files be assigned
    before anything is resolved, so a rank only reads the headers for the
    tensors it was given rather than for the whole checkpoint.
    """
    result: dict[str, int] = {}
    for nested_path, groups in sorted(
        nested_path_to_metadata.items(),
        key=lambda item: get_fqn_from_nested_path(item[0]),
    ):
        fqn = get_fqn_from_nested_path(nested_path)
        if fqn in result:
            raise ValueError(
                "FQN collision detected while converting checkpoint metadata "
                f"NestedPath values for HF safetensors: {fqn!r}"
            )
        _validate_sharding_metadata_groups(nested_path, groups)
        sharding_metadata = groups[0].sharding_metadata
        assert isinstance(sharding_metadata, DTensorShardingMetadata)
        torch_dtype = _torch_dtype_from_checkpoint_metadata(sharding_metadata.dtype)
        result[fqn] = math.prod(
            sharding_metadata.global_shape
        ) * torch._utils._element_size(torch_dtype)
    return result


def _source_ranks_for_fqns(
    nested_path_to_metadata: Mapping[NestedPath, list[GlobalObjectMetadata]],
    fqns: Container[str],
) -> set[int]:
    """Source ranks that hold any part of ``fqns``, from metadata alone.

    Ranks that wrote no file are left in; reading their metadata skips them.
    """
    return {
        source_rank
        for nested_path, groups in nested_path_to_metadata.items()
        if get_fqn_from_nested_path(nested_path) in fqns
        for group in groups
        for source_rank in group.ranks
    }


def consolidate_hf_safetensors_checkpoint(
    input_checkpoint_dir: str,
    *,
    output_dir: str | None = None,
    fqn_to_index_mapping: dict[str, int] | None = None,
    item_key: str = "model",
    storage_config: StorageConfig | None = None,
) -> None:
    """
    Consolidate a torch_checkpointing safetensors checkpoint into HF shards.

    ``metadata.pkl`` defines the tensors, their global layouts, and their source
    files. Rank-local safetensors headers supply byte layouts for those tensors.
    Contiguous destination slices are read directly into their final positions.
    Tensor shapes and dtypes are cross-checked between the two formats. Metadata tensors must exist in the declared safetensors files, and
    header FQNs absent from metadata are rejected. By default, consolidated files are written alongside the input
    checkpoint and prefixed with ``item_key``. Output files assigned to the same
    rank are processed sequentially while their tensors are read concurrently.
    """
    storage = (
        storage_config if storage_config is not None else LocalFileSystemStorageConfig()
    ).create_storage()
    if output_dir is None:
        output_dir = input_checkpoint_dir
    rank, world_size = _distributed_rank_and_world_size()
    item_metadata = _load_item_metadata(input_checkpoint_dir, item_key, storage)
    fqn_to_num_bytes = _fqn_to_num_bytes_from_metadata(
        item_metadata.nested_path_to_metadata
    )
    fqn_to_index_mapping = _validate_fqn_to_index_mapping(
        fqn_to_index_mapping,
        fqn_to_num_bytes.keys(),
    )
    if not fqn_to_num_bytes:
        return

    if fqn_to_index_mapping is None:
        fqn_to_index_mapping = _default_fqn_to_index_mapping(
            fqn_to_num_bytes,
            world_size,
        )

    output_file_sizes_by_index: dict[int, int] = {}
    for fqn, index in fqn_to_index_mapping.items():
        output_file_sizes_by_index[index] = (
            output_file_sizes_by_index.get(index, 0) + fqn_to_num_bytes[fqn]
        )
    # Grouping FQNs into files and assigning complete files to ranks are
    # separate: caller-provided mappings may be sparse, uneven, or contain
    # more output files than ranks.
    output_index_owners = _assign_items_to_owners_by_size(
        output_file_sizes_by_index,
        num_owners=world_size,
    )
    use_distributed = any(owner != 0 for owner in output_index_owners.values())
    logger.info(
        "HF safetensors consolidation mode=%s output_files=%s world_size=%s",
        "distributed" if use_distributed else "rank_zero",
        len(output_file_sizes_by_index),
        world_size,
    )
    max_index = max(output_file_sizes_by_index)
    output_file_by_index = {
        index: os.path.join(
            output_dir,
            _item_output_file_name(item_key, index, max_index),
        )
        for index in output_file_sizes_by_index
    }
    (
        participates,
        consolidation_process_group,
        created_consolidation_process_group,
    ) = _create_consolidation_process_group(
        rank,
        world_size,
        output_index_owners.values(),
        use_distributed,
    )
    if not participates:
        return

    try:
        assigned_indices = {
            index for index, owner in output_index_owners.items() if owner == rank
        }
        assigned_fqn_to_index_mapping = {
            fqn: index
            for fqn, index in fqn_to_index_mapping.items()
            if index in assigned_indices
        }
        # Only the assigned tensors are resolved, so only their source files are
        # opened. Reading every rank's header here would scale with the whole
        # checkpoint rather than with this rank's share of it.
        assigned_fqns = set(assigned_fqn_to_index_mapping)
        file_metadata_by_rank = _read_safetensors_file_metadata_by_rank(
            input_checkpoint_dir,
            {
                source_rank: item_metadata.rank_to_layout_info[source_rank]
                for source_rank in _source_ranks_for_fqns(
                    item_metadata.nested_path_to_metadata,
                    assigned_fqns,
                )
            },
            storage,
        )
        _validate_header_fqns_are_known(
            file_metadata_by_rank.values(),
            fqn_to_num_bytes.keys(),
        )
        _validate_input_output_paths_are_disjoint(
            input_file_paths=(
                file_metadata.file_path
                for file_metadata in file_metadata_by_rank.values()
            ),
            output_file_paths=output_file_by_index.values(),
        )
        fqn_to_tensor_slices = _build_fqn_to_tensor_slices(
            nested_path_to_metadata={
                nested_path: groups
                for nested_path, groups in (
                    item_metadata.nested_path_to_metadata.items()
                )
                if get_fqn_from_nested_path(nested_path) in assigned_fqns
            },
            rank_to_layout_info=item_metadata.rank_to_layout_info,
            file_metadata_by_rank=file_metadata_by_rank,
        )
        tensor_slices_by_index: dict[int, dict[str, list[TensorSlice]]] = {}
        for fqn, index in assigned_fqn_to_index_mapping.items():
            tensor_slices_by_index.setdefault(index, {})[fqn] = fqn_to_tensor_slices[
                fqn
            ]
        logger.info(
            "Rank %s/%s assigned HF safetensors consolidation output_files=%s/%s "
            "fqns=%s/%s bytes=%s/%s indices=%s",
            rank,
            world_size,
            len(tensor_slices_by_index),
            len(output_file_by_index),
            len(assigned_fqn_to_index_mapping),
            len(fqn_to_index_mapping),
            sum(fqn_to_num_bytes[fqn] for fqn in assigned_fqn_to_index_mapping),
            sum(fqn_to_num_bytes.values()),
            sorted(assigned_indices),
        )

        output_path = Path(output_dir)
        if rank == 0 and not storage.exists(output_path):
            storage.mkdir(output_path)
        if use_distributed:
            assert consolidation_process_group is not None
            dist.barrier(group=consolidation_process_group)
        _write_and_finalize_consolidated_output_files(
            tensor_slices_by_index=tensor_slices_by_index,
            fqn_to_index_mapping=fqn_to_index_mapping,
            fqn_to_num_bytes=fqn_to_num_bytes,
            output_file_by_index=output_file_by_index,
            output_dir=output_dir,
            storage=storage,
            rank=rank,
            use_distributed=use_distributed,
            consolidation_process_group=consolidation_process_group,
            item_key=item_key,
        )
    finally:
        if created_consolidation_process_group:
            assert consolidation_process_group is not None
            dist.destroy_process_group(consolidation_process_group)


def _read_safetensors_file_metadata_by_rank(
    input_checkpoint_dir: str,
    rank_to_layout_info: Mapping[int, LayoutInfo | None],
    storage: Storage,
) -> dict[int, SafetensorsFileMetadata]:
    result: dict[int, SafetensorsFileMetadata] = {}
    for rank, layout_info in sorted(rank_to_layout_info.items()):
        if layout_info is None:
            continue
        serialization_format = layout_info.serialization_format
        if not isinstance(serialization_format, SafetensorsSerialization):
            raise ValueError(
                "HF safetensors consolidation requires SafetensorsSerialization "
                f"layouts, but rank {rank} uses "
                f"{serialization_format.__class__.__name__!r}"
            )
        file_path = os.path.join(input_checkpoint_dir, layout_info.file_path)
        result[rank] = SafetensorsFileMetadata.from_file(
            storage=storage,
            file_path=file_path,
            source_rank=rank,
        )
    return result


def _item_output_file_name(item_key: str, index: int, max_index: int) -> str:
    return f"{item_key}-{index:05d}-of-{max_index:05d}.safetensors"


def _assign_items_to_owners_by_size(
    item_sizes: dict[_Item, int],
    num_owners: int,
) -> dict[_Item, int]:
    assert num_owners >= 1

    owner_loads = [0] * num_owners
    item_to_owner: dict[_Item, int] = {}
    for item, size in sorted(item_sizes.items(), key=lambda pair: (-pair[1], pair[0])):
        owner = min(range(num_owners), key=lambda i: (owner_loads[i], i))
        item_to_owner[item] = owner
        owner_loads[owner] += size

    return item_to_owner


def _default_fqn_to_index_mapping(
    fqn_to_num_bytes: dict[str, int],
    world_size: int,
) -> dict[str, int]:
    num_output_files = min(world_size, len(fqn_to_num_bytes))
    fqn_to_owner = _assign_items_to_owners_by_size(
        fqn_to_num_bytes,
        num_owners=num_output_files,
    )
    return {fqn: owner + 1 for fqn, owner in fqn_to_owner.items()}


def _create_consolidation_process_group(
    rank: int,
    world_size: int,
    output_index_owners: Iterable[int],
    use_distributed: bool,
) -> tuple[bool, dist.ProcessGroup | None, bool]:
    if not use_distributed:
        return rank == 0, None, False

    participating_ranks = sorted({0, *output_index_owners})
    if rank not in participating_ranks:
        return False, None, False
    if len(participating_ranks) == world_size:
        return True, dist.group.WORLD, False
    return (
        True,
        dist.new_group(
            ranks=participating_ranks,
            use_local_synchronization=True,
            group_desc="hf_safetensors_consolidation",
        ),
        True,
    )


def _write_and_finalize_consolidated_output_files(
    tensor_slices_by_index: dict[int, dict[str, list[TensorSlice]]],
    fqn_to_index_mapping: dict[str, int],
    fqn_to_num_bytes: dict[str, int],
    output_file_by_index: dict[int, str],
    output_dir: str,
    storage: Storage,
    rank: int,
    use_distributed: bool,
    consolidation_process_group: dist.ProcessGroup | None,
    item_key: str,
) -> None:
    _write_assigned_output_files(
        tensor_slices_by_index,
        output_file_by_index,
        storage,
    )
    if use_distributed:
        assert consolidation_process_group is not None
        dist.barrier(group=consolidation_process_group)
    if rank == 0:
        _write_overall_hf_index_file(
            output_dir,
            fqn_to_index_mapping,
            fqn_to_num_bytes,
            output_file_by_index,
            storage,
            item_key=item_key,
        )
    if use_distributed:
        assert consolidation_process_group is not None
        dist.barrier(group=consolidation_process_group)


def _build_fqn_to_tensor_slices(
    nested_path_to_metadata: Mapping[
        NestedPath,
        list[GlobalObjectMetadata],
    ],
    rank_to_layout_info: Mapping[int, LayoutInfo | None],
    file_metadata_by_rank: Mapping[int, SafetensorsFileMetadata],
) -> dict[str, list[TensorSlice]]:
    result: dict[str, list[TensorSlice]] = {}
    for nested_path, groups in sorted(
        nested_path_to_metadata.items(),
        key=lambda item: get_fqn_from_nested_path(item[0]),
    ):
        fqn = get_fqn_from_nested_path(nested_path)
        if fqn in result:
            raise ValueError(
                "FQN collision detected while converting checkpoint metadata "
                f"NestedPath values for HF safetensors: {fqn!r}"
            )
        result[fqn] = _resolve_distributed_tensor_slices(
            nested_path=nested_path,
            groups=groups,
            rank_to_layout_info=rank_to_layout_info,
            file_metadata_by_rank=file_metadata_by_rank,
        )

    return result


def _resolve_distributed_tensor_slices(
    nested_path: NestedPath,
    groups: list[GlobalObjectMetadata],
    rank_to_layout_info: Mapping[int, LayoutInfo | None],
    file_metadata_by_rank: Mapping[int, SafetensorsFileMetadata],
) -> list[TensorSlice]:
    fqn = get_fqn_from_nested_path(nested_path)
    _validate_sharding_metadata_groups(nested_path, groups)
    selected_slices: dict[tuple[tuple[int, ...], tuple[int, ...]], TensorSlice] = {}
    for group in groups:
        sharding_metadata = group.sharding_metadata
        assert isinstance(sharding_metadata, DTensorShardingMetadata)
        for source_rank in group.ranks:
            tensor_slice = _resolve_distributed_tensor_slice(
                fqn=fqn,
                source_rank=source_rank,
                sharding_metadata=sharding_metadata,
                rank_to_layout_info=rank_to_layout_info,
                file_metadata_by_rank=file_metadata_by_rank,
            )
            if tensor_slice is None:
                continue
            slice_key = (tensor_slice.offsets, tensor_slice.sizes)
            existing_slice = selected_slices.get(slice_key)
            if (
                existing_slice is None
                or tensor_slice.source_rank < existing_slice.source_rank
            ):
                selected_slices[slice_key] = tensor_slice

    if not selected_slices:
        raise ValueError(
            f"No written safetensors source slices found for metadata FQN {fqn!r}"
        )
    return sorted(
        selected_slices.values(),
        key=lambda tensor_slice: (tensor_slice.offsets, tensor_slice.source_rank),
    )


def _validate_sharding_metadata_groups(
    nested_path: NestedPath,
    groups: list[GlobalObjectMetadata],
) -> None:
    fqn = get_fqn_from_nested_path(nested_path)
    if not groups:
        raise ValueError(f"Missing sharding metadata for path {nested_path!r}")
    first_group = groups[0]
    if not first_group.ranks:
        raise ValueError(f"Missing source ranks for FQN {fqn!r}")
    first_sharding_metadata = first_group.sharding_metadata
    assert isinstance(first_sharding_metadata, DTensorShardingMetadata), (
        f"HF safetensors consolidation only supports DTensor metadata for "
        f"{nested_path!r}, got {first_sharding_metadata.__class__.__name__!r}"
    )
    for group in groups[1:]:
        if not group.ranks:
            raise ValueError(f"Missing source ranks for path {nested_path!r}")
        sharding_metadata = group.sharding_metadata
        assert isinstance(sharding_metadata, DTensorShardingMetadata), (
            f"HF safetensors consolidation only supports DTensor metadata for "
            f"{nested_path!r}, got {sharding_metadata.__class__.__name__!r}"
        )
        if (
            sharding_metadata.global_shape != first_sharding_metadata.global_shape
            or sharding_metadata.dtype != first_sharding_metadata.dtype
        ):
            raise ValueError(
                "HF safetensors consolidation requires all metadata groups for "
                f"path {nested_path!r} to describe the same global tensor, but "
                f"found {(sharding_metadata.global_shape, sharding_metadata.dtype)} "
                f"and {(first_sharding_metadata.global_shape, first_sharding_metadata.dtype)}"
            )


def _resolve_distributed_tensor_slice(
    fqn: str,
    source_rank: int,
    sharding_metadata: DTensorShardingMetadata,
    rank_to_layout_info: Mapping[int, LayoutInfo | None],
    file_metadata_by_rank: Mapping[int, SafetensorsFileMetadata],
) -> TensorSlice | None:
    if source_rank not in rank_to_layout_info:
        raise ValueError(
            f"HF safetensors consolidation has no file layout for source rank {source_rank}"
        )
    if rank_to_layout_info[source_rank] is None:
        return None
    file_metadata = file_metadata_by_rank[source_rank]
    try:
        written_slice = file_metadata.tensors[fqn]
    except KeyError as e:
        raise ValueError(
            f"Safetensors file {file_metadata.file_path!r} for rank {source_rank} "
            f"is missing checkpoint metadata FQN {fqn!r}"
        ) from e

    local_shape, global_offsets = compute_local_shard_info(
        sharding_metadata,
        source_rank,
    )
    torch_dtype = _torch_dtype_from_checkpoint_metadata(sharding_metadata.dtype)
    expected_local_shape = tuple(local_shape)
    _validate_written_tensor(
        fqn=fqn,
        tensor_slice=written_slice,
        expected_shape=expected_local_shape,
        expected_dtype=torch_dtype,
    )
    return written_slice.with_global_layout(
        global_shape=tuple(sharding_metadata.global_shape),
        global_offsets=tuple(global_offsets),
        local_shape=expected_local_shape,
        torch_dtype=torch_dtype,
    )


def _validate_written_tensor(
    fqn: str,
    tensor_slice: TensorSlice,
    expected_shape: tuple[int, ...],
    expected_dtype: torch.dtype,
) -> None:
    rank = tensor_slice.source_rank
    if tensor_slice.local_shape != expected_shape:
        raise ValueError(
            f"Cannot consolidate {fqn!r} from rank {rank}: safetensors shape "
            f"{tensor_slice.local_shape} does not match expected shape "
            f"{expected_shape}"
        )
    if tensor_slice.torch_dtype != expected_dtype:
        raise ValueError(
            f"Cannot consolidate {fqn!r} from rank {rank}: safetensors dtype "
            f"{_to_safetensors_dtype(tensor_slice.torch_dtype, fqn)!r} does not match "
            f"expected dtype {_to_safetensors_dtype(expected_dtype, fqn)!r}"
        )


def _validate_fqn_to_index_mapping(
    fqn_to_index_mapping: dict[str, int] | None,
    checkpoint_fqns: Iterable[str],
) -> dict[str, int] | None:
    checkpoint_fqns = set(checkpoint_fqns)

    if fqn_to_index_mapping is None:
        return None

    for fqn, index in fqn_to_index_mapping.items():
        if not isinstance(index, int):
            raise ValueError(
                f"HF safetensors shard index for {fqn!r} must be an integer, "
                f"got {type(index).__name__}"
            )
        if index < 1:
            raise ValueError(
                f"HF safetensors shard index for {fqn!r} must be >= 1, got {index}"
            )

    requested_fqns = set(fqn_to_index_mapping)
    missing_fqns = checkpoint_fqns - requested_fqns
    if missing_fqns:
        raise ValueError(
            "fqn_to_index_mapping is missing written checkpoint FQNs: "
            f"{sorted(missing_fqns)}"
        )
    # DCPv1's QuantizedHuggingFaceStorageReader can read auxiliary scale FQNs
    # while committing only the requested weight. See
    # fbcode/caffe2/torch/distributed/checkpoint/quantized_hf_storage.py.
    mapping_only_fqns = requested_fqns - checkpoint_fqns
    if mapping_only_fqns:
        logger.info(
            "Ignoring %s fqn_to_index_mapping FQNs absent from written "
            "safetensors headers; sample=%s",
            len(mapping_only_fqns),
            sorted(mapping_only_fqns)[:20],
        )
    return {
        fqn: index
        for fqn, index in fqn_to_index_mapping.items()
        if fqn in checkpoint_fqns
    }


def _to_safetensors_dtype(torch_dtype: torch.dtype, fqn: str) -> str:
    for safetensors_dtype, mapped_torch_dtype in safetensors_torch._TYPES.items():
        if mapped_torch_dtype == torch_dtype:
            return safetensors_dtype
    raise ValueError(
        f"Cannot consolidate {fqn!r}: checkpoint metadata dtype "
        f"{torch_dtype!r} is not supported by safetensors"
    )


def _torch_dtype_from_checkpoint_metadata(dtype: str) -> torch.dtype:
    torch_dtype = getattr(torch, dtype.removeprefix("torch."), None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"Unsupported checkpoint metadata dtype {dtype!r}")
    return torch_dtype


def _validate_header_fqns_are_known(
    file_metadata: Iterable[SafetensorsFileMetadata],
    checkpoint_fqns: Iterable[str],
) -> None:
    """Reject safetensors FQNs that ``metadata.pkl`` does not describe.

    Only the files this rank opened are checked; a rank does not read the files
    for tensors it was not assigned.
    """
    known_fqns = set(checkpoint_fqns)
    unexpected_fqns = sorted(
        {fqn for metadata in file_metadata for fqn in metadata.tensors} - known_fqns
    )
    if unexpected_fqns:
        raise ValueError(
            "Safetensors files contain FQNs absent from checkpoint metadata: "
            f"{unexpected_fqns}"
        )


def _validate_input_output_paths_are_disjoint(
    input_file_paths: Iterable[str],
    output_file_paths: Iterable[str],
) -> None:
    input_paths = {os.path.normpath(path) for path in input_file_paths}
    output_paths = {os.path.normpath(path) for path in output_file_paths}
    overlap = sorted(input_paths & output_paths)
    if overlap:
        raise ValueError(
            "HF safetensors consolidation output files must not overwrite "
            f"rank-local input safetensors files: {overlap}"
        )


def _prepare_metadata(
    tensor_slices_by_fqn: dict[str, list[TensorSlice]],
) -> bytes:
    metadata = {}
    current_offset = 0
    for fqn, tensor_slices in tensor_slices_by_fqn.items():
        tensor_slice = tensor_slices[0]
        end_offset = (
            current_offset + math.prod(tensor_slice.shape) * tensor_slice.dtype_size
        )
        metadata[fqn] = {
            SHAPE_KEY: list(tensor_slice.shape),
            DTYPE_KEY: _to_safetensors_dtype(tensor_slice.torch_dtype, fqn),
            DATA_OFFSETS_KEY: [current_offset, end_offset],
        }
        current_offset = end_offset

    json_bytes = json.dumps(metadata).encode("utf-8")
    metadata_bytes = struct.pack("<Q", len(json_bytes)) + json_bytes
    return metadata_bytes


def _write_assigned_output_files(
    tensor_slices_by_index: dict[int, dict[str, list[TensorSlice]]],
    output_file_by_index: dict[int, str],
    storage: Storage,
) -> None:
    if not tensor_slices_by_index:
        return

    work_count = sum(
        len(tensor_slices_by_fqn)
        for tensor_slices_by_fqn in tensor_slices_by_index.values()
    )
    assert work_count > 0
    # Memory is bounded by the budget, not by the thread count.
    reader_threads = min(work_count, _MAX_READER_THREADS)
    budget = _ReaderMemoryBudget(_READER_MEMORY_BUDGET_BYTES)
    handles, register_handles, close_handles = _reader_handle_pool()
    logger.info(
        "Writing assigned HF safetensors shards output_files=%s "
        "reader_threads=%s budget_bytes=%s",
        len(tensor_slices_by_index),
        reader_threads,
        _READER_MEMORY_BUDGET_BYTES,
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=reader_threads,
            initializer=register_handles,
        ) as read_executor:
            for index, tensor_slices_by_fqn in tensor_slices_by_index.items():
                _process_output_file(
                    output_file_by_index[index],
                    tensor_slices_by_fqn,
                    storage,
                    read_executor,
                    reader_threads,
                    handles,
                    budget,
                )
    finally:
        close_handles()
    if budget.exhausted_count:
        logger.info(
            "HF safetensors read-ahead was budget limited times=%s budget_bytes=%s",
            budget.exhausted_count,
            _READER_MEMORY_BUDGET_BYTES,
        )


class _ReaderMemoryBudget:
    """Bounds the consolidated-tensor bytes reader threads hold at once.

    Prefetch depth follows the tensors actually queued rather than an average
    over the whole checkpoint, so a run of large FQNs throttles itself instead
    of multiplying a small mean by the thread count. Reserving never blocks: the
    reader simply stops prefetching until the writer releases, which also keeps
    an ordered pipeline from deadlocking on a reservation held by a completed
    read further back in the queue.
    """

    def __init__(self, total_bytes: int) -> None:
        assert total_bytes > 0, f"Reader memory budget must be positive: {total_bytes}"
        self._available = total_bytes
        self._lock = threading.Lock()
        self._exhausted_count = 0

    def try_reserve(self, num_bytes: int, *, force: bool) -> bool:
        """Reserve ``num_bytes`` if the budget allows.

        ``force`` admits the read regardless, and is used for the first
        outstanding read so a tensor larger than the whole budget still makes
        progress.
        """
        with self._lock:
            if not force and num_bytes > self._available:
                self._exhausted_count += 1
                return False
            self._available -= num_bytes
            return True

    def release(self, num_bytes: int) -> None:
        with self._lock:
            self._available += num_bytes

    @property
    def exhausted_count(self) -> int:
        """Times prefetch was declined for want of budget."""
        with self._lock:
            return self._exhausted_count


def _tensor_memory_bytes(tensor_slices: list[TensorSlice]) -> int:
    """Peak bytes to consolidate one tensor.

    The output buffer, plus at most one staged source slice: slices are read one
    at a time and each is released before the next, so this is not multiplied by
    the slice count.
    """
    tensor_slice = tensor_slices[0]
    full_tensor_bytes = math.prod(tensor_slice.shape) * tensor_slice.dtype_size
    largest_buffered_source_bytes = max(
        (
            source_slice.byte_address.num_bytes
            for source_slice in tensor_slices
            if source_slice.contiguous_byte_range is None
        ),
        default=0,
    )
    return full_tensor_bytes + largest_buffered_source_bytes


def _process_output_file(
    output_file: str,
    tensor_slices_by_fqn: dict[str, list[TensorSlice]],
    storage: Storage,
    read_executor: concurrent.futures.ThreadPoolExecutor,
    reader_threads: int,
    handles: _ReaderHandles,
    budget: _ReaderMemoryBudget,
) -> None:
    metadata_bytes = _prepare_metadata(tensor_slices_by_fqn)
    total_tensor_bytes = sum(
        math.prod(tensor_slices[0].shape) * tensor_slices[0].dtype_size
        for tensor_slices in tensor_slices_by_fqn.values()
    )
    start_time = time.monotonic()
    logger.info(
        "Writing consolidated HF safetensors shard output_file=%s tensors=%s bytes=%s",
        output_file,
        len(tensor_slices_by_fqn),
        total_tensor_bytes,
    )
    written_tensor_bytes = 0
    next_progress_percent = 25
    with _atomic_stream_write(storage, Path(output_file)) as output_stream:
        output_stream.write(metadata_bytes)
        with closing(
            _read_full_tensors_in_order(
                tensor_slices_by_fqn,
                storage,
                read_executor,
                reader_threads,
                handles,
                budget,
            )
        ) as full_tensors:
            for full_tensor_mv in full_tensors:
                tensor_bytes = full_tensor_mv.nbytes
                try:
                    output_stream.write(full_tensor_mv)
                finally:
                    del full_tensor_mv
                written_tensor_bytes += tensor_bytes
                while (
                    total_tensor_bytes > 0
                    and next_progress_percent <= 75
                    and written_tensor_bytes * 100
                    >= total_tensor_bytes * next_progress_percent
                ):
                    logger.info(
                        "HF safetensors shard write progress output_file=%s "
                        "progress=%s%% bytes=%s/%s elapsed_seconds=%.2f",
                        output_file,
                        next_progress_percent,
                        written_tensor_bytes,
                        total_tensor_bytes,
                        time.monotonic() - start_time,
                    )
                    next_progress_percent += 25
    logger.info(
        "Finished consolidated HF safetensors shard output_file=%s tensors=%s "
        "bytes=%s elapsed_seconds=%.2f",
        output_file,
        len(tensor_slices_by_fqn),
        total_tensor_bytes,
        time.monotonic() - start_time,
    )


def _read_full_tensors_in_order(
    tensor_slices_by_fqn: Mapping[str, list[TensorSlice]],
    storage: Storage,
    read_executor: concurrent.futures.ThreadPoolExecutor,
    reader_threads: int,
    handles: _ReaderHandles,
    budget: _ReaderMemoryBudget,
) -> Generator[memoryview, None, None]:
    """Prefetch tensors in metadata order, bounded by the reader memory budget."""
    work = list(tensor_slices_by_fqn.values())
    work_sizes = [_tensor_memory_bytes(tensor_slices) for tensor_slices in work]
    pending: deque[tuple[concurrent.futures.Future[memoryview], int]] = deque()
    next_index = 0

    try:
        while next_index < len(work) or pending:
            while next_index < len(work) and len(pending) < reader_threads:
                # The first outstanding read is always admitted so an oversized
                # tensor cannot stall; the rest wait for the writer to release.
                if not budget.try_reserve(work_sizes[next_index], force=not pending):
                    break
                pending.append(
                    (
                        read_executor.submit(
                            _read_full_tensor,
                            work[next_index],
                            storage,
                            handles,
                        ),
                        work_sizes[next_index],
                    )
                )
                next_index += 1

            future, reserved_bytes = pending.popleft()
            try:
                full_tensor_mv = future.result()
            except BaseException:
                budget.release(reserved_bytes)
                raise
            try:
                yield full_tensor_mv
            finally:
                del full_tensor_mv
                budget.release(reserved_bytes)
    finally:
        for future, reserved_bytes in pending:
            future.cancel()
            budget.release(reserved_bytes)
        if pending:
            concurrent.futures.wait([future for future, _ in pending])


def _allocate_full_tensor(
    tensor_slices: list[TensorSlice],
) -> memoryview:
    output_tensor_slice = tensor_slices[0]
    return memoryview(
        bytearray(math.prod(output_tensor_slice.shape) * output_tensor_slice.dtype_size)
    )


class _ReaderHandles(threading.local):
    """Source file handles owned by a single reader thread.

    A handle carries one file position, so concurrent seek+readinto on a shared
    handle would race. Keeping the cache thread-local preserves that safety
    while holding opens to threads x source files rather than reopening every
    source file for each tensor.
    """

    def __init__(self) -> None:
        self.by_path: dict[str, Any] = {}


def _reader_handle_pool() -> tuple[
    _ReaderHandles,
    Callable[[], None],
    Callable[[], None],
]:
    """Thread-local handle cache plus its pool initializer and teardown.

    ``threading.local`` gives no way to enumerate instances, so each worker
    registers its own cache on startup and the caller closes them all once the
    pool has shut down.
    """
    handles = _ReaderHandles()
    registered: list[dict[str, Any]] = []
    lock = threading.Lock()

    def register() -> None:
        with lock:
            registered.append(handles.by_path)

    def close_all() -> None:
        with lock:
            for by_path in registered:
                for handle in by_path.values():
                    handle.close()
                by_path.clear()
            registered.clear()

    return handles, register, close_all


def _read_full_tensor(
    tensor_slices: list[TensorSlice],
    storage: Storage,
    handles: _ReaderHandles,
) -> memoryview:
    full_tensor_mv = _allocate_full_tensor(tensor_slices)
    _read_full_tensor_into_from_storage(
        tensor_slices,
        storage,
        full_tensor_mv,
        handles,
    )
    return full_tensor_mv


def _read_full_tensor_into_from_storage(
    tensor_slices: list[TensorSlice],
    storage: Storage,
    full_tensor_mv: memoryview,
    handles: _ReaderHandles,
) -> None:
    output_tensor_slice = tensor_slices[0]
    if not output_tensor_slice.shape:
        assert len(tensor_slices) == 1, (
            "Scalar tensors require exactly one source slice"
        )
    for tensor_slice in tensor_slices:
        byte_address = tensor_slice.byte_address
        file_handle = handles.by_path.get(byte_address.file_path)
        if file_handle is None:
            # Safetensors offsets and Python buffers are not guaranteed to
            # meet O_DIRECT alignment requirements.
            file_handle = storage.stream_read(
                Path(byte_address.file_path),
                ReadArgs(pre_read_full_file=False, direct_io=False),
            )
            handles.by_path[byte_address.file_path] = file_handle
        destination_range = tensor_slice.contiguous_byte_range
        if destination_range is not None:
            start, end = destination_range
            _read_tensor_data_into(
                file_handle,
                byte_address,
                full_tensor_mv[start:end],
            )
            continue
        data_to_write = _read_tensor_data(
            file_handle,
            byte_address,
        )
        _write_sub_tensor_to_file_optimized(
            full_tensor_mv,
            data_to_write,
            output_tensor_slice.dtype_size,
            list(output_tensor_slice.shape),
            list(tensor_slice.offsets),
            list(tensor_slice.sizes),
        )
        del data_to_write


def _read_tensor_data_into(
    f: Any,
    byte_address: HFSafetensorByteAddress,
    destination: memoryview,
) -> None:
    if destination.nbytes != byte_address.num_bytes:
        raise ValueError(
            f"Destination has {destination.nbytes} bytes but source tensor "
            f"{byte_address.file_path!r} has {byte_address.num_bytes} bytes"
        )
    f.seek(byte_address.start_byte_offset)
    bytes_read = 0
    while bytes_read < byte_address.num_bytes:
        read_size = f.readinto(destination[bytes_read:])
        if not read_size:
            raise EOFError(
                f"Expected {byte_address.num_bytes} bytes from "
                f"{byte_address.file_path!r}, got {bytes_read}"
            )
        bytes_read += read_size


def _read_tensor_data(
    f: Any,
    byte_address: HFSafetensorByteAddress,
) -> bytearray:
    data = bytearray(byte_address.num_bytes)
    _read_tensor_data_into(f, byte_address, memoryview(data))
    return data


def _write_overall_hf_index_file(
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    fqn_to_num_bytes: dict[str, int],
    output_file_by_index: dict[int, str],
    storage: Storage,
    *,
    item_key: str = "model",
) -> None:
    weight_map = {
        fqn: os.path.basename(output_file_by_index[index])
        for fqn, index in fqn_to_index_mapping.items()
    }

    metadata_file_name = f"{item_key}.safetensors.index.json"
    metadata_path = Path(output_dir) / metadata_file_name
    with _atomic_stream_write(storage, metadata_path) as metadata_file:
        metadata_file.write(
            json.dumps(
                {
                    "metadata": {"total_size": sum(fqn_to_num_bytes.values())},
                    "weight_map": weight_map,
                },
                indent=2,
            ).encode("utf-8")
        )
