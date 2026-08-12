# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for checkpoint consolidation.
"""

from __future__ import annotations

import json
import logging
import math
import os
import struct
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import safetensors.torch as safetensors_torch
import torch
from torch import distributed as dist
from torch.distributed.checkpoint._consolidate_hf_safetensors import (
    _write_sub_tensor_to_file_optimized,
)
from torch.distributed.checkpoint._hf_utils import (
    _gen_file_name,
    _get_safetensors_file_metadata,
    _metadata_fn,
    DATA_OFFSETS_KEY,
    DTYPE_KEY,
    SHAPE_KEY,
)

from ..checkpoint_layout import LayoutInfo, SafetensorsSerialization
from ..distributed_metadata import (
    GlobalObjectMetadata,
    load_distributed_metadata,
)
from ..dtensor_metadata import DTensorShardingMetadata
from ..dtensor_resharder import _compute_local_shard_info
from ..resharding_utils import get_fqn_from_nested_path
from ..storage.base_storage import ReadArgs, Storage, StorageConfig
from ..storage.filesystem import LocalFileSystemStorageConfig
from ..types import NestedPath

logger: logging.Logger = logging.getLogger(__name__)


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

    def to_safetensors_dtype(self, fqn: str) -> str:
        return _to_safetensors_dtype(self.torch_dtype, fqn)


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


def consolidate_hf_safetensors_checkpoint(
    input_checkpoint_dir: str,
    *,
    output_dir: str | None = None,
    fqn_to_index_mapping: dict[str, int] | None = None,
    item_key: str = "model",
    validate_tensor_metadata: bool = True,
    storage_config: StorageConfig | None = None,
) -> None:
    """
    Consolidate a torch_checkpointing safetensors checkpoint into HF shards.

    Rank-local safetensors headers define the tensors that were actually written.
    ``metadata.pkl`` supplies global layouts for distributed tensors. Set
    ``validate_tensor_metadata=False`` to skip cross-checking tensor shapes,
    dtypes, and byte spans between the two formats. Metadata tensors must still
    exist in the written safetensors files. By default, consolidated files are
    written alongside the input checkpoint.
    """
    storage = (
        storage_config if storage_config is not None else LocalFileSystemStorageConfig()
    ).create_storage()
    if output_dir is None:
        output_dir = input_checkpoint_dir
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    # Sync checkpoint save completion is rank-local. Rank 0 writes metadata.pkl,
    # so wait for every rank's save path before reading it for consolidation.
    _barrier_if_distributed()
    distributed_metadata = load_distributed_metadata(input_checkpoint_dir, storage)
    if distributed_metadata is None:
        raise FileNotFoundError(
            f"No distributed metadata found in {input_checkpoint_dir}"
        )

    try:
        item_metadata = distributed_metadata.metadata[item_key]
    except KeyError as e:
        raise ValueError(
            f"No distributed metadata for checkpoint item {item_key!r} "
            f"in {input_checkpoint_dir}"
        ) from e
    file_metadata_by_rank = _read_safetensors_file_metadata_by_rank(
        input_checkpoint_dir,
        item_metadata.rank_to_layout_info,
        storage,
    )
    fqn_to_tensor_slices = _build_fqn_to_tensor_slices(
        nested_path_to_metadata=item_metadata.nested_path_to_metadata,
        rank_to_layout_info=item_metadata.rank_to_layout_info,
        file_metadata_by_rank=file_metadata_by_rank,
        validate_tensor_metadata=validate_tensor_metadata,
    )
    fqn_to_index_mapping = _validate_fqn_to_index_mapping(
        fqn_to_index_mapping,
        fqn_to_tensor_slices.keys(),
    )
    if not fqn_to_index_mapping:
        return

    fqn_to_num_bytes = {
        fqn: math.prod(tensor_slices[0].shape) * tensor_slices[0].dtype_size
        for fqn, tensor_slices in fqn_to_tensor_slices.items()
    }
    max_index = max(fqn_to_index_mapping.values())
    output_file_by_index = {
        index: os.path.join(output_dir, _gen_file_name(index, max_index))
        for index in set(fqn_to_index_mapping.values())
    }
    assigned_indices = {
        index for index in output_file_by_index if (index - 1) % world_size == rank
    }
    assigned_fqn_to_index_mapping = {
        fqn: index
        for fqn, index in fqn_to_index_mapping.items()
        if index in assigned_indices
    }
    _validate_input_output_paths_are_disjoint(
        input_file_paths=(
            file_metadata.file_path for file_metadata in file_metadata_by_rank.values()
        ),
        output_file_paths=output_file_by_index.values(),
    )
    tensor_slices_by_index: dict[int, dict[str, list[TensorSlice]]] = {}
    for fqn, index in assigned_fqn_to_index_mapping.items():
        tensor_slices_by_index.setdefault(index, {})[fqn] = fqn_to_tensor_slices[fqn]
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
    _barrier_if_distributed()
    _write_assigned_output_files(
        tensor_slices_by_index,
        output_file_by_index,
        storage,
    )

    _barrier_if_distributed()
    if rank == 0:
        _write_overall_hf_index_file(
            output_dir,
            fqn_to_index_mapping,
            fqn_to_num_bytes,
            output_file_by_index,
            storage,
        )
    _barrier_if_distributed()


def _barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()


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
        result[rank] = SafetensorsFileMetadata.from_file(
            storage=storage,
            file_path=os.path.join(input_checkpoint_dir, layout_info.file_path),
            source_rank=rank,
        )
    return result


def _build_fqn_to_tensor_slices(
    nested_path_to_metadata: Mapping[
        NestedPath,
        list[GlobalObjectMetadata],
    ],
    rank_to_layout_info: Mapping[int, LayoutInfo | None],
    file_metadata_by_rank: Mapping[int, SafetensorsFileMetadata],
    validate_tensor_metadata: bool,
) -> dict[str, list[TensorSlice]]:
    written_slices_by_fqn: dict[str, list[TensorSlice]] = {}
    for file_metadata in file_metadata_by_rank.values():
        for fqn, tensor_slice in file_metadata.tensors.items():
            written_slices_by_fqn.setdefault(fqn, []).append(tensor_slice)

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
            validate_tensor_metadata=validate_tensor_metadata,
        )

    for fqn in sorted(written_slices_by_fqn.keys() - result.keys()):
        written_slices = written_slices_by_fqn[fqn]
        assert written_slices
        representative = min(
            written_slices,
            key=lambda tensor_slice: tensor_slice.source_rank,
        )
        result[fqn] = [
            representative.with_global_layout(
                global_shape=representative.local_shape,
                global_offsets=(0,) * len(representative.local_shape),
            )
        ]

    return result


def _resolve_distributed_tensor_slices(
    nested_path: NestedPath,
    groups: list[GlobalObjectMetadata],
    rank_to_layout_info: Mapping[int, LayoutInfo | None],
    file_metadata_by_rank: Mapping[int, SafetensorsFileMetadata],
    validate_tensor_metadata: bool,
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
                validate_tensor_metadata=validate_tensor_metadata,
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
    validate_tensor_metadata: bool,
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

    local_shape, global_offsets = _compute_local_shard_info(
        sharding_metadata,
        source_rank,
    )
    torch_dtype = _torch_dtype_from_checkpoint_metadata(sharding_metadata.dtype)
    expected_local_shape = tuple(local_shape)
    if validate_tensor_metadata:
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
            f"{tensor_slice.to_safetensors_dtype(fqn)!r} does not match "
            f"expected dtype {_to_safetensors_dtype(expected_dtype, fqn)!r}"
        )
    expected_bytes = math.prod(expected_shape) * torch._utils._element_size(
        expected_dtype
    )
    if tensor_slice.byte_address.num_bytes != expected_bytes:
        raise ValueError(
            f"Cannot consolidate {fqn!r} from rank {rank}: safetensors byte span "
            f"{tensor_slice.byte_address.num_bytes} does not match expected byte "
            f"span {expected_bytes}"
        )


def _validate_fqn_to_index_mapping(
    fqn_to_index_mapping: dict[str, int] | None,
    checkpoint_fqns: Iterable[str],
) -> dict[str, int]:
    checkpoint_fqns = set(checkpoint_fqns)

    if fqn_to_index_mapping is None:
        return dict.fromkeys(sorted(checkpoint_fqns), 1)

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
            DTYPE_KEY: tensor_slice.to_safetensors_dtype(fqn),
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
    for index, tensor_slices_by_fqn in tensor_slices_by_index.items():
        _process_output_file(
            output_file_by_index[index],
            tensor_slices_by_fqn,
            storage,
        )


def _process_output_file(
    output_file: str,
    tensor_slices_by_fqn: dict[str, list[TensorSlice]],
    storage: Storage,
) -> None:
    metadata_bytes = _prepare_metadata(tensor_slices_by_fqn)
    file_handles = {}
    try:
        with _atomic_stream_write(storage, Path(output_file)) as output_stream:
            output_stream.write(metadata_bytes)
            for tensor_slices in tensor_slices_by_fqn.values():
                output_tensor_slice = tensor_slices[0]
                full_tensor_mv = memoryview(
                    bytearray(
                        math.prod(output_tensor_slice.shape)
                        * output_tensor_slice.dtype_size
                    )
                )

                for tensor_slice in tensor_slices:
                    byte_address = tensor_slice.byte_address
                    if byte_address.file_path not in file_handles:
                        # Safetensors offsets and Python buffers are not guaranteed
                        # to meet O_DIRECT alignment requirements.
                        file_handles[byte_address.file_path] = storage.stream_read(
                            Path(byte_address.file_path),
                            ReadArgs(pre_read_full_file=False, direct_io=False),
                        )
                    data_to_write = _read_tensor_data(
                        file_handles[byte_address.file_path],
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

                output_stream.write(full_tensor_mv)
    finally:
        for f in file_handles.values():
            f.close()


def _read_tensor_data(
    f: Any,
    byte_address: HFSafetensorByteAddress,
) -> bytearray:
    f.seek(byte_address.start_byte_offset)
    data = bytearray(byte_address.num_bytes)
    view = memoryview(data)
    bytes_read = 0
    while bytes_read < byte_address.num_bytes:
        read_size = f.readinto(view[bytes_read:])
        if not read_size:
            raise EOFError(
                f"Expected {byte_address.num_bytes} bytes from "
                f"{byte_address.file_path!r}, got {bytes_read}"
            )
        bytes_read += read_size
    return data


def _write_overall_hf_index_file(
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    fqn_to_num_bytes: dict[str, int],
    output_file_by_index: dict[int, str],
    storage: Storage,
) -> None:
    weight_map = {
        fqn: os.path.basename(output_file_by_index[index])
        for fqn, index in fqn_to_index_mapping.items()
    }

    metadata_path = Path(output_dir) / _metadata_fn
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
