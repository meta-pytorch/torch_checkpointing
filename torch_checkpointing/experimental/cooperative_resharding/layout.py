# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pointer-free tensor byte layouts for cooperative checkpoint loading."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import prod
from typing import Any

import torch

from ...resharding import LoadPlan


@dataclass(frozen=True, order=True, slots=True)
class ByteRange:
    """A half-open byte range represented by an offset and length."""

    offset: int
    length: int

    def __post_init__(self) -> None:
        if self.offset < 0:
            raise ValueError("byte range offset must be non-negative")
        if self.length < 0:
            raise ValueError("byte range length must be non-negative")

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True, slots=True)
class RepeatedStrideBytePattern:
    """A compact row-major traversal of repeated strided byte ranges."""

    start_offset: int
    block_bytes: int
    repeat_counts: tuple[int, ...] = ()
    repeat_strides_bytes: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.block_bytes < 0:
            raise ValueError("block_bytes must be non-negative")
        if len(self.repeat_counts) != len(self.repeat_strides_bytes):
            raise ValueError("repeat counts and strides must have the same rank")
        if any(count <= 0 for count in self.repeat_counts):
            raise ValueError("repeat counts must be positive")
        if any(stride < 0 for stride in self.repeat_strides_bytes):
            raise ValueError("repeat strides must be non-negative")
        if self.block_bytes == 0 and self.repeat_counts:
            raise ValueError("empty patterns cannot contain repeat dimensions")

    @property
    def range_count(self) -> int:
        if self.block_bytes == 0:
            return 0
        return prod(self.repeat_counts) if self.repeat_counts else 1

    @property
    def dense_nbytes(self) -> int:
        return self.range_count * self.block_bytes

    @property
    def bounding_range(self) -> ByteRange:
        if self.block_bytes == 0:
            return ByteRange(self.start_offset, 0)
        last_start = self.start_offset + sum(
            (count - 1) * stride
            for count, stride in zip(
                self.repeat_counts,
                self.repeat_strides_bytes,
            )
        )
        return ByteRange(
            self.start_offset,
            last_start + self.block_bytes - self.start_offset,
        )

    def iter_ranges(self) -> Iterator[ByteRange]:
        """Expand ranges lazily in logical tensor order."""

        if self.block_bytes == 0:
            return
        yield from self._iter_ranges_at(0, self.start_offset)

    def _iter_ranges_at(self, dimension: int, offset: int) -> Iterator[ByteRange]:
        if dimension == len(self.repeat_counts):
            yield ByteRange(offset, self.block_bytes)
            return
        stride = self.repeat_strides_bytes[dimension]
        for index in range(self.repeat_counts[dimension]):
            yield from self._iter_ranges_at(
                dimension + 1,
                offset + index * stride,
            )


@dataclass(frozen=True, slots=True)
class SourceTensorMetadata:
    """Archive-neutral physical metadata for one checkpoint tensor."""

    fqn: str
    checkpoint_offset_bytes: int
    storage_offset_elements: int
    storage_nbytes: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    element_size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.fqn, str):
            raise ValueError("fqn must be a string")
        if self.checkpoint_offset_bytes < 0:
            raise ValueError("checkpoint_offset_bytes must be non-negative")
        if self.storage_offset_elements < 0:
            raise ValueError("storage_offset_elements must be non-negative")
        if self.storage_nbytes < 0:
            raise ValueError("storage_nbytes must be non-negative")
        if len(self.shape) != len(self.stride):
            raise ValueError("shape and stride must have the same rank")
        if any(size < 0 for size in self.shape):
            raise ValueError("shape dimensions must be non-negative")
        if any(stride < 0 for stride in self.stride):
            raise ValueError("negative source strides are unsupported")
        if self.element_size_bytes <= 0:
            raise ValueError("element_size_bytes must be positive")
        dtype = _resolve_torch_dtype(self.dtype)
        if torch.empty((), dtype=dtype).element_size() != self.element_size_bytes:
            raise ValueError(
                f"dtype {self.dtype!r} does not use {self.element_size_bytes} bytes"
            )
        self._validate_storage_bounds()

    @property
    def numel(self) -> int:
        return prod(self.shape)

    def _validate_storage_bounds(self) -> None:
        if self.numel == 0:
            return
        last_element = self.storage_offset_elements + sum(
            (size - 1) * stride for size, stride in zip(self.shape, self.stride)
        )
        required_nbytes = (last_element + 1) * self.element_size_bytes
        if required_nbytes > self.storage_nbytes:
            raise ValueError(
                f"metadata for {self.fqn!r} addresses {required_nbytes} bytes, "
                f"but its storage contains {self.storage_nbytes} bytes"
            )


@dataclass(frozen=True, slots=True)
class TensorReadTarget:
    """Serializable source and destination layout for one load-plan entry."""

    target_fqn: str
    source_rank: int
    source_fqn: str
    source_pattern: RepeatedStrideBytePattern
    destination_pattern: RepeatedStrideBytePattern
    source_tensor_shape: tuple[int, ...]
    source_slice_shape: tuple[int, ...]
    target_tensor_shape: tuple[int, ...]
    target_slice_shape: tuple[int, ...]
    source_dtype: str
    target_dtype: str
    source_element_size_bytes: int
    target_element_size_bytes: int
    transpose_dims: tuple[int, ...]
    target_device: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_fqn, str) or not isinstance(self.source_fqn, str):
            raise ValueError("source and target FQNs must be strings")
        if self.source_rank < 0:
            raise ValueError("source_rank must be non-negative")
        source_numel = prod(self.source_slice_shape)
        target_numel = prod(self.target_slice_shape)
        if source_numel != target_numel:
            raise ValueError("source and target slices must contain the same elements")
        if self.source_pattern.dense_nbytes != (
            source_numel * self.source_element_size_bytes
        ):
            raise ValueError("source byte pattern does not match its slice")
        if self.destination_pattern.dense_nbytes != (
            target_numel * self.target_element_size_bytes
        ):
            raise ValueError("destination byte pattern does not match its slice")

    @property
    def numel(self) -> int:
        return prod(self.source_slice_shape)

    @property
    def source_span(self) -> ByteRange:
        return self.source_pattern.bounding_range

    @property
    def requires_transform(self) -> bool:
        return bool(self.transpose_dims) or self.source_dtype != self.target_dtype


def checkpoint_offset_bytes(tensor: torch.Tensor) -> int:
    """Return the resolved checkpoint data offset carried by a meta tensor."""

    checkpoint_offset = getattr(tensor.untyped_storage(), "_checkpoint_offset", None)
    if checkpoint_offset is None:
        raise ValueError(
            "tensor storage has no resolved _checkpoint_offset; use a "
            "checkpoint reader that preserves serialized tensor offsets"
        )
    return int(checkpoint_offset)


def storage_offset_and_span_elements(
    offsets: Sequence[int],
    sizes: Sequence[int],
    strides: Sequence[int],
) -> tuple[int, int]:
    """Return the first storage element and bounding span for one slice."""

    if not (len(offsets) == len(sizes) == len(strides)):
        raise ValueError("offsets, sizes, and strides must have the same rank")
    start = sum(int(offset) * int(stride) for offset, stride in zip(offsets, strides))
    if any(int(size) == 0 for size in sizes):
        return start, 0
    end = sum(
        (int(offset) + int(size) - 1) * int(stride)
        for offset, size, stride in zip(offsets, sizes, strides)
    )
    return start, end - start + 1


def storage_byte_ranges(
    *,
    slice_offsets: Sequence[int],
    slice_sizes: Sequence[int],
    strides: Sequence[int],
    initial_storage_offset_bytes: int,
    element_size_bytes: int,
) -> tuple[ByteRange, ...]:
    """Return storage byte ranges in logical tensor order."""

    pattern = build_repeated_stride_pattern(
        offsets=slice_offsets,
        sizes=slice_sizes,
        strides=strides,
        element_size_bytes=element_size_bytes,
        initial_offset_bytes=initial_storage_offset_bytes,
    )
    return tuple(pattern.iter_ranges())


def build_repeated_stride_pattern(
    *,
    offsets: Sequence[int],
    sizes: Sequence[int],
    strides: Sequence[int],
    element_size_bytes: int,
    initial_offset_bytes: int = 0,
) -> RepeatedStrideBytePattern:
    """Build a compact byte pattern for a non-negative-stride tensor slice."""

    normalized_offsets = tuple(int(offset) for offset in offsets)
    normalized_sizes = tuple(int(size) for size in sizes)
    normalized_strides = tuple(int(stride) for stride in strides)
    _validate_pattern_inputs(
        normalized_offsets,
        normalized_sizes,
        normalized_strides,
        element_size_bytes,
        initial_offset_bytes,
    )
    start_offset = initial_offset_bytes + sum(
        offset * stride * element_size_bytes
        for offset, stride in zip(normalized_offsets, normalized_strides)
    )
    if any(size == 0 for size in normalized_sizes):
        return RepeatedStrideBytePattern(start_offset=start_offset, block_bytes=0)

    suffix_start, block_elements = _dense_suffix(
        normalized_sizes,
        normalized_strides,
    )
    repeat_dimensions = tuple(
        (size, stride * element_size_bytes)
        for size, stride in zip(
            normalized_sizes[:suffix_start],
            normalized_strides[:suffix_start],
        )
        if size > 1
    )
    return RepeatedStrideBytePattern(
        start_offset=start_offset,
        block_bytes=block_elements * element_size_bytes,
        repeat_counts=tuple(size for size, _ in repeat_dimensions),
        repeat_strides_bytes=tuple(stride for _, stride in repeat_dimensions),
    )


def resolve_tensor_read_target(
    target_fqn: str,
    plan: LoadPlan,
    source_metadata: SourceTensorMetadata,
    target_tensor: torch.Tensor,
) -> TensorReadTarget:
    """Resolve one plan into file-relative and storage-relative byte patterns."""

    if plan.src_fqn != source_metadata.fqn:
        raise ValueError(
            f"plan source {plan.src_fqn!r} does not match metadata "
            f"{source_metadata.fqn!r}"
        )
    _validate_source_plan(plan, source_metadata)
    _validate_target_plan(target_fqn, plan, target_tensor)
    _validate_shape_transform(target_fqn, plan)
    _validate_plan_source_type(target_fqn, plan, source_metadata)

    source_pattern = _source_pattern_from_plan(plan, source_metadata)
    target_element_size = int(target_tensor.element_size())
    destination_pattern = _destination_pattern_from_plan(plan, target_tensor)
    _validate_resolved_pattern_bounds(
        target_fqn,
        plan,
        source_metadata,
        target_tensor,
        source_pattern,
        destination_pattern,
    )
    return TensorReadTarget(
        target_fqn=target_fqn,
        source_rank=plan.src_rank,
        source_fqn=plan.src_fqn,
        source_pattern=source_pattern,
        destination_pattern=destination_pattern,
        source_tensor_shape=source_metadata.shape,
        source_slice_shape=tuple(plan.src_sizes),
        target_tensor_shape=tuple(int(size) for size in target_tensor.shape),
        target_slice_shape=tuple(plan.sizes),
        source_dtype=_canonical_dtype_name(source_metadata.dtype),
        target_dtype=_canonical_dtype_name(str(target_tensor.dtype)),
        source_element_size_bytes=source_metadata.element_size_bytes,
        target_element_size_bytes=target_element_size,
        transpose_dims=tuple(plan.transpose_dims),
        target_device=str(target_tensor.device),
    )


def resolve_tensor_read_targets(
    load_plan: Mapping[str, Sequence[LoadPlan]],
    source_metadata_by_rank: Mapping[int, Mapping[str, SourceTensorMetadata]],
    target_state_dict: Mapping[str, Any],
) -> tuple[TensorReadTarget, ...]:
    """Resolve and validate every load-plan entry without exposing pointers."""

    targets: list[TensorReadTarget] = []
    for target_fqn in sorted(load_plan):
        plans = load_plan[target_fqn]
        target_value = target_state_dict.get(target_fqn)
        if not isinstance(target_value, torch.Tensor):
            raise TypeError(f"target {target_fqn!r} is not a torch.Tensor")
        _validate_target_coverage(target_fqn, plans, target_value)
        for plan in plans:
            rank_metadata = source_metadata_by_rank.get(plan.src_rank)
            if rank_metadata is None:
                raise KeyError(f"source rank {plan.src_rank} has no metadata")
            source_metadata = rank_metadata.get(plan.src_fqn)
            if source_metadata is None:
                raise KeyError(
                    f"source rank {plan.src_rank} has no metadata for {plan.src_fqn!r}"
                )
            targets.append(
                resolve_tensor_read_target(
                    target_fqn,
                    plan,
                    source_metadata,
                    target_value,
                )
            )
    resolved = tuple(targets)
    _validate_disjoint_destination_storage(resolved, target_state_dict)
    return resolved


def _resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    if not isinstance(dtype_name, str) or not dtype_name:
        raise ValueError("dtype must be a non-empty string")
    aliases = {
        "byte": "uint8",
        "char": "int8",
        "double": "float64",
        "float": "float32",
        "half": "float16",
        "int": "int32",
        "long": "int64",
        "short": "int16",
    }
    name = aliases.get(
        dtype_name.removeprefix("torch."), dtype_name.removeprefix("torch.")
    )
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {dtype_name!r}")
    return dtype


def _canonical_dtype_name(dtype_name: str) -> str:
    return str(_resolve_torch_dtype(dtype_name))


def _validate_disjoint_destination_storage(
    targets: Sequence[TensorReadTarget],
    target_state_dict: Mapping[str, Any],
) -> None:
    targets_by_storage: dict[tuple[str, int], list[TensorReadTarget]] = {}
    for target in targets:
        tensor = target_state_dict[target.target_fqn]
        assert isinstance(tensor, torch.Tensor)
        storage_key = (str(tensor.device), int(tensor.untyped_storage().data_ptr()))
        targets_by_storage.setdefault(storage_key, []).append(target)
    for storage_targets in targets_by_storage.values():
        fqns = {target.target_fqn for target in storage_targets}
        if len(fqns) == 1:
            tensor = target_state_dict[next(iter(fqns))]
            assert isinstance(tensor, torch.Tensor)
            if _strides_prove_nonoverlap(tensor.shape, tensor.stride()):
                continue
        _validate_shared_storage_targets(storage_targets, target_state_dict)


def _validate_shared_storage_targets(
    targets: Sequence[TensorReadTarget],
    target_state_dict: Mapping[str, Any],
) -> None:
    intervals: list[tuple[int, int, str]] = []
    for target in targets:
        tensor = target_state_dict[target.target_fqn]
        assert isinstance(tensor, torch.Tensor)
        storage_base = int(tensor.untyped_storage().data_ptr())
        for byte_range in target.destination_pattern.iter_ranges():
            intervals.append(
                (
                    storage_base + byte_range.offset,
                    storage_base + byte_range.end,
                    target.target_fqn,
                )
            )
    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise ValueError(
                "cooperative load targets overlap in destination storage: "
                f"{previous[2]!r} and {current[2]!r}"
            )


def _strides_prove_nonoverlap(shape: Sequence[int], strides: Sequence[int]) -> bool:
    span = 1
    for stride, size in sorted(
        (int(stride), int(size)) for size, stride in zip(shape, strides) if size > 1
    ):
        if stride < span:
            return False
        span += (size - 1) * stride
    return True


def _source_pattern_from_plan(
    plan: LoadPlan,
    metadata: SourceTensorMetadata,
) -> RepeatedStrideBytePattern:
    return build_repeated_stride_pattern(
        offsets=plan.src_offsets,
        sizes=plan.src_sizes,
        strides=metadata.stride,
        element_size_bytes=metadata.element_size_bytes,
        initial_offset_bytes=(
            metadata.checkpoint_offset_bytes
            + metadata.storage_offset_elements * metadata.element_size_bytes
        ),
    )


def _destination_pattern_from_plan(
    plan: LoadPlan,
    target_tensor: torch.Tensor,
) -> RepeatedStrideBytePattern:
    element_size = int(target_tensor.element_size())
    return build_repeated_stride_pattern(
        offsets=plan.offsets,
        sizes=plan.sizes,
        strides=target_tensor.stride(),
        element_size_bytes=element_size,
        initial_offset_bytes=int(target_tensor.storage_offset()) * element_size,
    )


def _validate_resolved_pattern_bounds(
    target_fqn: str,
    plan: LoadPlan,
    metadata: SourceTensorMetadata,
    target_tensor: torch.Tensor,
    source_pattern: RepeatedStrideBytePattern,
    destination_pattern: RepeatedStrideBytePattern,
) -> None:
    _validate_pattern_bound(
        f"source tensor {plan.src_fqn!r}",
        source_pattern,
        metadata.checkpoint_offset_bytes + metadata.storage_nbytes,
    )
    _validate_pattern_bound(
        f"target tensor {target_fqn!r}",
        destination_pattern,
        int(target_tensor.untyped_storage().nbytes()),
    )


def _validate_pattern_inputs(
    offsets: tuple[int, ...],
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
    element_size_bytes: int,
    initial_offset_bytes: int,
) -> None:
    if not (len(offsets) == len(sizes) == len(strides)):
        raise ValueError("offsets, sizes, and strides must have the same rank")
    if any(offset < 0 for offset in offsets):
        raise ValueError("slice offsets must be non-negative")
    if any(size < 0 for size in sizes):
        raise ValueError("slice sizes must be non-negative")
    if any(stride < 0 for stride in strides):
        raise ValueError("negative strides are unsupported")
    if element_size_bytes <= 0:
        raise ValueError("element_size_bytes must be positive")
    if initial_offset_bytes < 0:
        raise ValueError("initial_offset_bytes must be non-negative")


def _dense_suffix(
    sizes: tuple[int, ...],
    strides: tuple[int, ...],
) -> tuple[int, int]:
    suffix_start = len(sizes)
    block_elements = 1
    expected_stride = 1
    for dimension in range(len(sizes) - 1, -1, -1):
        size = sizes[dimension]
        stride = strides[dimension]
        if size == 1:
            suffix_start = dimension
            continue
        if stride != expected_stride:
            break
        suffix_start = dimension
        block_elements *= size
        expected_stride *= size
    return suffix_start, block_elements


def _validate_source_plan(
    plan: LoadPlan,
    metadata: SourceTensorMetadata,
) -> None:
    _validate_slice_bounds(
        f"source tensor {plan.src_fqn!r}",
        plan.src_offsets,
        plan.src_sizes,
        metadata.shape,
    )


def _validate_target_plan(
    target_fqn: str,
    plan: LoadPlan,
    target_tensor: torch.Tensor,
) -> None:
    if target_tensor.layout != torch.strided:
        raise ValueError(
            f"target tensor {target_fqn!r} has unsupported layout "
            f"{target_tensor.layout}"
        )
    target_shape = tuple(int(size) for size in target_tensor.shape)
    _validate_slice_bounds(
        f"target tensor {target_fqn!r}",
        plan.offsets,
        plan.sizes,
        target_shape,
    )
    for size, stride in zip(target_shape, target_tensor.stride()):
        if size > 1 and stride == 0:
            raise ValueError(f"target tensor {target_fqn!r} has overlapping storage")


def _validate_target_coverage(
    target_fqn: str,
    plans: Sequence[LoadPlan],
    target_tensor: torch.Tensor,
) -> None:
    for plan in plans:
        _validate_target_plan(target_fqn, plan, target_tensor)
    _validate_nonoverlapping_destination_slices(target_fqn, plans)
    expected_numel = int(target_tensor.numel())
    covered_numel = sum(prod(plan.sizes) for plan in plans)
    if covered_numel != expected_numel:
        raise ValueError(
            f"target {target_fqn!r} load plans cover {covered_numel} elements, "
            f"expected {expected_numel}"
        )


def _validate_slice_bounds(
    label: str,
    offsets: Sequence[int],
    sizes: Sequence[int],
    shape: Sequence[int],
) -> None:
    if not (len(offsets) == len(sizes) == len(shape)):
        raise ValueError(
            f"{label} rank mismatch: offsets={len(offsets)}, "
            f"sizes={len(sizes)}, shape={len(shape)}"
        )
    for dimension, (offset, size, extent) in enumerate(zip(offsets, sizes, shape)):
        if offset < 0 or size < 0 or offset + size > extent:
            raise ValueError(
                f"{label} slice is out of bounds at dimension {dimension}: "
                f"offset={offset}, size={size}, extent={extent}"
            )


def _validate_shape_transform(target_fqn: str, plan: LoadPlan) -> None:
    source_shape = tuple(plan.src_sizes)
    target_shape = tuple(plan.sizes)
    transpose_dims = tuple(plan.transpose_dims)
    if transpose_dims:
        if sorted(transpose_dims) != list(range(len(source_shape))):
            raise ValueError(
                f"target {target_fqn!r} has invalid transpose {transpose_dims}"
            )
        source_shape = tuple(source_shape[index] for index in transpose_dims)
    if prod(source_shape) != prod(target_shape):
        raise ValueError(
            f"target {target_fqn!r} source and target slices have different numel"
        )
    if _remove_singleton_dimensions(source_shape) != _remove_singleton_dimensions(
        target_shape
    ):
        raise ValueError(
            f"target {target_fqn!r} cannot reshape source slice {source_shape} "
            f"to target slice {target_shape}"
        )


def _remove_singleton_dimensions(shape: Sequence[int]) -> tuple[int, ...]:
    return tuple(size for size in shape if size != 1)


def _validate_plan_source_type(
    target_fqn: str,
    plan: LoadPlan,
    metadata: SourceTensorMetadata,
) -> None:
    if plan.src_elem_size not in (0, metadata.element_size_bytes):
        raise ValueError(
            f"target {target_fqn!r} plan source element size "
            f"{plan.src_elem_size} disagrees with checkpoint metadata "
            f"{metadata.element_size_bytes}"
        )
    if plan.src_dtype and _resolve_torch_dtype(plan.src_dtype) != _resolve_torch_dtype(
        metadata.dtype
    ):
        raise ValueError(
            f"target {target_fqn!r} plan source dtype {plan.src_dtype!r} "
            f"disagrees with checkpoint metadata {metadata.dtype!r}"
        )


def _validate_pattern_bound(
    label: str,
    pattern: RepeatedStrideBytePattern,
    storage_end: int,
) -> None:
    if pattern.bounding_range.end > storage_end:
        raise ValueError(
            f"{label} byte pattern ends at {pattern.bounding_range.end}, "
            f"past storage end {storage_end}"
        )


def _validate_nonoverlapping_destination_slices(
    target_fqn: str,
    plans: Sequence[LoadPlan],
) -> None:
    for index, left in enumerate(plans):
        for right in plans[index + 1 :]:
            if _slices_overlap(left.offsets, left.sizes, right.offsets, right.sizes):
                raise ValueError(
                    f"target {target_fqn!r} has overlapping destination slices"
                )


def _slices_overlap(
    left_offsets: Sequence[int],
    left_sizes: Sequence[int],
    right_offsets: Sequence[int],
    right_sizes: Sequence[int],
) -> bool:
    if not (
        len(left_offsets) == len(left_sizes) == len(right_offsets) == len(right_sizes)
    ):
        return False
    if any(size == 0 for size in (*left_sizes, *right_sizes)):
        return False
    return all(
        left_offset < right_offset + right_size
        and right_offset < left_offset + left_size
        for left_offset, left_size, right_offset, right_size in zip(
            left_offsets,
            left_sizes,
            right_offsets,
            right_sizes,
        )
    )
