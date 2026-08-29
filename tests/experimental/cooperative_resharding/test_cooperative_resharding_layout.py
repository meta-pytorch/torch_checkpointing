# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from dataclasses import asdict

import pytest
import torch
from torch_checkpointing.experimental.cooperative_resharding.layout import (
    build_repeated_stride_pattern,
    ByteRange,
    resolve_tensor_read_target,
    resolve_tensor_read_targets,
    SourceTensorMetadata,
)
from torch_checkpointing.resharding import LoadPlan


def _metadata(
    *,
    fqn: str = "source.weight",
    shape: tuple[int, ...] = (4, 8),
    stride: tuple[int, ...] = (8, 1),
    checkpoint_offset_bytes: int = 1_000,
    storage_nbytes: int = 128,
    dtype: str = "torch.float32",
    element_size_bytes: int = 4,
) -> SourceTensorMetadata:
    return SourceTensorMetadata(
        fqn=fqn,
        checkpoint_offset_bytes=checkpoint_offset_bytes,
        storage_offset_elements=0,
        storage_nbytes=storage_nbytes,
        shape=shape,
        stride=stride,
        dtype=dtype,
        element_size_bytes=element_size_bytes,
    )


def test_compacts_row_strided_slice() -> None:
    pattern = build_repeated_stride_pattern(
        offsets=(1, 2),
        sizes=(2, 3),
        strides=(8, 1),
        element_size_bytes=4,
        initial_offset_bytes=100,
    )

    assert pattern.start_offset == 140
    assert pattern.block_bytes == 12
    assert pattern.repeat_counts == (2,)
    assert pattern.repeat_strides_bytes == (32,)
    assert tuple(pattern.iter_ranges()) == (ByteRange(140, 12), ByteRange(172, 12))
    assert pattern.bounding_range == ByteRange(140, 44)
    assert pattern.dense_nbytes == 24


def test_preserves_logical_order_for_transposed_storage() -> None:
    pattern = build_repeated_stride_pattern(
        offsets=(0, 0),
        sizes=(3, 2),
        strides=(1, 3),
        element_size_bytes=4,
    )

    assert pattern.block_bytes == 4
    assert pattern.repeat_counts == (3, 2)
    assert [byte_range.offset for byte_range in pattern.iter_ranges()] == [
        0,
        12,
        4,
        16,
        8,
        20,
    ]
    assert pattern.dense_nbytes == 24


def test_zero_sized_slice_has_no_ranges() -> None:
    pattern = build_repeated_stride_pattern(
        offsets=(2, 0),
        sizes=(0, 8),
        strides=(8, 1),
        element_size_bytes=2,
    )

    assert tuple(pattern.iter_ranges()) == ()
    assert pattern.dense_nbytes == 0


def test_resolves_strided_source_and_destination_without_pointers() -> None:
    source_metadata = _metadata()
    target = torch.empty_strided((4, 4), (8, 2), dtype=torch.float32)
    plan = LoadPlan(
        offsets=(1, 1),
        sizes=(2, 3),
        src_rank=0,
        src_fqn=source_metadata.fqn,
        src_offsets=(1, 2),
        src_sizes=(2, 3),
        src_elem_size=4,
        src_dtype="float",
    )

    resolved = resolve_tensor_read_target(
        "target.weight",
        plan,
        source_metadata,
        target,
    )

    assert tuple(resolved.source_pattern.iter_ranges()) == (
        ByteRange(1_040, 12),
        ByteRange(1_072, 12),
    )
    assert [
        byte_range.offset for byte_range in resolved.destination_pattern.iter_ranges()
    ] == [40, 48, 56, 72, 80, 88]
    assert resolved.source_span == ByteRange(1_040, 44)
    assert resolved.numel == 6
    assert not resolved.requires_transform
    serialized = json.dumps(asdict(resolved), sort_keys=True)
    assert '"source_rank": 0' in serialized
    assert "data_ptr" not in serialized


def test_resolves_transpose_and_dtype_conversion() -> None:
    source_metadata = _metadata(shape=(2, 3), stride=(3, 1))
    target = torch.empty((1, 3, 2), dtype=torch.bfloat16)
    plan = LoadPlan(
        offsets=(0, 0, 0),
        sizes=(1, 3, 2),
        src_rank=0,
        src_fqn=source_metadata.fqn,
        src_offsets=(0, 0),
        src_sizes=(2, 3),
        transpose_dims=(1, 0),
        src_elem_size=4,
        src_dtype="float32",
    )

    resolved = resolve_tensor_read_target(
        "target.weight",
        plan,
        source_metadata,
        target,
    )

    assert resolved.transpose_dims == (1, 0)
    assert resolved.target_slice_shape == (1, 3, 2)
    assert resolved.requires_transform
    assert resolved.source_pattern.dense_nbytes == 24
    assert resolved.destination_pattern.dense_nbytes == 12


def test_rejects_out_of_bounds_source_slice() -> None:
    source_metadata = _metadata()
    plan = LoadPlan(
        offsets=(0, 0),
        sizes=(2, 3),
        src_rank=0,
        src_fqn=source_metadata.fqn,
        src_offsets=(3, 6),
        src_sizes=(2, 3),
    )

    with pytest.raises(ValueError, match="out of bounds"):
        resolve_tensor_read_target(
            "target.weight",
            plan,
            source_metadata,
            torch.empty((2, 3)),
        )


def test_rejects_overlapping_or_incomplete_target_coverage() -> None:
    source_metadata = _metadata(shape=(4,), stride=(1,), storage_nbytes=16)
    overlapping = (
        LoadPlan((0,), (3,), 0, source_metadata.fqn, (0,), (3,)),
        LoadPlan((2,), (2,), 0, source_metadata.fqn, (2,), (2,)),
    )
    incomplete = (LoadPlan((0,), (3,), 0, source_metadata.fqn, (0,), (3,)),)

    with pytest.raises(ValueError, match="overlapping destination"):
        resolve_tensor_read_targets(
            {"target.weight": overlapping},
            {0: {source_metadata.fqn: source_metadata}},
            {"target.weight": torch.empty(4)},
        )
    with pytest.raises(ValueError, match="cover 3 elements, expected 4"):
        resolve_tensor_read_targets(
            {"target.weight": incomplete},
            {0: {source_metadata.fqn: source_metadata}},
            {"target.weight": torch.empty(4)},
        )


def test_rejects_distinct_targets_that_overlap_storage() -> None:
    source_a = _metadata(fqn="source.a", shape=(2,), stride=(1,), storage_nbytes=8)
    source_b = _metadata(fqn="source.b", shape=(2,), stride=(1,), storage_nbytes=8)
    backing = torch.empty(3)
    targets = {"target.a": backing[:2], "target.b": backing[1:]}
    plans = {
        "target.a": [LoadPlan((0,), (2,), 0, "source.a", (0,), (2,))],
        "target.b": [LoadPlan((0,), (2,), 0, "source.b", (0,), (2,))],
    }

    with pytest.raises(ValueError, match="destination storage"):
        resolve_tensor_read_targets(
            plans,
            {0: {"source.a": source_a, "source.b": source_b}},
            targets,
        )


def test_resolves_complete_partitioned_target() -> None:
    source_metadata = _metadata(shape=(4,), stride=(1,), storage_nbytes=16)
    plans = (
        LoadPlan((0,), (2,), 0, source_metadata.fqn, (0,), (2,)),
        LoadPlan((2,), (2,), 0, source_metadata.fqn, (2,), (2,)),
    )

    resolved = resolve_tensor_read_targets(
        {"target.weight": plans},
        {0: {source_metadata.fqn: source_metadata}},
        {"target.weight": torch.empty(4)},
    )

    assert len(resolved) == 2
    assert sum(target.numel for target in resolved) == 4


def test_resolves_root_tensor_with_empty_fqn() -> None:
    source_metadata = _metadata(
        fqn="",
        shape=(4,),
        stride=(1,),
        storage_nbytes=16,
    )
    plan = LoadPlan((0,), (4,), 0, "", (0,), (4,))

    resolved = resolve_tensor_read_targets(
        {"": (plan,)},
        {0: {"": source_metadata}},
        {"": torch.empty(4)},
    )

    assert len(resolved) == 1
    assert resolved[0].target_fqn == ""
    assert resolved[0].source_fqn == ""


@pytest.mark.parametrize(
    ("dtype", "element_size"),
    [("float", 4), ("torch.bfloat16", 2), ("long", 8)],
)
def test_source_metadata_uses_generic_dtype_resolution(
    dtype: str,
    element_size: int,
) -> None:
    metadata = _metadata(
        shape=(1,),
        stride=(1,),
        storage_nbytes=element_size,
        dtype=dtype,
        element_size_bytes=element_size,
    )

    assert metadata.element_size_bytes == element_size


def test_source_metadata_rejects_unknown_or_mismatched_dtype() -> None:
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        _metadata(
            shape=(1,),
            stride=(1,),
            storage_nbytes=4,
            dtype="not_a_dtype",
        )
    with pytest.raises(ValueError, match="does not use 2 bytes"):
        _metadata(
            shape=(1,),
            stride=(1,),
            storage_nbytes=2,
            dtype="float32",
            element_size_bytes=2,
        )
