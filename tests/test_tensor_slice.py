# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: pytorch_checkpointing"]

import pytest
from torch_checkpointing.tensor_slice import TensorSlice


def test_tensor_slice_exposes_global_and_local_ranges() -> None:
    tensor_slice = TensorSlice(
        global_shape=(13, 4),
        global_offsets=(11, 1),
        local_offsets=(1, 0),
        slice_shape=(1, 2),
    )

    assert tensor_slice.global_slice == ((11, 12), (1, 3))
    assert tensor_slice.local_slice == ((1, 2), (0, 2))


@pytest.mark.parametrize(
    ("global_offsets", "local_offsets", "slice_shape"),
    [
        ((0,), (0, 0), (1,)),
        ((0,), (0,), (1, 1)),
        ((0, 0), (0,), (1,)),
    ],
)
def test_tensor_slice_rejects_mismatched_dimensions(
    global_offsets: tuple[int, ...],
    local_offsets: tuple[int, ...],
    slice_shape: tuple[int, ...],
) -> None:
    with pytest.raises(AssertionError, match="same number of dimensions"):
        TensorSlice(
            global_shape=(2,),
            global_offsets=global_offsets,
            local_offsets=local_offsets,
            slice_shape=slice_shape,
        )


@pytest.mark.parametrize(
    ("global_shape", "global_offsets", "local_offsets", "slice_shape"),
    [
        ((-1,), (0,), (0,), (0,)),
        ((2,), (-1,), (0,), (1,)),
        ((2,), (0,), (-1,), (1,)),
        ((2,), (0,), (0,), (-1,)),
    ],
)
def test_tensor_slice_rejects_negative_geometry(
    global_shape: tuple[int, ...],
    global_offsets: tuple[int, ...],
    local_offsets: tuple[int, ...],
    slice_shape: tuple[int, ...],
) -> None:
    with pytest.raises(AssertionError, match="nonnegative"):
        TensorSlice(
            global_shape=global_shape,
            global_offsets=global_offsets,
            local_offsets=local_offsets,
            slice_shape=slice_shape,
        )


def test_tensor_slice_rejects_global_range_out_of_bounds() -> None:
    with pytest.raises(AssertionError, match="does not fit in dimension 0"):
        TensorSlice(
            global_shape=(13,),
            global_offsets=(12,),
            local_offsets=(0,),
            slice_shape=(2,),
        )


@pytest.mark.parametrize(
    ("tensor_slice", "expected_range"),
    [
        (
            TensorSlice(
                global_shape=(4, 3),
                global_offsets=(1, 0),
                local_offsets=(0, 0),
                slice_shape=(2, 3),
            ),
            (3, 9),
        ),
        (
            TensorSlice(
                global_shape=(4, 3),
                global_offsets=(1, 1),
                local_offsets=(0, 0),
                slice_shape=(2, 1),
            ),
            None,
        ),
        (
            TensorSlice(
                global_shape=(4, 3),
                global_offsets=(4, 0),
                local_offsets=(0, 0),
                slice_shape=(0, 3),
            ),
            (0, 0),
        ),
    ],
)
def test_tensor_slice_contiguous_global_element_range(
    tensor_slice: TensorSlice,
    expected_range: tuple[int, int] | None,
) -> None:
    assert tensor_slice.contiguous_global_element_range == expected_range


def test_tensor_slice_supports_unresolved_global_geometry() -> None:
    tensor_slice = TensorSlice(
        global_shape=None,
        global_offsets=None,
        local_offsets=(1, 0),
        slice_shape=(2, 3),
    )

    assert not tensor_slice.is_resolved
    assert tensor_slice.local_slice == ((1, 3), (0, 3))
    with pytest.raises(AssertionError, match="unresolved"):
        _ = tensor_slice.global_slice
    with pytest.raises(AssertionError, match="unresolved"):
        _ = tensor_slice.contiguous_global_element_range


@pytest.mark.parametrize(
    ("global_shape", "global_offsets"),
    [
        (None, (0,)),
        ((2,), None),
    ],
)
def test_tensor_slice_rejects_partially_resolved_global_geometry(
    global_shape: tuple[int, ...] | None,
    global_offsets: tuple[int, ...] | None,
) -> None:
    with pytest.raises(AssertionError, match="both resolved or both unresolved"):
        TensorSlice(
            global_shape=global_shape,
            global_offsets=global_offsets,
            local_offsets=(0,),
            slice_shape=(1,),
        )


def test_tensor_slice_with_global_layout_preserves_local_geometry() -> None:
    unresolved = TensorSlice(
        global_shape=None,
        global_offsets=None,
        local_offsets=(1, 0),
        slice_shape=(2, 3),
    )

    resolved = unresolved.with_global_layout(
        global_shape=(7, 5),
        global_offsets=(4, 1),
    )

    assert not unresolved.is_resolved
    assert resolved.is_resolved
    assert resolved.global_shape == (7, 5)
    assert resolved.global_offsets == (4, 1)
    assert resolved.local_offsets == (1, 0)
    assert resolved.slice_shape == (2, 3)
    with pytest.raises(AssertionError, match="already resolved"):
        resolved.with_global_layout(
            global_shape=(7, 5),
            global_offsets=(4, 1),
        )


def test_tensor_slice_with_global_layout_rejects_out_of_bounds_slice() -> None:
    unresolved = TensorSlice(
        global_shape=None,
        global_offsets=None,
        local_offsets=(0,),
        slice_shape=(2,),
    )

    with pytest.raises(AssertionError, match="does not fit in dimension 0"):
        unresolved.with_global_layout(
            global_shape=(2,),
            global_offsets=(1,),
        )


def test_tensor_slice_supports_resolved_scalar_geometry() -> None:
    tensor_slice = TensorSlice(
        global_shape=(),
        global_offsets=(),
        local_offsets=(),
        slice_shape=(),
    )

    assert tensor_slice.is_resolved
    assert tensor_slice.global_slice == ()
    assert tensor_slice.local_slice == ()
    assert tensor_slice.contiguous_global_element_range == (0, 1)
