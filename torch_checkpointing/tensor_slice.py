# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from typing_extensions import Self


@dataclass(frozen=True)
class TensorSlice:
    """One N-D region in shared global and packed-local coordinates.

    ``slice_shape`` is the extent of the same region in both coordinate systems;
    ``global_offsets`` and ``local_offsets`` only choose where that region starts.
    """

    global_shape: tuple[int, ...] | None
    global_offsets: tuple[int, ...] | None
    local_offsets: tuple[int, ...]
    slice_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        assert (self.global_shape is None) == (self.global_offsets is None), (
            "TensorSlice global shape and offsets must be both resolved or both "
            "unresolved"
        )
        ndim = len(self.local_offsets)
        assert len(self.slice_shape) == ndim, (
            "TensorSlice fields must have the same number of dimensions"
        )
        assert all(offset >= 0 for offset in self.local_offsets), (
            f"TensorSlice geometry must be nonnegative: {self.local_offsets=}"
        )
        assert all(size >= 0 for size in self.slice_shape), (
            f"TensorSlice geometry must be nonnegative: {self.slice_shape=}"
        )
        if not self.is_resolved:
            return

        assert self.global_shape is not None
        assert self.global_offsets is not None
        assert len(self.global_shape) == len(self.global_offsets) == ndim, (
            "TensorSlice fields must have the same number of dimensions"
        )
        assert all(size >= 0 for size in self.global_shape), (
            f"TensorSlice geometry must be nonnegative: {self.global_shape=}"
        )
        assert all(offset >= 0 for offset in self.global_offsets), (
            f"TensorSlice geometry must be nonnegative: {self.global_offsets=}"
        )
        for dimension, (offset, size, global_size) in enumerate(
            zip(self.global_offsets, self.slice_shape, self.global_shape)
        ):
            assert offset + size <= global_size, (
                f"TensorSlice does not fit in dimension {dimension}: "
                f"{offset=}, {size=}, {global_size=}"
            )

    @property
    def is_resolved(self) -> bool:
        return self.global_shape is not None

    def with_global_layout(
        self,
        global_shape: tuple[int, ...],
        global_offsets: tuple[int, ...],
    ) -> Self:
        assert not self.is_resolved, "TensorSlice is already resolved"
        return replace(
            self,
            global_shape=global_shape,
            global_offsets=global_offsets,
        )

    @property
    def global_slice(self) -> tuple[tuple[int, int], ...]:
        assert self.global_offsets is not None, "TensorSlice is unresolved"
        return tuple(
            (offset, offset + size)
            for offset, size in zip(self.global_offsets, self.slice_shape)
        )

    @property
    def local_slice(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (offset, offset + size)
            for offset, size in zip(self.local_offsets, self.slice_shape)
        )

    @property
    def contiguous_global_element_range(self) -> tuple[int, int] | None:
        assert self.global_shape is not None, "TensorSlice is unresolved"
        assert self.global_offsets is not None, "TensorSlice is unresolved"
        num_elements = math.prod(self.slice_shape)
        if num_elements == 0:
            return (0, 0)

        strides = tuple(
            math.prod(self.global_shape[index + 1 :])
            for index in range(len(self.global_shape))
        )
        start_element = sum(
            offset * stride for offset, stride in zip(self.global_offsets, strides)
        )
        last_element = sum(
            (offset + size - 1) * stride
            for offset, size, stride in zip(
                self.global_offsets,
                self.slice_shape,
                strides,
            )
        )
        if last_element - start_element + 1 != num_elements:
            return None
        return start_element, start_element + num_elements
