# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Where a tensor's bytes live in a checkpoint file.

``TensorSlice`` is pure geometry: which region of a global tensor something is,
in mathematical coordinates. ``SerializedTensorSlice`` adds where those bytes
physically sit, which is what a reader needs in order to fetch them without
loading the file.

The name follows ``SerializationFormat``: the byte address, dtype and strides
are exactly what differs between the formats a checkpoint can be written in,
and are what a reader must be told regardless of which one wrote it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tensor_slice import TensorSlice


@dataclass(frozen=True)
class ByteAddress:
    """A half-open byte range within one file."""

    file_path: str
    start_byte_offset: int
    end_byte_offset: int

    @property
    def num_bytes(self) -> int:
        return self.end_byte_offset - self.start_byte_offset


@dataclass(frozen=True)
class SerializedTensorSlice(TensorSlice):
    """A tensor region together with the bytes that hold it.

    ``serialized_strides`` describes the layout of this region *as written in the
    file*, not the strides of the global tensor it belongs to. It is stored
    rather than derived from ``slice_shape`` because only some formats pack
    densely: safetensors always does, while a torch-serialized tensor may be an
    arbitrarily strided view whose strides carry information the shape does not.
    """

    source_rank: int
    torch_dtype: torch.dtype
    byte_address: ByteAddress
    serialized_strides: tuple[int, ...]

    @property
    def dtype_size(self) -> int:
        return torch._utils._element_size(self.torch_dtype)

    @property
    def contiguous_global_byte_range(self) -> tuple[int, int] | None:
        element_range = self.contiguous_global_element_range
        if element_range is None:
            return None
        start, end = element_range
        return start * self.dtype_size, end * self.dtype_size


def contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Row-major strides for a densely packed tensor of ``shape``."""
    strides = [0] * len(shape)
    stride = 1
    for dimension in range(len(shape) - 1, -1, -1):
        strides[dimension] = stride
        stride *= shape[dimension]
    return tuple(strides)
