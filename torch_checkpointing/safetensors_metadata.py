# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Reading the header of a safetensors file.

A safetensors file begins with a JSON header naming every tensor it holds, with
each one's dtype, shape, and byte range. Parsing it yields enough to locate any
tensor's bytes without reading them, which is what both consolidation and
resharding loads need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import safetensors.torch as safetensors_torch
from torch.distributed.checkpoint._hf_utils import (
    _get_safetensors_file_metadata,
    DATA_OFFSETS_KEY,
    DTYPE_KEY,
    SHAPE_KEY,
)

from .serialized_tensor_slice import (
    ByteAddress,
    contiguous_strides,
    SerializedTensorSlice,
)
from .storage.base_storage import ReadArgs, Storage


@dataclass(frozen=True)
class SafetensorsFileMetadata:
    file_path: str
    tensors: dict[str, SerializedTensorSlice]

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

        tensors: dict[str, SerializedTensorSlice] = {}
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
            local_shape = tuple(tensor_metadata[SHAPE_KEY])
            tensors[fqn] = SerializedTensorSlice(
                global_shape=None,
                global_offsets=None,
                local_offsets=(0,) * len(local_shape),
                slice_shape=local_shape,
                source_rank=source_rank,
                torch_dtype=torch_dtype,
                byte_address=ByteAddress(
                    file_path=file_path,
                    start_byte_offset=file_start_byte_offset + start,
                    end_byte_offset=file_start_byte_offset + end,
                ),
                # safetensors always writes each tensor densely
                serialized_strides=contiguous_strides(local_shape),
            )

        return cls(file_path=file_path, tensors=tensors)
