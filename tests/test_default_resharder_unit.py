# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    GlobalObjectMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    ReplicateSpec,
    ShardSpec,
)
from torch_checkpointing.storage.base_storage import ReadArgs


class _TrackingReader(io.BytesIO):
    def __init__(self, data: bytes, storage: "_TrackingStorage") -> None:
        super().__init__(data)
        self._storage = storage

    def read(self, size: int = -1) -> bytes:
        data = super().read(size)
        self._storage.bytes_read += len(data)
        return data

    def readinto(self, buffer: Any) -> int | None:
        bytes_read = super().readinto(buffer)
        self._storage.bytes_read += bytes_read or 0
        return bytes_read


class _TrackingStorage:
    def __init__(self, path: Path, data: bytes) -> None:
        self._path = path
        self._data = data
        self.bytes_read = 0
        self.read_args: list[ReadArgs | None] = []

    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> _TrackingReader:
        assert path == self._path
        self.read_args.append(read_args)
        return _TrackingReader(self._data, self)


def test_extract_sharding_metadata_treats_plain_tensors_as_replicated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_item = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "scalar": torch.tensor(7, dtype=torch.int64),
        "epoch": 4,
    }

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.default_resharder.dist.get_world_size",
            return_value=4,
        ),
        caplog.at_level(logging.WARNING),
    ):
        metadata = DefaultResharder().extract_sharding_metadata(
            "model",
            checkpoint_item,
        )

    assert set(metadata) == {("weight",), ("scalar",)}
    weight_metadata = metadata[("weight",)]
    assert isinstance(weight_metadata, DTensorShardingMetadata)
    assert weight_metadata.global_shape == (2, 3)
    assert weight_metadata.dtype == "torch.float32"
    assert weight_metadata.stride == (3, 1)
    assert weight_metadata.mesh_spec.device_type == "cpu"
    assert weight_metadata.mesh_spec.mesh_shape == (4,)
    assert weight_metadata.mesh_spec.mesh_data == (0, 1, 2, 3)
    assert weight_metadata.placements == (ReplicateSpec(),)
    assert weight_metadata.equivalent_ranks == (0, 1, 2, 3)

    scalar_metadata = metadata[("scalar",)]
    assert isinstance(scalar_metadata, DTensorShardingMetadata)
    assert scalar_metadata.global_shape == ()
    assert scalar_metadata.stride == ()
    assert scalar_metadata.dtype == "torch.int64"
    assert scalar_metadata.placements == (ReplicateSpec(),)

    assert "Found 2 plain tensors" in caplog.text
    assert "treating them as replicated tensors" in caplog.text


def test_load_reads_one_span_for_noncontiguous_source_slice() -> None:
    backing = torch.arange(1_000_007, dtype=torch.bfloat16)
    selected = backing.as_strided((6, 5), (200_000, 1), storage_offset=2)
    checkpoint = io.BytesIO()
    torch.save(
        {
            "unused": torch.zeros(1_000_000, dtype=torch.float32),
            "selected": selected,
        },
        checkpoint,
    )
    path = Path("checkpoint.pt")
    checkpoint_bytes = checkpoint.getvalue()
    storage = _TrackingStorage(path, checkpoint_bytes)
    target = {"selected": torch.zeros((3, 5), dtype=torch.float32)}
    source_sharding = DTensorShardingMetadata(
        global_shape=(6, 5),
        dtype="torch.bfloat16",
        stride=selected.stride(),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=(1,),
            mesh_data=(0,),
        ),
        placements=(ReplicateSpec(),),
    )
    target_sharding = DTensorShardingMetadata(
        global_shape=(6, 5),
        dtype="torch.float32",
        stride=(5, 1),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=(2,),
            mesh_data=(0, 1),
        ),
        placements=(ShardSpec(0),),
    )
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            ("selected",): [
                GlobalObjectMetadata(
                    sharding_metadata=source_sharding,
                    ranks=(0,),
                )
            ]
        },
        rank_to_layout_info={0: LayoutInfo("checkpoint.pt", TorchSerialization())},
    )

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.default_resharder.dist.get_rank",
            return_value=1,
        ),
    ):
        missing_paths = DefaultResharder().load(
            source_path=Path("."),
            item_key="model",
            target_metadata={("selected",): target_sharding},
            source_metadata=source_metadata,
            target=target,
            storage=storage,  # type: ignore[arg-type]
        )

    assert missing_paths == []
    torch.testing.assert_close(target["selected"], selected[3:6].float())
    # A single span read: the first requested element through the last, plus
    # the metadata pass. More than the 30-byte dense payload because the
    # source rows are strided, but a small fraction of the file.
    rows, cols = 3, 5
    row_stride, column_stride = selected.stride()
    span_bytes = (
        1 + (rows - 1) * row_stride + (cols - 1) * column_stride
    ) * selected.element_size()
    assert storage.bytes_read < span_bytes + 64 * 1024
    assert storage.bytes_read < len(checkpoint_bytes) // 5
    assert all(
        read_args is not None and not read_args.pre_read_full_file
        for read_args in storage.read_args
    )


def test_load_falls_back_for_quantized_source_tensor() -> None:
    source = torch.quantize_per_tensor(
        torch.arange(8, dtype=torch.float32),
        scale=0.25,
        zero_point=3,
        dtype=torch.quint8,
    )
    checkpoint = io.BytesIO()
    torch.save({"selected": source}, checkpoint)
    path = Path("checkpoint.pt")
    storage = _TrackingStorage(path, checkpoint.getvalue())
    target_tensor = torch.quantize_per_tensor(
        torch.zeros(4, dtype=torch.float32),
        scale=0.25,
        zero_point=3,
        dtype=torch.quint8,
    )
    target = {"selected": target_tensor}
    source_sharding = DTensorShardingMetadata(
        global_shape=(8,),
        dtype="torch.quint8",
        stride=(1,),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=(1,),
            mesh_data=(0,),
        ),
        placements=(ReplicateSpec(),),
    )
    target_sharding = DTensorShardingMetadata(
        global_shape=(8,),
        dtype="torch.quint8",
        stride=(1,),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=(2,),
            mesh_data=(0, 1),
        ),
        placements=(ShardSpec(0),),
    )
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            ("selected",): [
                GlobalObjectMetadata(
                    sharding_metadata=source_sharding,
                    ranks=(0,),
                )
            ]
        },
        rank_to_layout_info={0: LayoutInfo("checkpoint.pt", TorchSerialization())},
    )

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.default_resharder.dist.get_rank",
            return_value=1,
        ),
    ):
        missing_paths = DefaultResharder().load(
            source_path=Path("."),
            item_key="model",
            target_metadata={("selected",): target_sharding},
            source_metadata=source_metadata,
            target=target,
            storage=storage,  # type: ignore[arg-type]
        )

    assert missing_paths == []
    assert torch.equal(target_tensor.int_repr(), source[4:8].int_repr())
    assert target_tensor.q_scale() == source.q_scale()
    assert target_tensor.q_zero_point() == source.q_zero_point()
    assert len(storage.read_args) == 2
    assert storage.read_args[0] is not None
    assert not storage.read_args[0].pre_read_full_file
    assert storage.read_args[1] is None
