# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
from pathlib import Path
from typing import Any
from unittest.mock import patch, PropertyMock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization
from torch_checkpointing.default_resharder import (
    _slice_source_tensor,
    _validate_source_slice_bounds,
    DefaultResharder,
    ReshardingReadStrategy,
)
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
from torch_checkpointing.resharding import LoadPlan
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
    mmap_fill_workers: int | None = None
    mmap_fill_chunk_bytes: int | None = None

    def __init__(self, path: Path, data: bytes) -> None:
        self._path = path
        self._data = data
        self.bytes_read = 0
        self.read_args: list[ReadArgs | None] = []
        self.getsize_calls: list[Path] = []

    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> _TrackingReader:
        assert path == self._path
        self.read_args.append(read_args)
        return _TrackingReader(self._data, self)

    def getsize(self, path: Path) -> int:
        assert path == self._path
        self.getsize_calls.append(path)
        return len(self._data)


class _MultiFileTrackingStorage:
    mmap_fill_workers: int | None = None
    mmap_fill_chunk_bytes: int | None = None

    def __init__(self, files: dict[Path, bytes]) -> None:
        self._files = files
        self.reads: list[tuple[Path, ReadArgs | None]] = []
        self.getsize_calls: list[Path] = []

    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> io.BytesIO:
        self.reads.append((path, read_args))
        return io.BytesIO(self._files[path])

    def getsize(self, path: Path) -> int:
        self.getsize_calls.append(path)
        return len(self._files[path])


def _load_second_half(
    source: torch.Tensor,
    *,
    read_strategy: ReshardingReadStrategy = ReshardingReadStrategy.AUTO,
) -> tuple[torch.Tensor, _TrackingStorage]:
    checkpoint = io.BytesIO()
    torch.save({"selected": source}, checkpoint)
    path = Path("checkpoint.pt")
    storage = _TrackingStorage(path, checkpoint.getvalue())
    target_tensor = torch.zeros(source.shape[0] // 2, dtype=source.dtype)
    source_sharding = DTensorShardingMetadata(
        global_shape=tuple(source.shape),
        dtype=str(source.dtype),
        stride=source.stride(),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=(1,),
            mesh_data=(0,),
        ),
        placements=(ReplicateSpec(),),
    )
    target_sharding = DTensorShardingMetadata(
        global_shape=tuple(source.shape),
        dtype=str(source.dtype),
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
        missing_paths = DefaultResharder(read_strategy=read_strategy).load(
            source_path=Path("."),
            item_key="model",
            target_metadata={("selected",): target_sharding},
            source_metadata=source_metadata,
            target={"selected": target_tensor},
            storage=storage,  # type: ignore[arg-type]
        )

    assert missing_paths == []
    return target_tensor, storage


def _source_load_plan(
    src_offsets: tuple[int, ...],
    src_sizes: tuple[int, ...],
) -> LoadPlan:
    return LoadPlan(
        offsets=tuple(0 for _ in src_sizes),
        sizes=src_sizes,
        src_rank=0,
        src_fqn="model.weight",
        src_offsets=src_offsets,
        src_sizes=src_sizes,
    )


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


@pytest.mark.parametrize(
    ("source_shape", "src_offsets", "src_sizes", "match"),
    [
        ((3, 4), (0,), (1,), "rank"),
        ((3, 4), (-1, 0), (1, 1), "dimension 0"),
        ((3, 4), (0, 0), (-1, 1), "dimension 0"),
        ((3, 4), (2, 0), (2, 1), "dimension 0"),
        ((3, 4), (4, 0), (0, 1), "dimension 0"),
    ],
)
def test_validate_source_slice_bounds_rejects_invalid_geometry(
    source_shape: tuple[int, ...],
    src_offsets: tuple[int, ...],
    src_sizes: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _validate_source_slice_bounds(
            source_shape,
            _source_load_plan(src_offsets, src_sizes),
        )


@pytest.mark.parametrize(
    ("source_shape", "src_offsets", "src_sizes"),
    [
        ((), (), ()),
        ((3, 4), (3, 0), (0, 1)),
    ],
)
def test_validate_source_slice_bounds_accepts_boundary_geometry(
    source_shape: tuple[int, ...],
    src_offsets: tuple[int, ...],
    src_sizes: tuple[int, ...],
) -> None:
    _validate_source_slice_bounds(
        source_shape,
        _source_load_plan(src_offsets, src_sizes),
    )


def test_full_file_slice_rejects_out_of_bounds_plan() -> None:
    with pytest.raises(ValueError, match="out of bounds in dimension 0"):
        _slice_source_tensor(
            torch.arange(8),
            _source_load_plan(src_offsets=(7,), src_sizes=(2,)),
        )


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


def test_load_preserves_conjugate_view_in_offset_slice() -> None:
    base = torch.arange(12, dtype=torch.float32).to(torch.complex64) * (1 + 2j)
    source = base[1::2].conj()

    target, storage = _load_second_half(source)

    torch.testing.assert_close(target, source[3:])
    assert len(storage.read_args) == 1
    assert storage.read_args[0] is not None
    assert not storage.read_args[0].pre_read_full_file


def test_load_preserves_negative_view_in_offset_slice() -> None:
    base = torch.arange(12, dtype=torch.float32)
    source = base[1::2]._neg_view()

    target, storage = _load_second_half(source)

    torch.testing.assert_close(target, source[3:])
    assert len(storage.read_args) == 1
    assert storage.read_args[0] is not None
    assert not storage.read_args[0].pre_read_full_file


def test_load_preserves_conjugate_negative_view_in_offset_slice() -> None:
    base = torch.arange(12, dtype=torch.float32).to(torch.complex64) * (1 + 2j)
    source = base[1::2].conj()._neg_view()

    target, storage = _load_second_half(source)

    torch.testing.assert_close(target, source[3:])
    assert len(storage.read_args) == 1
    assert storage.read_args[0] is not None
    assert not storage.read_args[0].pre_read_full_file


def test_quantized_offset_fallback_has_specific_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = torch.arange(8, dtype=torch.float32)

    with patch.object(
        FakeTensor,
        "is_quantized",
        new_callable=PropertyMock,
        return_value=True,
    ):
        target, storage = _load_second_half(source)

    torch.testing.assert_close(target, source[4:])
    assert len(storage.read_args) == 2
    assert storage.getsize_calls == [Path("checkpoint.pt")]
    assert "Source 'selected' is quantized" in caplog.text
    assert "does not use a strided storage" not in caplog.text


def test_offset_strategy_propagates_unsupported_tensor() -> None:
    source = torch.arange(8, dtype=torch.float32)

    with (
        patch.object(
            FakeTensor,
            "is_quantized",
            new_callable=PropertyMock,
            return_value=True,
        ),
        pytest.raises(NotImplementedError, match="quantized"),
    ):
        _load_second_half(
            source,
            read_strategy=ReshardingReadStrategy.OFFSET,
        )


def test_full_file_strategy_skips_offset_metadata_read() -> None:
    source = torch.arange(1024, dtype=torch.float32)

    target, storage = _load_second_half(
        source,
        read_strategy=ReshardingReadStrategy.FULL_FILE,
    )

    torch.testing.assert_close(target, source[512:])
    assert storage.getsize_calls == [Path("checkpoint.pt")]
    assert len(storage.read_args) == 1
    assert storage.read_args[0] is not None
    assert not storage.read_args[0].pre_read_full_file


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
    assert storage.getsize_calls == [path]


def test_auto_stops_trying_offset_reads_after_first_unsupported_file(
    caplog: pytest.LogCaptureFixture,
) -> None:
    paths = [Path(f"rank_{rank}.pt") for rank in range(3)]
    source_shards = [
        torch.arange(start, start + 4, dtype=torch.float32) for start in (0, 4, 8)
    ]
    files = {}
    for path, source_shard in zip(paths, source_shards):
        checkpoint = io.BytesIO()
        torch.save({"selected": source_shard}, checkpoint)
        files[path] = checkpoint.getvalue()
    storage = _MultiFileTrackingStorage(files)
    target_tensor = torch.zeros(12, dtype=torch.float32)
    load_plans = [
        LoadPlan(
            offsets=(rank * 4,),
            sizes=(4,),
            src_rank=rank,
            src_fqn="selected",
            src_offsets=(0,),
            src_sizes=(4,),
        )
        for rank in range(3)
    ]

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "torch_checkpointing.default_resharder._validate_offset_read_archive",
            side_effect=(None, NotImplementedError("unsupported offset read")),
        ) as validate_archive,
    ):
        DefaultResharder()._execute_load_plans(
            src_path_fn=lambda rank: paths[rank],
            item_key="model",
            nested_path_to_load_plans={("selected",): load_plans},
            target={"selected": target_tensor},
            storage=storage,  # type: ignore[arg-type]
        )

    torch.testing.assert_close(target_tensor, torch.arange(12, dtype=torch.float32))
    assert validate_archive.call_count == 2
    assert [path for path, _ in storage.reads] == [
        paths[0],
        paths[1],
        paths[1],
        paths[2],
    ]
    assert storage.getsize_calls == paths[1:]
    assert caplog.text.count("Offset reads unavailable") == 1
