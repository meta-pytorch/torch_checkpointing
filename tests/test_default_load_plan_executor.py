# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save as serialize_safetensors, save_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torch_checkpointing.checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
    TorchSerialization,
)
from torch_checkpointing.default_load_plan_executor import (
    _default_file_read_workers,
    _slice_source_tensor,
    _validate_source_slice_bounds,
    DefaultLoadPlanExecutor,
    ReshardingReadStrategy,
)
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
from torch_checkpointing.load_plan_executor import LoadExecutionContext
from torch_checkpointing.resharding import LoadPlan, ReshardingInfo
from torch_checkpointing.storage.base_storage import ReadArgs
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import NestedPath

from .resharding_test_utils import plan_and_execute


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

    def read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> bytes:
        with self.stream_read(path, read_args) as stream:
            return stream.read()

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
        missing_paths = plan_and_execute(
            DefaultResharder(),
            executor=DefaultLoadPlanExecutor(read_strategy=read_strategy),
            source_path=Path("."),
            item_key="model",
            target_metadata={("selected",): target_sharding},
            source_metadata=source_metadata,
            target={"selected": target_tensor},
            storage=storage,  # type: ignore[arg-type]
        )

    assert missing_paths == []
    return target_tensor, storage


def _execute_load_plans(
    load_plans: dict[NestedPath, list[LoadPlan]],
    target: dict[str, torch.Tensor],
    source_layouts: dict[int, LayoutInfo],
    storage: Any,
    *,
    source_path: str | Path = ".",
    **executor_options: Any,
) -> None:
    """Read ``load_plans`` into the tensors of ``target`` from ``source_layouts``."""
    DefaultLoadPlanExecutor(**executor_options).execute(
        {
            "model": ReshardingInfo(
                nested_path_to_load_plans=load_plans,
                non_reshardable_paths=[],
                nested_path_to_target_tensors={
                    (name,): tensor for name, tensor in target.items()
                },
            )
        },
        LoadExecutionContext(
            checkpoint_path=Path(source_path),
            storage=storage,
            source_metadata={
                "model": DistributedItemMetadata(
                    nested_path_to_metadata={},
                    rank_to_layout_info=dict(source_layouts),
                )
            },
        ),
    )


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
        missing_paths = plan_and_execute(
            DefaultResharder(),
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


def test_full_file_strategy_loads_dotted_safetensors() -> None:
    source = torch.arange(8, dtype=torch.float32)
    fqn = "layers.0.weight"
    path = Path("model.safetensors")
    checkpoint_data = serialize_safetensors({fqn: source})
    storage = _TrackingStorage(path, checkpoint_data)
    plan = LoadPlan(
        offsets=(0,),
        sizes=(4,),
        src_rank=0,
        src_fqn=fqn,
        src_offsets=(4,),
        src_sizes=(4,),
    )

    target = {fqn: torch.empty(4)}
    _execute_load_plans(
        {(fqn,): [plan]},
        target,
        {0: LayoutInfo(str(path), SafetensorsSerialization())},
        storage,
        read_strategy=ReshardingReadStrategy.FULL_FILE,
    )

    torch.testing.assert_close(target[fqn], source[4:])
    assert storage.read_args == [ReadArgs(pre_read_full_file=False)]
    assert storage.bytes_read == len(checkpoint_data)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_auto_uses_offset_reads_for_safetensors(device: str) -> None:
    source = torch.arange(1024, dtype=torch.float32)
    path = Path("model.safetensors")
    checkpoint_data = serialize_safetensors({"weight": source})
    storage = _TrackingStorage(path, checkpoint_data)
    plan = LoadPlan(
        offsets=(0,),
        sizes=tuple(source.shape),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=tuple(source.shape),
    )
    target = {
        "first": torch.empty_like(source, device=device),
        "second": torch.empty_like(source, device=device),
    }
    span_bytes = source.numel() * source.element_size()
    assert span_bytes < len(checkpoint_data) <= 2 * span_bytes

    _execute_load_plans(
        {("first",): [plan], ("second",): [plan]},
        target,
        {0: LayoutInfo(str(path), SafetensorsSerialization())},
        storage,
    )

    torch.testing.assert_close(target["first"].cpu(), source)
    torch.testing.assert_close(target["second"].cpu(), source)
    assert storage.read_args == [ReadArgs(pre_read_full_file=False, direct_io=True)]


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


def test_load_parameter_source_uses_offset_read() -> None:
    source = torch.nn.Parameter(torch.arange(8, dtype=torch.float32))

    target, storage = _load_second_half(source)

    torch.testing.assert_close(target, source.detach()[4:])
    assert len(storage.read_args) == 1
    assert storage.read_args[0] is not None
    assert not storage.read_args[0].pre_read_full_file


def test_quantized_offset_fallback_has_specific_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = torch.arange(8, dtype=torch.float32)

    with patch.object(
        torch.Tensor,
        "is_quantized",
        property(lambda self: self.is_meta),
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
            torch.Tensor,
            "is_quantized",
            property(lambda self: self.is_meta),
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
        missing_paths = plan_and_execute(
            DefaultResharder(),
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


def test_auto_falls_back_to_full_file_reads_per_unsupported_file(
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
    source_layouts = {
        rank: LayoutInfo(str(path), TorchSerialization())
        for rank, path in enumerate(paths)
    }

    with (
        caplog.at_level(logging.WARNING),
        patch(
            "torch_checkpointing.default_load_plan_executor._validate_offset_read_archive",
            side_effect=(
                None,
                NotImplementedError("unsupported offset read"),
                NotImplementedError("unsupported offset read"),
            ),
        ) as validate_archive,
    ):
        # One worker, so files are read in order.
        _execute_load_plans(
            {("selected",): load_plans},
            {"selected": target_tensor},
            source_layouts,
            storage,
            file_read_workers=1,
        )

    torch.testing.assert_close(target_tensor, torch.arange(12, dtype=torch.float32))
    assert validate_archive.call_count == 3
    # Every file tries offset reads first; the unsupported ones are then read in
    # full, one at a time.
    assert [path for path, _ in storage.reads] == [
        paths[0],
        paths[1],
        paths[2],
        paths[1],
        paths[2],
    ]
    assert storage.getsize_calls == paths[1:]
    assert caplog.text.count("Offset reads unavailable") == 2


def test_full_file_reads_run_one_at_a_time_on_the_calling_thread() -> None:
    """A full-file read holds the whole file in memory, so they never overlap."""
    paths = [Path(f"rank_{rank}.pt") for rank in range(3)]
    files = {}
    for rank, path in enumerate(paths):
        checkpoint = io.BytesIO()
        torch.save({"selected": torch.full((4,), float(rank))}, checkpoint)
        files[path] = checkpoint.getvalue()
    read_threads = []

    class _ThreadRecordingStorage(_MultiFileTrackingStorage):
        def stream_read(
            self,
            path: Path,
            read_args: ReadArgs | None = None,
        ) -> io.BytesIO:
            read_threads.append(threading.current_thread())
            return super().stream_read(path, read_args)

    storage = _ThreadRecordingStorage(files)
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
        for rank in range(len(paths))
    ]

    _execute_load_plans(
        {("selected",): load_plans},
        {"selected": target_tensor},
        {
            rank: LayoutInfo(str(path), TorchSerialization())
            for rank, path in enumerate(paths)
        },
        storage,
        read_strategy=ReshardingReadStrategy.FULL_FILE,
        file_read_workers=len(paths),
    )

    torch.testing.assert_close(
        target_tensor, torch.arange(3, dtype=torch.float32).repeat_interleave(4)
    )
    assert read_threads == [threading.current_thread()] * len(paths)


@pytest.mark.parametrize(
    ("local_world_size", "workers"),
    [
        ("8", 3),
        ("4", 6),
        ("2", 12),
        ("32", 1),
        (None, 8),
        ("", 8),
        ("0", 8),
        ("-2", 8),
        ("eight", 8),
    ],
)
def test_default_file_read_workers_split_host_reads_across_local_ranks(
    monkeypatch: pytest.MonkeyPatch, local_world_size: str | None, workers: int
) -> None:
    if local_world_size is None:
        monkeypatch.delenv("LOCAL_WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)
    assert _default_file_read_workers() == workers


def test_file_read_workers_must_be_positive() -> None:
    with pytest.raises(ValueError, match="file_read_workers must be at least 1"):
        DefaultLoadPlanExecutor(file_read_workers=0)


def test_reads_source_files_concurrently() -> None:
    num_files = 4
    paths = [Path(f"model_{rank}.safetensors") for rank in range(num_files)]
    files = {
        path: serialize_safetensors(
            {"weight": torch.arange(rank * 4, rank * 4 + 4, dtype=torch.float32)}
        )
        for rank, path in enumerate(paths)
    }
    # Every file must be open at once to get past the barrier.
    all_open = threading.Barrier(num_files, timeout=30)

    class _BarrierStorage(_MultiFileTrackingStorage):
        def stream_read(
            self,
            path: Path,
            read_args: ReadArgs | None = None,
        ) -> io.BytesIO:
            all_open.wait()
            return super().stream_read(path, read_args)

    target_tensor = torch.zeros(num_files * 4, dtype=torch.float32)
    load_plans = [
        LoadPlan(
            offsets=(rank * 4,),
            sizes=(4,),
            src_rank=rank,
            src_fqn="weight",
            src_offsets=(0,),
            src_sizes=(4,),
        )
        for rank in range(num_files)
    ]

    storage = _BarrierStorage(files)
    _execute_load_plans(
        {("weight",): load_plans},
        {"weight": target_tensor},
        {
            rank: LayoutInfo(str(path), SafetensorsSerialization())
            for rank, path in enumerate(paths)
        },
        storage,
        file_read_workers=num_files,
    )

    torch.testing.assert_close(
        target_tensor, torch.arange(num_files * 4, dtype=torch.float32)
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_concurrent_reads_of_many_tensors_fill_shared_targets(
    tmp_path: Path, device: str
) -> None:
    """Every file writes into every target at once, including interleaved columns."""
    num_files, rows, cols = 4, 1024, 4096
    generator = torch.Generator().manual_seed(0)
    shards = [
        {
            "rows": torch.randn(rows, cols, generator=generator),
            "columns": torch.randn(rows, cols, generator=generator),
            "half": torch.randn(rows, cols, generator=generator).to(torch.bfloat16),
        }
        for _ in range(num_files)
    ]
    names = [f"model_{rank}.safetensors" for rank in range(num_files)]
    for name, shard in zip(names, shards):
        save_file(shard, tmp_path / name)
    storage = LocalFileSystemStorageConfig().create_storage()
    # Every file must be open at once to get past the barrier.
    all_open = threading.Barrier(num_files, timeout=30)
    open_stream = storage.stream_read

    def stream_read_once_all_are_open(
        path: Path, read_args: ReadArgs | None = None
    ) -> io.RawIOBase:
        all_open.wait()
        return open_stream(path, read_args)

    target = {
        "rows": torch.zeros(num_files * rows, cols, device=device),
        "columns": torch.zeros(rows, num_files * cols, device=device),
        "half": torch.zeros(
            num_files * rows, cols, dtype=torch.bfloat16, device=device
        ),
    }
    # Row blocks are contiguous in the target; column blocks interleave.
    target_offsets = {
        "rows": lambda rank: (rank * rows, 0),
        "columns": lambda rank: (0, rank * cols),
        "half": lambda rank: (rank * rows, 0),
    }
    load_plans = {
        (name,): [
            LoadPlan(
                offsets=target_offsets[name](rank),
                sizes=(rows, cols),
                src_rank=rank,
                src_fqn=name,
                src_offsets=(0, 0),
                src_sizes=(rows, cols),
            )
            for rank in range(num_files)
        ]
        for name in target
    }

    with patch.object(storage, "stream_read", stream_read_once_all_are_open):
        _execute_load_plans(
            load_plans,
            target,
            {
                rank: LayoutInfo(name, SafetensorsSerialization())
                for rank, name in enumerate(names)
            },
            storage,
            source_path=tmp_path,
            file_read_workers=num_files,
        )

    torch.testing.assert_close(
        target["rows"].cpu(), torch.cat([shard["rows"] for shard in shards])
    )
    torch.testing.assert_close(
        target["columns"].cpu(),
        torch.cat([shard["columns"] for shard in shards], dim=1),
    )
    torch.testing.assert_close(
        target["half"].cpu(), torch.cat([shard["half"] for shard in shards])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copies_follow_pending_work_on_the_callers_stream() -> None:
    source = torch.arange(1024, dtype=torch.float32)
    path = Path("model.safetensors")
    storage = _TrackingStorage(path, serialize_safetensors({"weight": source}))
    plan = LoadPlan(
        offsets=(0,),
        sizes=tuple(source.shape),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=tuple(source.shape),
    )
    target = torch.empty_like(source, device="cuda")

    def load() -> None:
        _execute_load_plans(
            {("weight",): [plan]},
            {"weight": target},
            {0: LayoutInfo(str(path), SafetensorsSerialization())},
            storage,
        )

    # First kernel launches and first-time host allocations can synchronize the
    # whole device, which would order the copy by accident. Do them once up front.
    torch.cuda._sleep(1)
    target.fill_(-1)
    load()
    torch.cuda.synchronize()

    with torch.cuda.stream(torch.cuda.Stream()):
        # Delay the caller's pending write, so a copy not ordered after it
        # finishes first and is overwritten.
        torch.cuda._sleep(100_000_000)
        target.fill_(-1)
        load()
    torch.cuda.synchronize()

    torch.testing.assert_close(target.cpu(), source)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="requires CUDA"
            ),
        ),
    ],
)
def test_aliased_targets_load_once_from_the_last_path(device: str) -> None:
    """Two files fill paths that share one tensor; the last path wins, as serially."""
    paths = [Path(f"model_{rank}.safetensors") for rank in range(2)]
    files = {
        path: serialize_safetensors({"weight": torch.full((4,), float(rank + 1))})
        for rank, path in enumerate(paths)
    }
    tied = torch.zeros(4, device=device)
    storage = _MultiFileTrackingStorage(files)
    _execute_load_plans(
        {
            (name,): [
                LoadPlan(
                    offsets=(0,),
                    sizes=(4,),
                    src_rank=rank,
                    src_fqn="weight",
                    src_offsets=(0,),
                    src_sizes=(4,),
                )
            ]
            for rank, name in [(1, "first"), (0, "second")]
        },
        {"first": tied, "second": tied},
        {
            rank: LayoutInfo(str(path), SafetensorsSerialization())
            for rank, path in enumerate(paths)
        },
        storage,
        file_read_workers=2,
    )

    torch.testing.assert_close(tied.cpu(), torch.full((4,), 1.0))
    # The dropped path's file is never read, so no concurrent write can race.
    assert {path for path, _ in storage.reads} == {paths[0]}


def test_aliased_targets_disable_concurrent_reads() -> None:
    """With any aliased targets, every file is read by the same single worker."""
    paths = [Path(f"model_{rank}.safetensors") for rank in range(3)]
    files = {
        path: serialize_safetensors({"weight": torch.full((4,), float(rank + 1))})
        for rank, path in enumerate(paths)
    }
    read_threads = []

    class _SlowThreadRecordingStorage(_MultiFileTrackingStorage):
        def stream_read(
            self,
            path: Path,
            read_args: ReadArgs | None = None,
        ) -> io.BytesIO:
            read_threads.append(threading.current_thread())
            # Keep each read busy so concurrent workers would overlap.
            time.sleep(0.1)
            return super().stream_read(path, read_args)

    tied = torch.zeros(4)
    other = torch.zeros(4)
    storage = _SlowThreadRecordingStorage(files)
    _execute_load_plans(
        {
            (name,): [
                LoadPlan(
                    offsets=(0,),
                    sizes=(4,),
                    src_rank=rank,
                    src_fqn="weight",
                    src_offsets=(0,),
                    src_sizes=(4,),
                )
            ]
            for rank, name in [(1, "first"), (0, "second"), (2, "other")]
        },
        {"first": tied, "second": tied, "other": other},
        {
            rank: LayoutInfo(str(path), SafetensorsSerialization())
            for rank, path in enumerate(paths)
        },
        storage,
        file_read_workers=3,
    )

    torch.testing.assert_close(tied, torch.full((4,), 1.0))
    torch.testing.assert_close(other, torch.full((4,), 3.0))
    assert len(read_threads) == 2
    assert len(set(read_threads)) == 1


_ONE_SAFETENSORS_FILE = {0: LayoutInfo("model.safetensors", SafetensorsSerialization())}


def _one_safetensors_file_storage() -> _MultiFileTrackingStorage:
    return _MultiFileTrackingStorage(
        {
            Path("model.safetensors"): serialize_safetensors(
                {"weight": torch.arange(4.0)}
            )
        }
    )


def _copy_plan(**changes: Any) -> LoadPlan:
    """A plan copying all of the source tensor ``weight`` into a 4-element target."""
    plan = LoadPlan(
        offsets=(0,),
        sizes=(4,),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=(4,),
    )
    return replace(plan, **changes)


@pytest.fixture
def cpu_mesh() -> Iterator[DeviceMesh]:
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("transpose_dims", [(), (0,)], ids=["empty", "identity"])
def test_unpermuted_transpose_dims_copy_the_source_unchanged(
    transpose_dims: tuple[int, ...],
) -> None:
    target = torch.zeros(4)

    _execute_load_plans(
        {("weight",): [_copy_plan(transpose_dims=transpose_dims)]},
        {"weight": target},
        _ONE_SAFETENSORS_FILE,
        _one_safetensors_file_storage(),
    )

    torch.testing.assert_close(target, torch.arange(4.0))


@pytest.mark.parametrize("read_strategy", list(ReshardingReadStrategy))
def test_permuting_source_dimensions_is_rejected_before_any_read(
    read_strategy: ReshardingReadStrategy,
) -> None:
    storage = _MultiFileTrackingStorage(
        {
            Path("model.safetensors"): serialize_safetensors(
                {"weight": torch.arange(4.0).reshape(2, 2)}
            )
        }
    )
    plan = _copy_plan(
        offsets=(0, 0),
        sizes=(2, 2),
        src_offsets=(0, 0),
        src_sizes=(2, 2),
        transpose_dims=(1, 0),
    )

    with pytest.raises(NotImplementedError, match="permutes source dimensions"):
        _execute_load_plans(
            {("weight",): [plan]},
            {"weight": torch.zeros(2, 2)},
            _ONE_SAFETENSORS_FILE,
            storage,
            read_strategy=read_strategy,
        )
    assert storage.reads == []


@pytest.mark.parametrize(
    ("plan", "match"),
    [
        (_copy_plan(offsets=(1,)), "out of bounds in dimension 0"),
        (_copy_plan(sizes=(2,)), r"source chunk of shape \(4,\) into a target"),
        (
            _copy_plan(
                offsets=(0, 0), sizes=(1, 4), src_offsets=(0, 0), src_sizes=(1, 4)
            ),
            "rank does not match target tensor 'model::weight'",
        ),
    ],
    ids=["out_of_bounds", "shape_mismatch", "rank_mismatch"],
)
def test_plan_that_does_not_fit_its_target_is_rejected_before_any_read(
    plan: LoadPlan, match: str
) -> None:
    storage = _one_safetensors_file_storage()

    with pytest.raises(ValueError, match=match):
        _execute_load_plans(
            {("weight",): [plan]},
            {"weight": torch.zeros(4)},
            _ONE_SAFETENSORS_FILE,
            storage,
        )
    assert storage.reads == []


@pytest.mark.parametrize(
    "target",
    [torch.empty(4, device="meta"), torch.zeros(4).to_sparse()],
    ids=["meta", "sparse"],
)
def test_target_without_strided_storage_is_rejected(target: torch.Tensor) -> None:
    storage = _one_safetensors_file_storage()

    with pytest.raises(ValueError, match="has no strided storage to load into"):
        _execute_load_plans(
            {("weight",): [_copy_plan()]},
            {"weight": target},
            _ONE_SAFETENSORS_FILE,
            storage,
        )
    assert storage.reads == []


def test_dtensor_target_is_rejected(cpu_mesh: DeviceMesh) -> None:
    target = distribute_tensor(torch.zeros(4), cpu_mesh, [Replicate()])
    storage = _one_safetensors_file_storage()

    with pytest.raises(ValueError, match="is a DTensor, not its local tensor"):
        _execute_load_plans(
            {("weight",): [_copy_plan()]},
            {"weight": target},
            _ONE_SAFETENSORS_FILE,
            storage,
        )
    assert storage.reads == []


def test_every_item_is_validated_before_any_file_is_read() -> None:
    storage = _one_safetensors_file_storage()
    items = {
        key: ReshardingInfo(
            nested_path_to_load_plans={("weight",): [plan]},
            non_reshardable_paths=[],
            nested_path_to_target_tensors={("weight",): torch.zeros(4)},
        )
        for key, plan in [("model", _copy_plan()), ("optim", _copy_plan(offsets=(1,)))]
    }
    context = LoadExecutionContext(
        checkpoint_path=Path("."),
        storage=storage,  # type: ignore[arg-type]
        source_metadata={
            key: DistributedItemMetadata(
                nested_path_to_metadata={}, rank_to_layout_info=_ONE_SAFETENSORS_FILE
            )
            for key in items
        },
    )

    with pytest.raises(ValueError, match="'optim::weight'"):
        DefaultLoadPlanExecutor().execute(items, context)
    assert storage.reads == []
