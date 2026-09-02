# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: pytorch_checkpointing"]

import concurrent.futures
import io
import json
import logging
import os
import pickle
import socket
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_checkpointing.hf.consolidation as consolidation_module
from safetensors.torch import (
    load_file as safetensors_load_file,
    save_file as safetensors_save_file,
)
from torch_checkpointing.checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
)
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
    ShardSpec,
    StridedShardSpec,
)
from torch_checkpointing.hf.consolidation import (
    _assign_items_to_owners_by_size,
    _atomic_stream_write,
    _process_output_file,
    _read_safetensors_file_metadata_by_rank,
    _read_tensor_data,
    consolidate_hf_safetensors_checkpoint,
)
from torch_checkpointing.safetensors_metadata import SafetensorsFileMetadata
from torch_checkpointing.serialized_tensor_slice import (
    ByteAddress,
    contiguous_strides,
    SerializedTensorSlice,
)
from torch_checkpointing.storage.filesystem import (
    LocalFileSystemStorage,
    LocalFileSystemStorageConfig,
)
from torch_checkpointing.types import NestedPath


class _NonIdempotentLocalFileSystemStorage(LocalFileSystemStorage):
    def mkdir(self, path: Path, recursive: bool = True) -> None:
        os.mkdir(path)


class _NonIdempotentLocalFileSystemStorageConfig(LocalFileSystemStorageConfig):
    def create_storage(self) -> LocalFileSystemStorage:
        return _NonIdempotentLocalFileSystemStorage(config=self)


def _layout(file_path: str) -> LayoutInfo:
    return LayoutInfo(file_path, SafetensorsSerialization())


def _two_element_file_metadata(source_rank: int) -> SafetensorsFileMetadata:
    file_path = f"model_{source_rank}.safetensors"
    return SafetensorsFileMetadata(
        file_path=file_path,
        tensors={
            "weight": SerializedTensorSlice(
                global_shape=None,
                global_offsets=None,
                local_offsets=(0,),
                slice_shape=(2,),
                source_rank=source_rank,
                torch_dtype=torch.float32,
                byte_address=ByteAddress(
                    file_path=file_path,
                    start_byte_offset=16,
                    end_byte_offset=24,
                ),
                serialized_strides=contiguous_strides((2,)),
            )
        },
    )


def test_atomic_stream_write_replaces_file_after_success(tmp_path: Path) -> None:
    path = tmp_path / "output.safetensors"
    path.write_bytes(b"old")
    storage = LocalFileSystemStorageConfig().create_storage()

    with _atomic_stream_write(storage, path) as stream:
        stream.write(b"new")
        assert path.read_bytes() == b"old"

    assert path.read_bytes() == b"new"
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_stream_write_preserves_file_after_failure(tmp_path: Path) -> None:
    path = tmp_path / "output.safetensors"
    path.write_bytes(b"old")
    storage = LocalFileSystemStorageConfig().create_storage()

    with pytest.raises(RuntimeError, match="write failed"):
        with _atomic_stream_write(storage, path) as stream:
            stream.write(b"partial")
            raise RuntimeError("write failed")

    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def _dtensor_metadata(
    placements: tuple[ShardSpec | ReplicateSpec, ...] | None = None,
) -> DTensorShardingMetadata:
    if placements is None:
        placements = (ShardSpec(dim=0),)
    return DTensorShardingMetadata(
        global_shape=(4, 2),
        dtype=str(torch.float32),
        stride=(2, 1),
        mesh_spec=get_device_mesh_spec(
            device_type="cpu",
            mesh_shape=(2,),
            mesh_data=(0, 1),
            mesh_dim_names=None,
        ),
        placements=placements,
    )


def _replicated_tensor_metadata(tensor: torch.Tensor) -> DTensorShardingMetadata:
    return DTensorShardingMetadata(
        global_shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        stride=tuple(tensor.stride()),
        mesh_spec=get_device_mesh_spec(
            device_type="cpu",
            mesh_shape=(2,),
            mesh_data=(0, 1),
            mesh_dim_names=None,
        ),
        placements=(ReplicateSpec(),),
    )


def _write_metadata(
    checkpoint_path: Path,
    rank_to_file_path: dict[int, str | None] | None = None,
    sharding_metadata: DTensorShardingMetadata | None = None,
    nested_path: NestedPath = ("weight",),
    item_key: str = "model",
    metadata_by_path: dict[NestedPath, DTensorShardingMetadata] | None = None,
    nested_paths: tuple[NestedPath, ...] | None = None,
) -> None:
    if rank_to_file_path is None:
        rank_to_file_path = {
            0: "model_0.safetensors",
            1: "model_1.safetensors",
        }
    if metadata_by_path is None:
        if sharding_metadata is None:
            sharding_metadata = _dtensor_metadata()
        if nested_paths is None:
            nested_paths = (nested_path,)
        metadata_by_path = {path: sharding_metadata for path in nested_paths}
    nested_path_to_metadata = {
        path: [
            GlobalObjectMetadata(
                sharding_metadata=path_metadata,
                ranks=(0, 1),
            )
        ]
        for path, path_metadata in metadata_by_path.items()
    }
    metadata = DistributedMetadata(
        metadata={
            item_key: DistributedItemMetadata(
                nested_path_to_metadata=nested_path_to_metadata,
                rank_to_layout_info={
                    rank: _layout(file_path) if file_path is not None else None
                    for rank, file_path in rank_to_file_path.items()
                },
            )
        },
        world_size=2,
    )
    with open(checkpoint_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata.to_dict(), f)


def test_fqn_to_num_bytes_from_metadata_sizes_without_reading_headers() -> None:
    sizes = consolidation_module._fqn_to_num_bytes_from_metadata(
        {
            ("weight",): [
                GlobalObjectMetadata(
                    sharding_metadata=_dtensor_metadata(), ranks=(0, 1)
                )
            ],
            ("plain",): [
                GlobalObjectMetadata(
                    sharding_metadata=_replicated_tensor_metadata(torch.ones(3)),
                    ranks=(0, 1),
                )
            ],
        }
    )

    # (4, 2) float32 and (3,) float32, taken from metadata.pkl alone.
    assert sizes == {"weight": 4 * 2 * 4, "plain": 3 * 4}


def test_source_ranks_for_fqns_covers_only_the_requested_tensors() -> None:
    nested_path_to_metadata = {
        ("weight",): [
            GlobalObjectMetadata(sharding_metadata=_dtensor_metadata(), ranks=(0, 1))
        ],
        ("other",): [
            GlobalObjectMetadata(sharding_metadata=_dtensor_metadata(), ranks=(2, 3))
        ],
    }
    # "other" was not requested, so its ranks are not read.
    assert consolidation_module._source_ranks_for_fqns(
        nested_path_to_metadata,
        {"weight"},
    ) == {0, 1}
    assert consolidation_module._source_ranks_for_fqns(
        nested_path_to_metadata,
        {"weight", "other"},
    ) == {0, 1, 2, 3}


def test_consolidate_hf_safetensors_rejects_metadata_without_source_ranks(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    safetensors_save_file(
        {"weight": torch.arange(8, dtype=torch.float32).reshape(4, 2)},
        checkpoint_path / "model_0.safetensors",
    )
    metadata = DistributedMetadata(
        metadata={
            "model": DistributedItemMetadata(
                nested_path_to_metadata={
                    ("weight",): [
                        GlobalObjectMetadata(
                            sharding_metadata=_dtensor_metadata(),
                            ranks=(),
                        )
                    ]
                },
                rank_to_layout_info={0: _layout("model_0.safetensors")},
            )
        },
        world_size=1,
    )
    with open(checkpoint_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata.to_dict(), f)

    with pytest.raises(ValueError, match="Missing source ranks for FQN 'weight'"):
        consolidate_hf_safetensors_checkpoint(os.fspath(checkpoint_path))


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_consolidation_worker(
    rank: int,
    world_size: int,
    checkpoint_path: str,
    port: int,
    output_dir: str | None = None,
    non_idempotent_mkdir: bool = False,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        checkpoint_dir = Path(checkpoint_path)
        full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        full_tensors = {"weight": full_weight}
        if non_idempotent_mkdir:
            full_tensors["other"] = full_weight + 100
        rank_slice = slice(0, 2) if rank == 0 else slice(2, 4)
        safetensors_save_file(
            {
                fqn: tensor[rank_slice].contiguous()
                for fqn, tensor in full_tensors.items()
            },
            checkpoint_dir / f"model_{rank}.safetensors",
        )
        if rank == 0:
            _write_metadata(
                checkpoint_dir,
                nested_paths=tuple((fqn,) for fqn in full_tensors),
            )
        dist.barrier()

        consolidate_hf_safetensors_checkpoint(
            checkpoint_path,
            output_dir=output_dir,
            fqn_to_index_mapping={
                fqn: index for index, fqn in enumerate(full_tensors, start=1)
            },
            item_key="model",
            storage_config=(
                _NonIdempotentLocalFileSystemStorageConfig(use_direct_io=False)
                if non_idempotent_mkdir
                else None
            ),
        )

        if rank == 0:
            consolidated_dir = Path(output_dir or checkpoint_path)
            max_index = len(full_tensors)
            for index, (fqn, expected) in enumerate(full_tensors.items(), start=1):
                consolidated = safetensors_load_file(
                    consolidated_dir
                    / f"model-{index:05d}-of-{max_index:05d}.safetensors"
                )
                torch.testing.assert_close(consolidated[fqn], expected)
            assert (consolidated_dir / "model.safetensors.index.json").exists()
    finally:
        dist.destroy_process_group()


def _distributed_default_mapping_worker(
    rank: int,
    world_size: int,
    checkpoint_path: str,
    port: int,
) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        checkpoint_dir = Path(checkpoint_path)
        full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        full_other = full_weight + 100
        if rank < 2:
            rank_slice = slice(0, 2) if rank == 0 else slice(2, 4)
            safetensors_save_file(
                {
                    "weight": full_weight[rank_slice].contiguous(),
                    "other": full_other[rank_slice].contiguous(),
                },
                checkpoint_dir / f"model_{rank}.safetensors",
            )
        if rank == 0:
            _write_metadata(
                checkpoint_dir,
                nested_paths=(("weight",), ("other",)),
            )
        dist.barrier()

        original_barrier = dist.barrier
        barrier_calls = 0

        def counted_barrier(*args, **kwargs):
            nonlocal barrier_calls
            barrier_calls += 1
            return original_barrier(*args, **kwargs)

        with mock.patch("torch.distributed.barrier", side_effect=counted_barrier):
            consolidate_hf_safetensors_checkpoint(
                checkpoint_path,
                item_key="model",
            )
        assert barrier_calls == (3 if rank < 2 else 0)
    finally:
        dist.destroy_process_group()


def test_safetensors_file_metadata_reads_unresolved_file_header(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "model.safetensors"
    safetensors_save_file(
        {
            "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "scale": torch.arange(4, dtype=torch.bfloat16),
        },
        file_path,
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    file_metadata = SafetensorsFileMetadata.from_file(
        storage=storage,
        file_path=os.fspath(file_path),
        source_rank=3,
    )

    assert file_metadata.file_path == os.fspath(file_path)
    assert set(file_metadata.tensors) == {"weight", "scale"}
    weight = file_metadata.tensors["weight"]
    assert isinstance(weight, SerializedTensorSlice)
    assert weight.source_rank == 3
    assert weight.torch_dtype == torch.float32
    assert weight.global_shape is None
    assert weight.global_offsets is None
    assert weight.local_offsets == (0, 0)
    assert weight.slice_shape == (2, 3)
    address = weight.byte_address
    assert address.file_path == os.fspath(file_path)
    assert address.start_byte_offset > 8
    assert address.num_bytes == 6 * torch.empty((), dtype=torch.float32).element_size()

    resolved_weight = weight.with_global_layout(
        global_shape=(4, 3),
        global_offsets=(2, 0),
    )
    assert isinstance(resolved_weight, SerializedTensorSlice)
    assert resolved_weight.source_rank == weight.source_rank
    assert resolved_weight.torch_dtype == weight.torch_dtype
    assert resolved_weight.byte_address == weight.byte_address
    assert resolved_weight.local_offsets == weight.local_offsets
    assert resolved_weight.slice_shape == weight.slice_shape


def test_build_fqn_to_tensor_slices_selects_one_complete_metadata_group() -> None:
    # Both groups shard the same tensor two ways, over disjoint rank pairs.
    groups = [
        GlobalObjectMetadata(
            sharding_metadata=DTensorShardingMetadata(
                global_shape=(4,),
                dtype=str(torch.float32),
                stride=(1,),
                mesh_spec=get_device_mesh_spec(
                    device_type="cpu",
                    mesh_shape=(2,),
                    mesh_data=group_ranks,
                    mesh_dim_names=None,
                ),
                placements=(ShardSpec(dim=0),),
            ),
            ranks=group_ranks,
        )
        for group_ranks in ((1, 2), (0, 3))
    ]
    ranks = (0, 1, 2, 3)

    result = consolidation_module._build_fqn_to_tensor_slices(
        {("weight",): groups},
        {rank: _layout(f"model_{rank}.safetensors") for rank in ranks},
        {rank: _two_element_file_metadata(rank) for rank in ranks},
    )["weight"]

    assert {tensor_slice.source_rank for tensor_slice in result} == {0, 3}
    assert [tensor_slice.global_offsets for tensor_slice in result] == [(0,), (2,)]


def test_build_fqn_to_tensor_slices_resolves_strided_shard(tmp_path: Path) -> None:
    """FSDP + TP emits StridedShard; each rank still holds one contiguous block."""
    sharding_metadata = DTensorShardingMetadata(
        global_shape=(8, 2),
        dtype=str(torch.float32),
        stride=(2, 1),
        mesh_spec=get_device_mesh_spec(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_data=(0, 1, 2, 3),
            mesh_dim_names=("dp", "tp"),
        ),
        placements=(StridedShardSpec(dim=0, split_factor=2), ShardSpec(dim=0)),
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    file_metadata_by_rank = {}
    for rank in range(4):
        file_path = tmp_path / f"model_{rank}.safetensors"
        safetensors_save_file({"weight": torch.zeros(2, 2)}, file_path)
        file_metadata_by_rank[rank] = SafetensorsFileMetadata.from_file(
            storage=storage,
            file_path=os.fspath(file_path),
            source_rank=rank,
        )

    tensor_slices = consolidation_module._build_fqn_to_tensor_slices(
        {
            ("weight",): [
                GlobalObjectMetadata(
                    sharding_metadata=sharding_metadata, ranks=(0, 1, 2, 3)
                )
            ]
        },
        {rank: _layout(f"model_{rank}.safetensors") for rank in range(4)},
        file_metadata_by_rank,
    )["weight"]

    assert {
        tensor_slice.source_rank: tensor_slice.global_offsets
        for tensor_slice in tensor_slices
    } == {0: (0, 0), 1: (4, 0), 2: (2, 0), 3: (6, 0)}
    assert all(tensor_slice.slice_shape == (2, 2) for tensor_slice in tensor_slices)


def test_read_safetensors_file_metadata_by_rank_uses_only_declared_layouts(
    tmp_path: Path,
) -> None:
    explicit_file = tmp_path / "optimizer_7.safetensors"
    undeclared_file = tmp_path / "optimizer_10.safetensors"
    safetensors_save_file({"explicit": torch.ones(2)}, explicit_file)
    safetensors_save_file({"undeclared": torch.zeros(2)}, undeclared_file)
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    file_metadata_by_rank = _read_safetensors_file_metadata_by_rank(
        input_checkpoint_dir=os.fspath(tmp_path),
        rank_to_layout_info={3: _layout(explicit_file.name), 5: None},
        storage=storage,
    )

    assert set(file_metadata_by_rank) == {3}
    assert file_metadata_by_rank[3].file_path == os.fspath(explicit_file)
    assert file_metadata_by_rank[3].tensors["explicit"].source_rank == 3


def test_process_output_file_rejects_multiple_scalar_slices(tmp_path: Path) -> None:
    source_files = [
        tmp_path / "scalar_0.safetensors",
        tmp_path / "scalar_1.safetensors",
    ]
    for source_rank, source_file in enumerate(source_files):
        safetensors_save_file({"scalar": torch.tensor(source_rank)}, source_file)
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    tensor_slices = [
        SafetensorsFileMetadata.from_file(
            storage=storage,
            file_path=os.fspath(source_file),
            source_rank=source_rank,
        )
        .tensors["scalar"]
        .with_global_layout((), ())
        for source_rank, source_file in enumerate(source_files)
    ]

    handles, register_handles, close_handles = (
        consolidation_module._reader_handle_pool()
    )
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            initializer=register_handles,
        ) as read_executor:
            with pytest.raises(
                AssertionError,
                match="Scalar tensors require exactly one source slice",
            ):
                _process_output_file(
                    os.fspath(tmp_path / "output.safetensors"),
                    {"scalar": tensor_slices},
                    storage,
                    read_executor,
                    1,
                    handles,
                    consolidation_module._ReaderMemoryBudget(1 << 30),
                )
    finally:
        close_handles()


def test_read_tensor_data_handles_short_reads() -> None:
    class ShortReadBytesIO(io.BytesIO):
        def readinto(self, buffer) -> int:
            return super().readinto(buffer[:2])

    stream = ShortReadBytesIO(b"prefixpayloadsuffix")
    address = ByteAddress(
        file_path="checkpoint.safetensors",
        start_byte_offset=6,
        end_byte_offset=13,
    )

    assert bytes(_read_tensor_data(stream, address)) == b"payload"


def test_read_tensor_data_rejects_truncated_input() -> None:
    stream = io.BytesIO(b"short")
    address = ByteAddress(
        file_path="checkpoint.safetensors",
        start_byte_offset=0,
        end_byte_offset=10,
    )

    with pytest.raises(EOFError, match="Expected 10 bytes"):
        _read_tensor_data(stream, address)


@pytest.mark.parametrize(
    (
        "global_shape",
        "global_offsets",
        "local_shape",
        "torch_dtype",
        "expected",
    ),
    [
        pytest.param((), (), (), torch.float32, (0, 4), id="scalar"),
        pytest.param((0,), (0,), (0,), torch.float32, (0, 0), id="empty"),
        pytest.param((10,), (4,), (3,), torch.float16, (8, 14), id="one_dimensional"),
        pytest.param(
            (4, 5), (1, 0), (2, 5), torch.float32, (20, 60), id="dim_zero_shard"
        ),
        pytest.param((4, 5), (0, 1), (4, 2), torch.float32, None, id="dim_one_shard"),
        pytest.param((2, 3), (0, 0), (2, 3), torch.int8, (0, 6), id="replica"),
        pytest.param((4, 5), (3, 0), (1, 5), torch.float64, (120, 160), id="tail_row"),
        pytest.param(
            (8, 4, 6),
            (3, 0, 0),
            (2, 4, 6),
            torch.bfloat16,
            (144, 240),
            id="expert_slab",
        ),
        pytest.param(
            (8, 4, 6),
            (0, 1, 0),
            (8, 2, 6),
            torch.float32,
            None,
            id="input_dimension_shard",
        ),
        pytest.param(
            (8, 4, 6),
            (0, 0, 1),
            (8, 4, 2),
            torch.float32,
            None,
            id="output_dimension_shard",
        ),
    ],
)
def test_safetensors_tensor_slice_contiguous_global_byte_range(
    global_shape: tuple[int, ...],
    global_offsets: tuple[int, ...],
    local_shape: tuple[int, ...],
    torch_dtype: torch.dtype,
    expected: tuple[int, int] | None,
) -> None:
    tensor_slice = SerializedTensorSlice(
        global_shape=global_shape,
        global_offsets=global_offsets,
        local_offsets=(0,) * len(local_shape),
        slice_shape=local_shape,
        source_rank=0,
        torch_dtype=torch_dtype,
        byte_address=ByteAddress(
            file_path="model.safetensors",
            start_byte_offset=0,
            end_byte_offset=0,
        ),
        serialized_strides=contiguous_strides(local_shape),
    )

    assert tensor_slice.contiguous_global_byte_range == expected


def test_consolidate_hf_safetensors_reads_contiguous_slices_into_output_buffer(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[2:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

    with mock.patch(
        "torch_checkpointing.hf.consolidation._read_tensor_data",
        wraps=_read_tensor_data,
    ) as read_tensor_data:
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )

    read_tensor_data.assert_not_called()

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)
    assert (checkpoint_path / "model.safetensors.index.json").exists()


def test_reader_handles_are_reused_across_tensors(tmp_path: Path) -> None:
    source_file = tmp_path / "model_0.safetensors"
    safetensors_save_file(
        {"a": torch.ones(2, 2), "b": torch.zeros(2, 2)},
        source_file,
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    file_metadata = SafetensorsFileMetadata.from_file(
        storage=storage,
        file_path=os.fspath(source_file),
        source_rank=0,
    )
    handles, register_handles, close_handles = (
        consolidation_module._reader_handle_pool()
    )
    register_handles()

    real_stream_read = storage.stream_read
    with mock.patch.object(
        storage, "stream_read", side_effect=real_stream_read
    ) as stream_read:
        try:
            for fqn in ("a", "b"):
                tensor_slice = file_metadata.tensors[fqn].with_global_layout(
                    (2, 2), (0, 0)
                )
                consolidation_module._read_full_tensor_into_from_storage(
                    [tensor_slice],
                    storage,
                    memoryview(bytearray(2 * 2 * 4)),
                    handles,
                )
            # Both tensors live in one file, so the second read reuses the
            # handle the first opened rather than reopening per tensor.
            assert stream_read.call_count == 1
        finally:
            close_handles()

    assert not handles.by_path


def test_reader_memory_budget_always_admits_the_first_read() -> None:
    budget = consolidation_module._ReaderMemoryBudget(8)

    # Larger than the whole budget, but forced, so an oversized tensor cannot
    # stall the pipeline.
    assert budget.try_reserve(64, force=True)
    assert not budget.try_reserve(1, force=False)
    assert budget.exhausted_count == 1

    budget.release(64)
    assert budget.try_reserve(8, force=False)


def test_reader_memory_budget_rejects_non_positive_total() -> None:
    with pytest.raises(AssertionError, match="must be positive"):
        consolidation_module._ReaderMemoryBudget(0)


@pytest.mark.parametrize(
    ("budget_bytes", "expected_max_active_reads"),
    [(32, 1), (64, 2)],
)
def test_consolidate_hf_safetensors_bounds_reads_by_memory_budget(
    tmp_path: Path,
    budget_bytes: int,
    expected_max_active_reads: int,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    full_other = full_weight + 100
    for rank, rank_slice in enumerate((slice(0, 2), slice(2, 4))):
        safetensors_save_file(
            {
                "weight": full_weight[rank_slice].contiguous(),
                "other": full_other[rank_slice].contiguous(),
            },
            checkpoint_path / f"model_{rank}.safetensors",
        )
    _write_metadata(
        checkpoint_path,
        nested_paths=(("weight",), ("other",)),
    )

    active_reads = 0
    max_active_reads = 0
    read_lock = threading.Lock()
    read_tensor_data_into = consolidation_module._read_tensor_data_into

    def tracked_read(*args, **kwargs) -> None:
        nonlocal active_reads, max_active_reads
        with read_lock:
            active_reads += 1
            max_active_reads = max(max_active_reads, active_reads)
        try:
            time.sleep(0.05)
            read_tensor_data_into(*args, **kwargs)
        finally:
            with read_lock:
                active_reads -= 1

    # Each consolidated tensor is 32 bytes, so a 32-byte budget admits only the
    # forced first read while 64 bytes admits a second alongside it.
    with (
        mock.patch.object(
            consolidation_module,
            "_READER_MEMORY_BUDGET_BYTES",
            budget_bytes,
        ),
        mock.patch.object(
            consolidation_module,
            "_read_tensor_data_into",
            side_effect=tracked_read,
        ),
    ):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1, "other": 1},
            item_key="model",
        )

    assert max_active_reads == expected_max_active_reads
    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)
    torch.testing.assert_close(consolidated["other"], full_other)


def test_consolidate_hf_safetensors_writes_to_separate_output_dir(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "checkpoint"
    input_checkpoint_dir = tmp_path / "sharded"
    input_checkpoint_dir.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        input_checkpoint_dir / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[2:].contiguous()},
        input_checkpoint_dir / "model_1.safetensors",
    )
    _write_metadata(input_checkpoint_dir)
    assert not output_dir.exists()

    consolidate_hf_safetensors_checkpoint(
        os.fspath(input_checkpoint_dir),
        output_dir=os.fspath(output_dir),
        fqn_to_index_mapping={"weight": 1},
        item_key="model",
    )

    consolidated = safetensors_load_file(
        output_dir / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)
    assert (output_dir / "model.safetensors.index.json").exists()
    assert not (input_checkpoint_dir / "model.safetensors.index.json").exists()


def test_consolidate_hf_safetensors_buffers_noncontiguous_destination_slices(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:, :1].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[:, 1:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        sharding_metadata=_dtensor_metadata(placements=(ShardSpec(dim=1),)),
    )

    with mock.patch(
        "torch_checkpointing.hf.consolidation._read_tensor_data",
        wraps=_read_tensor_data,
    ) as read_tensor_data:
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )

    assert read_tensor_data.call_count == 2
    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)


def test_consolidate_hf_safetensors_converts_nested_path_at_hf_boundary(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"layer.weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"layer.weight": full_weight[2:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path, nested_path=("layer", "weight"))

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        fqn_to_index_mapping={"layer.weight": 1},
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["layer.weight"], full_weight)


def test_consolidate_hf_safetensors_rejects_nested_path_fqn_collision(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"layer.weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"layer.weight": full_weight[2:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    groups = [
        GlobalObjectMetadata(
            sharding_metadata=_dtensor_metadata(),
            ranks=(0, 1),
        )
    ]
    metadata = DistributedMetadata(
        metadata={
            "model": DistributedItemMetadata(
                nested_path_to_metadata={
                    ("layer", "weight"): groups,
                    ("layer.weight",): groups,
                },
                rank_to_layout_info={
                    0: _layout("model_0.safetensors"),
                    1: _layout("model_1.safetensors"),
                },
            )
        },
        world_size=2,
    )
    with open(checkpoint_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata.to_dict(), f)

    with pytest.raises(ValueError, match="FQN collision.*layer.weight"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            item_key="model",
        )


def test_consolidate_hf_safetensors_defaults_single_fqn_to_one_file(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    plain = torch.tensor([3.0, 4.0])
    safetensors_save_file(
        {
            "weight": full_weight[:2].contiguous(),
            "plain": plain,
        },
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {
            "weight": full_weight[2:].contiguous(),
            "plain": plain,
        },
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        metadata_by_path={
            ("weight",): _dtensor_metadata(),
            ("plain",): _replicated_tensor_metadata(plain),
        },
    )

    caplog.set_level(logging.INFO, logger=consolidation_module.__name__)
    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)
    torch.testing.assert_close(consolidated["plain"], plain)
    assert set(consolidated) == {"weight", "plain"}
    assert any(
        message.startswith("Writing consolidated HF safetensors shard")
        for message in caplog.messages
    )
    for progress in (25, 50, 75):
        assert any(f"progress={progress}%" in message for message in caplog.messages)
    assert any(
        message.startswith("Finished consolidated HF safetensors shard")
        for message in caplog.messages
    )
    assert any("phase=read_source_headers" in message for message in caplog.messages)
    assert any(
        message.startswith("Finished HF safetensors consolidation")
        for message in caplog.messages
    )
    metric_records = {
        record.metric_name: record
        for record in caplog.records
        if getattr(record, "metric_name", None)
    }
    metric_prefix = "train.checkpoint_write.execute.hf_consolidation"
    expected_metrics = {
        f"{metric_prefix}.read_source_headers.latency_ms",
        f"{metric_prefix}.write_output_files.latency_ms",
        f"{metric_prefix}.output_shard.read_and_write.latency_ms",
        f"{metric_prefix}.output_shard.finalize.latency_ms",
        f"{metric_prefix}.output_shard.e2e.latency_ms",
        f"{metric_prefix}.e2e.latency_ms",
    }
    assert expected_metrics <= metric_records.keys()
    assert all(metric_records[name].value >= 0 for name in expected_metrics)
    assert metric_records[f"{metric_prefix}.e2e.latency_ms"].end_to_end


def test_consolidate_hf_safetensors_exports_plain_only_item(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    plain = torch.tensor([3.0, 4.0])
    safetensors_save_file(
        {"plain": plain},
        checkpoint_path / "model_0.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        rank_to_file_path={0: "model_0.safetensors", 1: None},
        nested_path=("plain",),
        sharding_metadata=_replicated_tensor_metadata(plain),
    )

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["plain"], plain)


def test_consolidate_hf_safetensors_exports_plain_item_from_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    exp_avg = torch.tensor([1.0, 2.0])
    step = torch.tensor(3)
    for rank in range(2):
        safetensors_save_file(
            {"exp_avg": exp_avg, "step": step},
            checkpoint_path / f"optimizer_{rank}.safetensors",
        )
    _write_metadata(
        checkpoint_path,
        rank_to_file_path={
            0: "optimizer_0.safetensors",
            1: "optimizer_1.safetensors",
        },
        item_key="optimizer",
        metadata_by_path={
            ("exp_avg",): _replicated_tensor_metadata(exp_avg),
            ("step",): _replicated_tensor_metadata(step),
        },
    )

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        fqn_to_index_mapping={"exp_avg": 1, "step": 1},
        item_key="optimizer",
    )

    output_file = checkpoint_path / "optimizer-00001-of-00001.safetensors"
    consolidated = safetensors_load_file(output_file)
    torch.testing.assert_close(consolidated["exp_avg"], exp_avg)
    torch.testing.assert_close(consolidated["step"], step)
    with open(checkpoint_path / "optimizer.safetensors.index.json") as f:
        index = json.load(f)
    assert index["weight_map"] == {
        "exp_avg": output_file.name,
        "step": output_file.name,
    }


def test_consolidate_hf_safetensors_rejects_header_fqn_missing_from_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    plain = torch.tensor([3.0, 4.0])
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {
            "weight": full_weight[2:].contiguous(),
            "plain": plain,
        },
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

    with pytest.raises(ValueError, match="absent from checkpoint metadata.*plain"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            item_key="model",
        )


def test_consolidate_hf_safetensors_rejects_missing_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"

    with pytest.raises(FileNotFoundError, match="No distributed metadata"):
        consolidate_hf_safetensors_checkpoint(os.fspath(checkpoint_path))
    assert not checkpoint_path.exists()


def test_consolidate_hf_safetensors_rejects_missing_item_key(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    _write_metadata(checkpoint_path)

    with pytest.raises(ValueError, match="optimizer"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            item_key="optimizer",
        )


def test_consolidate_hf_safetensors_rejects_item_without_source_files(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    _write_metadata(
        checkpoint_path,
        rank_to_file_path={0: None, 1: None},
    )

    with pytest.raises(ValueError, match="No rank-local safetensors files"):
        consolidate_hf_safetensors_checkpoint(os.fspath(checkpoint_path))


def test_consolidate_hf_safetensors_destroys_created_group_if_planning_fails(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    full_other = full_weight + 100
    for rank, rank_slice in enumerate((slice(0, 2), slice(2, 4))):
        safetensors_save_file(
            {
                "weight": full_weight[rank_slice].contiguous(),
                "other": full_other[rank_slice].contiguous(),
            },
            checkpoint_path / f"model_{rank}.safetensors",
        )
    _write_metadata(
        checkpoint_path,
        nested_paths=(("weight",), ("other",)),
    )
    consolidation_group = mock.sentinel.consolidation_group

    with (
        mock.patch.object(dist, "is_available", return_value=True),
        mock.patch.object(dist, "is_initialized", return_value=True),
        mock.patch.object(dist, "get_rank", return_value=0),
        mock.patch.object(dist, "get_world_size", return_value=2),
        mock.patch.object(
            consolidation_module,
            "_create_consolidation_process_group",
            return_value=(True, consolidation_group, True),
        ),
        mock.patch.object(
            consolidation_module,
            "_validate_input_output_paths_are_disjoint",
            side_effect=ValueError("planning failed"),
        ),
        mock.patch.object(dist, "destroy_process_group") as destroy_process_group,
        pytest.raises(ValueError, match="planning failed"),
    ):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1, "other": 2},
            item_key="model",
        )

    destroy_process_group.assert_called_once_with(consolidation_group)


def test_consolidate_hf_safetensors_uses_rank_zero_for_single_output_file(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    mp.spawn(
        _distributed_consolidation_worker,
        args=(2, os.fspath(checkpoint_path), _get_free_port()),
        nprocs=2,
        join=True,
    )


def test_consolidate_hf_safetensors_creates_output_dir_once(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    output_dir = tmp_path / "consolidated"

    mp.spawn(
        _distributed_consolidation_worker,
        args=(
            2,
            os.fspath(checkpoint_path),
            _get_free_port(),
            os.fspath(output_dir),
            True,
        ),
        nprocs=2,
        join=True,
    )


def test_consolidate_hf_safetensors_balances_default_mapping_across_ranks(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    mp.spawn(
        _distributed_default_mapping_worker,
        args=(3, os.fspath(checkpoint_path), _get_free_port()),
        nprocs=3,
        join=True,
    )

    first_shard = safetensors_load_file(
        checkpoint_path / "model-00001-of-00002.safetensors"
    )
    second_shard = safetensors_load_file(
        checkpoint_path / "model-00002-of-00002.safetensors"
    )
    assert set(first_shard) | set(second_shard) == {"weight", "other"}
    assert set(first_shard).isdisjoint(second_shard)
    consolidated = {**first_shard, **second_shard}
    torch.testing.assert_close(
        consolidated["weight"],
        torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )
    torch.testing.assert_close(consolidated["other"], consolidated["weight"] + 100)

    with open(checkpoint_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    assert index["weight_map"] == {
        fqn: filename
        for filename, shard in (
            ("model-00001-of-00002.safetensors", first_shard),
            ("model-00002-of-00002.safetensors", second_shard),
        )
        for fqn in shard
    }


@pytest.mark.parametrize(
    ("item_sizes", "expected_owners"),
    [
        ({1: 9, 2: 8, 3: 7}, {1: 0, 2: 1, 3: 1}),
        ({1: 10, 2: 1}, {1: 0, 2: 1}),
    ],
)
def test_assign_items_to_owners_by_size_balances_owners(
    item_sizes: dict[int, int],
    expected_owners: dict[int, int],
) -> None:
    assert (
        _assign_items_to_owners_by_size(
            item_sizes,
            num_owners=2,
        )
        == expected_owners
    )


def test_consolidate_hf_safetensors_ignores_and_logs_mapping_fqns_not_in_checkpoint(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[2:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

    with caplog.at_level(logging.INFO):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={
                "weight": 1,
                "linear.weight_scale_inv": 99,
            },
            item_key="model",
        )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)
    with open(checkpoint_path / "model.safetensors.index.json") as f:
        index = json.load(f)
    assert index["weight_map"] == {"weight": "model-00001-of-00001.safetensors"}
    assert "linear.weight_scale_inv" in caplog.text


def test_consolidate_hf_safetensors_validates_mapping_only_indices(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[2:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

    with pytest.raises(ValueError, match="linear.weight_scale_inv"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={
                "weight": 1,
                "linear.weight_scale_inv": 0,
            },
            item_key="model",
        )


def test_consolidate_hf_safetensors_rejects_tensor_missing_from_mapping(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    other = torch.ones(2)
    safetensors_save_file(
        {
            "weight": full_weight[:2].contiguous(),
            "other": other,
        },
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {
            "weight": full_weight[2:].contiguous(),
            "other": other,
        },
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        metadata_by_path={
            ("weight",): _dtensor_metadata(),
            ("other",): _replicated_tensor_metadata(other),
        },
    )

    with pytest.raises(ValueError, match="other"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )


def test_consolidate_hf_safetensors_uses_replicated_metadata_for_plain_tensor(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    safetensors_save_file(
        {"plain": torch.ones(2)},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"plain": torch.zeros(2)},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        nested_path=("plain",),
        sharding_metadata=_replicated_tensor_metadata(torch.ones(2)),
    )

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["plain"], torch.ones(2))


def test_consolidate_hf_safetensors_rejects_shape_mismatch_against_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[2:].reshape(4).contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

    # Rank 1 saved its shard flattened, so the header shape disagrees with the
    # shape the sharding metadata implies even though the byte count matches.
    with pytest.raises(ValueError, match="safetensors shape"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )


def test_consolidate_hf_safetensors_rejects_dtype_mismatch_against_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[2:].to(torch.float16).contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

    # Rank 1 saved half precision, so the header dtype disagrees with the dtype
    # the sharding metadata declares.
    with pytest.raises(ValueError, match="safetensors dtype"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )


def test_consolidate_hf_safetensors_ignores_unused_rank_layout(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight.contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        rank_to_file_path={0: "model_0.safetensors", 1: None},
        sharding_metadata=_dtensor_metadata(placements=(ReplicateSpec(),)),
    )

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        fqn_to_index_mapping={"weight": 1},
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)


def test_consolidate_hf_safetensors_validates_all_replicated_sources(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight.reshape(2, 4)},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        sharding_metadata=_dtensor_metadata(placements=(ReplicateSpec(),)),
    )

    with pytest.raises(ValueError, match="safetensors shape"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )


def test_consolidate_hf_safetensors_rejects_input_output_path_overlap(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()

    full_weight = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model-00001-of-00001.safetensors",
    )
    safetensors_save_file(
        {"weight": full_weight[2:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(
        checkpoint_path,
        rank_to_file_path={
            0: "model-00001-of-00001.safetensors",
            1: "model_1.safetensors",
        },
    )

    with pytest.raises(ValueError, match="must not overwrite"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )
