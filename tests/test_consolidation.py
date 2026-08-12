# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Owner(s): ["oncall: pytorch_checkpointing"]

import io
import json
import logging
import os
import pickle
import socket
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
)
from torch_checkpointing.hf.consolidation import (
    _assign_items_to_owners_by_size,
    _atomic_stream_write,
    _process_output_file,
    _read_safetensors_file_metadata_by_rank,
    _read_tensor_data,
    consolidate_hf_safetensors_checkpoint,
    HFSafetensorByteAddress,
    SafetensorsFileMetadata,
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


def test_safetensors_file_metadata_reads_file_header(
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
    assert weight.source_rank == 3
    assert weight.local_shape == (2, 3)
    assert weight.torch_dtype == torch.float32
    assert weight.global_shape is None
    assert weight.global_offsets is None
    address = weight.byte_address
    assert address.file_path == os.fspath(file_path)
    assert address.start_byte_offset > 8
    assert address.num_bytes == 6 * torch.empty((), dtype=torch.float32).element_size()


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

    with pytest.raises(
        AssertionError,
        match="Scalar tensors require exactly one source slice",
    ):
        _process_output_file(
            os.fspath(tmp_path / "output.safetensors"),
            {"scalar": tensor_slices},
            storage,
        )


def test_read_tensor_data_handles_short_reads() -> None:
    class ShortReadBytesIO(io.BytesIO):
        def readinto(self, buffer) -> int:
            return super().readinto(buffer[:2])

    stream = ShortReadBytesIO(b"prefixpayloadsuffix")
    address = HFSafetensorByteAddress(
        file_path="checkpoint.safetensors",
        start_byte_offset=6,
        end_byte_offset=13,
    )

    assert bytes(_read_tensor_data(stream, address)) == b"payload"


def test_read_tensor_data_rejects_truncated_input() -> None:
    stream = io.BytesIO(b"short")
    address = HFSafetensorByteAddress(
        file_path="checkpoint.safetensors",
        start_byte_offset=0,
        end_byte_offset=10,
    )

    with pytest.raises(EOFError, match="Expected 10 bytes"):
        _read_tensor_data(stream, address)


def test_consolidate_hf_safetensors_reads_torch_checkpointing_metadata(
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

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        fqn_to_index_mapping={"weight": 1},
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)
    assert (checkpoint_path / "model.safetensors.index.json").exists()


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


@pytest.mark.parametrize("validate_tensor_metadata", [False, True])
def test_consolidate_hf_safetensors_uses_replicated_metadata_for_plain_tensor(
    tmp_path: Path,
    validate_tensor_metadata: bool,
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
        validate_tensor_metadata=validate_tensor_metadata,
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["plain"], torch.ones(2))


def test_consolidate_hf_safetensors_can_disable_tensor_metadata_validation(
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

    with pytest.raises(ValueError, match="safetensors shape"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        fqn_to_index_mapping={"weight": 1},
        item_key="model",
        validate_tensor_metadata=False,
    )
    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)


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
