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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    _atomic_stream_write,
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


def _write_metadata(
    checkpoint_path: Path,
    rank_to_file_path: dict[int, str | None] | None = None,
    sharding_metadata: DTensorShardingMetadata | None = None,
    nested_path: NestedPath = ("weight",),
    include_sharding_metadata: bool = True,
    nested_paths: tuple[NestedPath, ...] | None = None,
) -> None:
    if rank_to_file_path is None:
        rank_to_file_path = {
            0: "model_0.safetensors",
            1: "model_1.safetensors",
        }
    nested_path_to_metadata = {}
    if include_sharding_metadata:
        if sharding_metadata is None:
            sharding_metadata = _dtensor_metadata()
        if nested_paths is None:
            nested_paths = (nested_path,)
        nested_path_to_metadata = {
            path: [
                GlobalObjectMetadata(
                    sharding_metadata=sharding_metadata,
                    ranks=(0, 1),
                )
            ]
            for path in nested_paths
        }
    metadata = DistributedMetadata(
        metadata={
            "model": DistributedItemMetadata(
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


def test_consolidate_hf_safetensors_defaults_all_fqns_to_one_file(
    tmp_path: Path,
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
        {"weight": full_weight[2:].contiguous()},
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

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
        include_sharding_metadata=False,
    )

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["plain"], plain)


def test_consolidate_hf_safetensors_discovers_plain_tensor_on_other_rank(
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

    consolidate_hf_safetensors_checkpoint(
        os.fspath(checkpoint_path),
        item_key="model",
    )

    consolidated = safetensors_load_file(
        checkpoint_path / "model-00001-of-00001.safetensors"
    )
    torch.testing.assert_close(consolidated["weight"], full_weight)
    torch.testing.assert_close(consolidated["plain"], plain)


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


def test_consolidate_hf_safetensors_synchronizes_two_ranks(
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
    safetensors_save_file(
        {"weight": full_weight[:2].contiguous()},
        checkpoint_path / "model_0.safetensors",
    )
    safetensors_save_file(
        {
            "weight": full_weight[2:].contiguous(),
            "other": full_weight[2:].clone(),
        },
        checkpoint_path / "model_1.safetensors",
    )
    _write_metadata(checkpoint_path)

    with pytest.raises(ValueError, match="other"):
        consolidate_hf_safetensors_checkpoint(
            os.fspath(checkpoint_path),
            fqn_to_index_mapping={"weight": 1},
            item_key="model",
        )


@pytest.mark.parametrize("validate_tensor_metadata", [False, True])
def test_consolidate_hf_safetensors_treats_plain_tensor_on_multiple_ranks_as_replicated(
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
    _write_metadata(checkpoint_path, include_sharding_metadata=False)

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
