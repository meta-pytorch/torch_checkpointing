# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torch_checkpointing import default_resharder as _dr
from torch_checkpointing.checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
    TorchSerialization,
)
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    ShardSpec,
)
from torch_checkpointing.hf.metadata import (
    HuggingFaceSafetensorsDistributedMetadataFormat,
)
from torch_checkpointing.metadata_serialization import (
    load_distributed_metadata,
    METADATA_FILE_NAME,
    RankAddressableDistributedMetadataFormat,
    TorchDistributedMetadataFormat,
)
from torch_checkpointing.safetensors_metadata import SafetensorsFileMetadata
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig


def _manager() -> CheckpointManager:
    config = CheckpointManager.Config.with_sync_save()
    config.items = {
        "model": ItemSpec(resharder=DefaultResharder()),
    }
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    return config.build()


def test_manager_loads_single_file_hf_safetensors_in_place(tmp_path: Path) -> None:
    expected = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    target_tensor = torch.zeros((3, 4), dtype=torch.float32)
    target = {"weight": target_tensor}

    manager = _manager()
    try:
        loaded = manager.load(
            tmp_path,
            into={"model": target},
            strict=True,
            metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
        )
    finally:
        manager.close()

    assert loaded["model"] is target
    assert target["weight"] is target_tensor
    torch.testing.assert_close(target_tensor, expected.float())
    assert not (tmp_path / "metadata.pkl").exists()


def test_manager_rejects_hf_load_without_a_resharder(tmp_path: Path) -> None:
    """A direct read resolves one shard, which is wrong for any shard count.

    Rejecting it uniformly keeps the rule independent of how many files the
    export happens to have, so a model that outgrows one file cannot start
    loading partial weights.
    """
    expected = torch.arange(6, dtype=torch.float32)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    torch.save(7, tmp_path / "epoch_0.pt")
    target = {"weight": torch.zeros_like(expected)}
    config = CheckpointManager.Config.with_sync_save()
    config.items = {"model": ItemSpec(), "epoch": ItemSpec()}
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    try:
        with pytest.raises(ValueError, match="requires a resharder"):
            manager.load(
                tmp_path,
                into={"model": target, "epoch": 0},
                strict=True,
                metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
            )
    finally:
        manager.close()


def test_manager_loads_single_file_hf_with_a_resharder(tmp_path: Path) -> None:
    expected = torch.arange(6, dtype=torch.float32)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    torch.save(7, tmp_path / "epoch_0.pt")
    target = {"weight": torch.zeros_like(expected)}
    config = CheckpointManager.Config.with_sync_save()
    config.items = {
        "model": ItemSpec(resharder=DefaultResharder()),
        "epoch": ItemSpec(),
    }
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    try:
        loaded = manager.load(
            tmp_path,
            into={"model": target, "epoch": 0},
            strict=True,
            metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
        )
    finally:
        manager.close()

    torch.testing.assert_close(target["weight"], expected)
    assert loaded["epoch"] == 7


def test_single_file_hf_metadata_reads_header_once(tmp_path: Path) -> None:
    save_file({"weight": torch.arange(6)}, tmp_path / "model.safetensors")
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    with patch.object(
        storage,
        "stream_read",
        wraps=storage.stream_read,
    ) as stream_read:
        metadata = HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(
            tmp_path, storage
        )

    assert metadata is not None
    assert stream_read.call_count == 1


def test_hf_source_metadata_overrides_the_configured_item_layout(
    tmp_path: Path,
) -> None:
    expected = torch.arange(6, dtype=torch.float32)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    torch.save(
        {"weight": torch.full_like(expected, -1)},
        tmp_path / "configured-source.pt",
    )
    config = CheckpointManager.Config.with_sync_save()
    config.items = {
        "model": ItemSpec(
            layout=LayoutInfo("configured-source.pt", TorchSerialization()),
            resharder=DefaultResharder(),
        )
    }
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()
    target = {"weight": torch.zeros_like(expected)}

    try:
        manager.load(
            tmp_path,
            into={"model": target},
            strict=True,
            metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
        )
    finally:
        manager.close()

    torch.testing.assert_close(target["weight"], expected)


def test_direct_load_prefers_configured_layout_to_hf_metadata(tmp_path: Path) -> None:
    hf_value = torch.arange(6, dtype=torch.float32)
    configured_value = torch.full_like(hf_value, -1)
    save_file({"weight": hf_value}, tmp_path / "model.safetensors")
    torch.save(
        {"weight": configured_value},
        tmp_path / "configured-source.pt",
    )
    config = CheckpointManager.Config.with_sync_save()
    config.items = {
        "model": ItemSpec(
            layout=LayoutInfo("configured-source.pt", TorchSerialization()),
        )
    }
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()
    target = {"weight": torch.zeros_like(hf_value)}

    try:
        manager.load(str(tmp_path), into={"model": target}, strict=True)
    finally:
        manager.close()

    torch.testing.assert_close(target["weight"], configured_value)


def test_distributed_metadata_loader_recognizes_hf_safetensors(
    tmp_path: Path,
) -> None:
    expected = torch.arange(6, dtype=torch.float32)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    metadata = load_distributed_metadata(
        tmp_path,
        LocalFileSystemStorageConfig(use_direct_io=False).create_storage(),
        formats=(HuggingFaceSafetensorsDistributedMetadataFormat,),
    )

    assert metadata is not None
    assert set(metadata.metadata) == {"model"}
    assert set(metadata.metadata["model"].nested_path_to_metadata) == {("weight",)}
    assert all(
        layout is not None
        and isinstance(layout.serialization_format, SafetensorsSerialization)
        for layout in metadata.metadata["model"].rank_to_layout_info.values()
    )


def test_native_metadata_takes_precedence_over_hf_single_file(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "native"
    expected = torch.arange(6, dtype=torch.float32)
    config = CheckpointManager.Config.with_sync_save()
    config.items = {
        "model": ItemSpec(
            layout=LayoutInfo("model.safetensors", SafetensorsSerialization()),
            resharder=DefaultResharder(),
        )
    }
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    writer = config.build()
    try:
        writer.save(str(checkpoint_path), {"model": {"weight": expected}})
    finally:
        writer.close()

    assert (checkpoint_path / "metadata.pkl").exists()
    assert (checkpoint_path / "model.safetensors").exists()
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    assert (
        HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(
            checkpoint_path, storage
        )
        is not None
    )
    target = {"weight": torch.zeros_like(expected)}
    reader = config.build()
    try:
        reader.load(str(checkpoint_path), into={"model": target}, strict=True)
    finally:
        reader.close()

    torch.testing.assert_close(target["weight"], expected)


def test_manager_loads_indexed_hf_safetensors_and_ignores_extra_keys(
    tmp_path: Path,
) -> None:
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = torch.arange(4, dtype=torch.int64)
    save_file(
        {"first": first, "extra": torch.ones(2)},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"second": second},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {},
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "extra": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                },
            }
        )
    )
    target = {
        "first": torch.zeros_like(first),
        "second": torch.zeros_like(second),
    }

    manager = _manager()
    try:
        manager.load(
            tmp_path,
            into={"model": target},
            strict=True,
            metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
        )
    finally:
        manager.close()

    torch.testing.assert_close(target["first"], first)
    torch.testing.assert_close(target["second"], second)


def test_manager_loads_indexed_hf_safetensors_with_dotted_keys(
    tmp_path: Path,
) -> None:
    first = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    second = torch.arange(4, dtype=torch.float32)
    save_file(
        {"layers.0.weight": first},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"layers.1.weight": second},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.weight": "model-00001-of-00002.safetensors",
                    "layers.1.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    target = {
        "layers.0.weight": torch.zeros_like(first),
        "layers.1.weight": torch.zeros_like(second),
    }

    manager = _manager()
    try:
        manager.load(
            tmp_path,
            into={"model": target},
            strict=True,
            metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
        )
    finally:
        manager.close()

    torch.testing.assert_close(target["layers.0.weight"], first)
    torch.testing.assert_close(target["layers.1.weight"], second)


def test_manager_loads_zero_length_hf_tensors(tmp_path: Path) -> None:
    expected = {
        "empty_first": torch.empty(0, dtype=torch.float32),
        "value": torch.arange(3, dtype=torch.float32),
        "empty_last": torch.empty(0, dtype=torch.float32),
    }
    save_file(expected, tmp_path / "model.safetensors")
    target = {key: torch.zeros_like(value) for key, value in expected.items()}

    manager = _manager()
    try:
        manager.load(
            tmp_path,
            into={"model": target},
            strict=True,
            metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
        )
    finally:
        manager.close()

    for key in expected:
        torch.testing.assert_close(target[key], expected[key])


def test_manager_strict_hf_load_reports_missing_requested_key(tmp_path: Path) -> None:
    save_file({"present": torch.ones(2)}, tmp_path / "model.safetensors")
    target = {
        "present": torch.zeros(2),
        "missing": torch.zeros(3),
    }

    manager = _manager()
    try:
        with pytest.raises(RuntimeError, match="missing"):
            manager.load(
                tmp_path,
                into={"model": target},
                strict=True,
                metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
            )
    finally:
        manager.close()


def test_manager_reports_missing_hf_index_shard(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )

    manager = _manager()
    try:
        with pytest.raises(FileNotFoundError):
            manager.load(
                tmp_path,
                into={"model": {"weight": torch.zeros(2)}},
                strict=True,
                metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
            )
    finally:
        manager.close()


@pytest.mark.parametrize(
    "shard_name",
    [
        "../model-00001-of-00001.safetensors",
        "nested/../../model-00001-of-00001.safetensors",
        "/model-00001-of-00001.safetensors",
        "",
        ".",
        "..",
    ],
)
def test_hf_index_rejects_shard_names_outside_the_checkpoint_directory(
    tmp_path: Path, shard_name: str
) -> None:
    save_file(
        {"weight": torch.arange(2, dtype=torch.float32)},
        tmp_path / "model-00001-of-00001.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard_name}})
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    with pytest.raises(ValueError, match="is not a file inside"):
        HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(tmp_path, storage)


def test_hf_index_canonicalizes_shard_paths_inside_the_checkpoint_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "nested").mkdir()
    save_file(
        {"first": torch.arange(2, dtype=torch.float32)},
        tmp_path / "model-00001-of-00002.safetensors",
    )
    save_file(
        {"second": torch.arange(3, dtype=torch.float32)},
        tmp_path / "nested" / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    # Two spellings of the same top-level shard, and a shard in a
                    # subdirectory reached through a "..".
                    "first": "./model-00001-of-00002.safetensors",
                    "first.alias": "model-00001-of-00002.safetensors",
                    "second": "nested/../nested/model-00002-of-00002.safetensors",
                }
            }
        )
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    metadata = HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(
        tmp_path, storage
    )

    assert metadata is not None
    layouts = metadata.metadata["model"].rank_to_layout_info
    assert sorted(layout.file_path for layout in layouts.values()) == [
        "model-00001-of-00002.safetensors",
        "nested/model-00002-of-00002.safetensors",
    ]


def test_hf_resharder_reads_only_target_shard_range(tmp_path: Path) -> None:
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file(
        {
            "unused": torch.ones(1_000_000, dtype=torch.float32),
            "weight": expected,
        },
        tmp_path / "model.safetensors",
    )
    target_metadata = {
        ("weight",): DTensorShardingMetadata(
            global_shape=(3, 4),
            dtype=str(expected.dtype),
            stride=(4, 1),
            mesh_spec=DeviceMeshSpec(
                device_type="cpu",
                mesh_shape=(2,),
                mesh_data=(0, 1),
            ),
            placements=(ShardSpec(1),),
        )
    }
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    resharder = DefaultResharder()
    source_metadata = (lambda m: None if m is None else m.metadata["model"])(
        HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(tmp_path, storage)
    )
    assert source_metadata is not None
    target_tensor = torch.zeros((3, 2), dtype=torch.float32)

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.default_resharder.dist.get_rank",
            return_value=1,
        ),
        patch.object(
            _dr,
            "_read_exact",
            wraps=_dr._read_exact,
        ) as read_exact,
    ):
        missing = resharder.load(
            source_path=tmp_path,
            item_key="model",
            target_metadata=target_metadata,
            source_metadata=source_metadata,
            target={"weight": target_tensor},
            storage=storage,
        )

    assert missing == []
    torch.testing.assert_close(target_tensor, expected[:, 2:])
    # One span read covering rows 0..2 of the two requested columns: far less
    # than the file, which also holds a 4 MB tensor we never touch.
    assert [len(call.args[2]) for call in read_exact.call_args_list] == [40]


def test_hf_resharder_reuses_shard_headers_read_for_metadata(tmp_path: Path) -> None:
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    target_metadata = {
        ("weight",): DTensorShardingMetadata(
            global_shape=(3, 4),
            dtype=str(expected.dtype),
            stride=(4, 1),
            mesh_spec=DeviceMeshSpec(
                device_type="cpu",
                mesh_shape=(2,),
                mesh_data=(0, 1),
            ),
            placements=(ShardSpec(1),),
        )
    }
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    # Building the metadata reads each shard header once.
    source_metadata = (lambda m: None if m is None else m.metadata["model"])(
        HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(tmp_path, storage)
    )
    assert source_metadata is not None
    target_tensor = torch.zeros((3, 2), dtype=torch.float32)

    with (
        patch(
            "torch_checkpointing.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.default_resharder.dist.get_rank",
            return_value=1,
        ),
        patch.object(
            SafetensorsFileMetadata,
            "from_stream",
            side_effect=AssertionError("shard header read a second time"),
        ),
    ):
        missing = DefaultResharder().load(
            source_path=tmp_path,
            item_key="model",
            target_metadata=target_metadata,
            source_metadata=source_metadata,
            target={"weight": target_tensor},
            storage=storage,
        )

    assert missing == []
    torch.testing.assert_close(target_tensor, expected[:, 2:])


def test_hf_resharder_falls_through_for_native_checkpoint_without_metadata(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "native"
    expected = torch.arange(6, dtype=torch.float32)
    native_config = CheckpointManager.Config.with_sync_save()
    native_config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    writer = native_config.build()
    try:
        writer.save(str(checkpoint_path), {"model": {"weight": expected}})
    finally:
        writer.close()

    target = {"weight": torch.zeros_like(expected)}
    reader = _manager()
    try:
        reader.load(str(checkpoint_path), into={"model": target}, strict=True)
    finally:
        reader.close()

    torch.testing.assert_close(target["weight"], expected)


def test_hf_metadata_uses_configured_layout_for_an_undescribed_item(
    tmp_path: Path,
) -> None:
    save_file({"weight": torch.ones(2)}, tmp_path / "model.safetensors")
    torch.save({"momentum": torch.full((2,), 3.0)}, tmp_path / "optimizer.pt")
    torch.save({"position": 11}, tmp_path / "dataloader_0.pt")
    config = CheckpointManager.Config.with_sync_save()
    config.items = {
        "model": ItemSpec(resharder=DefaultResharder()),
        "optimizer": ItemSpec(layout=LayoutInfo("optimizer.pt", TorchSerialization())),
        "dataloader": ItemSpec(),
    }
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()

    try:
        with patch(
            "torch_checkpointing.storage.filesystem.LocalFileSystemStorage.ls",
            side_effect=AssertionError("storage.ls must not be called"),
        ):
            loaded = manager.load(
                tmp_path,
                into={
                    "model": {"weight": torch.zeros(2)},
                    "optimizer": {"momentum": torch.zeros(2)},
                    "dataloader": {"position": 0},
                },
                strict=True,
                metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
            )
    finally:
        manager.close()

    torch.testing.assert_close(loaded["model"]["weight"], torch.ones(2))
    torch.testing.assert_close(loaded["optimizer"]["momentum"], torch.full((2,), 3.0))
    assert loaded["dataloader"]["position"] == 11


def test_native_metadata_takes_precedence_over_hf_index(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "native"
    expected = torch.arange(6, dtype=torch.float32)
    writer = _manager()
    try:
        writer.save(str(checkpoint_path), {"model": {"weight": expected}})
    finally:
        writer.close()

    shard_path = checkpoint_path / "hf-model.safetensors"
    save_file({"weight": torch.full_like(expected, -1)}, shard_path)
    (checkpoint_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": shard_path.name}})
    )

    target = {"weight": torch.zeros_like(expected)}
    reader = _manager()
    try:
        loaded = reader.load(
            str(checkpoint_path),
            into={"model": target},
            strict=True,
        )
    finally:
        reader.close()

    assert loaded["model"] is target
    torch.testing.assert_close(target["weight"], expected)


def test_hf_index_describes_what_it_holds_not_what_was_asked_for(
    tmp_path: Path,
) -> None:
    """The directory describes itself; absent keys surface as missing, later."""
    shard_path = tmp_path / "other.safetensors"
    save_file({"other": torch.arange(6, dtype=torch.float32)}, shard_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"other": shard_path.name}})
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    described = HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(
        tmp_path, storage
    )

    assert described is not None
    assert set(described.metadata["model"].nested_path_to_metadata) == {("other",)}

    manager = _manager()
    try:
        with pytest.raises(RuntimeError):
            manager.load(
                tmp_path,
                into={"model": {"weight": torch.zeros(6)}},
                strict=True,
                metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
            )
    finally:
        manager.close()


def test_hf_directory_without_shards_is_not_claimed(tmp_path: Path) -> None:
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()
    assert (
        HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(tmp_path, storage)
        is None
    )


def test_manager_rejects_indexed_hf_direct_load_instead_of_reading_one_shard(
    tmp_path: Path,
) -> None:
    """Without this, rank 0 resolves shard 0 and 'second' stays zero.

    With into=None the omission is invisible: the walker has no target to
    compare against, so strict=True reports nothing missing either.
    """
    save_file(
        {"first": torch.tensor([10.0])}, tmp_path / "model-00001-of-00002.safetensors"
    )
    save_file(
        {"second": torch.tensor([20.0])}, tmp_path / "model-00002-of-00002.safetensors"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    config = CheckpointManager.Config.with_sync_save()
    config.items = {"model": ItemSpec()}
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()
    target = {"first": torch.zeros(1), "second": torch.zeros(1)}

    try:
        with pytest.raises(ValueError, match="requires a resharder"):
            manager.load(
                tmp_path,
                into={"model": target},
                metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
            )
    finally:
        manager.close()

    assert target["second"].item() == 0.0


class _SkippingDefaultResharder(DefaultResharder):
    @property
    def skip_resharding(self) -> bool:
        return True


def test_manager_rejects_a_direct_hf_item_when_another_item_has_a_resharder(
    tmp_path: Path,
) -> None:
    """Another item's resharder avoids the direct-read fast path, not the check."""
    save_file(
        {"first": torch.tensor([10.0])}, tmp_path / "model-00001-of-00002.safetensors"
    )
    save_file(
        {"second": torch.tensor([20.0])}, tmp_path / "model-00002-of-00002.safetensors"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    torch.save({"step": torch.tensor([1.0])}, tmp_path / "extra_0.pt")
    config = CheckpointManager.Config.with_sync_save()
    config.items = {
        "model": ItemSpec(resharder=_SkippingDefaultResharder()),
        "extra": ItemSpec(resharder=DefaultResharder()),
    }
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    manager = config.build()
    target = {"first": torch.zeros(1), "second": torch.zeros(1)}

    try:
        with pytest.raises(ValueError, match=r"part of \['model'\]"):
            manager.load(
                tmp_path,
                into={"model": target, "extra": {"step": torch.zeros(1)}},
                metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
            )
    finally:
        manager.close()

    assert target["first"].item() == 0.0


def test_hf_metadata_format_is_not_rank_addressable() -> None:
    assert not issubclass(
        HuggingFaceSafetensorsDistributedMetadataFormat,
        RankAddressableDistributedMetadataFormat,
    )
    assert issubclass(
        TorchDistributedMetadataFormat, RankAddressableDistributedMetadataFormat
    )


def test_plain_path_does_not_read_hf_metadata(tmp_path: Path) -> None:
    """Hugging Face metadata is read only when the location names its format."""
    save_file(
        {"first": torch.tensor([10.0])}, tmp_path / "model-00001-of-00002.safetensors"
    )
    save_file(
        {"second": torch.tensor([20.0])}, tmp_path / "model-00002-of-00002.safetensors"
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "first": "model-00001-of-00002.safetensors",
                    "second": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    target = {"first": torch.zeros(1), "second": torch.zeros(1)}

    manager = _manager()
    try:
        with pytest.raises(
            FileNotFoundError,
            match=r"Missing file .*model_0\.pt for checkpoint item 'model'",
        ):
            manager.load(str(tmp_path), into={"model": target}, strict=True)
    finally:
        manager.close()

    assert target["first"].item() == 0.0


def test_explicit_metadata_format_must_be_present(tmp_path: Path) -> None:
    save_file({"weight": torch.ones(2)}, tmp_path / "model.safetensors")

    manager = _manager()
    try:
        with pytest.raises(
            FileNotFoundError, match=r"tried \['TorchDistributedMetadataFormat'\]"
        ):
            manager.load(
                tmp_path,
                into={"model": {"weight": torch.zeros(2)}},
                metadata_format=TorchDistributedMetadataFormat,
            )
    finally:
        manager.close()


def test_hf_tensor_in_two_shards_is_rejected(tmp_path: Path) -> None:
    save_file({"weight": torch.ones(2)}, tmp_path / "model-00001-of-00002.safetensors")
    save_file(
        {"weight": torch.zeros(2), "bias": torch.zeros(2)},
        tmp_path / "model-00002-of-00002.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "weight": "model-00001-of-00002.safetensors",
                    "bias": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    with pytest.raises(ValueError, match="'weight' appears in more than one"):
        HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(tmp_path, storage)


def test_hf_index_with_an_empty_weight_map_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}})
    )
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    with pytest.raises(ValueError, match="contains no tensors"):
        HuggingFaceSafetensorsDistributedMetadataFormat.maybe_load(tmp_path, storage)


def _direct_manager() -> CheckpointManager:
    config = CheckpointManager.Config.with_sync_save()
    config.items = {"model": ItemSpec()}
    config.default = None
    config.storage_config = LocalFileSystemStorageConfig(use_direct_io=False)
    return config.build()


def test_named_rank_addressable_format_is_not_loaded_for_a_direct_read(
    tmp_path: Path,
) -> None:
    expected = torch.arange(4, dtype=torch.float32)
    torch.save({"weight": expected}, tmp_path / "model_0.pt")
    target = {"weight": torch.zeros_like(expected)}

    manager = _direct_manager()
    try:
        manager.load(
            tmp_path,
            into={"model": target},
            strict=True,
            metadata_format=TorchDistributedMetadataFormat,
        )
    finally:
        manager.close()

    assert not (tmp_path / METADATA_FILE_NAME).exists()
    torch.testing.assert_close(target["weight"], expected)


def test_direct_read_of_a_misplaced_file_does_not_load_metadata(
    tmp_path: Path,
) -> None:
    torch.save({"weight": torch.ones(2)}, tmp_path / "elsewhere.pt")

    manager = _direct_manager()
    try:
        with (
            patch.object(
                TorchDistributedMetadataFormat,
                "maybe_load",
                side_effect=AssertionError("a direct read loaded metadata"),
            ),
            pytest.raises(
                FileNotFoundError,
                match=r"Missing file .*model_0\.pt for checkpoint item 'model'",
            ),
        ):
            manager.load(
                tmp_path,
                into={"model": {"weight": torch.zeros(2)}},
                metadata_format=TorchDistributedMetadataFormat,
            )
    finally:
        manager.close()


@pytest.fixture
def cpu_device_mesh(tmp_path: Path) -> Iterator[DeviceMesh]:
    dist.init_process_group(
        backend="gloo",
        init_method=(tmp_path / "process_group").as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def test_hf_export_reshards_into_a_dtensor_with_matching_sharding(
    tmp_path: Path,
    cpu_device_mesh: DeviceMesh,
) -> None:
    """The export's bare tensors cannot be copied into a DTensor directly.

    The synthesized sharding (replicated on a one-rank CPU mesh, same dtype)
    matches the target exactly, so only the safetensors rule keeps this load
    on the resharder.
    """
    expected = torch.arange(4, dtype=torch.float32)
    save_file({"weight": expected}, tmp_path / "model.safetensors")
    target = distribute_tensor(torch.zeros(4), cpu_device_mesh, [Replicate()])

    manager = _manager()
    try:
        manager.load(
            tmp_path,
            into={"model": {"weight": target}},
            strict=True,
            metadata_format=HuggingFaceSafetensorsDistributedMetadataFormat,
        )
    finally:
        manager.close()

    torch.testing.assert_close(target.to_local(), expected)
