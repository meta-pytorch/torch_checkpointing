# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import _StridedShard, Placement
from torch_checkpointing.checkpoint_base import CheckpointInfo, CheckpointItem
from torch_checkpointing.checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
    TorchSerialization,
)
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.checkpointable_tensor_metadata import (
    CheckpointableTensorShardingMetadata,
)
from torch_checkpointing.checkpointable_tensor_resharder import (
    CheckpointableTensorResharder,
)
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
    ShardingMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    ReplicateSpec,
)
from torch_checkpointing.metadata_serialization import METADATA_FILE_NAME
from torch_checkpointing.resharding import LoadPlan
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import CheckpointPath, NestedPath, RankInfo

from .resharding_test_utils import (
    dtensor_local_shard,
    fake_process_group,
    plan_and_execute,
)

_GLOBAL = torch.arange(48, dtype=torch.float32).reshape(8, 6)


def _rows(*blocks: tuple[int, int], local_rows: int | None = None) -> torch.Tensor:
    """A local tensor of row blocks of ``_GLOBAL``, annotated as one shard each.

    The blocks are stacked in order, after padding rows that bring the local
    tensor to ``local_rows`` rows, if it is set.
    """
    shard_rows = sum(stop - start for start, stop in blocks)
    padding = 0 if local_rows is None else local_rows - shard_rows
    local = torch.cat(
        [torch.full((padding, 6), -1.0)]
        + [_GLOBAL[start:stop] for start, stop in blocks]
    )
    local_starts = [padding]
    for start, stop in blocks[:-1]:
        local_starts.append(local_starts[-1] + stop - start)
    return _annotate(
        local,
        global_offsets=tuple((start, 0) for start, _ in blocks),
        local_offsets=tuple((row, 0) for row in local_starts),
        local_sizes=tuple((stop - start, 6) for start, stop in blocks),
    )


def _annotate(
    local: torch.Tensor,
    *,
    global_offsets: tuple[tuple[int, ...], ...],
    local_offsets: tuple[tuple[int, ...], ...],
    local_sizes: tuple[tuple[int, ...], ...],
    global_shape: tuple[int, ...] = (8, 6),
) -> torch.Tensor:
    """Give ``local`` the attributes of a ``CheckpointableTensor``."""
    local.global_shape = global_shape
    local.global_offsets = global_offsets
    local.local_offsets = local_offsets
    local.local_sizes = local_sizes
    return local


def _metadata(tensor: torch.Tensor) -> CheckpointableTensorShardingMetadata:
    return CheckpointableTensorShardingMetadata.from_tensor(tensor)


def _source(
    path_to_rank_metadata: dict[NestedPath, dict[int, ShardingMetadata]],
    rank_to_layout_info: dict[int, LayoutInfo | None],
) -> DistributedItemMetadata:
    """Source metadata grouping the ranks that saved equal metadata, as on save."""
    nested_path_to_metadata: dict[NestedPath, list[GlobalObjectMetadata]] = {}
    for path, rank_metadata in path_to_rank_metadata.items():
        ranks_by_metadata: dict[ShardingMetadata, list[int]] = {}
        for rank, metadata in rank_metadata.items():
            ranks_by_metadata.setdefault(metadata, []).append(rank)
        nested_path_to_metadata[path] = [
            GlobalObjectMetadata(sharding_metadata=metadata, ranks=tuple(ranks))
            for metadata, ranks in ranks_by_metadata.items()
        ]
    return DistributedItemMetadata(
        nested_path_to_metadata=nested_path_to_metadata,
        rank_to_layout_info=rank_to_layout_info,
    )


def _dtensor(
    global_tensor: torch.Tensor,
    mesh_shape: tuple[int, ...],
    placements: Sequence[Placement],
) -> DTensor:
    """This rank's DTensor; call inside its fake process group."""
    mesh = init_device_mesh("cpu", mesh_shape)
    coordinate = mesh.get_coordinate()
    assert coordinate is not None
    local = dtensor_local_shard(global_tensor, mesh_shape, coordinate, placements)
    return DTensor.from_local(
        local.clone(),
        mesh,
        placements,
        run_check=False,
        shape=global_tensor.shape,
        stride=global_tensor.stride(),
    )


def test_extract_sharding_metadata_records_annotated_tensors_and_dtensors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    annotated = _rows((2, 4), (6, 8), local_rows=5)
    with fake_process_group(rank=1, world_size=4):
        dtensor = _dtensor(_GLOBAL, (4,), [Shard(0)])
        item = {"annotated": annotated, "dtensor": dtensor, "plain": torch.ones(2)}

        with caplog.at_level(logging.WARNING):
            metadata = CheckpointableTensorResharder().extract_sharding_metadata(
                "model", {**item, "step": 3}
            )

        assert metadata == {
            ("annotated",): CheckpointableTensorShardingMetadata(
                global_shape=(8, 6),
                global_offsets=((2, 0), (6, 0)),
                local_offsets=((1, 0), (3, 0)),
                local_sizes=((2, 6), (2, 6)),
                dtype="torch.float32",
            ),
            ("dtensor",): CheckpointableTensorShardingMetadata(
                global_shape=(8, 6),
                global_offsets=((2, 0),),
                local_offsets=((0, 0),),
                local_sizes=((2, 6),),
                dtype="torch.float32",
            ),
        }
    assert "Found 1 tensors in checkpoint item 'model'" in caplog.text
    assert "[('plain',)]" in caplog.text


def test_generate_load_plans_reads_each_block_once_from_the_lowest_rank() -> None:
    # Ranks 2 and 3 each saved the whole tensor, rank 1 saved rows 2-6, and
    # rank 0 saved rows 4-6 and nothing else of it.
    source = _source(
        {
            ("weight",): {
                3: _metadata(_rows((0, 8))),
                2: _metadata(_rows((0, 8))),
                1: _metadata(_rows((2, 6))),
                0: _metadata(_rows((4, 6), local_rows=3)),
            }
        },
        dict.fromkeys(range(4)),
    )
    weight = _rows((0, 8))

    info = CheckpointableTensorResharder().generate_load_plans(
        "model", {"weight": weight}, {("weight",): _metadata(weight)}, source
    )

    assert info.non_reshardable_paths == []
    assert info.nested_path_to_target_tensors.keys() == {("weight",)}
    assert info.nested_path_to_target_tensors[("weight",)] is weight

    def plan(start: int, stop: int, rank: int, src_start: int) -> LoadPlan:
        return LoadPlan(
            offsets=(start, 0),
            sizes=(stop - start, 6),
            src_rank=rank,
            src_fqn="weight",
            src_offsets=(src_start, 0),
            src_sizes=(stop - start, 6),
        )

    assert info.nested_path_to_load_plans == {
        ("weight",): [
            plan(4, 6, rank=0, src_start=1),
            plan(2, 4, rank=1, src_start=0),
            plan(0, 2, rank=2, src_start=0),
            plan(6, 8, rank=2, src_start=6),
        ]
    }


_REPLICATED_DTENSOR = DTensorShardingMetadata(
    global_shape=(8, 6),
    dtype="torch.float32",
    stride=(6, 1),
    mesh_spec=DeviceMeshSpec(device_type="cpu", mesh_shape=(1,), mesh_data=(0,)),
    placements=(ReplicateSpec(),),
)


@pytest.mark.parametrize(
    "saved",
    [
        pytest.param({}, id="missing-from-source"),
        pytest.param({0: _REPLICATED_DTENSOR}, id="saved-as-dtensor-metadata"),
        pytest.param(
            {
                0: CheckpointableTensorShardingMetadata(
                    global_shape=(8, 5),
                    global_offsets=((0, 0),),
                    local_offsets=((0, 0),),
                    local_sizes=((8, 5),),
                    dtype="torch.float32",
                )
            },
            id="different-global-shape",
        ),
        pytest.param(
            {0: _metadata(_rows((0, 4))), 1: _metadata(_rows((5, 8)))},
            id="row-4-never-saved",
        ),
    ],
)
def test_generate_load_plans_reports_targets_the_source_cannot_fill(
    saved: dict[int, ShardingMetadata],
    caplog: pytest.LogCaptureFixture,
) -> None:
    source = _source({("weight",): saved} if saved else {}, dict.fromkeys(range(2)))
    weight = _rows((0, 8))

    with caplog.at_level(logging.WARNING):
        info = CheckpointableTensorResharder().generate_load_plans(
            "model", {"weight": weight}, {("weight",): _metadata(weight)}, source
        )

    assert info.nested_path_to_load_plans == {}
    assert info.non_reshardable_paths == [("weight",)]
    assert "weight" in caplog.text


def test_generate_load_plans_reports_tensors_without_annotations() -> None:
    weight = _rows((0, 8))
    source = _source(
        {("weight",): {0: _metadata(weight)}, ("bias",): {0: _metadata(weight)}},
        {0: None},
    )
    target = {"weight": weight, "bias": torch.zeros(8, 6), "step": 3}

    info = CheckpointableTensorResharder().generate_load_plans(
        "model", target, {("weight",): _metadata(weight)}, source
    )

    assert list(info.nested_path_to_load_plans) == [("weight",)]
    assert info.non_reshardable_paths == [("bias",)]


@pytest.mark.parametrize(
    ("saved_by_rank_0", "saved_by_rank_1", "layout", "expected"),
    [
        pytest.param((0, 4), (4, 8), None, False, id="this-rank-saved-the-same"),
        pytest.param((4, 8), (0, 4), None, True, id="only-another-rank-matches"),
        pytest.param(
            (0, 4),
            (4, 8),
            LayoutInfo("model_0.safetensors", SafetensorsSerialization()),
            True,
            id="safetensors",
        ),
    ],
)
def test_should_reshard_compares_what_this_rank_saved(
    saved_by_rank_0: tuple[int, int],
    saved_by_rank_1: tuple[int, int],
    layout: LayoutInfo | None,
    expected: bool,
) -> None:
    source = _source(
        {
            ("weight",): {
                0: _metadata(_rows(saved_by_rank_0)),
                1: _metadata(_rows(saved_by_rank_1)),
            }
        },
        {0: layout, 1: None},
    )
    target = {("weight",): _metadata(_rows((0, 4)))}

    resharder = CheckpointableTensorResharder()

    assert resharder.should_reshard(source, target) is expected
    assert resharder.should_reshard(None, target) is False


@pytest.mark.parametrize("serialization", ["torch", "safetensors"])
def test_load_fills_an_annotated_target_from_annotated_sources(
    tmp_path: Path, serialization: str
) -> None:
    # Each source rank holds two row blocks that are not adjacent globally.
    saved = {0: _rows((0, 2), (4, 6), local_rows=5), 1: _rows((2, 4), (6, 8))}
    layouts: dict[int, LayoutInfo | None] = {}
    for rank, local in saved.items():
        if serialization == "torch":
            layouts[rank] = LayoutInfo(f"model_{rank}.pt", TorchSerialization())
            torch.save({"weight": local}, tmp_path / f"model_{rank}.pt")
        else:
            layouts[rank] = LayoutInfo(
                f"model_{rank}.safetensors", SafetensorsSerialization()
            )
            save_file({"weight": local}, tmp_path / f"model_{rank}.safetensors")
    source = _source(
        {("weight",): {rank: _metadata(local) for rank, local in saved.items()}},
        layouts,
    )
    # The target holds columns 3-6 of every row.
    weight = _annotate(
        torch.zeros(8, 3),
        global_offsets=((0, 3),),
        local_offsets=((0, 0),),
        local_sizes=((8, 3),),
    )

    missing = plan_and_execute(
        CheckpointableTensorResharder(),
        source_path=tmp_path,
        item_key="model",
        target_metadata={("weight",): _metadata(weight)},
        source_metadata=source,
        target={"weight": weight},
        storage=LocalFileSystemStorageConfig(use_direct_io=False).create_storage(),
    )

    assert missing == []
    torch.testing.assert_close(weight, _GLOBAL[:, 3:])


_MESH_PLACEMENTS = {
    "row-shard": ((4,), [Shard(0)]),
    "column-shard": ((4,), [Shard(1)]),
    "2d-shard": ((2, 2), [Shard(0), Shard(1)]),
    "strided-then-shard": ((2, 2), [_StridedShard(0, split_factor=2), Shard(0)]),
    "shard-then-strided": ((2, 2), [Shard(0), _StridedShard(0, split_factor=2)]),
    "replicate-then-shard": ((2, 2), [Replicate(), Shard(1)]),
}


@pytest.mark.parametrize(
    ("saved_as", "loaded_as"),
    [
        ("row-shard", "column-shard"),
        ("2d-shard", "strided-then-shard"),
        ("shard-then-strided", "replicate-then-shard"),
        ("replicate-then-shard", "shard-then-strided"),
        ("strided-then-shard", "2d-shard"),
    ],
)
def test_load_reshards_dtensors_across_placements(
    tmp_path: Path, saved_as: str, loaded_as: str
) -> None:
    global_tensor = torch.arange(35, dtype=torch.float32).reshape(7, 5)
    resharder = CheckpointableTensorResharder()
    saved: dict[int, ShardingMetadata] = {}
    for rank in range(4):
        with fake_process_group(rank, world_size=4):
            dtensor = _dtensor(global_tensor, *_MESH_PLACEMENTS[saved_as])
            saved[rank] = resharder.extract_sharding_metadata(
                "model", {"weight": dtensor}
            )[("weight",)]
        # Saved as safetensors would save it: the bare local tensor.
        torch.save({"weight": dtensor._local_tensor}, tmp_path / f"model_{rank}.pt")
    source = _source({("weight",): saved}, dict.fromkeys(range(4)))
    storage = LocalFileSystemStorageConfig(use_direct_io=False).create_storage()

    for rank in range(4):
        with fake_process_group(rank, world_size=4):
            expected = _dtensor(global_tensor, *_MESH_PLACEMENTS[loaded_as])
            target = {
                "weight": _dtensor(torch.zeros(7, 5), *_MESH_PLACEMENTS[loaded_as])
            }
            missing = plan_and_execute(
                resharder,
                source_path=tmp_path,
                item_key="model",
                target_metadata=resharder.extract_sharding_metadata("model", target),
                source_metadata=source,
                target=target,
                storage=storage,
            )

        assert missing == []
        torch.testing.assert_close(
            target["weight"]._local_tensor, expected._local_tensor
        )


def test_reader_reshards_annotated_tensors_and_reports_the_others(
    tmp_path: Path,
) -> None:
    resharder = CheckpointableTensorResharder()
    saved: dict[int, ShardingMetadata] = {}
    for rank, rows in enumerate([(0, 4), (4, 8)]):
        value = {"weight": _rows(rows), "bias": torch.full((6,), float(rank))}
        saved[rank] = resharder.extract_sharding_metadata("model", value)[("weight",)]
        torch.save(value, tmp_path / f"model_{rank}.pt")
    source = _source(
        {("weight",): saved},
        {rank: LayoutInfo(f"model_{rank}.pt", TorchSerialization()) for rank in saved},
    )
    with open(tmp_path / METADATA_FILE_NAME, "wb") as stream:
        pickle.dump(
            DistributedMetadata(metadata={"model": source}, world_size=2).to_dict(),
            stream,
        )
    weight = _annotate(
        torch.zeros(8, 6),
        global_offsets=((0, 0),),
        local_offsets=((0, 0),),
        local_sizes=((8, 6),),
    )
    target = {"weight": weight, "bias": torch.zeros(6)}
    reader = CheckpointReader(
        rank_info=RankInfo(
            global_rank=0, global_world_size=1, role_rank=0, role_world_size=1
        ),
        storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
    )

    _, missing = reader.read(
        str(tmp_path),
        CheckpointInfo(
            checkpoint_items={
                "model": CheckpointItem(value=target, resharder=resharder)
            }
        ).for_reads({"model": resharder.extract_sharding_metadata("model", target)}),
    )

    torch.testing.assert_close(weight, _GLOBAL)
    torch.testing.assert_close(target["bias"], torch.zeros(6))
    assert missing == [str(CheckpointPath("model", ("bias",)))]
