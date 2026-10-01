# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import random
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import (
    _StridedShard as DTensorStridedShard,
    Replicate as DTensorReplicate,
    Shard as DTensorShard,
)
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    GlobalObjectMetadata,
    ShardingMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    ReplicateSpec,
    ShardSpec,
    StridedShardSpec,
)
from torch_checkpointing.experimental.cooperative_resharding.default_resharder import (
    _load_plan_template_cache_key,
    _LoadPlanTemplate,
    _LoadPlanTemplateCache,
    _LoadPlanTemplateEntry,
    _ShardGeometryCache,
    compute_local_shard_info,
    DefaultResharder,
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


def _make_sharding_metadata(
    *,
    global_shape: tuple[int, ...] = (16, 4),
    mesh_shape: tuple[int, ...] = (4,),
    mesh_data: tuple[int, ...] = (7, 3, 11, 5),
    placements: tuple[ReplicateSpec | ShardSpec | StridedShardSpec, ...],
) -> DTensorShardingMetadata:
    return DTensorShardingMetadata(
        global_shape=global_shape,
        dtype="torch.float32",
        stride=(4, 1),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=mesh_shape,
            mesh_data=mesh_data,
        ),
        placements=placements,
    )


def _make_repeated_shape_planning_case(
    path_count: int,
    *,
    source_world_size: int = 128,
    target_world_size: int = 32,
) -> tuple[
    dict[tuple[str, int, str], DTensorShardingMetadata],
    DistributedItemMetadata,
]:
    global_shape = (source_world_size * target_world_size, 128)
    source_sharding = DTensorShardingMetadata(
        global_shape=global_shape,
        dtype="torch.bfloat16",
        stride=(global_shape[1], 1),
        mesh_spec=DeviceMeshSpec(
            device_type="cuda",
            mesh_shape=(source_world_size,),
            mesh_data=tuple(range(source_world_size)),
        ),
        placements=(ShardSpec(0),),
    )
    target_sharding = DTensorShardingMetadata(
        global_shape=global_shape,
        dtype="torch.bfloat16",
        stride=(global_shape[1], 1),
        mesh_spec=DeviceMeshSpec(
            device_type="cuda",
            mesh_shape=(target_world_size,),
            mesh_data=tuple(range(target_world_size)),
        ),
        placements=(ShardSpec(0),),
    )
    source_group = GlobalObjectMetadata(
        sharding_metadata=source_sharding,
        ranks=tuple(range(source_world_size)),
    )
    paths = [("layers", index, "weight") for index in range(path_count)]
    return (
        {path: target_sharding for path in paths},
        DistributedItemMetadata(
            nested_path_to_metadata={path: [source_group] for path in paths},
            rank_to_layout_info={},
        ),
    )


def _reference_local_shard_info(
    metadata: DTensorShardingMetadata,
    rank: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    coordinate = metadata.mesh_spec.get_coordinate(rank)
    assert coordinate is not None
    placements = []
    for placement in metadata.placements:
        if isinstance(placement, StridedShardSpec):
            placements.append(
                DTensorStridedShard(
                    placement.dim,
                    split_factor=placement.split_factor,
                )
            )
        elif isinstance(placement, ShardSpec):
            placements.append(DTensorShard(placement.dim))
        else:
            assert isinstance(placement, ReplicateSpec)
            placements.append(DTensorReplicate())
    return _compute_local_shape_and_global_offset(
        torch.Size(metadata.global_shape),
        torch.Size(metadata.mesh_spec.mesh_shape),
        list(coordinate),
        placements,
    )


def _random_fast_path_metadata(
    rng: random.Random,
) -> DTensorShardingMetadata:
    ndim = rng.randrange(5)
    global_shape = tuple(rng.randrange(33) for _ in range(ndim))
    mesh_ndim = rng.randrange(1, 4)
    mesh_shape = tuple(rng.randrange(1, 5) for _ in range(mesh_ndim))
    world_size = 1
    for size in mesh_shape:
        world_size *= size
    mesh_data = list(range(101, 101 + world_size * 3, 3))
    rng.shuffle(mesh_data)

    available_dims = list(range(ndim))
    rng.shuffle(available_dims)
    placements = []
    for _ in mesh_shape:
        if not available_dims or rng.randrange(3) == 0:
            placements.append(ReplicateSpec())
            continue
        shard_dim = available_dims.pop()
        if rng.randrange(2):
            placements.append(ShardSpec(shard_dim))
        else:
            placements.append(
                StridedShardSpec(
                    shard_dim,
                    split_factor=rng.randrange(1, 9),
                )
            )
    return DTensorShardingMetadata(
        global_shape=global_shape,
        dtype="torch.float32",
        stride=(1,) * ndim,
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=mesh_shape,
            mesh_data=tuple(mesh_data),
        ),
        placements=tuple(placements),
    )


def test_device_mesh_spec_indexes_noncontiguous_ranks() -> None:
    mesh_spec = DeviceMeshSpec(
        device_type="cpu",
        mesh_shape=(2, 2),
        mesh_data=(8, 2, 9, 4),
    )

    assert mesh_spec.get_coordinate(8) == (0, 0)
    assert mesh_spec.get_coordinate(2) == (0, 1)
    assert mesh_spec.get_coordinate(9) == (1, 0)
    assert mesh_spec.get_coordinate(4) == (1, 1)
    assert mesh_spec.get_coordinate(3) is None
    assert mesh_spec.get_coordinate(2) == (0, 1)


def test_compute_local_shard_info_fast_path_matches_dtensor_randomized() -> None:
    rng = random.Random(20260823)
    metadata_cases = [
        _make_sharding_metadata(
            global_shape=(),
            mesh_shape=(2,),
            mesh_data=(5, 9),
            placements=(ReplicateSpec(),),
        ),
        _make_sharding_metadata(
            global_shape=(0, 11),
            mesh_shape=(4, 3),
            mesh_data=tuple(range(12)),
            placements=(ShardSpec(0), StridedShardSpec(1, split_factor=5)),
        ),
        _make_sharding_metadata(
            global_shape=(17, 0, 5),
            mesh_shape=(2, 3, 4),
            mesh_data=tuple(range(24)),
            placements=(
                StridedShardSpec(2, split_factor=7),
                ReplicateSpec(),
                ShardSpec(0),
            ),
        ),
        _make_sharding_metadata(
            global_shape=(128, 1024, 4096),
            mesh_shape=(8, 4),
            mesh_data=tuple(range(32)),
            placements=(
                StridedShardSpec(0, split_factor=4),
                ShardSpec(0),
            ),
        ),
        _make_sharding_metadata(
            global_shape=(128, 4096, 1024),
            mesh_shape=(8, 4),
            mesh_data=tuple(range(32)),
            placements=(
                StridedShardSpec(0, split_factor=4),
                ShardSpec(0),
            ),
        ),
    ]
    metadata_cases.extend(_random_fast_path_metadata(rng) for _ in range(500))

    with patch(
        "torch_checkpointing.experimental.cooperative_resharding.default_resharder._compute_local_shape_and_global_offset",
        wraps=_compute_local_shape_and_global_offset,
    ) as fallback:
        checked_rank_count = 0
        for metadata in metadata_cases:
            ranks = metadata.mesh_spec.mesh_data
            selected_ranks = (
                ranks if len(ranks) <= 5 else (ranks[0], ranks[-1], ranks[2])
            )
            for rank in selected_ranks:
                assert compute_local_shard_info(
                    metadata,
                    rank,
                ) == _reference_local_shard_info(metadata, rank)
                checked_rank_count += 1

    assert checked_rank_count > 1_000
    fallback.assert_not_called()


def test_compute_local_shard_info_matches_randomized_two_stage_shards() -> None:
    rng = random.Random(20260824)
    checked_rank_count = 0
    with patch(
        "torch_checkpointing.experimental.cooperative_resharding.default_resharder._compute_local_shape_and_global_offset",
        wraps=_compute_local_shape_and_global_offset,
    ) as fallback:
        for _ in range(500):
            mesh_shape = (
                rng.randrange(1, 7),
                rng.randrange(1, 5),
                rng.randrange(1, 7),
            )
            world_size = mesh_shape[0] * mesh_shape[1] * mesh_shape[2]
            mesh_data = list(range(1001, 1001 + world_size * 2, 2))
            rng.shuffle(mesh_data)
            metadata = _make_sharding_metadata(
                global_shape=(
                    rng.randrange(33) * mesh_shape[0] * mesh_shape[2],
                    rng.randrange(33),
                ),
                mesh_shape=mesh_shape,
                mesh_data=tuple(mesh_data),
                placements=(
                    StridedShardSpec(0, split_factor=mesh_shape[2]),
                    ShardSpec(1),
                    ShardSpec(0),
                ),
            )
            ranks = metadata.mesh_spec.mesh_data
            selected_ranks = (ranks[0], ranks[len(ranks) // 2], ranks[-1])
            for rank in selected_ranks:
                assert compute_local_shard_info(
                    metadata,
                    rank,
                ) == _reference_local_shard_info(metadata, rank)
                checked_rank_count += 1

    assert checked_rank_count == 1_500
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "global_shape",
    [(128, 1024, 4096), (128, 4096, 1024)],
)
def test_compute_local_shard_info_fast_path_matches_balanced_repeated_shards(
    global_shape: tuple[int, ...],
) -> None:
    metadata = _make_sharding_metadata(
        global_shape=global_shape,
        mesh_shape=(8, 4),
        mesh_data=tuple(range(32)),
        placements=(StridedShardSpec(0, split_factor=4), ShardSpec(0)),
    )

    with patch(
        "torch_checkpointing.experimental.cooperative_resharding.default_resharder._compute_local_shape_and_global_offset",
        wraps=_compute_local_shape_and_global_offset,
    ) as fallback:
        for rank in metadata.mesh_spec.mesh_data:
            local_shape, global_offset = compute_local_shard_info(metadata, rank)
            assert (local_shape, global_offset) == _reference_local_shard_info(
                metadata,
                rank,
            )
            assert local_shape[0] == 4

    fallback.assert_not_called()


@pytest.mark.parametrize(
    "placements",
    [
        (ShardSpec(0), ShardSpec(0)),
        (StridedShardSpec(0, split_factor=2), ShardSpec(0)),
        (StridedShardSpec(0, split_factor=3), ShardSpec(0)),
        (ShardSpec(1), StridedShardSpec(1, split_factor=2)),
        (
            StridedShardSpec(0, split_factor=3),
            StridedShardSpec(0, split_factor=2),
        ),
        (ShardSpec(-1), ReplicateSpec()),
    ],
)
def test_compute_local_shard_info_falls_back_for_complex_placements(
    placements: tuple[ReplicateSpec | ShardSpec | StridedShardSpec, ...],
) -> None:
    metadata = _make_sharding_metadata(
        global_shape=(17, 11),
        mesh_shape=(2, 2),
        mesh_data=(7, 3, 11, 5),
        placements=placements,
    )
    expected = _reference_local_shard_info(metadata, 5)

    with patch(
        "torch_checkpointing.experimental.cooperative_resharding.default_resharder._compute_local_shape_and_global_offset",
        wraps=_compute_local_shape_and_global_offset,
    ) as fallback:
        assert compute_local_shard_info(metadata, 5) == expected

    fallback.assert_called_once()


def test_compute_local_shard_info_preserves_unsupported_placement_error() -> None:
    metadata = _make_sharding_metadata(
        placements=(cast(Any, object()),),
    )

    with pytest.raises(ValueError, match="Unsupported placement type"):
        compute_local_shard_info(metadata, 7)


@pytest.mark.parametrize(
    "placements",
    [
        (ReplicateSpec(),),
        (ShardSpec(0),),
        (StridedShardSpec(0, split_factor=2),),
    ],
)
def test_shard_geometry_cache_preserves_geometry(
    placements: tuple[ReplicateSpec | ShardSpec | StridedShardSpec, ...],
) -> None:
    metadata = _make_sharding_metadata(placements=placements)
    cache = _ShardGeometryCache()

    expected = compute_local_shard_info(metadata, 11)

    assert cache.get_local_shard_info(metadata, 11) == expected
    assert cache.get_local_shard_info(metadata, 11) == expected
    assert cache.metrics.geometry_misses == 1
    assert cache.metrics.geometry_hits == 1
    assert cache.metrics.geometry_fast_paths == 1
    assert cache.metrics.geometry_fallbacks == 0
    assert cache.placement_entry_count == 0


def test_shard_geometry_cache_is_bounded() -> None:
    cache = _ShardGeometryCache(
        geometry_max_entries=2,
        placement_max_entries=2,
    )
    metadata = [
        _make_sharding_metadata(
            global_shape=(size, 4),
            mesh_shape=(2, 2),
            placements=placements,
        )
        for size, placements in (
            (8, (ShardSpec(0), ShardSpec(0))),
            (12, (ShardSpec(0), StridedShardSpec(0, split_factor=2))),
            (16, (ShardSpec(1), ShardSpec(1))),
        )
    ]

    for sharding_metadata in metadata:
        cache.get_local_shard_info(sharding_metadata, 7)
    cache.get_local_shard_info(
        _make_sharding_metadata(
            global_shape=(20, 4),
            mesh_shape=(2, 2),
            placements=(ShardSpec(0), StridedShardSpec(0, split_factor=2)),
        ),
        7,
    )

    assert cache.geometry_entry_count == 2
    assert cache.placement_entry_count == 2
    assert cache.metrics.geometry_evictions == 2
    assert cache.metrics.geometry_fast_paths == 0
    assert cache.metrics.geometry_fallbacks == 4
    assert cache.metrics.placement_evictions == 1
    assert cache.metrics.placement_hits == 1


def test_load_plan_template_cache_is_bounded_lru() -> None:
    cache = _LoadPlanTemplateCache(max_entries=2)
    entry = _LoadPlanTemplateEntry(
        plans=(
            _LoadPlanTemplate(
                offsets=(0,),
                sizes=(1,),
                src_rank=0,
                src_offsets=(0,),
                src_sizes=(1,),
                transpose_dims=(),
                src_elem_size=0,
                src_dtype="",
            ),
        ),
        source_group_count=1,
        source_rank_candidate_count=1,
        duplicate_source_slice_count=0,
        target_has_elements=True,
    )
    source_sharding = _make_sharding_metadata(placements=(ShardSpec(0),))
    source_groups = [
        GlobalObjectMetadata(sharding_metadata=source_sharding, ranks=(7, 3, 11, 5))
    ]
    keys = [
        _load_plan_template_cache_key(
            rank,
            _make_sharding_metadata(
                global_shape=(16 + rank * 4, 4),
                placements=(ShardSpec(0),),
            ),
            source_groups,
        )
        for rank in range(3)
    ]
    assert all(key is not None for key in keys)

    for key in keys[:2]:
        assert key is not None
        assert cache.get(key) is None
        cache.put(key, entry)
    assert keys[0] is not None
    assert cache.get(keys[0]) == entry
    assert keys[2] is not None
    assert cache.get(keys[2]) is None
    cache.put(keys[2], entry)

    assert cache.entry_count == 2
    assert cache.metrics.plan_template_hits == 1
    assert cache.metrics.plan_template_misses == 3
    assert cache.metrics.plan_template_evictions == 1
    assert keys[1] is not None
    assert cache.get(keys[1]) is None


def test_generate_load_plans_reuses_geometry_template_and_rebinds_fqn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target_metadata, source_metadata = _make_repeated_shape_planning_case(3)
    for index, path in enumerate(target_metadata):
        if index == 0:
            continue
        target_metadata[path] = replace(
            target_metadata[path],
            dtype="torch.float32",
            stride=(1, target_metadata[path].global_shape[0]),
        )
        source_group = source_metadata.nested_path_to_metadata[path][0]
        assert isinstance(source_group.sharding_metadata, DTensorShardingMetadata)
        source_metadata.nested_path_to_metadata[path] = [
            replace(
                source_group,
                sharding_metadata=replace(
                    source_group.sharding_metadata,
                    dtype="torch.float32",
                    stride=(1, source_group.sharding_metadata.global_shape[0]),
                ),
            )
        ]

    with (
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
            return_value=0,
        ),
        caplog.at_level(logging.INFO),
    ):
        result = DefaultResharder()._generate_load_plans(
            target_metadata,
            source_metadata,
        )

    assert list(result.nested_path_to_load_plans) == list(target_metadata)
    for path, plans in result.nested_path_to_load_plans.items():
        expected_fqn = ".".join(str(component) for component in path)
        assert len(plans) == 4
        assert [plan.src_rank for plan in plans] == list(range(4))
        assert all(plan.src_fqn == expected_fqn for plan in plans)
    assert result.non_reshardable_paths == []
    assert "source_rank_candidates=384" in caplog.text
    assert "source_rank_scans=128" in caplog.text
    assert "plan_template_cache_hits=2" in caplog.text
    assert "plan_template_cache_misses=1" in caplog.text


def test_load_plan_template_cache_preserves_non_reshardable_source() -> None:
    target_metadata, source_metadata = _make_repeated_shape_planning_case(3)
    invalid_path = list(target_metadata)[1]
    source_metadata.nested_path_to_metadata[invalid_path] = [
        GlobalObjectMetadata(
            sharding_metadata=cast(ShardingMetadata, object()),
            ranks=(0,),
        )
    ]

    with (
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
            return_value=0,
        ),
    ):
        result = DefaultResharder()._generate_load_plans(
            target_metadata,
            source_metadata,
        )

    assert list(result.nested_path_to_load_plans) == [
        list(target_metadata)[0],
        list(target_metadata)[2],
    ]
    assert result.non_reshardable_paths == [invalid_path]


def test_load_plan_template_cache_randomized_differential() -> None:
    rng = random.Random(20260823)
    source_ranks = tuple(range(8))
    target_ranks = tuple(range(4))
    shape_choices = ((32, 16), (64, 8), (128, 4))
    source_placement_choices = (
        (ShardSpec(0),),
        (ReplicateSpec(),),
        (StridedShardSpec(0, split_factor=2),),
    )
    target_placement_choices = (
        (ShardSpec(0),),
        (ShardSpec(1),),
        (ReplicateSpec(),),
    )
    geometry_choices = [
        (
            rng.choice(shape_choices),
            rng.choice(source_placement_choices),
            rng.choice(target_placement_choices),
            source_ranks if rng.randrange(2) else tuple(reversed(source_ranks)),
        )
        for _ in range(12)
    ]
    target_metadata = {}
    source_path_metadata = {}
    for index in range(240):
        shape, source_placements, target_placements, ranks = rng.choice(
            geometry_choices
        )
        path = ("layers", index, "weight")
        source_sharding = DTensorShardingMetadata(
            global_shape=shape,
            dtype=rng.choice(("torch.bfloat16", "torch.float32")),
            stride=(shape[1], 1),
            mesh_spec=DeviceMeshSpec(
                device_type="cuda",
                mesh_shape=(len(source_ranks),),
                mesh_data=source_ranks,
            ),
            placements=source_placements,
        )
        target_sharding = DTensorShardingMetadata(
            global_shape=shape,
            dtype=rng.choice(("torch.bfloat16", "torch.float32")),
            stride=(shape[1], 1),
            mesh_spec=DeviceMeshSpec(
                device_type="cuda",
                mesh_shape=(len(target_ranks),),
                mesh_data=target_ranks,
            ),
            placements=target_placements,
        )
        target_metadata[path] = target_sharding
        source_path_metadata[path] = [
            GlobalObjectMetadata(
                sharding_metadata=source_sharding,
                ranks=ranks,
            )
        ]

    missing_path = ("missing", 0, "weight")
    target_metadata[missing_path] = next(iter(target_metadata.values()))
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata=source_path_metadata,
        rank_to_layout_info={},
    )

    with (
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
            return_value=2,
        ),
        patch.object(_LoadPlanTemplateCache, "get", return_value=None),
    ):
        uncached = DefaultResharder()._generate_load_plans(
            target_metadata,
            source_metadata,
        )

    with (
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
            return_value=2,
        ),
    ):
        cached = DefaultResharder()._generate_load_plans(
            target_metadata,
            source_metadata,
        )

    assert cached == uncached
    assert cached.non_reshardable_paths == [missing_path]


def test_load_plan_template_cache_large_synthetic_scan_reduction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    target_metadata, source_metadata = _make_repeated_shape_planning_case(256)
    with (
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
            return_value=0,
        ),
        caplog.at_level(logging.INFO),
    ):
        cached = DefaultResharder()._generate_load_plans(
            target_metadata,
            source_metadata,
        )

    assert len(cached.nested_path_to_load_plans) == 256
    assert (
        sum(len(plans) for plans in cached.nested_path_to_load_plans.values()) == 1024
    )
    assert "source_rank_candidates=32768" in caplog.text
    assert "source_rank_scans=128" in caplog.text
    assert "geometry_fast_paths=129" in caplog.text
    assert "geometry_fallbacks=0" in caplog.text
    assert "plan_template_cache_hits=255" in caplog.text
    assert "plan_template_cache_misses=1" in caplog.text


def test_generate_load_plans_preserves_exact_noncontiguous_mesh_plan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_sharding = _make_sharding_metadata(placements=(ShardSpec(0),))
    target_sharding = _make_sharding_metadata(
        global_shape=(16, 4),
        mesh_shape=(2,),
        mesh_data=(7, 5),
        placements=(ShardSpec(0),),
    )
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            ("weight",): [
                GlobalObjectMetadata(
                    sharding_metadata=source_sharding,
                    ranks=(7, 3, 11, 5),
                )
            ]
        },
        rank_to_layout_info={},
    )

    with (
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
            return_value=5,
        ),
        caplog.at_level(logging.INFO),
    ):
        result = DefaultResharder()._generate_load_plans(
            {("weight",): target_sharding},
            source_metadata,
        )

    plans = result.nested_path_to_load_plans[("weight",)]
    assert [
        (
            plan.offsets,
            plan.sizes,
            plan.src_rank,
            plan.src_fqn,
            plan.src_offsets,
            plan.src_sizes,
            plan.transpose_dims,
        )
        for plan in plans
    ] == [
        ((0, 0), (4, 4), 11, "weight", (0, 0), (4, 4), ()),
        ((4, 0), (4, 4), 5, "weight", (0, 0), (4, 4), ()),
    ]
    assert result.non_reshardable_paths == []
    assert (
        sum("DefaultResharder plan metrics" in message for message in caplog.messages)
        == 1
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
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_world_size",
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
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
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
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "torch_checkpointing.experimental.cooperative_resharding.default_resharder.dist.get_rank",
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
