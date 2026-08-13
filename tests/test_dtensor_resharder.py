# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for DTensorResharder.

Tests resharding scenarios including placement changes and mesh topology changes
using DTensor's native placement APIs.
"""

import logging
import os
import pickle
from typing import Any

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor._utils import _compute_local_shape_and_global_offset
from torch.distributed.tensor.placement_types import _StridedShard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir
from torch_checkpointing.checkpoint_base import (
    CheckpointInfo,
    CheckpointItem,
    CheckpointReadInfo,
)
from torch_checkpointing.checkpoint_layout import default_layout_info
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    GlobalObjectMetadata,
    METADATA_FILE_NAME,
    ShardingMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ShardSpec,
    StridedShardSpec,
)
from torch_checkpointing.dtensor_resharder import (
    compute_local_shard_info,
    DTensorResharder,
)
from torch_checkpointing.metadata_manager import (
    CheckpointMetadata,
    DefaultMetadataManager,
)
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo

# These tests spawn a multi-process group and exercise the async saver's
# accelerator staging, so they require an accelerator.
pytestmark = [
    pytest.mark.gpus_needed_4,
    pytest.mark.skipif(
        not torch.accelerator.is_available(),
        reason="requires an accelerator (multi-process DTensor resharding)",
    ),
]


class _UnsupportedShardingMetadata(ShardingMetadata):
    def to_dict(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_UnsupportedShardingMetadata":
        return cls()

    @property
    def equivalent_ranks(self) -> tuple[int, ...] | None:
        return None


def _test_dtensor_metadata(
    *,
    mesh_shape: tuple[int, ...] = (1,),
    mesh_data: tuple[int, ...] = (0,),
) -> DTensorShardingMetadata:
    return DTensorShardingMetadata(
        global_shape=(8, 8),
        dtype="torch.float32",
        stride=(8, 1),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=mesh_shape,
            mesh_data=mesh_data,
        ),
        placements=(ShardSpec(0),),
    )


def test_generate_load_plans_skips_path_with_unsupported_source_metadata() -> None:
    nested_path = ("weight",)
    dtensor_metadata = _test_dtensor_metadata()
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            nested_path: [
                GlobalObjectMetadata(
                    sharding_metadata=dtensor_metadata,
                    ranks=(0,),
                ),
                GlobalObjectMetadata(
                    sharding_metadata=_UnsupportedShardingMetadata(),
                    ranks=(0,),
                ),
            ],
        },
        rank_to_layout_info={},
    )

    resharding_info = DTensorResharder()._generate_load_plans(
        target_metadata={nested_path: dtensor_metadata},
        source_metadata=source_metadata,
    )

    assert resharding_info.nested_path_to_load_plans == {}
    assert resharding_info.non_reshardable_paths == [nested_path]


def test_generate_load_plans_marks_non_intersecting_target_shard_non_reshardable(
    caplog,
) -> None:
    nested_path = ("weight",)
    dtensor_metadata = _test_dtensor_metadata(
        mesh_shape=(2,),
        mesh_data=(0, 1),
    )
    source_metadata = DistributedItemMetadata(
        nested_path_to_metadata={
            nested_path: [
                GlobalObjectMetadata(
                    sharding_metadata=dtensor_metadata,
                    ranks=(1,),
                ),
            ],
        },
        rank_to_layout_info={},
    )

    with caplog.at_level(logging.WARNING):
        resharding_info = DTensorResharder()._generate_load_plans(
            target_metadata={nested_path: dtensor_metadata},
            source_metadata=source_metadata,
        )

    assert resharding_info.nested_path_to_load_plans == {}
    assert resharding_info.non_reshardable_paths == [nested_path]
    assert "No source DTensor shard intersects target shard" in caplog.text


class TestDTensorResharder(DTensorTestBase):
    """Integration tests for DTensorResharder resharding across topologies."""

    temp_dir: str  # Set by @with_temp_dir decorator

    @property
    def world_size(self) -> int:
        return 4

    @property
    def backend(self) -> str:
        return "gloo"

    def setUp(self) -> None:
        super().setUp()

    def _create_rank_info(self) -> RankInfo:
        """Create RankInfo for current distributed setup."""
        if dist.is_initialized():
            return RankInfo(
                global_rank=dist.get_rank(),
                global_world_size=dist.get_world_size(),
                role_rank=dist.get_rank(),
                role_world_size=dist.get_world_size(),
            )
        else:
            return RankInfo(
                global_rank=0,
                global_world_size=1,
                role_rank=0,
                role_world_size=1,
            )

    def _compute_checkpoint_metadata(
        self, checkpoint_info: CheckpointInfo, keys_to_compute: list[str]
    ) -> CheckpointMetadata:
        """Compute checkpoint metadata for the given checkpoint info."""
        metadata_manager = DefaultMetadataManager(
            rank_info=self.rank_info,
            process_group=dist.distributed_c10d._get_default_group(),
        )
        result = metadata_manager.compute_metadata(checkpoint_info)
        assert result is not None, "Expected fresh metadata computation"
        return result

    def _save_checkpoint_info(
        self,
        checkpoint_info: CheckpointInfo,
        checkpoint_metadata: CheckpointMetadata,
        checkpoint_dir: str,
    ) -> None:
        """Save checkpoint items and metadata to disk."""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save checkpoint items
        for key, item in checkpoint_info.checkpoint_items.items():
            if item.layout is not None:
                file_path = os.path.join(checkpoint_dir, item.layout.file_path)
            else:
                layout = default_layout_info(key, self.rank)
                file_path = os.path.join(checkpoint_dir, layout.file_path)

            torch.save(item.value, file_path)

        # Write metadata file (rank 0 only)
        if self.rank == 0:
            metadata_path = os.path.join(checkpoint_dir, METADATA_FILE_NAME)
            with open(metadata_path, "wb") as f:
                pickle.dump(
                    checkpoint_metadata.distributed_metadata.to_dict(),
                    f,
                )

        dist.barrier()

    def _create_checkpoint_read_info(
        self,
        items: dict[str, CheckpointItem],
        checkpoint_metadata: CheckpointMetadata,
    ) -> CheckpointReadInfo:
        """Create CheckpointReadInfo for reading operations."""
        checkpoint_info = CheckpointInfo(checkpoint_items=items)
        return checkpoint_info.for_reads(checkpoint_metadata)

    def _create_arange_tensor(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> torch.Tensor:
        """Create a tensor filled with arange values (deterministic)."""
        numel = 1
        for dim in shape:
            numel *= dim
        return torch.arange(numel, dtype=dtype, device=device).reshape(shape)

    def _verify_dtensor_data(
        self,
        loaded_dtensor: DTensor,
        expected_global_tensor: torch.Tensor,
    ) -> None:
        """Verify loaded DTensor contains correct data by comparing local tensors."""
        expected_dtensor = distribute_tensor(
            expected_global_tensor,
            loaded_dtensor.device_mesh,
            loaded_dtensor.placements,
        )

        torch.testing.assert_close(
            loaded_dtensor._local_tensor,
            expected_dtensor._local_tensor,
        )

    @with_comms()
    @with_temp_dir
    def test_1d_resharding(self):
        """Test 1D resharding: save with Shard(0) on 4-rank mesh, load with Shard(1)."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape = (8, 8)

        source_mesh = init_device_mesh("cpu", (4,))
        target_mesh = init_device_mesh("cpu", (4,))

        # Source: Shard(0)
        source_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        self._create_arange_tensor(shape),
                        source_mesh,
                        [Shard(0)],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        # Target: Shard(1)
        target_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        torch.randn(shape, device="cpu"),
                        target_mesh,
                        [Shard(1)],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["model"]["weight"], self._create_arange_tensor(shape)
        )

    @with_comms()
    @with_temp_dir
    def test_root_dtensor_item(self):
        """Test resharding when the checkpoint item value is itself a DTensor."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape = (8, 8)

        source_mesh = init_device_mesh("cpu", (4,))
        target_mesh = init_device_mesh("cpu", (4,))

        source_items = {
            "weight": CheckpointItem(
                value=distribute_tensor(
                    self._create_arange_tensor(shape),
                    source_mesh,
                    [Shard(0)],
                ),
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["weight"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        target_items = {
            "weight": CheckpointItem(
                value=distribute_tensor(
                    torch.randn(shape, device="cpu"),
                    target_mesh,
                    [Shard(1)],
                ),
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["weight"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["weight"], self._create_arange_tensor(shape)
        )

    @with_comms()
    @with_temp_dir
    def test_sequence_nested_dtensor_item(self):
        """Test resharding when a DTensor is nested under a sequence."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape = (8, 8)

        source_mesh = init_device_mesh("cpu", (4,))
        target_mesh = init_device_mesh("cpu", (4,))

        source_items = {
            "model": CheckpointItem(
                value={
                    "layers": [
                        {
                            "weight": distribute_tensor(
                                self._create_arange_tensor(shape),
                                source_mesh,
                                [Shard(0)],
                            )
                        }
                    ]
                },
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        target_items = {
            "model": CheckpointItem(
                value={
                    "layers": [
                        {
                            "weight": distribute_tensor(
                                torch.randn(shape, device="cpu"),
                                target_mesh,
                                [Shard(1)],
                            )
                        }
                    ]
                },
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["model"]["layers"][0]["weight"],
            self._create_arange_tensor(shape),
        )

    @with_comms()
    @with_temp_dir
    def test_2d_resharding(self):
        """Test 2D resharding: save with (2,2) + [Shard(0), Shard(1)], load with (4,1) + [Shard(0), Replicate()]."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape = (8, 8)

        source_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        target_mesh = init_device_mesh("cpu", (4, 1), mesh_dim_names=("dp", "tp"))

        # Source: [Shard(0), Shard(1)]
        source_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        self._create_arange_tensor(shape),
                        source_mesh,
                        [Shard(0), Shard(1)],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        # Target: [Shard(0), Replicate()]
        target_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        torch.randn(shape, device="cpu"),
                        target_mesh,
                        [Shard(0), Replicate()],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["model"]["weight"], self._create_arange_tensor(shape)
        )

    @with_comms()
    @with_temp_dir
    def test_same_topology_no_resharding(self):
        """Test that same topology skips resharding but still loads correct data."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape = (8, 8)

        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

        # Source and target: same mesh and placements
        source_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        self._create_arange_tensor(shape),
                        mesh,
                        [Shard(0), Shard(1)],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        # Target: same mesh/placements, but random init
        target_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        torch.randn(shape, device="cpu"),
                        mesh,
                        [Shard(0), Shard(1)],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model"]
        )

        # Verify should_reshard returns False for identical topologies
        resharder = DTensorResharder()
        target_sharding = resharder.extract_sharding_metadata(
            "model", target_items["model"].value
        )
        source_item_metadata = source_metadata.distributed_metadata.metadata.get(
            "model"
        )
        self.assertFalse(
            resharder.should_reshard(source_item_metadata, target_sharding)
        )

        # Still load correctly
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["model"]["weight"], self._create_arange_tensor(shape)
        )

    @with_comms()
    @with_temp_dir
    def test_replicate_to_shard(self):
        """Test resharding from Replicate() to Shard(0)."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape = (8, 4)

        source_mesh = init_device_mesh("cpu", (4,))
        target_mesh = init_device_mesh("cpu", (4,))

        # Source: Replicate()
        source_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        self._create_arange_tensor(shape),
                        source_mesh,
                        [Replicate()],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        # Target: Shard(0)
        target_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        torch.randn(shape, device="cpu"),
                        target_mesh,
                        [Shard(0)],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["model"]["weight"], self._create_arange_tensor(shape)
        )

    @with_comms()
    @with_temp_dir
    def test_shard_to_replicate(self):
        """Test resharding from Shard(0) to Replicate()."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape = (8, 4)

        source_mesh = init_device_mesh("cpu", (4,))
        target_mesh = init_device_mesh("cpu", (4,))

        # Source: Shard(0)
        source_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        self._create_arange_tensor(shape),
                        source_mesh,
                        [Shard(0)],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        # Target: Replicate()
        target_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        torch.randn(shape, device="cpu"),
                        target_mesh,
                        [Replicate()],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["model"]["weight"], self._create_arange_tensor(shape)
        )

    @with_comms()
    @with_temp_dir
    def test_multiple_items(self):
        """Test resharding with multiple checkpoint items (model + optimizer)."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        shape1 = (8, 8)
        shape2 = (4, 4)

        source_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

        # Source: two items with different placements
        source_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        self._create_arange_tensor(shape1),
                        source_mesh,
                        [Shard(0), Shard(1)],
                    )
                },
                resharder=DTensorResharder(),
            ),
            "optimizer": CheckpointItem(
                value={
                    "state": distribute_tensor(
                        self._create_arange_tensor(shape2),
                        source_mesh,
                        [Shard(0), Replicate()],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model", "optimizer"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        # Target: different mesh, different placements
        target_mesh = init_device_mesh("cpu", (4, 1), mesh_dim_names=("dp", "tp"))

        target_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        torch.randn(shape1, device="cpu"),
                        target_mesh,
                        [Shard(0), Replicate()],
                    )
                },
                resharder=DTensorResharder(),
            ),
            "optimizer": CheckpointItem(
                value={
                    "state": distribute_tensor(
                        torch.randn(shape2, device="cpu"),
                        target_mesh,
                        [Shard(0), Replicate()],
                    )
                },
                resharder=DTensorResharder(),
            ),
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model", "optimizer"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        self.assertEqual(missing_keys, [])
        self._verify_dtensor_data(
            result_dict["model"]["weight"], self._create_arange_tensor(shape1)
        )
        self._verify_dtensor_data(
            result_dict["optimizer"]["state"], self._create_arange_tensor(shape2)
        )

    @with_comms()
    @with_temp_dir
    def test_mixed_items_reshard_and_no_reshard(self):
        """Test mix of items: one with DTensorResharder, one without."""
        self.rank_info = self._create_rank_info()
        self.reader = CheckpointReader(
            rank_info=self.rank_info, storage_config=LocalFileSystemStorageConfig()
        )

        checkpoint_dir = os.path.join(self.temp_dir, "ckpt")
        global_shape = (8, 8)

        source_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))

        source_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        self._create_arange_tensor(global_shape),
                        source_mesh,
                        [Shard(0), Shard(1)],
                    )
                },
                resharder=DTensorResharder(),
            ),
            "epoch": CheckpointItem(value=42),  # Non-tensor, no resharder
        }

        source_checkpoint_info = CheckpointInfo(checkpoint_items=source_items)
        source_metadata = self._compute_checkpoint_metadata(
            source_checkpoint_info, ["model"]
        )
        self._save_checkpoint_info(
            source_checkpoint_info, source_metadata, checkpoint_dir
        )

        # Target: model has resharder (different mesh), epoch does not
        target_mesh = init_device_mesh("cpu", (4, 1), mesh_dim_names=("dp", "tp"))

        target_items = {
            "model": CheckpointItem(
                value={
                    "weight": distribute_tensor(
                        torch.randn(global_shape, device="cpu"),
                        target_mesh,
                        [Shard(0), Replicate()],
                    )
                },
                resharder=DTensorResharder(),
            ),
            "epoch": CheckpointItem(value=None),  # No resharder
        }

        target_metadata = self._compute_checkpoint_metadata(
            CheckpointInfo(checkpoint_items=target_items), ["model"]
        )
        target_checkpoint_info = self._create_checkpoint_read_info(
            target_items, target_metadata
        )

        result_dict, missing_keys = self.reader.read(
            path=checkpoint_dir,
            checkpoint_info=target_checkpoint_info,
        )

        # Verify model loaded with resharding
        self._verify_dtensor_data(
            result_dict["model"]["weight"],
            self._create_arange_tensor(global_shape),
        )

        # Verify epoch loaded without resharding
        self.assertEqual(result_dict["epoch"], 42)


@pytest.mark.parametrize("rank", range(4))
def test_compute_local_shard_info_matches_dtensor_for_strided_shard(rank: int) -> None:
    """Strided shard geometry must agree with DTensor's own computation."""
    placements = (StridedShardSpec(dim=0, split_factor=2), ShardSpec(dim=0))
    metadata = DTensorShardingMetadata(
        global_shape=(8, 2),
        dtype=str(torch.float32),
        stride=(2, 1),
        mesh_spec=get_device_mesh_spec(
            device_type="cpu",
            mesh_shape=(2, 2),
            mesh_data=(0, 1, 2, 3),
            mesh_dim_names=("dp", "tp"),
        ),
        placements=placements,
    )

    local_shape, global_offset = compute_local_shard_info(metadata, rank)

    expected_shape, expected_offset = _compute_local_shape_and_global_offset(
        torch.Size((8, 2)),
        torch.Size((2, 2)),
        [rank // 2, rank % 2],
        [_StridedShard(0, split_factor=2), Shard(0)],
    )
    assert tuple(local_shape) == tuple(expected_shape)
    assert tuple(global_offset) == tuple(expected_offset)
