# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end saves and loads of DTensors whose local tensors are views into larger
per-rank buffers, through ``CheckpointManager``.

A DTensor's wrapper takes the storage offset of its local tensor, so a local
tensor that is a view at a nonzero offset gives the wrapper a nonzero
``data_ptr()`` although the wrapper's storage holds no data.
"""

import os
import uuid
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.placement_types import Placement
from torch_checkpointing.checkpoint_layout import LayoutInfo, SafetensorsSerialization
from torch_checkpointing.checkpoint_manager import CheckpointManager
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    CheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig

ITEM = "model"
# Two DTensors per checkpoint. Each one's source local tensor is one row of its
# own (ROWS, COLS) buffer per rank.
KEYS = ("w", "v")
ROWS, COLS = 4, 16
SAVE_WORLD_SIZE = 2
SENTINEL = -1.0
TIMEOUT_SECS = 60

DEVICE_TYPES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpus_needed_2,
            pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="requires 2 CUDA devices"
            ),
        ],
    ),
]


def _source_buffer(rank: int, key: str) -> torch.Tensor:
    # Every element is unique across ranks and keys.
    numel = ROWS * COLS
    start = (rank * len(KEYS) + KEYS.index(key)) * numel
    return torch.arange(start, start + numel, dtype=torch.float32).view(ROWS, COLS)


def _expected_global(key: str, source_row: int) -> torch.Tensor:
    # The saved DTensors are sharded with Shard(0) across SAVE_WORLD_SIZE ranks.
    return torch.cat(
        [_source_buffer(rank, key)[source_row] for rank in range(SAVE_WORLD_SIZE)]
    )


def _device(device_type: str) -> torch.device:
    # Each rank's process sets its own current CUDA device.
    if device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _manager(fmt: str, save: CheckpointSaverConfig) -> CheckpointManager:
    layout = (
        LayoutInfo(f"{ITEM}_{{rank}}.safetensors", SafetensorsSerialization())
        if fmt == "safetensors"
        else None  # the default torch.save layout
    )
    return CheckpointManager.Config(
        items={ITEM: ItemSpec(resharder=DefaultResharder(), layout=layout)},
        save=save,
        storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
    ).build()


def _load_into_views_and_check(
    manager: CheckpointManager,
    checkpoint_path: str,
    mesh: DeviceMesh,
    placement: Placement,
    source_row: int,
) -> None:
    """
    Load into DTensors whose local tensors are adjacent rows of one shared,
    sentinel-filled buffer, at nonzero offsets, and check the whole buffer.
    """
    rank = mesh.get_rank()
    expected = {
        key: (
            _expected_global(key, source_row)
            if isinstance(placement, Replicate)
            else _expected_global(key, source_row).chunk(mesh.size())[rank]
        )
        for key in KEYS
    }
    shape = (len(KEYS) + 2, len(expected["w"]))
    target_buffer = torch.full(shape, SENTINEL, device=_device(mesh.device_type))
    targets = {
        key: DTensor.from_local(
            target_buffer[i + 1], mesh, [placement], run_check=False
        )
        for i, key in enumerate(KEYS)
    }

    loaded = manager.load(checkpoint_path, into={ITEM: targets})

    expected_buffer = torch.full(shape, SENTINEL)
    for i, key in enumerate(KEYS):
        assert loaded[ITEM][key] is targets[key]
        expected_buffer[i + 1] = expected[key]
    torch.testing.assert_close(target_buffer.cpu(), expected_buffer, rtol=0, atol=0)


def _run_rank(
    rank: int,
    world_size: int,
    store_path: str,
    device_type: str,
    fn: Callable[..., None],
    args: tuple[Any, ...],
) -> None:
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl" if device_type == "cuda" else "gloo",
        store=dist.FileStore(store_path, world_size),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=TIMEOUT_SECS),
    )
    try:
        fn(init_device_mesh(device_type, (world_size,)), *args)
    finally:
        dist.destroy_process_group()


def _spawn(
    tmp_path: Path,
    world_size: int,
    device_type: str,
    fn: Callable[..., None],
    *args: Any,
) -> None:
    store_path = os.fspath(tmp_path / f"store_{uuid.uuid4().hex}")
    mp.spawn(
        _run_rank,
        args=(world_size, store_path, device_type, fn, args),
        nprocs=world_size,
        join=True,
    )


def _save_then_load_on_save_mesh(
    mesh: DeviceMesh, fmt: str, source_row: int, checkpoint_path: str
) -> None:
    rank = mesh.get_rank()
    device = _device(mesh.device_type)
    on_cuda = mesh.device_type == "cuda"
    state = {
        key: DTensor.from_local(
            _source_buffer(rank, key).to(device)[source_row],
            mesh,
            [Shard(0)],
            run_check=False,
        )
        for key in KEYS
    }
    assert all(
        dtensor.to_local().storage_offset() == source_row * COLS
        for dtensor in state.values()
    )

    manager = _manager(
        fmt,
        AsyncCheckpointSaverConfig(
            wait_timeout_secs=TIMEOUT_SECS,
            staging_config=CheckpointStagerConfig(
                use_pinned_memory=on_cuda, use_non_blocking_copy=on_cuda
            ),
        ),
    )
    try:
        write_future = manager.save(checkpoint_path, {ITEM: state})
        assert write_future is not None
        write_future.result(timeout=TIMEOUT_SECS)
        dist.barrier()

        # Same mesh and placements as the save: with the default layout, this
        # takes the reader's path without resharding.
        _load_into_views_and_check(manager, checkpoint_path, mesh, Shard(0), source_row)
        # Shard(0) to Replicate on the same ranks: resharding.
        _load_into_views_and_check(
            manager, checkpoint_path, mesh, Replicate(), source_row
        )
    finally:
        manager.close()


def _load_replicated(
    mesh: DeviceMesh, fmt: str, source_row: int, checkpoint_path: str
) -> None:
    manager = _manager(fmt, SyncCheckpointSaverConfig())
    try:
        _load_into_views_and_check(
            manager, checkpoint_path, mesh, Replicate(), source_row
        )
    finally:
        manager.close()


@pytest.mark.parametrize("device_type", DEVICE_TYPES)
@pytest.mark.parametrize("fmt", ["torch", "safetensors"])
@pytest.mark.parametrize(
    "source_row", [0, 2], ids=["view_at_offset_0", "view_at_nonzero_offset"]
)
def test_save_and_load_dtensors_whose_local_tensors_are_views(
    tmp_path: Path, device_type: str, fmt: str, source_row: int
) -> None:
    checkpoint_path = os.fspath(tmp_path / "checkpoint")
    _spawn(
        tmp_path,
        SAVE_WORLD_SIZE,
        device_type,
        _save_then_load_on_save_mesh,
        fmt,
        source_row,
        checkpoint_path,
    )

    suffix = ".safetensors" if fmt == "safetensors" else ".pt"
    assert sorted(
        path.name for path in Path(checkpoint_path).glob(f"{ITEM}_*{suffix}")
    ) == [f"{ITEM}_{rank}{suffix}" for rank in range(SAVE_WORLD_SIZE)]

    # Two ranks Shard(0) to one rank Replicate: resharding to another world size.
    _spawn(tmp_path, 1, device_type, _load_replicated, fmt, source_row, checkpoint_path)


@pytest.fixture
def cpu_mesh() -> Iterator[DeviceMesh]:
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.xfail(
    strict=False,
    raises=RuntimeError,
    reason=(
        "PyTorch limitation: torch.save tests data_ptr() == 0 to recognize a "
        "wrapper subclass, so it pickles a DTensor whose local tensor is a view "
        "at a nonzero offset as a plain tensor, and raises on its placeholder "
        "storage. The synchronous saver does not stage, so torch.save gets the "
        "live DTensor."
    ),
)
def test_sync_save_of_dtensor_whose_local_tensor_is_a_view_at_nonzero_offset(
    cpu_mesh: DeviceMesh, tmp_path: Path
) -> None:
    base = torch.arange(12, dtype=torch.float32)
    dtensor = DTensor.from_local(base[2:6], cpu_mesh, [Shard(0)], run_check=False)
    target = DTensor.from_local(torch.zeros(4), cpu_mesh, [Shard(0)], run_check=False)
    checkpoint_path = os.fspath(tmp_path / "checkpoint")

    manager = _manager("torch", SyncCheckpointSaverConfig())
    try:
        manager.save(checkpoint_path, {ITEM: {"w": dtensor}})
        manager.load(checkpoint_path, into={ITEM: {"w": target}})
    finally:
        manager.close()

    torch.testing.assert_close(target.to_local(), base[2:6], rtol=0, atol=0)
