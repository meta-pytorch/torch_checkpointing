# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_checkpointing.experimental.cooperative_resharding.checkpoint_reader as checkpoint_reader_module
from torch_checkpointing.checkpoint_base import CheckpointInfo, CheckpointItem
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    SerializationFormat,
    TorchSerialization,
)
from torch_checkpointing.distributed_metadata import (
    CheckpointMetadata,
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
    METADATA_FILE_NAME,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    ReplicateSpec,
    ShardSpec,
)
from torch_checkpointing.experimental.cooperative_resharding.checkpoint_reader import (
    CheckpointReader,
)
from torch_checkpointing.experimental.cooperative_resharding.config import (
    CooperativeLoadResult,
)
from torch_checkpointing.experimental.cooperative_resharding.default_resharder import (
    DefaultResharder,
)
from torch_checkpointing.experimental.cooperative_resharding.loader import (
    CooperativeLoadFailure,
    CooperativeLoadRequest,
    CooperativeLoadUnsupported,
)
from torch_checkpointing.experimental.cooperative_resharding.metadata import (
    ArchiveMetadataPreflightError,
    MetadataIneligibilityReason,
    MetadataPreflightErrorKind,
    MetadataPreparationIneligible,
)
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo


class _DefaultResharderSubclass(DefaultResharder):
    pass


@dataclass
class _ReaderCase:
    reader: CheckpointReader
    checkpoint_info: Any
    targets: dict[str, torch.Tensor]
    expected: dict[str, torch.Tensor]


@dataclass
class _FakeCollectives:
    remote_item_modes: tuple[tuple[str, str], ...] | None = None
    remote_path: str | None = None
    remote_schedule_status: str = "empty"
    mismatch_source_contract: bool = False
    remote_checkpoint_exists: bool = True

    def all_gather_object(self, output: list[object], value: object) -> None:
        output[0] = value
        if isinstance(value, checkpoint_reader_module._ReadRankManifest):
            output[1] = replace(
                value,
                rank=1,
                hostname="127.0.0.2",
                checkpoint_path=(
                    value.checkpoint_path
                    if self.remote_path is None
                    else self.remote_path
                ),
                checkpoint_exists=self.remote_checkpoint_exists,
                item_modes=(
                    value.item_modes
                    if self.remote_item_modes is None
                    else self.remote_item_modes
                ),
            )
            return
        if isinstance(value, checkpoint_reader_module._SourceMetadataManifest):
            contract_digest = value.contract_digest
            if self.mismatch_source_contract:
                contract_digest = f"{contract_digest}-different"
            output[1] = replace(value, contract_digest=contract_digest)
            return
        if isinstance(value, checkpoint_reader_module._LocalScheduleManifest):
            output[1] = replace(
                value,
                statuses=tuple(
                    (key, self.remote_schedule_status) for key, _ in value.statuses
                ),
            )
            return
        raise AssertionError(f"unexpected collective payload: {type(value)}")


def _read_rank_manifest(
    rank: int,
    hostname: str,
) -> checkpoint_reader_module._ReadRankManifest:
    return checkpoint_reader_module._ReadRankManifest(
        rank=rank,
        hostname=hostname,
        checkpoint_path="/checkpoint",
        checkpoint_exists=True,
        path_error=None,
        runtime_eligible=True,
        item_modes=(("model", "default"),),
    )


def _mesh_metadata(
    *,
    world_size: int,
    sharded: bool,
) -> DTensorShardingMetadata:
    return DTensorShardingMetadata(
        global_shape=(4,),
        dtype="torch.float32",
        stride=(1,),
        mesh_spec=DeviceMeshSpec(
            device_type="cpu",
            mesh_shape=(world_size,),
            mesh_data=tuple(range(world_size)),
        ),
        placements=(ShardSpec(0) if sharded else ReplicateSpec(),),
    )


def _build_reader_case(
    checkpoint_path: Path,
    *,
    keys: tuple[str, ...] = ("model",),
    target_is_sharded: bool = True,
    resharder_type: type[DefaultResharder] = DefaultResharder,
    source_serialization: SerializationFormat | None = None,
    rank_info: RankInfo | None = None,
) -> _ReaderCase:
    checkpoint_path.mkdir()
    source_metadata = _mesh_metadata(world_size=1, sharded=False)
    target_metadata = (
        _mesh_metadata(world_size=2, sharded=True)
        if target_is_sharded
        else source_metadata
    )
    distributed_items: dict[str, DistributedItemMetadata] = {}
    checkpoint_items: dict[str, CheckpointItem] = {}
    local_metadata: dict[str, dict[tuple[str, ...], DTensorShardingMetadata]] = {}
    targets: dict[str, torch.Tensor] = {}
    expected: dict[str, torch.Tensor] = {}
    serialization = source_serialization or TorchSerialization()
    for index, key in enumerate(keys):
        values = torch.arange(4, dtype=torch.float32) + index * 10
        expected[key] = values[:2] if target_is_sharded else values
        target = torch.full_like(expected[key], -1)
        targets[key] = target
        layout = LayoutInfo(f"{key}_0.pt", serialization)
        torch.save({"weight": values}, checkpoint_path / layout.file_path)
        distributed_items[key] = DistributedItemMetadata(
            nested_path_to_metadata={
                ("weight",): [
                    GlobalObjectMetadata(
                        sharding_metadata=source_metadata,
                        ranks=(0,),
                    )
                ]
            },
            rank_to_layout_info={0: layout},
        )
        checkpoint_items[key] = CheckpointItem(
            value={"weight": target},
            layout=layout,
            resharder=resharder_type(),
        )
        local_metadata[key] = {("weight",): target_metadata}
    distributed_metadata = DistributedMetadata(
        metadata=distributed_items,
        world_size=1,
    )
    with (checkpoint_path / METADATA_FILE_NAME).open("wb") as stream:
        pickle.dump(distributed_metadata.to_dict(), stream)
    checkpoint_metadata = CheckpointMetadata(
        distributed_metadata=distributed_metadata,
        local_metadata=local_metadata,
    )
    reader = CheckpointReader(
        rank_info=rank_info
        or RankInfo(
            global_rank=0,
            global_world_size=2,
            role_rank=0,
            role_world_size=2,
        ),
        storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
    )
    return _ReaderCase(
        reader=reader,
        checkpoint_info=CheckpointInfo(checkpoint_items).for_reads(checkpoint_metadata),
        targets=targets,
        expected=expected,
    )


def _install_fake_distributed(
    monkeypatch: pytest.MonkeyPatch,
    collectives: _FakeCollectives,
) -> None:
    monkeypatch.setenv(
        "TORCH_CHECKPOINTING_ENABLE_COOPERATIVE_RESHARDING",
        "1",
    )
    monkeypatch.setenv(
        "TORCH_CHECKPOINTING_DISABLE_COOPERATIVE_RESHARDING",
        "0",
    )
    monkeypatch.setattr(checkpoint_reader_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint_reader_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(checkpoint_reader_module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        checkpoint_reader_module.dist,
        "all_gather_object",
        collectives.all_gather_object,
    )

    def broadcast_object_list(values: list[object], *, src: int) -> None:
        assert src == 0
        values[0] = "test-nonce"

    monkeypatch.setattr(
        checkpoint_reader_module.dist,
        "broadcast_object_list",
        broadcast_object_list,
    )
    monkeypatch.setattr(
        checkpoint_reader_module.dist.distributed_c10d,
        "_get_default_store",
        lambda: object(),
    )
    monkeypatch.setattr(
        checkpoint_reader_module,
        "_local_shared_memory_capable",
        lambda: True,
    )
    monkeypatch.setattr(
        checkpoint_reader_module.socket, "gethostname", lambda: "127.0.0.1"
    )


def test_cooperative_metric_logging_is_bounded_for_large_worlds() -> None:
    small = checkpoint_reader_module.RankTopology(
        global_rank=1,
        nodes=(checkpoint_reader_module.NodeMembership("node-a", (0, 1)),),
        coordination_world_count=1,
        job_id="small",
    )
    large_nodes = (
        checkpoint_reader_module.NodeMembership("node-a", (0,)),
        checkpoint_reader_module.NodeMembership("node-b", tuple(range(1, 129))),
    )
    large_leader = checkpoint_reader_module.RankTopology(
        global_rank=1,
        nodes=large_nodes,
        coordination_world_count=1,
        job_id="large-leader",
    )
    large_follower = checkpoint_reader_module.RankTopology(
        global_rank=2,
        nodes=large_nodes,
        coordination_world_count=1,
        job_id="large-follower",
    )

    assert checkpoint_reader_module._should_emit_cooperative_metrics(small)
    assert checkpoint_reader_module._should_emit_cooperative_metrics(large_leader)
    assert not checkpoint_reader_module._should_emit_cooperative_metrics(large_follower)


def test_node_memberships_use_leader_ranks_independent_of_hostnames() -> None:
    first = (
        _read_rank_manifest(3, "host-a"),
        _read_rank_manifest(0, "host-z"),
        _read_rank_manifest(2, "host-z"),
        _read_rank_manifest(1, "host-a"),
    )
    renamed_and_reordered = (
        _read_rank_manifest(2, "host-b"),
        _read_rank_manifest(1, "host-y"),
        _read_rank_manifest(0, "host-b"),
        _read_rank_manifest(3, "host-y"),
    )

    expected = (
        checkpoint_reader_module.NodeMembership(0, (0, 2)),
        checkpoint_reader_module.NodeMembership(1, (1, 3)),
    )
    assert checkpoint_reader_module._node_memberships(first) == expected
    assert checkpoint_reader_module._node_memberships(renamed_and_reordered) == expected


def test_reader_schedules_sorted_cooperative_keys_with_absent_peer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_reader_case(tmp_path / "checkpoint", keys=("zeta", "alpha"))
    _install_fake_distributed(
        monkeypatch,
        _FakeCollectives(remote_item_modes=()),
    )
    requests: list[CooperativeLoadRequest] = []

    def capture_request(
        request: CooperativeLoadRequest,
        *,
        _pool_cache: object | None = None,
    ) -> CooperativeLoadResult:
        requests.append(request)
        assert _pool_cache is checkpoint_reader_module._CHECKPOINT_READER_POOL_CACHE
        for target in request.target_state_dict.values():
            assert torch.equal(target, torch.full_like(target, -1))
        return CooperativeLoadResult(target_count=len(request.target_state_dict))

    monkeypatch.setattr(
        checkpoint_reader_module,
        "load_cooperatively",
        capture_request,
    )

    result, missing = case.reader.read(
        str(tmp_path / "checkpoint"),
        case.checkpoint_info,
    )

    assert missing == []
    assert [request.metadata_provider.item_key for request in requests] == [
        "alpha",
        "zeta",
    ]
    assert all(request.bind_host == "0.0.0.0" for request in requests)
    assert all(request.advertise_host == "127.0.0.1" for request in requests)
    assert requests[0].topology.nodes == (
        checkpoint_reader_module.NodeMembership(0, (0,)),
        checkpoint_reader_module.NodeMembership(1, (1,)),
    )
    assert result["alpha"]["weight"] is case.targets["alpha"]
    assert result["zeta"]["weight"] is case.targets["zeta"]


def test_should_reshard_false_rank_joins_empty_before_direct_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_reader_case(
        tmp_path / "checkpoint",
        target_is_sharded=False,
    )
    _install_fake_distributed(
        monkeypatch,
        _FakeCollectives(remote_schedule_status="ready"),
    )
    events: list[str] = []

    def capture_request(
        request: CooperativeLoadRequest,
        *,
        _pool_cache: object | None = None,
    ) -> CooperativeLoadResult:
        assert _pool_cache is checkpoint_reader_module._CHECKPOINT_READER_POOL_CACHE
        events.append("cooperative")
        assert request.local_load_plan == {}
        assert request.target_state_dict == {}
        return CooperativeLoadResult()

    original_direct_read = case.reader._read_without_resharding

    def direct_read(*args: Any, **kwargs: Any) -> Any:
        events.append("direct")
        return original_direct_read(*args, **kwargs)

    monkeypatch.setattr(checkpoint_reader_module, "load_cooperatively", capture_request)
    monkeypatch.setattr(case.reader, "_read_without_resharding", direct_read)

    result, missing = case.reader.read(
        str(tmp_path / "checkpoint"),
        case.checkpoint_info,
    )

    assert missing == []
    assert events == ["cooperative", "direct"]
    torch.testing.assert_close(result["model"]["weight"], case.expected["model"])


def test_unsupported_metadata_falls_back_before_target_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_reader_case(tmp_path / "checkpoint")
    _install_fake_distributed(
        monkeypatch,
        _FakeCollectives(remote_item_modes=()),
    )

    def prepare_ineligible(*args: Any, **kwargs: Any) -> MetadataPreparationIneligible:
        return MetadataPreparationIneligible(
            reason=MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
            detail="unsupported archive feature",
        )

    def run_metadata_preflight(
        request: CooperativeLoadRequest,
        *,
        _pool_cache: object | None = None,
    ) -> CooperativeLoadResult:
        assert _pool_cache is checkpoint_reader_module._CHECKPOINT_READER_POOL_CACHE
        assert torch.equal(
            request.target_state_dict["weight"],
            torch.full_like(request.target_state_dict["weight"], -1),
        )
        assert request.metadata_provider is not None
        request.metadata_provider.load_metadata(
            {0: {"weight"}},
            storage=request.storage,
            source_path_for_rank=request.source_path_for_rank,
            max_workers=1,
            timeout_seconds=1,
        )
        raise AssertionError("metadata provider should have vetoed the load")

    monkeypatch.setattr(
        checkpoint_reader_module,
        "prepare_source_tensor_metadata",
        prepare_ineligible,
    )
    monkeypatch.setattr(
        checkpoint_reader_module,
        "load_cooperatively",
        run_metadata_preflight,
    )

    result, missing = case.reader.read(
        str(tmp_path / "checkpoint"),
        case.checkpoint_info,
    )

    assert missing == []
    torch.testing.assert_close(result["model"]["weight"], case.expected["model"])


def test_fatal_metadata_error_never_falls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    case = _build_reader_case(checkpoint_path)
    _install_fake_distributed(
        monkeypatch,
        _FakeCollectives(remote_item_modes=()),
    )
    fatal = ArchiveMetadataPreflightError(
        MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
        checkpoint_path / "model_0.pt",
        "invalid archive",
        source_rank=0,
    )

    def fail_metadata(*args: Any, **kwargs: Any) -> Any:
        raise fatal

    def run_metadata_preflight(
        request: CooperativeLoadRequest,
        *,
        _pool_cache: object | None = None,
    ) -> CooperativeLoadResult:
        assert _pool_cache is checkpoint_reader_module._CHECKPOINT_READER_POOL_CACHE
        assert request.metadata_provider is not None
        request.metadata_provider.load_metadata(
            {0: {"weight"}},
            storage=request.storage,
            source_path_for_rank=request.source_path_for_rank,
            max_workers=1,
            timeout_seconds=1,
        )
        raise AssertionError("metadata provider should have failed")

    monkeypatch.setattr(
        checkpoint_reader_module,
        "prepare_source_tensor_metadata",
        fail_metadata,
    )
    monkeypatch.setattr(
        checkpoint_reader_module,
        "load_cooperatively",
        run_metadata_preflight,
    )

    with pytest.raises(ArchiveMetadataPreflightError) as error:
        case.reader.read(str(checkpoint_path), case.checkpoint_info)

    assert error.value is fatal
    assert torch.equal(
        case.targets["model"],
        torch.full_like(case.targets["model"], -1),
    )


def test_unsupported_after_completed_key_is_fatal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_reader_case(tmp_path / "checkpoint", keys=("zeta", "alpha"))
    _install_fake_distributed(
        monkeypatch,
        _FakeCollectives(remote_item_modes=()),
    )
    calls = 0

    def fail_second_key(
        request: CooperativeLoadRequest,
        *,
        _pool_cache: object | None = None,
    ) -> CooperativeLoadResult:
        nonlocal calls
        assert _pool_cache is checkpoint_reader_module._CHECKPOINT_READER_POOL_CACHE
        calls += 1
        if calls == 1:
            for target in request.target_state_dict.values():
                target.fill_(5)
            return CooperativeLoadResult(target_count=1)
        raise CooperativeLoadUnsupported("late veto")

    monkeypatch.setattr(
        checkpoint_reader_module,
        "load_cooperatively",
        fail_second_key,
    )

    with pytest.raises(CooperativeLoadFailure) as error:
        case.reader.read(str(tmp_path / "checkpoint"), case.checkpoint_info)

    assert error.value.target_writes_started
    assert calls == 2
    assert torch.equal(case.targets["alpha"], torch.full_like(case.targets["alpha"], 5))
    assert torch.equal(case.targets["zeta"], torch.full_like(case.targets["zeta"], -1))


@pytest.mark.parametrize(
    "veto",
    [
        "subclass",
        "path",
        "metadata",
        "role",
        "environment",
        "serialization",
        "shared_memory",
    ],
)
def test_collective_contract_veto_uses_legacy_resharding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    veto: str,
) -> None:
    rank_info = None
    resharder_type = DefaultResharder
    serialization: SerializationFormat | None = None
    collectives = _FakeCollectives(remote_item_modes=())
    if veto == "subclass":
        resharder_type = _DefaultResharderSubclass
    elif veto == "path":
        collectives.remote_path = "/different/checkpoint"
    elif veto == "metadata":
        collectives.mismatch_source_contract = True
    elif veto == "role":
        rank_info = RankInfo(0, 2, 1, 2)
    elif veto == "serialization":
        serialization = JsonSerialization()
    case = _build_reader_case(
        tmp_path / "checkpoint",
        resharder_type=resharder_type,
        source_serialization=serialization,
        rank_info=rank_info,
    )
    _install_fake_distributed(monkeypatch, collectives)
    if veto == "environment":
        monkeypatch.setenv(
            "TORCH_CHECKPOINTING_DISABLE_COOPERATIVE_RESHARDING",
            "1",
        )
    if veto == "shared_memory":
        monkeypatch.setattr(
            checkpoint_reader_module,
            "_local_shared_memory_capable",
            lambda: False,
        )

    def unexpected_cooperative_load(
        request: CooperativeLoadRequest,
        *,
        _pool_cache: object | None = None,
    ) -> CooperativeLoadResult:
        del _pool_cache
        raise AssertionError(f"unexpected cooperative load: {request}")

    monkeypatch.setattr(
        checkpoint_reader_module,
        "load_cooperatively",
        unexpected_cooperative_load,
    )

    result, missing = case.reader.read(
        str(tmp_path / "checkpoint"),
        case.checkpoint_info,
    )

    assert missing == []
    torch.testing.assert_close(result["model"]["weight"], case.expected["model"])


def test_missing_checkpoint_is_reported_collectively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = CheckpointReader(
        RankInfo(0, 2, 0, 2),
        LocalFileSystemStorageConfig(use_direct_io=False),
    )
    _install_fake_distributed(
        monkeypatch,
        _FakeCollectives(remote_checkpoint_exists=False),
    )

    with pytest.raises(FileNotFoundError, match=r"ranks \[0, 1\]"):
        reader.read(
            str(tmp_path / "missing"),
            CheckpointInfo({}).for_reads(),
        )


@pytest.mark.parametrize(
    ("gate", "use_resharder"),
    [("default", False), ("default", True), ("disabled", True)],
)
def test_default_or_disabled_reads_do_not_enter_collective_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gate: str,
    use_resharder: bool,
) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    expected = torch.tensor([1.0, 2.0])
    torch.save({"weight": expected}, checkpoint_path / "model_0.pt")
    if gate == "disabled":
        monkeypatch.setenv(
            "TORCH_CHECKPOINTING_ENABLE_COOPERATIVE_RESHARDING",
            "1",
        )
        monkeypatch.setenv(
            "TORCH_CHECKPOINTING_DISABLE_COOPERATIVE_RESHARDING",
            "1",
        )
    else:
        monkeypatch.delenv(
            "TORCH_CHECKPOINTING_ENABLE_COOPERATIVE_RESHARDING",
            raising=False,
        )
        monkeypatch.delenv(
            "TORCH_CHECKPOINTING_DISABLE_COOPERATIVE_RESHARDING",
            raising=False,
        )
    monkeypatch.setattr(checkpoint_reader_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(checkpoint_reader_module.dist, "get_world_size", lambda: 2)

    def unexpected_collective(output: list[object], value: object) -> None:
        raise AssertionError(f"unexpected collective payload: {value}")

    monkeypatch.setattr(
        checkpoint_reader_module.dist,
        "all_gather_object",
        unexpected_collective,
    )
    target = torch.full((2,), -1.0)
    item = CheckpointItem(
        value={"weight": target},
        layout=LayoutInfo("model_0.pt", TorchSerialization()),
        resharder=DefaultResharder() if use_resharder else None,
    )
    reader = CheckpointReader(
        RankInfo(0, 2, 0, 2),
        LocalFileSystemStorageConfig(use_direct_io=False),
    )

    result, missing = reader.read(
        str(checkpoint_path),
        CheckpointInfo({"model": item}).for_reads(),
    )

    assert missing == []
    torch.testing.assert_close(result["model"]["weight"], expected)


def _gloo_worker(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_path: str,
    result_path: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        source_metadata = _mesh_metadata(world_size=1, sharded=False)
        target_metadata = _mesh_metadata(world_size=world_size, sharded=True)
        distributed_metadata = DistributedMetadata(
            metadata={
                "model": DistributedItemMetadata(
                    nested_path_to_metadata={
                        ("weight",): [GlobalObjectMetadata(source_metadata, (0,))]
                    },
                    rank_to_layout_info={
                        0: LayoutInfo("model_0.pt", TorchSerialization())
                    },
                )
            },
            world_size=1,
        )
        target = torch.full((2,), -1.0)
        read_info = CheckpointInfo(
            {
                "model": CheckpointItem(
                    value={"weight": target},
                    layout=LayoutInfo("model_0.pt", TorchSerialization()),
                    resharder=DefaultResharder(),
                )
            }
        ).for_reads(
            CheckpointMetadata(
                distributed_metadata=distributed_metadata,
                local_metadata={"model": {("weight",): target_metadata}},
            )
        )
        reader = CheckpointReader(
            RankInfo(rank, world_size, rank, world_size),
            LocalFileSystemStorageConfig(use_direct_io=False),
        )
        with (
            mock.patch.dict(
                os.environ,
                {
                    "TORCH_CHECKPOINTING_DISABLE_COOPERATIVE_RESHARDING": "0",
                    "TORCH_CHECKPOINTING_ENABLE_COOPERATIVE_RESHARDING": "1",
                },
            ),
            mock.patch.object(
                checkpoint_reader_module.socket,
                "gethostname",
                return_value="127.0.0.1",
            ),
        ):
            result, missing = reader.read(checkpoint_path, read_info)
        assert missing == []
        torch.save(result["model"]["weight"], Path(result_path) / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_cooperative_read(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "checkpoint"
    checkpoint_path.mkdir()
    torch.save(
        {"weight": torch.arange(4, dtype=torch.float32)},
        checkpoint_path / "model_0.pt",
    )
    source_metadata = _mesh_metadata(world_size=1, sharded=False)
    distributed_metadata = DistributedMetadata(
        metadata={
            "model": DistributedItemMetadata(
                nested_path_to_metadata={
                    ("weight",): [GlobalObjectMetadata(source_metadata, (0,))]
                },
                rank_to_layout_info={0: LayoutInfo("model_0.pt", TorchSerialization())},
            )
        },
        world_size=1,
    )
    with (checkpoint_path / METADATA_FILE_NAME).open("wb") as stream:
        pickle.dump(distributed_metadata.to_dict(), stream)
    result_path = tmp_path / "results"
    result_path.mkdir()

    mp.spawn(
        _gloo_worker,
        args=(
            2,
            str(tmp_path / "gloo-init"),
            str(checkpoint_path),
            str(result_path),
        ),
        nprocs=2,
        join=True,
    )

    torch.testing.assert_close(
        torch.load(result_path / "rank-0.pt", weights_only=True),
        torch.tensor([0.0, 1.0]),
    )
    torch.testing.assert_close(
        torch.load(result_path / "rank-1.pt", weights_only=True),
        torch.tensor([2.0, 3.0]),
    )
