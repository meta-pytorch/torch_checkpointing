# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Checkpoint reader functionality for machine learning models.

This module provides classes for reading checkpoints from storage, including
determining checkpoint layout and configuring the reader.
"""

import atexit
import hashlib
import json
import logging
import os
import pickle
import secrets
import socket
import sys
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ...checkpoint_base import (
    CheckpointInfo,
    CheckpointItem,
    CheckpointReadInfo,
)
from ...checkpoint_layout import (
    default_layout_info,
    JsonSerialization,
    LayoutInfo,
    RawSerialization,
    SafetensorsSerialization,
    TorchSerialization,
)
from ...distributed_metadata import (
    CheckpointMetadata,
    DistributedItemMetadata,
    DistributedMetadata,
    METADATA_FILE_NAME,
    ShardingMetadata,
)
from ...logging_utils import EventLogger, EventType
from ...storage.base_storage import Storage, StorageConfig
from ...storage.torch_serialization import MmapFill
from ...types import CheckpointPath, NestedPath, RankInfo, STATE_DICT
from ...utils import from_dict
from ...walk_utils import walk_checkpoint_structure
from .default_resharder import (
    _PreparedCooperativeLoad,
    DefaultResharder,
)
from .layout import SourceTensorMetadata
from .loader import (
    _ChunkPoolCache,
    CooperativeLoadFailure,
    CooperativeLoadRequest,
    CooperativeLoadUnsupported,
    load_cooperatively,
    MetadataProvider,
    MetricValue,
)
from .metadata import (
    MetadataPreparationEligible,
    MetadataPreparationIneligible,
    prepare_source_tensor_metadata,
)
from .planning import NodeMembership, RankTopology
from .rendezvous import C10dStoreRendezvous
from .shared_memory import recommended_capacity_bytes

logger = logging.getLogger(__name__)

_DISABLE_COOPERATIVE_RESHARDING_ENV = (
    "TORCH_CHECKPOINTING_DISABLE_COOPERATIVE_RESHARDING"
)
_ENABLE_COOPERATIVE_RESHARDING_ENV = "TORCH_CHECKPOINTING_ENABLE_COOPERATIVE_RESHARDING"
_DETAILED_COOPERATIVE_METRIC_WORLD_SIZE_LIMIT = 128
_CHECKPOINT_READER_POOL_CACHE = _ChunkPoolCache()
atexit.register(_CHECKPOINT_READER_POOL_CACHE.close)


@dataclass(frozen=True, slots=True)
class _ReadRankManifest:
    rank: int
    hostname: str
    checkpoint_path: str
    checkpoint_exists: bool
    path_error: str | None
    runtime_eligible: bool
    item_modes: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class _CooperativeReadContext:
    candidate_keys: tuple[str, ...]
    nodes: tuple[NodeMembership, ...]
    advertise_host: str
    bind_host: str


@dataclass(frozen=True, slots=True)
class _SourceMetadataManifest:
    available: bool
    layouts_supported: bool
    contract_digest: str
    error: str | None


@dataclass(frozen=True, slots=True)
class _LocalScheduleManifest:
    statuses: tuple[tuple[str, str], ...]
    error: str | None


@dataclass(frozen=True, slots=True)
class _CooperativeSchedule:
    keys: tuple[str, ...]
    prepared_by_key: Mapping[str, _PreparedCooperativeLoad]


@dataclass(frozen=True, slots=True)
class _TorchArchiveMetadataProvider(MetadataProvider):
    item_key: str
    metric_callback: Callable[[str, Mapping[str, MetricValue]], None] | None = None

    def load_metadata(
        self,
        demands_by_rank: Mapping[int, Collection[str]],
        *,
        storage: Storage,
        source_path_for_rank: Callable[[int], Path],
        max_workers: int,
        timeout_seconds: float,
    ) -> Mapping[int, Mapping[str, SourceTensorMetadata]]:
        result = prepare_source_tensor_metadata(
            storage,
            {
                source_rank: source_path_for_rank(source_rank)
                for source_rank in demands_by_rank
            },
            demands_by_rank,
            item_key=self.item_key,
            timeout_seconds=timeout_seconds,
            _max_workers=max_workers,
            _metric_callback=self.metric_callback,
        )
        if isinstance(result, MetadataPreparationIneligible):
            location = "" if result.path is None else f" at {result.path}"
            raise CooperativeLoadUnsupported(
                f"{result.reason.value}{location}: {result.detail}"
            )
        assert isinstance(result, MetadataPreparationEligible)
        return result.metadata_by_rank


def _build_src_to_layout_info_mappings(
    distributed_metadata: "DistributedMetadata",
) -> dict[int, dict[str, LayoutInfo | None]]:
    """Build a mapping from source ranks to their per-item layout info.

    Pivots the per-item rank_to_layout_info into a per-rank item_to_layout_info
    structure needed by the checkpoint reader for file path resolution.
    """
    result: dict[int, dict[str, LayoutInfo | None]] = {}
    for item_key, item_metadata in distributed_metadata.metadata.items():
        for rank, layout_info in item_metadata.rank_to_layout_info.items():
            if rank not in result:
                result[rank] = {}
            result[rank][item_key] = layout_info
    return result


def _all_gather_objects(value: object, world_size: int) -> tuple[object, ...]:
    gathered: list[object] = [None] * world_size
    dist.all_gather_object(gathered, value)
    return tuple(gathered)


def _item_modes(checkpoint_info: CheckpointInfo) -> tuple[tuple[str, str], ...]:
    modes: list[tuple[str, str]] = []
    for key, item in sorted(checkpoint_info.checkpoint_items.items()):
        if item.resharder is None:
            mode = "none"
        elif type(item.resharder) is DefaultResharder:
            mode = "default"
        else:
            mode = "custom"
        modes.append((key, mode))
    return tuple(modes)


def _cooperative_candidate_keys(
    manifests: tuple[_ReadRankManifest, ...],
) -> tuple[str, ...] | None:
    modes_by_key: dict[str, set[str]] = defaultdict(set)
    for manifest in manifests:
        for key, mode in manifest.item_modes:
            if mode == "custom":
                return None
            modes_by_key[key].add(mode)
    if any(len(modes) > 1 for modes in modes_by_key.values()):
        return None
    return tuple(
        key for key in sorted(modes_by_key) if modes_by_key[key] == {"default"}
    )


def _node_memberships(
    manifests: tuple[_ReadRankManifest, ...],
) -> tuple[NodeMembership, ...]:
    ranks_by_hostname: dict[str, list[int]] = defaultdict(list)
    for manifest in manifests:
        ranks_by_hostname[manifest.hostname].append(manifest.rank)
    return tuple(
        NodeMembership(min(ranks), tuple(ranks))
        for ranks in sorted(ranks_by_hostname.values(), key=min)
    )


def _wildcard_bind_host(advertise_host: str) -> str:
    try:
        families = {
            entry[0]
            for entry in socket.getaddrinfo(
                advertise_host,
                None,
                type=socket.SOCK_STREAM,
            )
        }
    except OSError:
        return "0.0.0.0"
    return "0.0.0.0" if socket.AF_INET in families else "::"


def _local_shared_memory_capable() -> bool:
    directory = Path("/dev/shm")
    if not sys.platform.startswith("linux"):
        return False
    if not directory.is_dir() or not os.access(directory, os.W_OK | os.X_OK):
        return False
    try:
        return recommended_capacity_bytes(directory=directory) > 0
    except OSError:
        return False


def _cooperative_resharding_enabled() -> bool:
    return (
        os.environ.get(_ENABLE_COOPERATIVE_RESHARDING_ENV) == "1"
        and os.environ.get(_DISABLE_COOPERATIVE_RESHARDING_ENV) != "1"
    )


def _source_metadata_contract(
    checkpoint_path: Path,
    metadata: DistributedMetadata,
    candidate_keys: tuple[str, ...],
) -> tuple[bool, bool, str]:
    available = True
    layouts_supported = True
    item_contracts: list[object] = []
    for key in candidate_keys:
        item_metadata = metadata.metadata.get(key)
        if item_metadata is None:
            available = False
            item_contracts.append((key, None))
            continue
        paths: list[tuple[int, str, str]] = []
        for source_rank, declared_layout in sorted(
            item_metadata.rank_to_layout_info.items()
        ):
            layout = declared_layout or default_layout_info(key, source_rank)
            serialization = layout.serialization_format
            layouts_supported &= isinstance(serialization, TorchSerialization)
            paths.append(
                (
                    source_rank,
                    str(checkpoint_path / layout.file_path),
                    f"{type(serialization).__module__}.{type(serialization).__qualname__}",
                )
            )
        canonical_metadata = json.dumps(
            item_metadata.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        item_contracts.append(
            (
                key,
                tuple(paths),
                hashlib.sha256(canonical_metadata).hexdigest(),
            )
        )
    canonical_contract = json.dumps(
        {
            "items": item_contracts,
            "world_size": metadata.world_size,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return (
        available,
        layouts_supported,
        hashlib.sha256(canonical_contract).hexdigest(),
    )


def _cooperative_metric_logger(
    item_key: str,
    rank: int,
) -> Callable[[str, Mapping[str, MetricValue]], None]:
    def emit(event_name: str, fields: Mapping[str, MetricValue]) -> None:
        payload = dict(fields)
        payload.setdefault("rank", rank)
        logger.info(
            "Cooperative resharding metric item=%r event=%s fields=%s",
            item_key,
            event_name,
            payload,
        )

    return emit


def _should_emit_cooperative_metrics(topology: RankTopology) -> bool:
    world_size = sum(len(node.ranks) for node in topology.nodes)
    return (
        world_size <= _DETAILED_COOPERATIVE_METRIC_WORLD_SIZE_LIMIT
        or topology.is_node_leader
    )


def _partition_items_for_read(
    checkpoint_info: CheckpointReadInfo,
    checkpoint_metadata: CheckpointMetadata | None,
    source_metadata: DistributedMetadata | None,
) -> tuple[dict[str, CheckpointItem], dict[str, CheckpointItem]]:
    needing_reshard: dict[str, CheckpointItem] = {}
    not_needing_reshard: dict[str, CheckpointItem] = {}
    for key, item in checkpoint_info.checkpoint_items.items():
        source_item_metadata = (
            None if source_metadata is None else source_metadata.metadata.get(key)
        )
        target_metadata = (
            None
            if checkpoint_metadata is None
            else checkpoint_metadata.local_metadata.get(key)
        )
        if item.resharder is not None and item.resharder.should_reshard(
            source_item_metadata,
            target_metadata,
        ):
            needing_reshard[key] = item
        else:
            not_needing_reshard[key] = item
    return needing_reshard, not_needing_reshard


def _prepare_local_cooperative_items(
    candidate_keys: tuple[str, ...],
    items_needing_reshard: Mapping[str, CheckpointItem],
    checkpoint_metadata: CheckpointMetadata | None,
    source_metadata: DistributedMetadata,
) -> tuple[dict[str, _PreparedCooperativeLoad], tuple[tuple[str, str], ...]]:
    prepared_by_key: dict[str, _PreparedCooperativeLoad] = {}
    statuses: list[tuple[str, str]] = []
    for key in candidate_keys:
        item = items_needing_reshard.get(key)
        if item is None:
            statuses.append((key, "empty"))
            continue
        resharder = item.resharder
        if type(resharder) is not DefaultResharder:
            statuses.append((key, "unsupported"))
            continue
        target_metadata = (
            None
            if checkpoint_metadata is None
            else checkpoint_metadata.local_metadata.get(key)
        )
        source_item_metadata = source_metadata.metadata.get(key)
        if target_metadata is None or source_item_metadata is None:
            statuses.append((key, "unsupported"))
            continue
        prepared = resharder._prepare_cooperative_load(
            key,
            target_metadata,
            source_item_metadata,
            item.value,
        )
        if prepared.non_reshardable_paths:
            statuses.append((key, "unsupported"))
            continue
        prepared_by_key[key] = prepared
        statuses.append((key, "ready"))
    return prepared_by_key, tuple(statuses)


def _resolve_cooperative_schedule_keys(
    candidate_keys: tuple[str, ...],
    manifests: tuple[_LocalScheduleManifest, ...],
) -> tuple[str, ...] | None:
    status_maps = tuple(dict(item.statuses) for item in manifests)
    if any(set(status) != set(candidate_keys) for status in status_maps):
        return None
    if any("unsupported" in status.values() for status in status_maps):
        return None
    return tuple(
        key
        for key in candidate_keys
        if any(status[key] == "ready" for status in status_maps)
    )


class CheckpointReader:
    """
    Handles reading state dictionaries from storage.

    This class is responsible for reading model state dictionaries from storage according
    to the specified checkpoint layout. It supports synchronization barriers to ensure
    all ranks in a distributed setting complete their checkpoint operations.
    """

    def __init__(
        self,
        rank_info: RankInfo,
        storage_config: StorageConfig,
        disable_use_mmap_backed_storage_on_load: bool = False,
        mmap_fill_factory: Callable[[Storage], MmapFill | None] | None = None,
    ):
        """
        Initialize a CheckpointReader.

        Args:
            rank_info: Information about the current global/local rank.
            storage_config: Configuration for the storage backend.
            disable_use_mmap_backed_storage_on_load: If True, fall back to the
                BytesIO-based torch.load path for torch-serialized files. The
                default (False) routes loads through a single mmap-backed
                overall storage to reduce allocator fragmentation after load
                cleanup.
            mmap_fill_factory: Optional factory for a storage-specific mmap
                fill callback. It is resolved only for mmap-backed torch loads.
        """

        self._rank_info = rank_info
        self._storage: Storage = storage_config.create_storage()
        self._disable_use_mmap_backed_storage_on_load = (
            disable_use_mmap_backed_storage_on_load
        )
        self._mmap_fill_factory = mmap_fill_factory

    def _collective_cooperative_preflight(
        self,
        path: str,
        checkpoint_info: CheckpointReadInfo,
    ) -> _CooperativeReadContext | None:
        checkpoint_path = Path(path)
        if not dist.is_initialized() or dist.get_world_size() <= 1:
            if not self._storage.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint path {path} does not exist.")
            return None

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        path_error: str | None = None
        try:
            checkpoint_exists = self._storage.exists(checkpoint_path)
        except Exception as error:
            checkpoint_exists = False
            path_error = f"{type(error).__name__}: {error}"
        hostname = socket.gethostname()
        manifest = _ReadRankManifest(
            rank=rank,
            hostname=hostname,
            checkpoint_path=str(checkpoint_path.absolute()),
            checkpoint_exists=checkpoint_exists,
            path_error=path_error,
            runtime_eligible=(
                self._rank_info.global_rank == rank
                and self._rank_info.global_world_size == world_size
                and self._rank_info.role_rank == rank
                and self._rank_info.role_world_size == world_size
                and _cooperative_resharding_enabled()
                and _local_shared_memory_capable()
            ),
            item_modes=_item_modes(checkpoint_info),
        )
        gathered = _all_gather_objects(manifest, world_size)
        manifests = tuple(
            item for item in gathered if isinstance(item, _ReadRankManifest)
        )
        if len(manifests) != world_size:
            raise RuntimeError(
                "cooperative preflight received an invalid rank manifest"
            )
        manifest_by_rank = {item.rank: item for item in manifests}
        if set(manifest_by_rank) != set(range(world_size)):
            raise RuntimeError("cooperative preflight rank manifests are incomplete")
        ordered = tuple(manifest_by_rank[index] for index in range(world_size))
        errors = tuple(
            item.path_error for item in ordered if item.path_error is not None
        )
        if errors:
            raise RuntimeError(f"checkpoint path preflight failed: {errors[0]}")
        if not all(item.checkpoint_exists for item in ordered):
            missing_ranks = [
                item.rank for item in ordered if not item.checkpoint_exists
            ]
            raise FileNotFoundError(
                f"Checkpoint path {path} does not exist on ranks {missing_ranks}."
            )
        if not all(item.runtime_eligible and item.hostname for item in ordered):
            return None
        if len({item.checkpoint_path for item in ordered}) != 1:
            return None
        candidate_keys = _cooperative_candidate_keys(ordered)
        if not candidate_keys:
            return None
        return _CooperativeReadContext(
            candidate_keys=candidate_keys,
            nodes=_node_memberships(ordered),
            advertise_host=hostname,
            bind_host=_wildcard_bind_host(hostname),
        )

    def _load_collective_source_metadata(
        self,
        path: str,
        context: _CooperativeReadContext,
    ) -> tuple[DistributedMetadata | None, _CooperativeReadContext | None]:
        metadata: DistributedMetadata | None = None
        error_message: str | None = None
        local_error: Exception | None = None
        available = False
        layouts_supported = False
        contract_digest = ""
        try:
            metadata = self._load_metadata(path)
            if metadata is not None:
                available, layouts_supported, contract_digest = (
                    _source_metadata_contract(
                        Path(path),
                        metadata,
                        context.candidate_keys,
                    )
                )
        except Exception as error:
            local_error = error
            error_message = f"{type(error).__name__}: {error}"
        manifest = _SourceMetadataManifest(
            available=available,
            layouts_supported=layouts_supported,
            contract_digest=contract_digest,
            error=error_message,
        )
        gathered = _all_gather_objects(manifest, dist.get_world_size())
        manifests = tuple(
            item for item in gathered if isinstance(item, _SourceMetadataManifest)
        )
        if len(manifests) != dist.get_world_size():
            raise RuntimeError("cooperative preflight received invalid source metadata")
        errors = tuple(item.error for item in manifests if item.error is not None)
        if errors:
            failure = RuntimeError(f"checkpoint metadata preflight failed: {errors[0]}")
            if local_error is not None:
                raise failure from local_error
            raise failure
        if not all(item.available and item.layouts_supported for item in manifests):
            return metadata, None
        if len({item.contract_digest for item in manifests}) != 1:
            return metadata, None
        return metadata, context

    def _prepare_cooperative_schedule(
        self,
        context: _CooperativeReadContext,
        items_needing_reshard: Mapping[str, CheckpointItem],
        checkpoint_metadata: CheckpointMetadata | None,
        source_metadata: DistributedMetadata,
    ) -> _CooperativeSchedule | None:
        local_error: Exception | None = None
        try:
            prepared_by_key, statuses = _prepare_local_cooperative_items(
                context.candidate_keys,
                items_needing_reshard,
                checkpoint_metadata,
                source_metadata,
            )
        except Exception as error:
            prepared_by_key = {}
            statuses = ()
            local_error = error
        manifest = _LocalScheduleManifest(
            statuses=statuses,
            error=(
                None
                if local_error is None
                else f"{type(local_error).__name__}: {local_error}"
            ),
        )
        gathered = _all_gather_objects(manifest, dist.get_world_size())
        manifests = tuple(
            item for item in gathered if isinstance(item, _LocalScheduleManifest)
        )
        if len(manifests) != dist.get_world_size():
            raise RuntimeError(
                "cooperative preflight received an invalid load schedule"
            )
        errors = tuple(item.error for item in manifests if item.error is not None)
        if errors:
            failure = RuntimeError(f"cooperative load planning failed: {errors[0]}")
            if local_error is not None:
                raise failure from local_error
            raise failure
        keys = _resolve_cooperative_schedule_keys(
            context.candidate_keys,
            manifests,
        )
        if not keys:
            return None
        return _CooperativeSchedule(keys=keys, prepared_by_key=prepared_by_key)

    def _load_cooperative_items(
        self,
        path: str,
        context: _CooperativeReadContext | None,
        items_needing_reshard: dict[str, CheckpointItem],
        checkpoint_metadata: CheckpointMetadata | None,
        source_metadata: DistributedMetadata | None,
    ) -> dict[str, Any]:
        if context is None or source_metadata is None:
            return {}
        schedule = self._prepare_cooperative_schedule(
            context,
            items_needing_reshard,
            checkpoint_metadata,
            source_metadata,
        )
        if schedule is None:
            return {}
        if not self._execute_cooperative_schedule(
            path,
            context,
            schedule,
            source_metadata,
        ):
            return {}
        result: dict[str, Any] = {}
        for key in schedule.keys:
            item = items_needing_reshard.pop(key, None)
            if item is not None:
                result[key] = item.value
        return result

    def _execute_cooperative_schedule(
        self,
        path: str,
        context: _CooperativeReadContext,
        schedule: _CooperativeSchedule,
        source_metadata: DistributedMetadata,
    ) -> bool:
        nonce_values: list[object] = [
            secrets.token_hex(16) if dist.get_rank() == 0 else None
        ]
        dist.broadcast_object_list(nonce_values, src=0)
        nonce = nonce_values[0]
        if not isinstance(nonce, str) or not nonce:
            raise RuntimeError("cooperative load received an invalid session nonce")
        topology = RankTopology(
            global_rank=dist.get_rank(),
            nodes=context.nodes,
            coordination_world_count=1,
            job_id=f"checkpoint-read-{nonce}",
        )
        rendezvous = C10dStoreRendezvous(dist.distributed_c10d._get_default_store())
        completed_any = False
        for index, key in enumerate(schedule.keys):
            item_metadata = source_metadata.metadata[key]

            def source_path_for_rank(
                source_rank: int,
                *,
                metadata: DistributedItemMetadata = item_metadata,
                item_key: str = key,
            ) -> Path:
                return metadata.get_file_path(source_rank, Path(path), item_key)

            prepared = schedule.prepared_by_key.get(key)
            metric_callback = (
                _cooperative_metric_logger(key, topology.global_rank)
                if _should_emit_cooperative_metrics(topology)
                else None
            )
            request = CooperativeLoadRequest(
                topology=topology,
                rendezvous=rendezvous,
                session_token=f"{nonce}-{index}",
                storage=self._storage,
                source_path_for_rank=source_path_for_rank,
                target_state_dict=(
                    {} if prepared is None else prepared.target_state_dict
                ),
                local_load_plan=({} if prepared is None else prepared.local_load_plan),
                metadata_provider=_TorchArchiveMetadataProvider(
                    key,
                    metric_callback,
                ),
                metric_callback=metric_callback,
                bind_host=context.bind_host,
                advertise_host=context.advertise_host,
            )
            try:
                load_result = load_cooperatively(
                    request,
                    _pool_cache=_CHECKPOINT_READER_POOL_CACHE,
                )
            except CooperativeLoadUnsupported as error:
                if completed_any:
                    raise CooperativeLoadFailure(
                        "cooperative eligibility changed after target writes",
                        target_writes_started=True,
                    ) from error
                return False
            if topology.is_world_leader:
                logger.info(
                    "Cooperative resharding completed item=%r targets=%d "
                    "batches=%d unique_source_bytes=%d storage_bytes_read=%d "
                    "network_bytes_received=%d elapsed_seconds=%.6f "
                    "slowest_rank_seconds=%.6f",
                    key,
                    load_result.target_count,
                    load_result.batch_count,
                    load_result.unique_source_bytes,
                    load_result.storage_bytes_read,
                    load_result.network_bytes_received,
                    load_result.elapsed_seconds,
                    load_result.slowest_rank_seconds,
                )
            completed_any = True
        return True

    def read(
        self,
        path: str,
        checkpoint_info: CheckpointReadInfo,
        map_location: Any = None,
    ) -> tuple[STATE_DICT, list[str]]:
        """
        Reads a state dictionary from storage.

        Only keys defined in checkpoint_info will be loaded. Each file is loaded in full.

        File names are discovered by looking at the layout_info_mappings in checkpoint_info.

        In-place modification behavior:
            When checkpoint_info contains values (not None), loaded data is merged with
            those values. The following are modified IN-PLACE:
            - Mutable containers (dict, list, deque): updated in the existing objects
            - Tensors: data is copied via copy_() into the target tensors, preserving
              the target tensor's identity (same object, updated data)

            The following are NOT modified in-place:
            - Immutable containers (tuple): new containers are created
            - Non-tensor leaf values: source value replaces target value

            When checkpoint_info values are None, new objects are created from the
            loaded checkpoint data.

        Args:
            path (str): The path from which to read the checkpoint.
            checkpoint_info (CheckpointReadInfo): Encapsulates state_dict, layout_info_mappings,
                and optional checkpoint_metadata for resharding.
                Each item in checkpoint_info.checkpoint_items may have a resharder for resharding.
                checkpoint_metadata is used for resharding.
            map_location (Any): Device mapping function or device name for relocating tensors.

        Returns:
            STATE_DICT: The loaded state dictionary.
            list[str]: List of missing keys.
        """
        event_logger = EventLogger()
        logger.debug(
            f"Reading checkpoint from {path} for rank {self._rank_info.global_rank}"
        )

        if _cooperative_resharding_enabled():
            cooperative_context = self._collective_cooperative_preflight(
                path,
                checkpoint_info,
            )
        else:
            cooperative_context = None
            if not self._storage.exists(Path(path)):
                raise FileNotFoundError(f"Checkpoint path {path} does not exist.")

        # Check if any items have resharders configured
        has_any_resharder = any(
            item.resharder is not None
            for item in checkpoint_info.checkpoint_items.values()
        )

        # Check if all resharders have skip_resharding=True
        all_resharders_skip = all(
            item.resharder.skip_resharding
            for item in checkpoint_info.checkpoint_items.values()
            if item.resharder is not None
        )

        # Fast path: if no items have resharders OR all resharders have skip_resharding=True,
        # skip metadata loading entirely and use direct file reads
        if (
            not has_any_resharder or all_resharders_skip
        ) and cooperative_context is None:
            if all_resharders_skip:
                logger.info(
                    "Resharder skip_resharding=True: skipping metadata loading and using direct file reads"
                )
            else:
                logger.info(
                    "No resharders configured: skipping metadata loading and using direct file reads"
                )
            # _read_without_resharding loads full files and filters to requested keys
            result, missing_paths = self._read_without_resharding(
                path,
                checkpoint_info,
                map_location=map_location,
            )
            missing_keys = [str(checkpoint_path) for checkpoint_path in missing_paths]
            logger.info(
                f"Successfully read checkpoint file from {path}",
                extra=event_logger(EventType.LOG_METRIC, end_to_end=True),
            )
            return result, missing_keys

        # Normal path with resharding support
        checkpoint_metadata = checkpoint_info.checkpoint_metadata
        if cooperative_context is None:
            source_distributed_metadata = self._load_metadata(path)
        else:
            source_distributed_metadata, cooperative_context = (
                self._load_collective_source_metadata(path, cooperative_context)
            )
        logger.info(
            "Finished reading checkpoint metadata",
            extra=event_logger(
                EventType.LOG_METRIC,
                metric_name="train.checkpoint_read.execute.filesystem.metadata.read.latency_ms",
            ),
        )

        # If resharding is needed, use metadata to determine source ranks for loading.
        # This mapping uses source ranks as keys because the world size or device mesh
        # may differ between when the checkpoint was saved and when it is loaded.
        src_to_layout_info_mappings = (
            _build_src_to_layout_info_mappings(source_distributed_metadata)
            if source_distributed_metadata
            else None
        )

        items_needing_reshard, items_not_needing_reshard = _partition_items_for_read(
            checkpoint_info,
            checkpoint_metadata,
            source_distributed_metadata,
        )
        result_dict = self._load_cooperative_items(
            path,
            cooperative_context,
            items_needing_reshard,
            checkpoint_metadata,
            source_distributed_metadata,
        )
        missing_paths: list[CheckpointPath] = []

        # Load items that don't need resharding
        if items_not_needing_reshard:
            checkpoint_info_no_reshard = CheckpointInfo(
                checkpoint_items=items_not_needing_reshard
            )
            non_reshard_result, non_reshard_missing = self._read_without_resharding(
                path,
                checkpoint_info_no_reshard,
                map_location=map_location,
            )
            result_dict.update(non_reshard_result)
            missing_paths.extend(non_reshard_missing)

        # Load items that need resharding
        if items_needing_reshard:
            assert checkpoint_metadata is not None
            assert source_distributed_metadata is not None
            assert src_to_layout_info_mappings is not None
            checkpoint_info_reshard = CheckpointInfo(
                checkpoint_items=items_needing_reshard
            )
            reshard_result, reshard_missing = self._read_with_resharding(
                path,
                checkpoint_info_reshard,
                checkpoint_metadata,
                src_to_layout_info_mappings,
                source_distributed_metadata,
                map_location=map_location,
            )
            result_dict.update(reshard_result)
            missing_paths.extend(reshard_missing)

        missing_keys = [str(checkpoint_path) for checkpoint_path in missing_paths]
        if missing_keys:
            if len(missing_keys) > 10:
                logger.warning(
                    f"Missing {len(missing_keys)} keys from checkpoint: {missing_keys[:10]}... (and {len(missing_keys) - 10} more)"
                )
            else:
                logger.warning(
                    f"Missing {len(missing_keys)} keys from checkpoint: {missing_keys}"
                )
        logger.info(
            f"Successfully read checkpoint file from {path}",
            extra=event_logger(EventType.LOG_METRIC),
        )
        return result_dict, missing_keys

    def _load_metadata(
        self,
        checkpoint_dir: str | Path,
    ) -> DistributedMetadata | None:
        """
        Load distributed metadata from the checkpoint directory.

        Args:
            checkpoint_dir: Path to the checkpoint directory.

        Returns:
            DistributedMetadata if METADATA_FILE_NAME exists, None otherwise.
        """
        metadata_path = Path(checkpoint_dir) / METADATA_FILE_NAME
        metadata = None
        if self._storage.exists(metadata_path):
            data = self._storage.read(metadata_path)
            metadata_dict = pickle.loads(data)
            metadata = DistributedMetadata.from_dict(metadata_dict)
        return metadata

    def _read_without_resharding(
        self,
        path: str,
        checkpoint_info: CheckpointInfo,
        *,
        map_location: Any = None,
    ) -> tuple[STATE_DICT, list[CheckpointPath]]:
        """
        Load checkpoint without resharding support.

        This method loads checkpoint data using the standard layout without any
        resharding logic. It reads from the local rank's checkpoint files only.
        Each file is loaded in full.

        Args:
            path: Path to the checkpoint directory.
            checkpoint_info: Encapsulates state_dict and layout_info_mappings.
            map_location: Device mapping for tensor relocation.

        Returns:
            Tuple of (loaded_state_dict, missing_paths).
        """
        event_logger = EventLogger()
        logger.debug(
            f"Reading checkpoint from {path} for rank {self._rank_info.global_rank}"
        )

        result_dict: dict[str, Any] = {}
        missing_paths: list[CheckpointPath] = []

        for key in checkpoint_info.keys:
            if key not in checkpoint_info.layout_info_mappings:
                logger.warning(
                    f"Item {key=} not found in layout_info_mappings. Skipping."
                )

                # Add all leaf keys to missing_keys
                def collect_all_paths(
                    checkpoint_path: CheckpointPath, src: Any, tgt: Any
                ) -> Any:
                    missing_paths.append(checkpoint_path)
                    return src

                walk_checkpoint_structure(
                    item_key=key,
                    source=checkpoint_info.checkpoint_items[key].value,
                    target=None,
                    leaf_fn=collect_all_paths,
                )
                continue

            layout_info = checkpoint_info.layout_info_mappings[key]
            if layout_info is None:
                layout_info = default_layout_info(key, self._rank_info.global_rank)

            file_path = Path(path) / layout_info.file_path
            if not self._storage.exists(file_path):
                raise RuntimeError(f"Missing file {file_path} for key {key}.")

            loaded_data = self._load_full_file(
                file_path, layout_info, map_location=map_location
            )
            # Filter loaded data to only include keys present in the requested structure
            # Also track any missing keys within the nested structure
            requested_value = checkpoint_info.checkpoint_items[key].value
            # safetensors only supports flat dict[str, Tensor], so the writer flattens
            # nested inputs with '.' separators. Re-nest here to match the target's shape
            # — otherwise walk_checkpoint_structure (which descends source + target in
            # parallel) would misalign and silently drop everything.
            if (
                isinstance(layout_info.serialization_format, SafetensorsSerialization)
                and requested_value is not None
                and isinstance(loaded_data, dict)
            ):
                loaded_data = SafetensorsSerialization.unflatten_to_target(
                    loaded_data, requested_value
                )
            result_dict[key], item_missing_paths = walk_checkpoint_structure(
                item_key=key,
                source=loaded_data,
                target=requested_value,
            )
            missing_paths.extend(item_missing_paths)
            logger.info(
                f"Done Loading {key} checkpoint from {file_path} without resharder.",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"train.checkpoint_read.execute.filesystem.{key}.read.latency_ms",
                ),
            )
        logger.info(
            f"Successfully read checkpoint file from {path} without resharding",
            extra=event_logger(EventType.LOG_METRIC, end_to_end=True),
        )
        return result_dict, missing_paths

    def _read_with_resharding(
        self,
        path: str,
        checkpoint_info: CheckpointInfo,
        checkpoint_metadata: CheckpointMetadata,
        src_to_layout_info_mappings: dict[int, dict[str, LayoutInfo | None]],
        distributed_metadata: DistributedMetadata,
        *,
        map_location: Any = None,
    ) -> tuple[STATE_DICT, list[CheckpointPath]]:
        """
        Load checkpoint with resharding support.

        This method handles loading checkpoint data when the distributed configuration
        differs between save time and load time. It uses resharders to generate load
        plans that determine which source ranks to read from and how to redistribute
        the data to match the current distributed layout.

        All items in checkpoint_info are expected to need resharding with valid resharders.

        Args:
            path: Path to the checkpoint directory.
            checkpoint_info: Encapsulates state_dict and layout_info_mappings.
                All items must have non-None resharders.
            checkpoint_metadata: Metadata for the target checkpoint, containing
                local metadata for generating load plans.
            src_to_layout_info_mappings: Mapping from source ranks to their layout info,
                used to locate checkpoint files from different source ranks.
            distributed_metadata: Source distributed metadata from the saved checkpoint,
                used for resharding decisions.
            map_location: Device mapping for tensor relocation.

        Returns:
            Tuple of (loaded_state_dict, unhandled_paths) where unhandled_paths contains
            CheckpointPaths for keys that could not be resharded.
        """
        event_logger = EventLogger()
        logger.debug(
            f"Reading checkpoint from {path} for rank {self._rank_info.global_rank}"
        )

        result_dict: dict[str, Any] = checkpoint_info.state_dict  # type: ignore
        unhandled_paths: list[CheckpointPath] = []

        for key, item in checkpoint_info.checkpoint_items.items():
            logger.info(
                f"Loading {key} checkpoint with resharder.",
                extra=event_logger(EventType.LOG_METRIC),
            )
            resharder = item.resharder
            assert resharder is not None  # API contract guarantees this

            # Get target metadata for this item (direct access by item_key)
            target_metadata: dict[NestedPath, ShardingMetadata] | None = (
                checkpoint_metadata.local_metadata.get(key)
            )

            # Get item-level source metadata (direct access by item_key)
            source_item_metadata = distributed_metadata.metadata.get(key)

            if not target_metadata or source_item_metadata is None:
                logger.warning(f"Missing metadata for item {key}, skipping resharding")
                continue

            # Load and reshard checkpoint data using per-path API
            unhandled_nested_paths = resharder.load(
                source_path=Path(path),
                item_key=key,
                target_metadata=target_metadata,
                source_metadata=source_item_metadata,
                target=result_dict[key],
                storage=self._storage,
            )

            # Convert NestedPath to CheckpointPath for reporting
            for nested_path in unhandled_nested_paths:
                unhandled_paths.append(CheckpointPath(key, nested_path))

            logger.info(
                f"Done Loading {key} checkpoint with resharder.",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"train.checkpoint_read.execute.filesystem.{key}.read.latency_ms",
                ),
            )

        return result_dict, unhandled_paths

    def _load_full_file(
        self,
        file_path: Path,
        layout_info: Any,
        *,
        map_location: Any = None,
    ) -> Any:
        """
        Load an entire file based on its serialization format.

        Supports TorchSerialization, JsonSerialization, and RawSerialization formats.

        Args:
            file_path: Path to the file to load.
            layout_info: LayoutInfo containing serialization format information.
            map_location: Device mapping for tensor relocation (torch files only).

        Returns:
            The deserialized content of the file.

        Raises:
            ValueError: If the serialization format is not supported.
        """
        if isinstance(layout_info.serialization_format, TorchSerialization):
            if not self._disable_use_mmap_backed_storage_on_load:
                from ...storage.torch_serialization import (
                    load_torch_serialized_from_storage,
                )

                return load_torch_serialized_from_storage(
                    file_path,
                    self._storage,
                    map_location=map_location,
                    mmap_fill=(
                        self._mmap_fill_factory(self._storage)
                        if self._mmap_fill_factory is not None
                        else None
                    ),
                )

            with self._storage.stream_read(file_path) as f:
                state_dict = torch.load(
                    f,  # type: ignore[arg-type]
                    map_location=map_location,
                    weights_only=False,
                )
            return state_dict
        elif isinstance(layout_info.serialization_format, JsonSerialization):
            data = self._storage.read(file_path)
            json_data = json.loads(data.decode("utf-8"))
            if layout_info.serialization_format.cls is None:
                return json_data
            else:
                return from_dict(layout_info.serialization_format.cls, json_data)
        elif isinstance(layout_info.serialization_format, RawSerialization):
            return self._storage.read(file_path)
        elif isinstance(layout_info.serialization_format, SafetensorsSerialization):
            from safetensors.torch import load as safetensors_load

            data = self._storage.read(file_path)
            loaded = safetensors_load(data)
            if map_location is not None:
                # safetensors has no callable/dict remapper concept like torch.load —
                # only a concrete destination device is meaningful here. Reject other
                # forms loudly so callers don't get a confusing `Tensor.to(<function ...>)`.
                if not isinstance(map_location, (str, torch.device)):
                    raise ValueError(
                        f"SafetensorsSerialization map_location must be a str or "
                        f"torch.device, got {type(map_location).__name__}. Callables "
                        f"and dict remappings (accepted by torch.load) are not supported."
                    )
                loaded = {k: v.to(map_location) for k, v in loaded.items()}
            return loaded
        else:
            raise ValueError(
                f"Unsupported serialization format: {layout_info.serialization_format}"
            )
