# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Application-neutral orchestration for cooperative checkpoint loading."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import threading
import time
import zlib
from bisect import bisect_right
from collections import Counter, defaultdict, OrderedDict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import (
    as_completed,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Protocol, TypeAlias
from urllib.parse import urlsplit

import torch

from ...resharding import LoadPlan
from ...storage.base_storage import ReadArgs, Storage
from .config import CooperativeLoadConfig, CooperativeLoadResult
from .layout import (
    _resolve_torch_dtype,
    resolve_tensor_read_targets,
    SourceTensorMetadata,
    TensorReadTarget,
)
from .metadata import (
    _build_partitioned_source_tensor_metadata_wire,
    _decode_trusted_source_tensor_metadata_wire,
    _materialize_trusted_source_tensor_metadata_wire,
    _merge_partitioned_source_tensor_metadata_wire,
    _select_trusted_source_tensor_metadata_wire,
)
from .planning import (
    _execution_plan_to_wire_from_validated,
    _merge_canonical_fqn_demand_wire_payloads,
    _merge_fqn_demand_wire_payloads_to_canonical_wire,
    _plan_cooperative_resharding_from_merged_demands,
    _rebuild_batch_node_works_from_canonical,
    BatchNodeWork,
    ByteRange as SourceByteRange,
    FqnDemand,
    NodeId,
    project_execution_plan_wire,
    ProjectedBatchDownload,
    ProjectedExecutionPlan,
    ProjectedSourceSchedule,
    RankTopology,
)
from .rendezvous import Rendezvous, RendezvousNamespace
from .scatter import (
    can_receive_directly_to_cpu,
    direct_cpu_destination_buffer,
    PinnedBufferPool,
    scatter_buffer_slice,
    scatter_flat_buffer_chunk,
)
from .shared_memory import (
    ChunkPool,
    ChunkReservation,
    RangeSpec,
    readinto_exact,
    recommended_capacity_bytes,
    RegisteredRange,
    SegmentSlice,
)
from .transport import (
    NodeClient,
    NodeServer,
    RangeNotReadyError,
    RangeRequest,
    TransportError,
)

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = 6
_BYTE_DEMAND_PAYLOAD_VERSION = 2
_CONTROL_BYTES_LIMIT = 256 * 1024 * 1024
_DECOMPRESSED_CONTROL_BYTES_LIMIT = _CONTROL_BYTES_LIMIT
_METADATA_CONTROL_COMPRESSION_LEVEL = 1
_ERROR_MESSAGE_BYTES_LIMIT = 16 * 1024
_MAX_FETCH_RANGES = 4096
_MAX_DATA_CLIENTS_PER_THREAD = 4
_ERROR_MONITOR_BASE_INTERVAL_SECONDS = 0.5
_ERROR_MONITOR_JITTER_SECONDS = 0.5
_ERROR_MONITOR_MAX_REQUEST_SECONDS = 5.0
_PRIVATE_POOL_CACHE_MAX_BYTES = 512 * 1024**3

MetricValue: TypeAlias = bool | int | float | str
MetricCallback: TypeAlias = Callable[[str, Mapping[str, MetricValue]], None]
VisibilityProbe: TypeAlias = Callable[[Path, bytes], bool]


class MetadataProvider(Protocol):
    """Loads the demanded physical tensor metadata for a subset of archives."""

    def load_metadata(
        self,
        demands_by_rank: Mapping[int, Collection[str]],
        *,
        storage: Storage,
        source_path_for_rank: Callable[[int], Path],
        max_workers: int,
        timeout_seconds: float,
    ) -> Mapping[int, Mapping[str, SourceTensorMetadata]]: ...


@dataclass(frozen=True, slots=True)
class CooperativeLoadRequest:
    """All application-supplied adapters and local state for one load."""

    topology: RankTopology
    rendezvous: Rendezvous
    session_token: str
    storage: Storage
    source_path_for_rank: Callable[[int], Path]
    target_state_dict: Mapping[str, Any]
    local_load_plan: Mapping[str, Sequence[LoadPlan]] = field(default_factory=dict)
    local_targets: Sequence[TensorReadTarget] | None = None
    metadata_provider: MetadataProvider | None = None
    metric_callback: MetricCallback | None = None
    shared_memory_visibility_probe: VisibilityProbe | None = None
    bind_host: str = "127.0.0.1"
    advertise_host: str | None = None
    shared_memory_directory: str | Path = "/dev/shm"
    shared_memory_capacity_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.session_token:
            raise ValueError("session_token must not be empty")
        if not self.bind_host:
            raise ValueError("bind_host must not be empty")
        if self.advertise_host is not None:
            _format_url_host(self.advertise_host)
            if _is_wildcard_host(self.advertise_host):
                raise ValueError("advertise_host must not be a wildcard address")
        if len(self.topology.coordination_world.node_ids) > 1 and (
            self.advertise_host is None
        ):
            raise ValueError("advertise_host is required for multi-node loading")
        if self.advertise_host is None and _is_wildcard_host(self.bind_host):
            raise ValueError("advertise_host is required with a wildcard bind_host")
        if self.shared_memory_capacity_bytes is not None and (
            self.shared_memory_capacity_bytes <= 0
        ):
            raise ValueError("shared_memory_capacity_bytes must be positive")
        if (
            self.local_targets is None
            and self.local_load_plan
            and self.metadata_provider is None
        ):
            raise ValueError(
                "metadata_provider is required when local targets are not supplied"
            )


class CooperativeLoadFailure(RuntimeError):
    """A terminal cooperative-load failure with target-write state."""

    def __init__(self, message: str, *, target_writes_started: bool) -> None:
        super().__init__(message)
        self.target_writes_started = target_writes_started


class CooperativeLoadUnsupported(RuntimeError):
    """A collective preflight veto for which fallback remains safe."""


class _ChunkPoolCacheBusy(CooperativeLoadUnsupported):
    pass


@dataclass(frozen=True, slots=True)
class _ChunkPoolSpec:
    directory: str
    capacity_bytes: int
    chunk_bytes: int


@dataclass(slots=True)
class _ChunkPoolLease:
    cache: _ChunkPoolCache
    token: object
    spec: _ChunkPoolSpec
    pool: ChunkPool
    retainable: bool
    reused: bool

    def release(self) -> None:
        self.cache._release(self)

    def discard(self) -> None:
        self.cache._discard(self)


class _ChunkPoolCache:
    """One-entry exclusive cache for reusable shared-memory backing chunks."""

    def __init__(
        self,
        *,
        max_retained_capacity_bytes: int = _PRIVATE_POOL_CACHE_MAX_BYTES,
    ) -> None:
        if max_retained_capacity_bytes <= 0:
            raise ValueError("max_retained_capacity_bytes must be positive")
        self._max_retained_capacity_bytes = max_retained_capacity_bytes
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._idle_spec: _ChunkPoolSpec | None = None
        self._idle_pool: ChunkPool | None = None
        self._active_token: object | None = None
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=self._reset_after_fork)

    @property
    def max_retained_capacity_bytes(self) -> int:
        return self._max_retained_capacity_bytes

    def try_acquire(
        self,
        spec: _ChunkPoolSpec,
        factory: Callable[[], ChunkPool],
        *,
        cleanup_timeout: float,
    ) -> _ChunkPoolLease | None:
        retainable = self.can_retain(spec)
        with self._lock:
            self._reset_for_pid_locked()
            if self._active_token is not None:
                return None
            token = object()
            self._active_token = token
            idle_spec, self._idle_spec = self._idle_spec, None
            idle_pool, self._idle_pool = self._idle_pool, None
        reused = retainable and idle_pool is not None and idle_spec == spec
        try:
            if idle_pool is not None and not reused:
                idle_pool.cleanup(timeout=cleanup_timeout)
                idle_pool = None
            if idle_pool is not None:
                try:
                    idle_pool.index.assert_quiescent()
                    if not idle_pool.reuse_supported:
                        raise RuntimeError("cached chunk pool no longer supports reuse")
                except Exception:
                    idle_pool.cleanup(timeout=cleanup_timeout)
                    idle_pool = None
                    reused = False
            pool = idle_pool if idle_pool is not None else factory()
        except BaseException:
            with self._lock:
                if self._active_token is token:
                    self._active_token = None
            raise
        return _ChunkPoolLease(self, token, spec, pool, retainable, reused)

    def can_retain(self, spec: _ChunkPoolSpec) -> bool:
        return spec.capacity_bytes <= self._max_retained_capacity_bytes

    def close(self, timeout: float | None = None) -> None:
        with self._lock:
            self._reset_for_pid_locked()
            if self._active_token is not None:
                return
            pool, self._idle_pool = self._idle_pool, None
            self._idle_spec = None
        if pool is not None:
            pool.cleanup(timeout=timeout)

    def _release(self, lease: _ChunkPoolLease) -> None:
        with self._lock:
            self._reset_for_pid_locked()
            self._assert_active_lease_locked(lease)
            if not lease.retainable:
                raise RuntimeError("one-shot chunk-pool lease cannot be retained")
            if self._idle_pool is not None:
                raise RuntimeError("chunk-pool cache already contains an idle pool")
            self._idle_spec = lease.spec
            self._idle_pool = lease.pool
            self._active_token = None

    def _discard(self, lease: _ChunkPoolLease) -> None:
        with self._lock:
            self._reset_for_pid_locked()
            self._assert_active_lease_locked(lease)
            self._active_token = None

    def _assert_active_lease_locked(self, lease: _ChunkPoolLease) -> None:
        if lease.cache is not self or self._active_token is not lease.token:
            raise RuntimeError("chunk-pool cache lease is no longer active")

    def _reset_after_fork(self) -> None:
        self._owner_pid = os.getpid()
        self._lock = threading.Lock()
        self._idle_spec = None
        self._idle_pool = None
        self._active_token = None

    def _reset_for_pid_locked(self) -> None:
        if os.getpid() == self._owner_pid:
            return
        self._owner_pid = os.getpid()
        self._idle_spec = None
        self._idle_pool = None
        self._active_token = None


@dataclass(frozen=True, slots=True)
class _ScatterWorkItem:
    owner: NodeId
    source_rank: int
    targets: tuple[TensorReadTarget, ...]
    dense_nbytes: int
    range_count: int
    uses_grouped_fetch: bool
    first_target_index: int
    predicted_download_frontier_bytes: int


@dataclass(frozen=True, slots=True)
class _ScatterBatchPlan:
    work_items: tuple[_ScatterWorkItem, ...]
    target_count: int
    source_range_count: int
    coalesced_group_count: int
    coalesced_target_count: int
    buffered_fetch_count: int
    fallback_target_count: int
    max_group_bytes: int
    max_group_ranges: int
    max_group_targets: int
    max_predicted_download_frontier_bytes: int
    source_group_count: int


@dataclass(slots=True)
class _OpenScatterGroup:
    indexed_targets: list[tuple[int, TensorReadTarget]]
    dense_nbytes: int = 0
    range_count: int = 0
    predicted_download_frontier_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _ScatterTargetCandidate:
    target_index: int
    target: TensorReadTarget
    owner: NodeId
    predicted_download_frontier_bytes: int


@dataclass(frozen=True, slots=True)
class _GlobalByteDemandPlan:
    demands: tuple[FqnDemand, ...]
    source_consumer_bytes_by_node: Mapping[NodeId, Mapping[int, int]]


class _Metrics:
    def __init__(self, callback: MetricCallback | None, rank: int) -> None:
        self._callback = callback
        self._rank = rank
        self._retry_lock = threading.Lock()
        self._retry_operation_count = 0
        self._retry_count = 0
        self._retry_local_operation_count = 0
        self._retry_remote_operation_count = 0
        self._retry_total_ns = 0
        self._retry_max_ns = 0

    def emit(self, event: str, **fields: MetricValue) -> None:
        callback = self._callback
        if callback is None:
            return
        payload: dict[str, MetricValue] = {"rank": self._rank, **fields}
        try:
            callback(event, payload)
        except Exception:
            logger.warning("cooperative metric callback failed", exc_info=True)

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        started_ns = time.perf_counter_ns()
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            self.emit(
                "phase",
                name=name,
                latency_ms=elapsed_ns / 1_000_000,
                elapsed_seconds=elapsed_ns / 1_000_000_000,
                succeeded=succeeded,
            )

    @contextmanager
    def latency(
        self,
        name: str,
        **fields: MetricValue,
    ) -> Iterator[dict[str, MetricValue]]:
        started_ns = time.perf_counter_ns()
        succeeded = False
        mutable_fields = dict(fields)
        try:
            yield mutable_fields
            succeeded = True
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            self.emit(
                "latency_ms",
                name=name,
                latency_ms=elapsed_ns / 1_000_000,
                elapsed_seconds=elapsed_ns / 1_000_000_000,
                succeeded=succeeded,
                **mutable_fields,
            )

    def record_node_capacities(self, capacities: Sequence[int]) -> None:
        self.emit(
            "node_capacities",
            node_count=len(capacities),
            min_capacity_bytes=min(capacities, default=0),
            max_capacity_bytes=max(capacities, default=0),
        )

    def record_latency(
        self,
        name: str,
        elapsed_ns: int,
        **fields: MetricValue,
    ) -> None:
        self.emit(
            "latency_ms",
            name=name,
            latency_ms=elapsed_ns / 1_000_000,
            elapsed_seconds=elapsed_ns / 1_000_000_000,
            succeeded=True,
            **fields,
        )

    def record_batch_phase(self, name: str, batch_index: int, started_ns: int) -> None:
        elapsed_ns = time.perf_counter_ns() - started_ns
        self.emit(
            "batch_phase",
            name=name,
            batch_index=batch_index,
            latency_ms=elapsed_ns / 1_000_000,
            elapsed_seconds=elapsed_ns / 1_000_000_000,
        )

    def record_scatter_plan(self, batch_index: int, plan: _ScatterBatchPlan) -> None:
        self.emit(
            "scatter_plan",
            batch_index=batch_index,
            target_count=plan.target_count,
            work_item_count=len(plan.work_items),
            buffered_fetch_count=plan.buffered_fetch_count,
            source_range_count=plan.source_range_count,
            coalesced_group_count=plan.coalesced_group_count,
            coalesced_target_count=plan.coalesced_target_count,
            fallback_target_count=plan.fallback_target_count,
            max_group_bytes=plan.max_group_bytes,
            max_group_ranges=plan.max_group_ranges,
            max_group_targets=plan.max_group_targets,
            max_predicted_download_frontier_bytes=(
                plan.max_predicted_download_frontier_bytes
            ),
            source_group_count=plan.source_group_count,
        )

    def record_download_registration(
        self,
        *,
        batch_index: int,
        byte_count: int,
        elapsed_ns: int,
        range_count: int,
        segment_count: int,
        max_segment_bytes: int,
    ) -> None:
        self.emit(
            "download_registration",
            batch_index=batch_index,
            byte_count=byte_count,
            latency_ms=elapsed_ns / 1_000_000,
            elapsed_seconds=elapsed_ns / 1_000_000_000,
            range_count=range_count,
            segment_count=segment_count,
            max_segment_bytes=max_segment_bytes,
        )

    def record_readiness_retry_window(
        self, *, elapsed_ns: int, local: bool, retry_count: int
    ) -> None:
        if not retry_count:
            return
        with self._retry_lock:
            self._retry_operation_count += 1
            self._retry_count += retry_count
            if local:
                self._retry_local_operation_count += 1
            else:
                self._retry_remote_operation_count += 1
            self._retry_total_ns += elapsed_ns
            self._retry_max_ns = max(self._retry_max_ns, elapsed_ns)

    def record_download_batch(
        self, batch_index: int, started_ns: int, *, completed: bool
    ) -> None:
        elapsed_ns = time.perf_counter_ns() - started_ns
        self.emit(
            "download_batch",
            batch_index=batch_index,
            completed=completed,
            latency_ms=elapsed_ns / 1_000_000,
            elapsed_seconds=elapsed_ns / 1_000_000_000,
        )

    def record_download_executor_creation(self, worker_count: int) -> None:
        self.emit("download_executor", worker_count=worker_count)

    def record_stream_read(
        self,
        *,
        active_worker_count: int,
        active_worker_peak: int,
        batch_index: int,
        source_rank: int,
        queue_ns: int,
        open_ns: int,
        read_ns: int,
        close_ns: int,
        range_count: int,
        byte_count: int,
        opened: bool,
    ) -> None:
        worker_ns = open_ns + read_ns + close_ns
        self.emit(
            "stream_read",
            batch_index=batch_index,
            source_rank=source_rank,
            opened=opened,
            range_count=range_count,
            byte_count=byte_count,
            active_worker_count=active_worker_count,
            active_worker_peak=active_worker_peak,
            queue_latency_ms=queue_ns / 1_000_000,
            latency_ms=worker_ns / 1_000_000,
            open_latency_ms=open_ns / 1_000_000,
            read_latency_ms=read_ns / 1_000_000,
            close_latency_ms=close_ns / 1_000_000,
            open_seconds=open_ns / 1_000_000_000,
            read_seconds=read_ns / 1_000_000_000,
            close_seconds=close_ns / 1_000_000_000,
        )

    def emit_batch_summaries(self, *, succeeded: bool) -> None:
        with self._retry_lock:
            fields = {
                "readiness_retry_operation_count": self._retry_operation_count,
                "readiness_retry_count": self._retry_count,
                "readiness_retry_local_operation_count": (
                    self._retry_local_operation_count
                ),
                "readiness_retry_remote_operation_count": (
                    self._retry_remote_operation_count
                ),
                "readiness_retry_summed_latency_ms": self._retry_total_ns / 1_000_000,
                "readiness_retry_max_latency_ms": self._retry_max_ns / 1_000_000,
            }
        self.emit("batch_summary", succeeded=succeeded, **fields)

    def emit_download_summaries(
        self,
        started_ns: int,
        *,
        peak_worker_count: int,
        planned_batch_count: int,
        succeeded: bool,
        worker_count: int,
    ) -> None:
        elapsed_ns = time.perf_counter_ns() - started_ns
        self.emit(
            "download_summary",
            latency_ms=elapsed_ns / 1_000_000,
            elapsed_seconds=elapsed_ns / 1_000_000_000,
            peak_worker_count=peak_worker_count,
            planned_batch_count=planned_batch_count,
            succeeded=succeeded,
            worker_count=worker_count,
        )


def load_cooperatively(
    request: CooperativeLoadRequest,
    *,
    config: CooperativeLoadConfig | None = None,
    _pool_cache: _ChunkPoolCache | None = None,
) -> CooperativeLoadResult:
    """Execute one cooperative load without selecting or invoking a fallback."""

    session = _CooperativeLoadSession(
        request,
        config or CooperativeLoadConfig(),
        pool_cache=_pool_cache,
    )
    try:
        return session.run()
    except CooperativeLoadUnsupported:
        if session.target_writes_started:
            raise AssertionError(
                "an unsupported outcome cannot be raised after target writes"
            )
        raise
    except CooperativeLoadFailure:
        raise
    except Exception as error:
        raise CooperativeLoadFailure(
            f"cooperative load failed: {error}",
            target_writes_started=session.target_writes_started,
        ) from error


class _CooperativeLoadSession:
    def __init__(
        self,
        request: CooperativeLoadRequest,
        config: CooperativeLoadConfig,
        *,
        pool_cache: _ChunkPoolCache | None = None,
    ) -> None:
        self._request = request
        self._config = config
        self._topology = request.topology
        self._world = self._topology.coordination_world
        self._world_nodes = tuple(self._world.node_ids)
        self._world_ranks = tuple(
            rank
            for node in self._topology.nodes
            if node.node_id in self._world_nodes
            for rank in node.ranks
        )
        self._namespace = RendezvousNamespace(
            _PROTOCOL_VERSION,
            self._world.rendezvous_id,
            request.session_token,
        )
        self._metrics = _Metrics(request.metric_callback, self._topology.global_rank)
        self._server: NodeServer | None = None
        self._pool: ChunkPool | None = None
        self._pool_cache = pool_cache
        self._pool_lease: _ChunkPoolLease | None = None
        self._completed_successfully = False
        self._node_control_urls: dict[NodeId, str] = {}
        self._node_data_urls: dict[NodeId, str] = {}
        self._node_capacities: dict[NodeId, int] = {}
        self._coordinator_control_url: str | None = None
        self._control_coordinator: NodeClient | None = None
        self._local_control: NodeClient | None = None
        self._error_monitor_client: NodeClient | None = None
        self._error_monitor_thread: threading.Thread | None = None
        self._error_monitor_stop = threading.Event()
        self._error_monitor_started = False
        self._error_monitor_lock = threading.Lock()
        self._remote_error: str | None = None
        self._error_monitor_failure: Exception | None = None
        self._local_error_forwarded: str | None = None
        self._download_thread: threading.Thread | None = None
        self._download_stop = threading.Event()
        self._download_error: BaseException | None = None
        self._download_error_lock = threading.Lock()
        self._reservation_lock = threading.Lock()
        self._reservations: dict[int, ChunkReservation] = {}
        self._inflight = threading.Semaphore(config.max_inflight_batches)
        self._data_clients_lock = threading.Lock()
        self._data_clients: dict[int, OrderedDict[NodeId, NodeClient]] = {}
        self._metrics_lock = threading.Lock()
        self._storage_bytes = 0
        self._network_bytes = 0
        self._download_worker_lock = threading.Lock()
        self._active_download_workers = 0
        self._peak_download_workers = 0
        self._target_write_lock = threading.Lock()
        self._target_writes_started = False
        self._local_shared_memory_visible = self._topology.is_node_leader
        self._visibility_probe_path: Path | None = None

    @property
    def target_writes_started(self) -> bool:
        with self._target_write_lock:
            return self._target_writes_started

    def run(self) -> CooperativeLoadResult:
        started = time.monotonic()
        started_ns = time.perf_counter_ns()
        primary_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        self._metrics.emit(
            "config",
            batch_target_bytes=self._config.batch_target_bytes,
            download_workers=self._config.download_workers,
            fetch_workers=self._config.fetch_workers,
            node_ids=json.dumps(self._world_nodes, separators=(",", ":")),
            server_workers=self._config.server_workers,
            pinned_buffer_bytes=self._config.pinned_buffer_bytes,
            pinned_buffer_count=self._config.pinned_buffer_count,
        )
        try:
            with self._metrics.phase("bootstrap"):
                self._bootstrap()
            with self._metrics.phase("prepare_targets"):
                local_targets = self._prepare_targets()
            with self._metrics.phase("exchange_demands"):
                global_demand_plan = self._exchange_byte_demands(local_targets)
            with self._metrics.phase("exchange_plan"):
                execution_plan = self._exchange_plan(
                    global_demand_plan,
                    local_targets,
                )
            with self._metrics.phase("execute_batches"):
                if any(execution_plan.active_node_ids_by_batch):
                    with self._target_write_lock:
                        self._target_writes_started = True
                self._execute_batches(local_targets, execution_plan)
            with self._metrics.phase("exchange_metrics"):
                result = self._exchange_metrics(
                    started=started,
                    target_count=len(local_targets),
                    demands=(
                        None
                        if global_demand_plan is None
                        else global_demand_plan.demands
                    ),
                    batch_count=len(execution_plan.batch_indices),
                )
            self._completed_successfully = True
            return result
        except CooperativeLoadUnsupported as error:
            primary_error = error
            if isinstance(error, _ChunkPoolCacheBusy):
                self._publish_error(error)
            raise
        except BaseException as error:
            primary_error = error
            self._publish_error(error)
            raise
        finally:
            try:
                self._cleanup()
            except BaseException as error:
                cleanup_error = error
                if primary_error is None:
                    raise
                logger.exception("cooperative cleanup failed after the primary error")
            finally:
                elapsed_ns = time.perf_counter_ns() - started_ns
                self._metrics.emit(
                    "load_total",
                    latency_ms=elapsed_ns / 1_000_000,
                    elapsed_seconds=elapsed_ns / 1_000_000_000,
                    succeeded=primary_error is None and cleanup_error is None,
                )

    def _bootstrap(self) -> None:
        if self._topology.is_node_leader:
            self._start_node_server()
            assert self._server is not None
            probe_path, probe_value = self._create_visibility_probe()
            self._request.rendezvous.put_blob(
                self._namespace,
                _bootstrap_node_tag(
                    "control", self._world_index(self._topology.node_id)
                ),
                self._advertised_server_url(self._server.control_base_url).encode(),
            )
            self._request.rendezvous.put_blob(
                self._namespace,
                _bootstrap_node_tag("data", self._world_index(self._topology.node_id)),
                self._advertised_server_url(self._server.data_base_url).encode(),
            )
            self._request.rendezvous.put_blob(
                self._namespace,
                _bootstrap_node_tag(
                    "capacity", self._world_index(self._topology.node_id)
                ),
                str(self._require_pool().capacity_bytes).encode(),
            )
            self._request.rendezvous.put_blob(
                self._namespace,
                _bootstrap_node_tag(
                    "probe-path", self._world_index(self._topology.node_id)
                ),
                str(probe_path).encode(),
            )
            self._request.rendezvous.put_blob(
                self._namespace,
                _bootstrap_node_tag(
                    "probe-value", self._world_index(self._topology.node_id)
                ),
                probe_value,
            )
            if self._topology.is_world_leader:
                self._request.rendezvous.put_blob(
                    self._namespace,
                    "coordinator-control-url",
                    self._advertised_server_url(self._server.control_base_url).encode(),
                )

        local_index = self._world_index(self._topology.node_id)
        local_control_url = self._wait_bootstrap_blob(
            _bootstrap_node_tag("control", local_index)
        ).decode()
        self._coordinator_control_url = self._wait_bootstrap_blob(
            "coordinator-control-url"
        ).decode()
        self._local_control = self._new_control_client(local_control_url)
        self._local_control.health()
        self._control_coordinator = self._new_control_client(
            self._coordinator_control_url
        )
        self._control_coordinator.health()
        self._probe_local_shared_memory(local_index)

        node_payload: bytes | None = None
        if self._topology.is_world_leader:
            entries = []
            for node_index, node_id in enumerate(self._world_nodes):
                entries.append(
                    {
                        "capacity": int(
                            self._wait_bootstrap_blob(
                                _bootstrap_node_tag("capacity", node_index)
                            ).decode()
                        ),
                        "control_url": self._wait_bootstrap_blob(
                            _bootstrap_node_tag("control", node_index)
                        ).decode(),
                        "data_url": self._wait_bootstrap_blob(
                            _bootstrap_node_tag("data", node_index)
                        ).decode(),
                        "node_id": node_id,
                    }
                )
            node_payload = _encode_control(entries)
        decoded = _sequence(
            _decode_control(self._broadcast_bytes("node-configuration", node_payload)),
            "node configuration",
        )
        for item in decoded:
            entry = _mapping(item, "node configuration entry")
            node_id = _node_id(entry["node_id"])
            self._node_control_urls[node_id] = str(entry["control_url"])
            self._node_data_urls[node_id] = str(entry["data_url"])
            self._node_capacities[node_id] = int(entry["capacity"])
        if set(self._node_data_urls) != set(self._world_nodes):
            raise ValueError("node configuration does not match the coordination world")
        if any(value <= 0 for value in self._node_capacities.values()):
            raise ValueError("node shared-memory capacities must be positive")
        self._metrics.record_node_capacities(tuple(self._node_capacities.values()))
        self._start_error_monitor()

    def _create_visibility_probe(self) -> tuple[Path, bytes]:
        probe_path = self._require_pool().job_directory / "visibility_probe"
        probe_value = self._namespace.load_token.encode()
        with probe_path.open("wb") as stream:
            stream.write(probe_value)
        self._visibility_probe_path = probe_path
        return probe_path, probe_value

    def _probe_local_shared_memory(self, node_index: int) -> None:
        probe_path = Path(
            self._wait_bootstrap_blob(
                _bootstrap_node_tag("probe-path", node_index)
            ).decode()
        )
        expected = self._wait_bootstrap_blob(
            _bootstrap_node_tag("probe-value", node_index)
        )
        probe = self._request.shared_memory_visibility_probe
        if probe is None:
            try:
                visible = probe_path.read_bytes() == expected
            except OSError:
                visible = False
        else:
            try:
                visible = bool(probe(probe_path, expected))
            except OSError:
                visible = False
        self._local_shared_memory_visible = visible
        self._put_local(
            _local_rank_tag("shared-memory-visible", self._topology.global_rank),
            b"1" if visible else b"0",
        )
        if self._topology.is_node_leader:
            visible_count = 0
            for rank in self._topology.node_ranks:
                visible_count += (
                    self._take_local(_local_rank_tag("shared-memory-visible", rank))
                    == b"1"
                )
            self._remove_visibility_probe()
            self._metrics.emit(
                "shared_memory_visibility",
                local_rank_count=len(self._topology.node_ranks),
                visible_rank_count=visible_count,
            )

    def _remove_visibility_probe(self) -> None:
        path, self._visibility_probe_path = self._visibility_probe_path, None
        if path is not None:
            path.unlink(missing_ok=True)

    def _start_node_server(self) -> None:
        spec = _shared_memory_pool_spec(self._request, self._config)

        def create_pool() -> ChunkPool:
            return ChunkPool(
                capacity_bytes=spec.capacity_bytes,
                chunk_bytes=spec.chunk_bytes,
                job_token=(
                    f"{self._namespace.load_token}-"
                    f"{self._world_index(self._topology.node_id)}"
                ),
                directory=spec.directory,
            )

        lease = None
        if self._pool_cache is not None:
            lease = self._pool_cache.try_acquire(
                spec,
                create_pool,
                cleanup_timeout=self._config.progress_timeout_seconds,
            )
            if lease is None:
                raise _ChunkPoolCacheBusy(
                    "another cooperative load already owns the reusable "
                    "shared-memory pool"
                )
        self._pool_lease = lease
        self._pool = lease.pool if lease is not None else create_pool()
        self._metrics.emit(
            "shared_memory_pool",
            cache_requested=self._pool_cache is not None,
            cache_retained=lease is not None and lease.retainable,
            cache_retain_ceiling_bytes=(
                self._pool_cache.max_retained_capacity_bytes
                if self._pool_cache is not None
                else 0
            ),
            cache_retained_capacity_bytes=(
                spec.capacity_bytes if lease is not None and lease.retainable else 0
            ),
            cache_reused=lease is not None and lease.reused,
            capacity_bytes=spec.capacity_bytes,
            chunk_bytes=spec.chunk_bytes,
        )
        self._server = NodeServer(
            self._pool.index,
            protocol_version=_PROTOCOL_VERSION,
            load_token=self._namespace.load_token,
            host=self._request.bind_host,
            data_worker_count=self._config.server_workers,
            max_control_body_bytes=_CONTROL_BYTES_LIMIT,
            max_fetch_ranges=_MAX_FETCH_RANGES,
        ).start()

    def _advertised_server_url(self, bound_url: str) -> str:
        return _replace_url_host(
            bound_url,
            self._request.advertise_host or self._request.bind_host,
        )

    def _prepare_targets(self) -> tuple[TensorReadTarget, ...]:
        with self._metrics.latency("prepare_targets.target_mode"):
            self._require_uniform_target_mode()
        if self._request.local_targets is not None:
            with self._metrics.latency(
                "prepare_targets.dedupe_resolved",
                input_target_count=len(self._request.local_targets),
            ) as fields:
                targets = _dedupe_resolved_targets(
                    tuple(self._request.local_targets),
                    self._request.target_state_dict,
                )
                fields["output_target_count"] = len(targets)
            return targets
        with self._metrics.latency(
            "prepare_targets.dedupe_aliases",
            input_fqn_count=len(self._request.local_load_plan),
        ) as fields:
            load_plan, aliases = dedupe_aliased_targets(
                self._request.local_load_plan,
                self._request.target_state_dict,
            )
            fields["output_fqn_count"] = len(load_plan)
            fields["alias_count"] = len(aliases)
        if aliases:
            self._metrics.emit("aliases", alias_count=len(aliases))
        provider = self._request.metadata_provider
        with self._metrics.latency("prepare_targets.build_source_demands") as fields:
            local_demands = _source_demands_for_plan(load_plan)
            fields["source_rank_count"] = len(local_demands)
            fields["fqn_count"] = sum(len(fqns) for fqns in local_demands.values())
        with self._metrics.latency("prepare_targets.exchange_metadata_demands"):
            rank_demands = self._exchange_source_metadata_demands(local_demands)
        with self._metrics.latency(
            "prepare_targets.publish_metadata_outcome",
            assigned_worker_count=self._config.download_workers,
        ):
            self._publish_metadata_outcome(provider)
        with self._metrics.latency("prepare_targets.collect_metadata_outcomes"):
            metadata_by_node, status_payload = self._collect_metadata_outcomes(
                rank_demands
            )
        with self._metrics.latency("prepare_targets.metadata_status"):
            self._require_metadata_eligible(status_payload)
        with self._metrics.latency("prepare_targets.scatter_metadata") as fields:
            encoded_group = self._scatter_node_payloads(
                "rank-metadata", metadata_by_node
            )
            fields["compressed_bytes"] = len(encoded_group)
        with self._metrics.latency("prepare_targets.decode_metadata") as fields:
            node_metadata = _decode_trusted_source_tensor_metadata_wire(
                _decode_control(encoded_group)
            )
            selected_metadata = _select_trusted_source_tensor_metadata_wire(
                node_metadata,
                (local_demands,),
            )
            local_metadata = _materialize_trusted_source_tensor_metadata_wire(
                selected_metadata
            )
            fields["node_union_source_rank_count"] = node_metadata.source_rank_count
            fields["node_union_tensor_count"] = node_metadata.tensor_count
            fields["source_rank_count"] = len(local_metadata)
            fields["tensor_count"] = sum(
                len(metadata) for metadata in local_metadata.values()
            )
        with self._metrics.latency("prepare_targets.resolve_targets") as fields:
            targets = tuple(
                resolve_tensor_read_targets(
                    load_plan,
                    local_metadata,
                    self._request.target_state_dict,
                )
            )
            fields["target_count"] = len(targets)
        return targets

    def _require_uniform_target_mode(self) -> None:
        local_mode = (
            "resolved" if self._request.local_targets is not None else "planned"
        )
        local_payloads = self._gather_local_rank_payloads(
            "target-mode", local_mode.encode()
        )
        if self._topology.is_node_leader:
            assert local_payloads is not None
            self._put_global(
                _global_node_tag(
                    "target-mode", self._world_index(self._topology.node_id)
                ),
                _encode_control(
                    {
                        str(rank): value.decode()
                        for rank, value in local_payloads.items()
                    }
                ),
            )

        status_payload: bytes | None = None
        if self._topology.is_world_leader:
            rank_modes: dict[int, str] = {}
            for node_index in range(len(self._world_nodes)):
                node_modes = _mapping(
                    _decode_control(
                        self._take_global(_global_node_tag("target-mode", node_index))
                    ),
                    "node target modes",
                )
                for raw_rank, raw_mode in node_modes.items():
                    rank = int(raw_rank)
                    if rank in rank_modes:
                        raise ValueError(f"duplicate target mode for rank {rank}")
                    if raw_mode not in ("planned", "resolved"):
                        raise ValueError(f"invalid target mode for rank {rank}")
                    rank_modes[rank] = raw_mode
            if set(rank_modes) != set(self._world_ranks):
                raise ValueError("target modes do not cover the coordination world")
            planned_ranks = sorted(
                rank for rank, mode in rank_modes.items() if mode == "planned"
            )
            resolved_ranks = sorted(
                rank for rank, mode in rank_modes.items() if mode == "resolved"
            )
            status_payload = _encode_control(
                {
                    "mode": local_mode if not planned_ranks else "planned",
                    "planned_ranks": planned_ranks,
                    "resolved_ranks": resolved_ranks,
                    "status": (
                        "mixed" if planned_ranks and resolved_ranks else "uniform"
                    ),
                }
            )

        status = _mapping(
            _decode_control(
                self._broadcast_bytes("target-mode-status", status_payload)
            ),
            "target mode status",
        )
        if status.get("status") == "mixed":
            raise ValueError(
                "all ranks must use the same cooperative target mode; "
                f"planned ranks {status.get('planned_ranks')}, "
                f"resolved ranks {status.get('resolved_ranks')}"
            )
        if status.get("status") != "uniform" or status.get("mode") != local_mode:
            raise ValueError("cooperative target mode status is invalid")

    def _exchange_source_metadata_demands(
        self,
        local_demands: Mapping[int, Collection[str]],
    ) -> dict[int, dict[int, frozenset[str]]] | None:
        encoded_local = _encode_control(_source_demands_to_wire(local_demands))
        local_payloads = self._gather_local_rank_payloads(
            "source-demands", encoded_local
        )
        if self._topology.is_node_leader:
            assert local_payloads is not None
            self._put_global(
                _global_node_tag(
                    "source-demands", self._world_index(self._topology.node_id)
                ),
                _encode_control(
                    {
                        str(rank): _decode_control(value)
                        for rank, value in local_payloads.items()
                    }
                ),
            )
        if not self._topology.is_world_leader:
            return None
        rank_demands: dict[int, dict[int, frozenset[str]]] = {}
        for node_index in range(len(self._world_nodes)):
            node_payload = _mapping(
                _decode_control(
                    self._take_global(_global_node_tag("source-demands", node_index))
                ),
                "node source demands",
            )
            for raw_rank, raw_demands in node_payload.items():
                rank = int(raw_rank)
                if rank in rank_demands:
                    raise ValueError(f"duplicate source demands for rank {rank}")
                rank_demands[rank] = _source_demands_from_wire(raw_demands)
        if set(rank_demands) != set(self._world_ranks):
            raise ValueError("source demands do not cover the coordination world")
        merged_demands = _merge_source_demands(rank_demands.values())
        assigned = _partition_metadata_demands(merged_demands, self._world_nodes)
        for node_index, node_id in enumerate(self._world_nodes):
            self._put_global(
                _global_node_tag("metadata-demands", node_index),
                _encode_control(_source_demands_to_wire(assigned[node_id])),
            )
        return rank_demands

    def _publish_metadata_outcome(self, provider: MetadataProvider | None) -> None:
        if not self._topology.is_node_leader:
            return
        demands = _source_demands_from_wire(
            _decode_control(
                self._take_global(
                    _global_node_tag(
                        "metadata-demands",
                        self._world_index(self._topology.node_id),
                    )
                )
            )
        )
        try:
            if demands and provider is None:
                raise ValueError(
                    "metadata_provider is required when this node is assigned "
                    "metadata demands"
                )
            with self._metrics.latency(
                "prepare_targets.inspect_archives",
                source_rank_count=len(demands),
                demanded_fqn_count=sum(len(fqns) for fqns in demands.values()),
                worker_count=self._config.download_workers,
            ) as fields:
                metadata = (
                    provider.load_metadata(
                        demands,
                        storage=self._request.storage,
                        source_path_for_rank=self._request.source_path_for_rank,
                        max_workers=self._config.download_workers,
                        timeout_seconds=self._config.progress_timeout_seconds,
                    )
                    if demands
                    else {}
                )
                partition = _build_partitioned_source_tensor_metadata_wire(
                    metadata,
                    demands,
                )
                fields["tensor_count"] = partition.tensor_count
            with self._metrics.latency(
                "prepare_targets.encode_metadata_outcome"
            ) as fields:
                encoded_outcome = _encode_metadata_control(
                    {
                        "metadata": partition.payload,
                        "status": "eligible",
                    }
                )
                fields["compressed_bytes"] = len(encoded_outcome)
        except CooperativeLoadUnsupported as error:
            encoded_outcome = _encode_control(
                {
                    "detail": str(error),
                    "status": "unsupported",
                }
            )
        except Exception as error:
            encoded_outcome = _encode_control(
                {
                    "detail": _error_message(
                        self._topology.global_rank,
                        error,
                    ),
                    "status": "fatal",
                }
            )
        self._put_global(
            _global_node_tag(
                "metadata-outcome", self._world_index(self._topology.node_id)
            ),
            encoded_outcome,
        )

    def _collect_metadata_outcomes(
        self,
        rank_demands: Mapping[int, Mapping[int, Collection[str]]] | None,
    ) -> tuple[dict[NodeId, bytes] | None, bytes | None]:
        if not self._topology.is_world_leader:
            return None, None
        assert rank_demands is not None
        merged_demands = _merge_source_demands(rank_demands.values())
        assigned_demands = _partition_metadata_demands(
            merged_demands,
            self._world_nodes,
        )
        metadata_partitions: list[tuple[object, Mapping[int, Collection[str]]]] = []
        unsupported: tuple[NodeId, str] | None = None
        fatal: tuple[NodeId, str] | None = None
        with self._metrics.latency(
            "prepare_targets.merge_metadata_outcomes",
            node_count=len(self._world_nodes),
        ) as fields:
            compressed_bytes = 0
            for node_index in range(len(self._world_nodes)):
                raw_outcome = self._take_global(
                    _global_node_tag("metadata-outcome", node_index)
                )
                compressed_bytes += len(raw_outcome)
                outcome = _mapping(
                    _decode_control(raw_outcome),
                    "metadata outcome",
                )
                status = str(outcome.get("status"))
                if status == "unsupported":
                    if unsupported is None:
                        unsupported = (
                            self._world_nodes[node_index],
                            str(outcome.get("detail", "metadata is unsupported")),
                        )
                    continue
                if status == "fatal":
                    if fatal is None:
                        fatal = (
                            self._world_nodes[node_index],
                            str(outcome.get("detail", "metadata preparation failed")),
                        )
                    continue
                if status != "eligible":
                    raise ValueError(f"unknown metadata outcome {status!r}")
                metadata_partitions.append(
                    (
                        outcome["metadata"],
                        assigned_demands[self._world_nodes[node_index]],
                    )
                )
            merged_metadata = _merge_partitioned_source_tensor_metadata_wire(
                metadata_partitions
            )
            fields["compressed_bytes"] = compressed_bytes
            fields["source_rank_count"] = merged_metadata.source_rank_count
            fields["tensor_count"] = merged_metadata.tensor_count
            fields["duplicate_tensor_count"] = 0
        if fatal is not None:
            return None, _encode_control(
                {
                    "detail": fatal[1],
                    "node_id": fatal[0],
                    "status": "fatal",
                }
            )
        if unsupported is not None:
            return None, _encode_control(
                {
                    "detail": unsupported[1],
                    "node_id": unsupported[0],
                    "status": "unsupported",
                }
            )
        metadata_by_node: dict[NodeId, bytes] = {}
        with self._metrics.latency(
            "prepare_targets.encode_node_metadata",
            node_count=len(self._world_nodes),
            rank_count=len(rank_demands),
        ) as fields:
            emitted_tensor_count = 0
            for node_id in self._world_nodes:
                selected = _select_trusted_source_tensor_metadata_wire(
                    merged_metadata,
                    (rank_demands[rank] for rank in self._node_members(node_id)),
                )
                emitted_tensor_count += selected.tensor_count
                metadata_by_node[node_id] = _encode_metadata_control(selected.payload)
            fields["emitted_tensor_count"] = emitted_tensor_count
            fields["globally_unique_tensor_count"] = merged_metadata.tensor_count
            fields["compressed_bytes"] = sum(
                len(payload) for payload in metadata_by_node.values()
            )
        return metadata_by_node, _encode_control({"status": "eligible"})

    def _require_metadata_eligible(self, status_payload: bytes | None) -> None:
        status = _mapping(
            _decode_control(self._broadcast_bytes("metadata-status", status_payload)),
            "metadata status",
        )
        if status.get("status") == "unsupported":
            if self.target_writes_started:
                raise AssertionError("metadata veto occurred after target writes")
            raise CooperativeLoadUnsupported(
                f"node {status.get('node_id')!r}: "
                f"{status.get('detail', 'metadata is unsupported')}"
            )
        if status.get("status") == "fatal":
            if self.target_writes_started:
                raise AssertionError("metadata failure occurred after target writes")
            raise ValueError(
                f"node {status.get('node_id')!r}: "
                f"{status.get('detail', 'metadata preparation failed')}"
            )
        if status.get("status") != "eligible":
            raise ValueError("metadata status is invalid")

    def _exchange_byte_demands(
        self, local_targets: Sequence[TensorReadTarget]
    ) -> _GlobalByteDemandPlan | None:
        with self._metrics.latency(
            "exchange_demands.encode_local",
            local_target_count=len(local_targets),
        ) as fields:
            local_demands = _targets_to_demands(local_targets)
            local_consumer_bytes = _target_consumer_bytes(local_targets)
            encoded = _encode_control(
                _byte_demand_payload_to_wire(
                    local_demands,
                    local_consumer_bytes,
                )
            )
            fields["local_demand_count"] = len(local_demands)
            fields["local_range_count"] = sum(
                len(demand.ranges) for demand in local_demands
            )
            fields["local_consumer_bytes"] = sum(local_consumer_bytes.values())
            fields["compressed_bytes"] = len(encoded)
        with self._metrics.latency("exchange_demands.gather_local"):
            local_payloads = self._gather_local_rank_payloads("byte-demands", encoded)
        if self._topology.is_node_leader:
            assert local_payloads is not None
            with self._metrics.latency(
                "exchange_demands.decode_local",
                input_payload_count=len(local_payloads),
                compressed_input_bytes=sum(
                    len(value) for value in local_payloads.values()
                ),
            ) as fields:
                decoded_local = tuple(
                    _byte_demand_payload_from_wire(_decode_control(value))
                    for value in local_payloads.values()
                )
                fields["decoded_demand_count"] = sum(
                    len(demands) for demands, _ in decoded_local
                )
            with self._metrics.latency(
                "exchange_demands.deduplicate_local",
                input_payload_count=len(decoded_local),
            ) as fields:
                unique_demands = _unique_demand_wire_payloads(
                    demands for demands, _ in decoded_local
                )
                fields["unique_payload_count"] = len(unique_demands)
            node_consumer_bytes: Counter[int] = Counter()
            for _, consumer_bytes in decoded_local:
                node_consumer_bytes.update(consumer_bytes)
            with self._metrics.latency(
                "exchange_demands.merge_node",
                input_payload_count=len(local_payloads),
                unique_payload_count=len(unique_demands),
                compressed_input_bytes=sum(
                    len(value) for value in local_payloads.values()
                ),
            ) as fields:
                node_merge_result = _merge_fqn_demand_wire_payloads_to_canonical_wire(
                    unique_demands
                )
                node_encoded = _encode_control(
                    _byte_demand_payload_to_wire_from_canonical(
                        node_merge_result.payload,
                        node_consumer_bytes,
                    )
                )
                fields["input_demand_count"] = node_merge_result.input_demand_count
                fields["input_range_count"] = node_merge_result.input_range_count
                fields["output_demand_count"] = node_merge_result.output_demand_count
                fields["output_range_count"] = node_merge_result.output_range_count
                fields["output_compressed_bytes"] = len(node_encoded)
                fields["consumer_source_count"] = len(node_consumer_bytes)
                fields["consumer_bytes"] = sum(node_consumer_bytes.values())
                fields["merge_decode_ms"] = node_merge_result.decode_ns / 1_000_000
                fields["merge_input_iteration_ms"] = (
                    node_merge_result.decode_ns / 1_000_000
                )
                fields["merge_union_ms"] = node_merge_result.union_ns / 1_000_000
                fields["merge_finalize_ms"] = node_merge_result.finalize_ns / 1_000_000
            with self._metrics.latency("exchange_demands.publish_node"):
                self._put_global(
                    _global_node_tag(
                        "byte-demands", self._world_index(self._topology.node_id)
                    ),
                    node_encoded,
                )
        if not self._topology.is_world_leader:
            return None
        with self._metrics.latency("exchange_demands.collect_nodes"):
            encoded_nodes = [
                self._take_global(_global_node_tag("byte-demands", node_index))
                for node_index in range(len(self._world_nodes))
            ]
        with self._metrics.latency(
            "exchange_demands.decode_nodes",
            input_payload_count=len(encoded_nodes),
            compressed_input_bytes=sum(len(value) for value in encoded_nodes),
        ) as fields:
            decoded_nodes = tuple(
                _byte_demand_payload_from_wire(_decode_control(value))
                for value in encoded_nodes
            )
            fields["decoded_demand_count"] = sum(
                len(demands) for demands, _ in decoded_nodes
            )
        with self._metrics.latency(
            "exchange_demands.deduplicate_nodes",
            input_payload_count=len(decoded_nodes),
        ) as fields:
            unique_node_demands = _unique_demand_wire_payloads(
                demands for demands, _ in decoded_nodes
            )
            fields["unique_payload_count"] = len(unique_node_demands)
        source_consumer_bytes_by_node = {
            node_id: consumer_bytes
            for node_id, (_, consumer_bytes) in zip(
                self._world_nodes,
                decoded_nodes,
                strict=True,
            )
        }
        with self._metrics.latency(
            "exchange_demands.merge_world",
            input_payload_count=len(encoded_nodes),
            unique_payload_count=len(unique_node_demands),
            compressed_input_bytes=sum(len(value) for value in encoded_nodes),
        ) as fields:
            world_merge_result = _merge_canonical_fqn_demand_wire_payloads(
                unique_node_demands
            )
            fields["input_demand_count"] = world_merge_result.input_demand_count
            fields["input_range_count"] = world_merge_result.input_range_count
            fields["output_demand_count"] = len(world_merge_result.demands)
            fields["output_range_count"] = sum(
                len(demand.ranges) for demand in world_merge_result.demands
            )
            fields["merge_decode_ms"] = world_merge_result.decode_ns / 1_000_000
            fields["merge_input_iteration_ms"] = (
                world_merge_result.decode_ns / 1_000_000
            )
            fields["merge_union_ms"] = world_merge_result.union_ns / 1_000_000
            fields["merge_finalize_ms"] = world_merge_result.finalize_ns / 1_000_000
            fields["consumer_bytes"] = sum(
                sum(consumer_bytes.values())
                for consumer_bytes in source_consumer_bytes_by_node.values()
            )
        return _GlobalByteDemandPlan(
            demands=world_merge_result.demands,
            source_consumer_bytes_by_node=source_consumer_bytes_by_node,
        )

    def _exchange_plan(
        self,
        global_demand_plan: _GlobalByteDemandPlan | None,
        local_targets: Sequence[TensorReadTarget],
    ) -> ProjectedExecutionPlan:
        encoded: bytes | None = None
        if self._topology.is_world_leader:
            if global_demand_plan is None:
                raise RuntimeError("world leader has no global byte demands")
            global_demands = global_demand_plan.demands
            demand_count = len(global_demands)
            range_count = sum(len(demand.ranges) for demand in global_demands)
            batch_budget = min(
                self._config.batch_target_bytes,
                min(self._node_capacities.values()),
            )
            with self._metrics.latency(
                "exchange_plan.fused_build",
                demand_count=demand_count,
                range_count=range_count,
                node_count=len(self._world_nodes),
                batch_budget_bytes=batch_budget,
            ):
                planning_result = _plan_cooperative_resharding_from_merged_demands(
                    global_demands,
                    self._world_nodes,
                    batch_budget,
                    self._config.range_consolidation_gap_bytes,
                    source_consumer_bytes_by_node=(
                        global_demand_plan.source_consumer_bytes_by_node
                    ),
                )
            assignment_result = planning_result.assignment_result
            assignment = assignment_result.assignment
            baseline = assignment_result.baseline_locality
            locality = assignment_result.chosen_locality
            digest_started_ns = time.perf_counter_ns()
            assignment_digest = hashlib.sha256(
                json.dumps(
                    assignment.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            assignment_ns = (
                planning_result.assignment_ns
                + time.perf_counter_ns()
                - digest_started_ns
            )
            self._metrics.emit(
                "exchange_plan.assignment_locality",
                assignment_digest=assignment_digest,
                baseline_local_consumer_bytes=baseline.local_consumer_bytes,
                baseline_remote_consumer_bytes=baseline.remote_consumer_bytes,
                theoretical_max_local_consumer_bytes=(
                    assignment_result.theoretical_max_local_consumer_bytes
                ),
                total_consumer_bytes=locality.total_consumer_bytes,
                local_consumer_bytes=locality.local_consumer_bytes,
                remote_consumer_bytes=locality.remote_consumer_bytes,
                local_fraction=(
                    locality.local_consumer_bytes / locality.total_consumer_bytes
                    if locality.total_consumer_bytes
                    else 1.0
                ),
            )
            for node_index, node_id in enumerate(locality.node_ids):
                self._metrics.emit(
                    "exchange_plan.assignment_node",
                    node_id=node_id,
                    source_count=len(assignment.node_source_ranks[node_index]),
                    read_bytes=assignment.node_bytes[node_index],
                    total_consumer_bytes=locality.node_total_consumer_bytes[node_index],
                    local_consumer_bytes=locality.node_local_consumer_bytes[node_index],
                    remote_consumer_bytes=locality.node_remote_consumer_bytes[
                        node_index
                    ],
                )
            self._metrics.record_latency(
                "exchange_plan.assign_sources",
                assignment_ns,
                demand_count=demand_count,
                range_count=range_count,
                node_count=len(self._world_nodes),
                assignment_digest=assignment_digest,
                baseline_local_consumer_bytes=baseline.local_consumer_bytes,
                baseline_remote_consumer_bytes=baseline.remote_consumer_bytes,
                chosen_local_consumer_bytes=locality.local_consumer_bytes,
                chosen_remote_consumer_bytes=locality.remote_consumer_bytes,
                theoretical_max_local_consumer_bytes=(
                    assignment_result.theoretical_max_local_consumer_bytes
                ),
            )
            batches = planning_result.batches
            self._metrics.record_latency(
                "exchange_plan.plan_batches",
                planning_result.batching_ns,
                batch_budget_bytes=batch_budget,
                demand_count=demand_count,
                range_count=range_count,
                batch_count=len(batches),
            )
            effective_gap_bytes = self._config.range_consolidation_gap_bytes
            works = planning_result.works
            work_ns = planning_result.work_ns
            if _oversized_node_works(works, self._node_capacities):
                rebuild_started_ns = time.perf_counter_ns()
                effective_gap_bytes = 0
                works = _rebuild_batch_node_works_from_canonical(
                    works,
                    effective_gap_bytes,
                )
                work_ns += time.perf_counter_ns() - rebuild_started_ns
            self._metrics.record_latency(
                "exchange_plan.build_works",
                work_ns,
                batch_count=len(batches),
                consolidate_gap_bytes=self._config.range_consolidation_gap_bytes,
                work_count=len(works),
                download_range_count=sum(len(work.download_ranges) for work in works),
                download_bytes=sum(work.download_bytes for work in works),
                effective_consolidate_gap_bytes=effective_gap_bytes,
            )
            oversized = _oversized_node_works(works, self._node_capacities)
            if oversized:
                work = oversized[0]
                encoded = _encode_control(
                    {
                        "detail": (
                            f"batch {work.batch_index} needs {work.download_bytes} "
                            f"bytes on node {work.node_id!r}, whose capacity is "
                            f"{self._node_capacities[work.node_id]} bytes"
                        ),
                        "status": "unsupported",
                    }
                )
            else:
                with self._metrics.latency("exchange_plan.encode") as fields:
                    encoded = _encode_control(
                        {
                            "plan": _execution_plan_to_wire_from_validated(
                                assignment,
                                batches,
                                works,
                            ),
                            "status": "eligible",
                        }
                    )
                    fields["compressed_bytes"] = len(encoded)
        with self._metrics.latency("exchange_plan.broadcast") as fields:
            plan_payload = self._broadcast_bytes("pipeline-plan", encoded)
            fields["compressed_bytes"] = len(plan_payload)
        with self._metrics.latency("exchange_plan.decode"):
            payload = _mapping(_decode_control(plan_payload), "pipeline plan")
        if payload.get("status") == "unsupported":
            if self.target_writes_started:
                raise AssertionError("pipeline-plan veto occurred after target writes")
            raise CooperativeLoadUnsupported(
                str(payload.get("detail", "pipeline plan exceeds node capacity"))
            )
        if payload.get("status") != "eligible":
            raise ValueError("pipeline plan status is invalid")
        with self._metrics.latency("exchange_plan.project") as fields:
            execution_plan = project_execution_plan_wire(
                payload["plan"],
                expected_node_ids=self._world_nodes,
                local_node_id=self._topology.node_id,
                local_targets=local_targets,
                node_capacities=self._node_capacities,
            )
            fields["batch_count"] = len(execution_plan.batch_indices)
            fields["source_count"] = len(execution_plan.source_owners)
            fields["source_schedule_count"] = len(execution_plan.source_schedules)
            fields["local_target_count"] = sum(
                len(indices) for indices in execution_plan.local_target_indices_by_batch
            )
            fields["local_download_range_count"] = sum(
                len(download.download_ranges)
                for download in execution_plan.local_downloads
            )
            fields["local_download_bytes"] = sum(
                download.download_bytes for download in execution_plan.local_downloads
            )
        return execution_plan

    def _execute_batches(
        self,
        local_targets: Sequence[TensorReadTarget],
        execution_plan: ProjectedExecutionPlan,
    ) -> None:
        pool = _make_receive_pool(self._config)
        self._start_download_pipeline(execution_plan.local_downloads)
        primary_error: BaseException | None = None
        try:
            with ThreadPoolExecutor(
                max_workers=self._config.fetch_workers,
                thread_name_prefix="cooperative-fetch",
            ) as executor:
                for batch_index in execution_plan.batch_indices:
                    active_node_ids = execution_plan.active_node_ids_by_batch[
                        batch_index
                    ]
                    started_ns = time.perf_counter_ns()
                    self._wait_batch_registered(batch_index, active_node_ids)
                    self._metrics.record_batch_phase(
                        "wait_registered", batch_index, started_ns
                    )
                    started_ns = time.perf_counter_ns()
                    self._scatter_batch(
                        batch_index,
                        tuple(
                            local_targets[target_index]
                            for target_index in execution_plan.local_target_indices_by_batch[
                                batch_index
                            ]
                        ),
                        execution_plan,
                        pool,
                        executor,
                    )
                    self._metrics.record_batch_phase("scatter", batch_index, started_ns)
                    started_ns = time.perf_counter_ns()
                    self._complete_and_retire_batch(
                        batch_index,
                        active_node_ids,
                    )
                    self._metrics.record_batch_phase("retire", batch_index, started_ns)
        except BaseException as error:
            primary_error = error
            self._download_stop.set()
            raise
        finally:
            succeeded = primary_error is None
            try:
                self._finish_batch_execution(pool, primary_error)
            except BaseException:
                succeeded = False
                raise
            finally:
                self._metrics.emit_batch_summaries(succeeded=succeeded)

    def _finish_batch_execution(
        self,
        pool: PinnedBufferPool,
        primary_error: BaseException | None,
    ) -> None:
        errors: list[Exception] = []
        try:
            with self._metrics.latency("execute_batches.finish_download_pipeline"):
                self._finish_download_pipeline(
                    timeout=(
                        0
                        if primary_error is not None
                        else self._config.progress_timeout_seconds
                    )
                )
        except Exception as error:
            errors.append(error)
        try:
            with self._metrics.latency("execute_batches.close_receive_pool"):
                pool.close()
        except Exception as error:
            errors.append(error)
        if not errors:
            return
        if primary_error is not None:
            for error in errors:
                logger.error(
                    "cooperative batch cleanup failed after %s",
                    type(primary_error).__name__,
                    exc_info=(type(error), error, error.__traceback__),
                )
            return
        if len(errors) == 1:
            raise errors[0]
        raise RuntimeError(
            "multiple cooperative batch cleanup failures: "
            + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
        ) from errors[0]

    def _start_download_pipeline(
        self,
        downloads: Sequence[ProjectedBatchDownload],
    ) -> None:
        if not self._topology.is_node_leader:
            return
        if any(download.node_id != self._topology.node_id for download in downloads):
            raise ValueError("projected downloads belong to another node")
        self._download_thread = threading.Thread(
            target=self._download_batches,
            args=(tuple(downloads),),
            name="cooperative-download",
            daemon=True,
        )
        self._download_thread.start()

    def _download_batches(self, works: Sequence[ProjectedBatchDownload]) -> None:
        started_ns = time.perf_counter_ns()
        succeeded = False
        client: NodeClient | None = None
        try:
            client = self._new_control_client(self._require_coordinator_control_url())
            self._metrics.record_download_executor_creation(
                self._config.download_workers
            )
            with ThreadPoolExecutor(
                max_workers=self._config.download_workers,
                thread_name_prefix="cooperative-storage",
            ) as executor:
                for work in works:
                    if self._download_stop.is_set():
                        return
                    if work.download_bytes and not self._download_one_batch(
                        work, executor, client
                    ):
                        return
            succeeded = True
        except Exception as error:
            self._record_download_error(error)
        finally:
            try:
                if client is not None:
                    client.close()
            except Exception as error:
                succeeded = False
                self._record_download_error(error)
            finally:
                self._metrics.emit_download_summaries(
                    started_ns,
                    peak_worker_count=self._peak_download_workers,
                    planned_batch_count=len(works),
                    succeeded=succeeded,
                    worker_count=self._config.download_workers,
                )

    def _download_one_batch(
        self,
        work: ProjectedBatchDownload,
        executor: ThreadPoolExecutor,
        control_client: NodeClient,
    ) -> bool:
        started_ns = time.perf_counter_ns()
        completed = False
        try:
            while not self._download_stop.is_set():
                if self._inflight.acquire(timeout=0.5):
                    break
            else:
                return False
            owner_id = (
                f"batch-{work.batch_index}-node-{self._world_index(work.node_id)}"
            )
            reservation: ChunkReservation | None = None
            try:
                reservation = self._reserve_batch(
                    self._require_pool(), owner_id, work.download_bytes
                )
                if reservation is None:
                    self._inflight.release()
                    return False
                registration_started_ns = time.perf_counter_ns()
                registered_ranges = reservation.register_many(
                    tuple(
                        RangeSpec(
                            str(byte_range.source_rank),
                            byte_range.offset,
                            byte_range.length,
                        )
                        for byte_range in work.download_ranges
                    )
                )
                if reservation.used_bytes != work.download_bytes:
                    raise RuntimeError(
                        f"batch {work.batch_index} registered "
                        f"{reservation.used_bytes} bytes, expected "
                        f"{work.download_bytes}"
                    )
                with self._reservation_lock:
                    self._reservations[work.batch_index] = reservation
                control_client.put_blob(
                    _global_node_tag(
                        f"registered/{work.batch_index}",
                        self._world_index(work.node_id),
                    ),
                    b"1",
                )
                self._metrics.record_download_registration(
                    batch_index=work.batch_index,
                    byte_count=work.download_bytes,
                    elapsed_ns=time.perf_counter_ns() - registration_started_ns,
                    range_count=len(registered_ranges),
                    segment_count=sum(
                        len(registered_range.segments)
                        for registered_range in registered_ranges
                    ),
                    max_segment_bytes=max(
                        (
                            segment.length
                            for registered_range in registered_ranges
                            for segment in registered_range.segments
                        ),
                        default=0,
                    ),
                )
                self._download_registered_ranges(
                    work,
                    reservation,
                    registered_ranges,
                    executor,
                )
                with self._metrics_lock:
                    self._storage_bytes += work.download_bytes
                control_client.put_blob(
                    _global_node_tag(
                        f"downloaded/{work.batch_index}",
                        self._world_index(work.node_id),
                    ),
                    b"1",
                )
                completed = True
                return True
            except BaseException as error:
                self._record_download_error(error)
                if reservation is not None:
                    reservation.retire(self._config.progress_timeout_seconds)
                    with self._reservation_lock:
                        self._reservations.pop(work.batch_index, None)
                self._inflight.release()
                raise
        finally:
            self._metrics.record_download_batch(
                work.batch_index, started_ns, completed=completed
            )

    def _download_registered_ranges(
        self,
        work: ProjectedBatchDownload,
        reservation: ChunkReservation,
        registered_ranges: Sequence[RegisteredRange],
        executor: ThreadPoolExecutor,
    ) -> None:
        ranges_by_source: dict[int, list[RegisteredRange]] = defaultdict(list)
        for byte_range, registered_range in zip(
            work.download_ranges,
            registered_ranges,
            strict=True,
        ):
            ranges_by_source[byte_range.source_rank].append(registered_range)
        futures = []
        for source_rank, ranges in sorted(ranges_by_source.items()):
            submitted_ns = time.perf_counter_ns()
            futures.append(
                executor.submit(
                    self._download_file_ranges,
                    reservation,
                    work.batch_index,
                    source_rank,
                    tuple(ranges),
                    submitted_ns,
                )
            )
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException as error:
            self._record_download_error(error)
            for pending in futures:
                pending.cancel()
            raise

    def _record_download_error(self, error: BaseException) -> None:
        with self._download_error_lock:
            if self._download_error is not None:
                return
            self._download_error = error
        self._download_stop.set()
        self._publish_error(error)

    def _reserve_batch(
        self, pool: ChunkPool, owner_id: str, required_bytes: int
    ) -> ChunkReservation | None:
        deadline = time.monotonic() + self._config.progress_timeout_seconds
        while not self._download_stop.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out reserving shared memory for {owner_id}")
            try:
                return pool.reserve(
                    owner_id,
                    required_bytes,
                    timeout=min(remaining, 1.0),
                )
            except TimeoutError:
                continue
        return None

    def _download_file_ranges(
        self,
        reservation: ChunkReservation,
        batch_index: int,
        source_rank: int,
        registered_ranges: Sequence[RegisteredRange],
        submitted_ns: int,
    ) -> None:
        path = self._request.source_path_for_rank(source_rank)
        started_ns = time.perf_counter_ns()
        with self._download_worker_lock:
            self._active_download_workers += 1
            active_worker_count = self._active_download_workers
            self._peak_download_workers = max(
                self._peak_download_workers,
                active_worker_count,
            )
            active_worker_peak = self._peak_download_workers
        opened_ns: int | None = None
        read_finished_ns: int | None = None
        try:
            with self._request.storage.stream_read(
                path,
                ReadArgs(
                    direct_io=False,
                    pre_read_full_file=False,
                    timeout_us=max(
                        1,
                        int(self._config.progress_timeout_seconds * 1_000_000),
                    ),
                ),
            ) as reader:
                opened_ns = time.perf_counter_ns()
                for registered_range in registered_ranges:
                    reader.seek(registered_range.source_offset)
                    reservation.write_registered_progressively(registered_range, reader)
                read_finished_ns = time.perf_counter_ns()
        finally:
            finished_ns = time.perf_counter_ns()
            with self._download_worker_lock:
                self._active_download_workers -= 1
            self._metrics.record_stream_read(
                active_worker_count=active_worker_count,
                active_worker_peak=active_worker_peak,
                batch_index=batch_index,
                source_rank=source_rank,
                queue_ns=started_ns - submitted_ns,
                open_ns=(opened_ns or finished_ns) - started_ns,
                read_ns=(read_finished_ns or finished_ns) - (opened_ns or finished_ns),
                close_ns=finished_ns - (read_finished_ns or finished_ns),
                range_count=len(registered_ranges),
                byte_count=sum(item.length for item in registered_ranges),
                opened=opened_ns is not None,
            )

    def _wait_batch_registered(
        self,
        batch_index: int,
        active_node_ids: Sequence[NodeId],
    ) -> None:
        payload: bytes | None = None
        if self._topology.is_world_leader:
            for node_id in active_node_ids:
                self._take_global(
                    _global_node_tag(
                        f"registered/{batch_index}",
                        self._world_index(node_id),
                    )
                )
            payload = b"1"
        self._broadcast_bytes(f"scatter-start/{batch_index}", payload)

    def _scatter_batch(
        self,
        batch_index: int,
        targets: Sequence[TensorReadTarget],
        execution_plan: ProjectedExecutionPlan,
        pool: PinnedBufferPool,
        executor: ThreadPoolExecutor,
    ) -> None:
        plan = _plan_scatter_work(
            targets,
            batch_index,
            execution_plan,
            slot_bytes=pool.slot_bytes,
            enable_fast_scatter=self._config.enable_fast_scatter,
        )
        self._metrics.record_scatter_plan(batch_index, plan)
        work_iterator = iter(plan.work_items)
        pending: set[Future[None]] = set()
        for work_item in itertools.islice(
            work_iterator,
            self._config.fetch_workers,
        ):
            pending.add(executor.submit(self._scatter_work_item, work_item, pool))
        while pending:
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            try:
                for future in completed:
                    future.result()
            except Exception as error:
                self._download_stop.set()
                self._publish_error(error)
                for future in pending:
                    future.cancel()
                wait(pending)
                raise
            for work_item in itertools.islice(work_iterator, len(completed)):
                pending.add(executor.submit(self._scatter_work_item, work_item, pool))

    def _scatter_work_item(
        self,
        work_item: _ScatterWorkItem,
        pool: PinnedBufferPool,
    ) -> None:
        if not work_item.uses_grouped_fetch:
            if len(work_item.targets) != 1:
                raise ValueError("fallback scatter work must contain one target")
            self._scatter_target(work_item.targets[0], work_item.owner, pool)
            return
        is_local = work_item.owner == self._topology.node_id
        requests = tuple(
            RangeRequest(
                str(target.source_rank), source_range.offset, source_range.length
            )
            for target in work_item.targets
            for source_range in target.source_pattern.iter_ranges()
        )
        if len(requests) != work_item.range_count:
            raise RuntimeError("scatter work range count changed after planning")
        with pool.acquire(
            work_item.dense_nbytes,
            timeout=self._config.progress_timeout_seconds,
        ) as slot:
            destination = slot.view[: work_item.dense_nbytes]
            try:
                self._fetch_requests(work_item.owner, requests, destination)
            finally:
                destination.release()
            cursor = 0
            for target in work_item.targets:
                event = scatter_buffer_slice(
                    target,
                    self._request.target_state_dict,
                    slot,
                    offset_bytes=cursor,
                    non_blocking=self._non_blocking_scatter(target),
                )
                if event is not None:
                    slot.pending_cuda_events.append(event)
                cursor += target.source_pattern.dense_nbytes
            if cursor != work_item.dense_nbytes:
                raise RuntimeError("scatter work byte count changed after planning")
        if not is_local:
            with self._metrics_lock:
                self._network_bytes += work_item.dense_nbytes

    def _scatter_target(
        self,
        target: TensorReadTarget,
        owner: NodeId,
        pool: PinnedBufferPool,
    ) -> None:
        is_local = owner == self._topology.node_id
        if self._config.enable_fast_scatter and can_receive_directly_to_cpu(target):
            with direct_cpu_destination_buffer(
                target, self._request.target_state_dict
            ) as destination:
                self._fetch_target(owner, target, destination)
        else:
            self._fetch_and_scatter_buffered(owner, target, pool)
        if not is_local:
            with self._metrics_lock:
                self._network_bytes += target.source_pattern.dense_nbytes

    def _fetch_and_scatter_buffered(
        self,
        owner: NodeId,
        target: TensorReadTarget,
        pool: PinnedBufferPool,
    ) -> None:
        if target.source_pattern.dense_nbytes <= pool.slot_bytes:
            with pool.acquire(
                target.source_pattern.dense_nbytes,
                timeout=self._config.progress_timeout_seconds,
            ) as slot:
                self._fetch_target(owner, target, slot.view)
                event = scatter_buffer_slice(
                    target,
                    self._request.target_state_dict,
                    slot,
                    non_blocking=self._non_blocking_scatter(target),
                )
                if event is not None:
                    slot.pending_cuda_events.append(event)
            return
        self._fetch_and_scatter_chunks(owner, target, pool)

    def _fetch_and_scatter_chunks(
        self,
        owner: NodeId,
        target: TensorReadTarget,
        pool: PinnedBufferPool,
    ) -> None:
        if target.transpose_dims:
            self._fetch_and_scatter_transposed_chunks(owner, target, pool)
            return
        destination_ranges = iter(target.destination_pattern.iter_ranges())
        destination = next(destination_ranges, None)
        destination_consumed = 0
        max_elements = pool.slot_bytes // target.source_element_size_bytes
        if max_elements <= 0:
            raise MemoryError("receive slot cannot hold one source element")
        for source_range in target.source_pattern.iter_ranges():
            source_consumed = 0
            while source_consumed < source_range.length:
                if destination is None:
                    raise ValueError("source pattern exceeds destination pattern")
                destination_remaining = destination.length - destination_consumed
                source_remaining = source_range.length - source_consumed
                numel = min(
                    max_elements,
                    source_remaining // target.source_element_size_bytes,
                    destination_remaining // target.target_element_size_bytes,
                )
                if numel <= 0:
                    raise ValueError(
                        "source or destination pattern is not element-aligned"
                    )
                self._fetch_and_scatter_chunk(
                    owner,
                    target,
                    pool,
                    RangeRequest(
                        str(target.source_rank),
                        source_range.offset + source_consumed,
                        numel * target.source_element_size_bytes,
                    ),
                    destination.offset + destination_consumed,
                    numel,
                )
                source_consumed += numel * target.source_element_size_bytes
                destination_consumed += numel * target.target_element_size_bytes
                if destination_consumed == destination.length:
                    destination = next(destination_ranges, None)
                    destination_consumed = 0
        if destination is not None:
            raise ValueError("destination pattern exceeds source pattern")

    def _fetch_and_scatter_transposed_chunks(
        self,
        owner: NodeId,
        target: TensorReadTarget,
        pool: PinnedBufferPool,
    ) -> None:
        source_element_offset = 0
        for requests, request_bytes in _bounded_source_request_groups(
            target, pool.slot_bytes
        ):
            with pool.acquire(
                request_bytes,
                timeout=self._config.progress_timeout_seconds,
            ) as slot:
                view = slot.view[:request_bytes]
                try:
                    self._fetch_requests(owner, requests, view)
                    event = _scatter_transposed_flat_chunk(
                        target,
                        self._request.target_state_dict,
                        view,
                        source_element_offset=source_element_offset,
                        numel=request_bytes // target.source_element_size_bytes,
                        non_blocking=self._non_blocking_scatter(target),
                    )
                    if event is not None:
                        slot.pending_cuda_events.append(event)
                finally:
                    view.release()
            source_element_offset += request_bytes // target.source_element_size_bytes
        if source_element_offset != target.numel:
            raise ValueError(
                f"transposed source supplied {source_element_offset} elements, "
                f"expected {target.numel}"
            )

    def _fetch_and_scatter_chunk(
        self,
        owner: NodeId,
        target: TensorReadTarget,
        pool: PinnedBufferPool,
        request: RangeRequest,
        destination_offset_bytes: int,
        numel: int,
    ) -> None:
        with pool.acquire(
            request.length,
            timeout=self._config.progress_timeout_seconds,
        ) as slot:
            view = slot.view[: request.length]
            try:
                self._fetch_requests(owner, (request,), view)
                event = scatter_flat_buffer_chunk(
                    target,
                    self._request.target_state_dict,
                    view,
                    destination_offset_bytes=destination_offset_bytes,
                    numel=numel,
                    non_blocking=self._non_blocking_scatter(target),
                )
                if event is not None:
                    slot.pending_cuda_events.append(event)
            finally:
                view.release()

    def _non_blocking_scatter(self, target: TensorReadTarget) -> bool:
        return self._config.enable_fast_scatter and target.target_device.startswith(
            "cuda"
        )

    def _data_client(self, owner: NodeId) -> NodeClient:
        thread_id = threading.get_ident()
        evicted: NodeClient | None = None
        with self._data_clients_lock:
            thread_clients = self._data_clients.setdefault(
                thread_id,
                OrderedDict(),
            )
            client = thread_clients.pop(owner, None)
            if client is None:
                client = self._new_data_client(self._node_data_url(owner))
            thread_clients[owner] = client
            if len(thread_clients) > _MAX_DATA_CLIENTS_PER_THREAD:
                _, evicted = thread_clients.popitem(last=False)
        if evicted is not None:
            evicted.close()
        return client

    def _fetch_target(
        self,
        owner: NodeId,
        target: TensorReadTarget,
        destination: memoryview,
    ) -> None:
        cursor = 0
        ranges = target.source_pattern.iter_ranges()
        while True:
            group = tuple(itertools.islice(ranges, _MAX_FETCH_RANGES))
            if not group:
                return
            requests = tuple(
                RangeRequest(str(target.source_rank), item.offset, item.length)
                for item in group
            )
            length = sum(request.length for request in requests)
            view = destination[cursor : cursor + length]
            try:
                self._fetch_requests(owner, requests, view)
            finally:
                view.release()
            cursor += length

    def _fetch_requests(
        self,
        owner: NodeId,
        requests: Sequence[RangeRequest],
        destination: memoryview,
    ) -> None:
        is_local = owner == self._topology.node_id
        if is_local and self._topology.is_node_leader:
            self._read_local_index_with_error_check(requests, destination)
            return
        client = self._data_client(owner)
        if (
            is_local
            and self._local_shared_memory_visible
            and self._config.enable_fast_scatter
        ):
            self._resolve_local_with_error_check(client, requests, destination)
        else:
            self._fetch_with_error_check(client, requests, destination)

    def _read_local_index_with_error_check(
        self,
        requests: Sequence[RangeRequest],
        destination: memoryview,
    ) -> None:
        deadline = time.monotonic() + self._config.progress_timeout_seconds
        specs = tuple(request.to_spec() for request in requests)
        attempt = 0
        started_ns = time.perf_counter_ns()
        try:
            while True:
                if self._download_stop.is_set():
                    self._raise_if_failed()
                    raise RuntimeError("cooperative local fetch was cancelled")
                try:
                    with self._require_pool().index.acquire_many(specs) as lease:
                        _read_segment_slices(lease.ranges, destination)
                    return
                except RangeNotReadyError:
                    attempt += 1
                    self._raise_if_failed()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out resolving local byte ranges")
                    delay = min(
                        self._config.retry_backoff_seconds * (2 ** min(attempt - 1, 6)),
                        1.0,
                        remaining,
                    )
                    if delay > 0:
                        self._download_stop.wait(delay)
        finally:
            self._metrics.record_readiness_retry_window(
                elapsed_ns=time.perf_counter_ns() - started_ns,
                local=True,
                retry_count=attempt,
            )

    def _resolve_local_with_error_check(
        self,
        client: NodeClient,
        requests: Sequence[RangeRequest],
        destination: memoryview,
    ) -> None:
        deadline = time.monotonic() + self._config.progress_timeout_seconds
        retry_count = 0
        started_ns = time.perf_counter_ns()
        try:
            while True:
                if self._download_stop.is_set():
                    self._raise_if_failed()
                    raise RuntimeError("cooperative local fetch was cancelled")
                try:
                    resolved = client.resolve_ranges(
                        requests,
                        ready_timeout=_remaining_retry_window(deadline),
                    )
                    _read_segment_slices(resolved, destination)
                    return
                except RangeNotReadyError:
                    retry_count += 1
                    self._raise_if_failed()
                    if time.monotonic() >= deadline:
                        raise TimeoutError("timed out resolving local byte ranges")
        finally:
            self._metrics.record_readiness_retry_window(
                elapsed_ns=time.perf_counter_ns() - started_ns,
                local=True,
                retry_count=retry_count,
            )

    def _fetch_with_error_check(
        self,
        client: NodeClient,
        requests: Sequence[RangeRequest],
        destination: memoryview,
    ) -> None:
        deadline = time.monotonic() + self._config.progress_timeout_seconds
        retry_count = 0
        started_ns = time.perf_counter_ns()
        try:
            while True:
                if self._download_stop.is_set():
                    self._raise_if_failed()
                    raise RuntimeError("cooperative remote fetch was cancelled")
                try:
                    client.fetch_into(
                        requests,
                        destination,
                        ready_timeout=_remaining_retry_window(deadline),
                    )
                    return
                except RangeNotReadyError:
                    retry_count += 1
                    self._raise_if_failed()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "timed out waiting for cooperative byte ranges"
                        )
        finally:
            self._metrics.record_readiness_retry_window(
                elapsed_ns=time.perf_counter_ns() - started_ns,
                local=False,
                retry_count=retry_count,
            )

    def _complete_and_retire_batch(
        self,
        batch_index: int,
        active_node_ids: Sequence[NodeId],
    ) -> None:
        local_values = self._gather_local_rank_payloads(f"done/{batch_index}", b"1")
        if self._topology.is_node_leader:
            assert local_values is not None
            self._put_global(
                _global_node_tag(
                    f"done/{batch_index}", self._world_index(self._topology.node_id)
                ),
                b"1",
            )
        retire_payload: bytes | None = None
        if self._topology.is_world_leader:
            for node_id in active_node_ids:
                self._take_global(
                    _global_node_tag(
                        f"downloaded/{batch_index}",
                        self._world_index(node_id),
                    )
                )
            for node_index in range(len(self._world_nodes)):
                self._take_global(_global_node_tag(f"done/{batch_index}", node_index))
            retire_payload = b"1"
        self._broadcast_bytes(f"retire/{batch_index}", retire_payload)
        if self._topology.is_node_leader:
            with self._reservation_lock:
                reservation = self._reservations.pop(batch_index, None)
            if reservation is not None:
                reservation.retire(self._config.progress_timeout_seconds)
                self._inflight.release()

    def _exchange_metrics(
        self,
        *,
        started: float,
        target_count: int,
        demands: Sequence[FqnDemand] | None,
        batch_count: int,
    ) -> CooperativeLoadResult:
        local = _encode_control(
            {
                "elapsed_seconds": time.monotonic() - started,
                "network_bytes": self._network_bytes,
                "storage_bytes": self._storage_bytes,
                "target_count": target_count,
            }
        )
        rank_metrics = self._gather_local_rank_payloads("metrics", local)
        if self._topology.is_node_leader:
            assert rank_metrics is not None
            decoded = [
                _mapping(_decode_control(value), "rank metrics")
                for value in rank_metrics.values()
            ]
            node_metric = {
                "elapsed_seconds": max(
                    (float(item["elapsed_seconds"]) for item in decoded),
                    default=0.0,
                ),
                "network_bytes": sum(int(item["network_bytes"]) for item in decoded),
                "storage_bytes": sum(int(item["storage_bytes"]) for item in decoded),
                "target_count": sum(int(item["target_count"]) for item in decoded),
            }
            self._put_global(
                _global_node_tag("metrics", self._world_index(self._topology.node_id)),
                _encode_control(node_metric),
            )
        encoded_result: bytes | None = None
        if self._topology.is_world_leader:
            if demands is None:
                raise RuntimeError("world leader has no global byte demands")
            node_metrics = [
                _mapping(
                    _decode_control(
                        self._take_global(_global_node_tag("metrics", index))
                    ),
                    "node metrics",
                )
                for index in range(len(self._world_nodes))
            ]
            slowest = max(
                (float(item["elapsed_seconds"]) for item in node_metrics),
                default=0.0,
            )
            result = CooperativeLoadResult(
                unique_source_bytes=sum(demand.total_bytes for demand in demands),
                storage_bytes_read=sum(
                    int(item["storage_bytes"]) for item in node_metrics
                ),
                network_bytes_received=sum(
                    int(item["network_bytes"]) for item in node_metrics
                ),
                target_count=sum(int(item["target_count"]) for item in node_metrics),
                batch_count=batch_count,
                elapsed_seconds=slowest,
                slowest_rank_seconds=slowest,
            )
            encoded_result = _encode_control(asdict(result))
        payload = _mapping(
            _decode_control(self._broadcast_bytes("load-result", encoded_result)),
            "load result",
        )
        return CooperativeLoadResult(
            unique_source_bytes=int(payload["unique_source_bytes"]),
            storage_bytes_read=int(payload["storage_bytes_read"]),
            network_bytes_received=int(payload["network_bytes_received"]),
            target_count=int(payload["target_count"]),
            batch_count=int(payload["batch_count"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            slowest_rank_seconds=float(payload["slowest_rank_seconds"]),
        )

    def _finish_download_pipeline(self, *, timeout: float) -> None:
        thread = self._download_thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            raise TimeoutError("cooperative download pipeline did not stop")
        self._raise_if_failed()

    def _start_error_monitor(self) -> None:
        if self._error_monitor_started:
            raise RuntimeError("cooperative error monitor is already started")
        thread = threading.Thread(
            target=self._monitor_remote_errors,
            name="cooperative-error-monitor",
            daemon=True,
        )
        self._error_monitor_thread = thread
        try:
            thread.start()
        except BaseException:
            self._error_monitor_thread = None
            raise
        self._error_monitor_started = True

    def _monitor_remote_errors(self) -> None:
        try:
            while not self._error_monitor_stop.is_set():
                remote: str | None
                if self._topology.is_node_leader:
                    local = (
                        self._server.get_error(0) if self._server is not None else None
                    )
                    self._forward_local_error_if_needed(local)
                    remote = self._sample_error_url(
                        self._require_coordinator_control_url()
                    )
                    if remote is None:
                        remote = self._poll_bootstrap_error()
                else:
                    remote = self._sample_error_url(
                        self._node_control_urls[self._topology.node_id]
                    )
                if self._error_monitor_stop.is_set():
                    return
                if remote is not None:
                    self._cache_remote_error(remote)
                    self._download_stop.set()
                    return
                if self._error_monitor_stop.wait(
                    self._error_monitor_interval_seconds()
                ):
                    return
        except Exception as error:
            with self._error_monitor_lock:
                if self._error_monitor_failure is None:
                    self._error_monitor_failure = error
            self._download_stop.set()

    def _sample_error_url(self, url: str) -> str | None:
        client = self._new_error_monitor_client(url)
        with self._error_monitor_lock:
            stopping = self._error_monitor_stop.is_set()
            if not stopping:
                self._error_monitor_client = client
        if stopping:
            client.close()
            return None
        remote: str | None = None
        try:
            try:
                remote = client.get_error(timeout=0)
            except TransportError:
                logger.debug("cooperative error-monitor request failed", exc_info=True)
        finally:
            try:
                client.close()
            finally:
                with self._error_monitor_lock:
                    if self._error_monitor_client is client:
                        self._error_monitor_client = None
        return remote

    def _error_monitor_interval_seconds(self) -> float:
        jitter_fraction = (self._topology.global_rank * 0.6180339887498949) % 1
        return (
            _ERROR_MONITOR_BASE_INTERVAL_SECONDS
            + _ERROR_MONITOR_JITTER_SECONDS * jitter_fraction
        )

    def _poll_bootstrap_error(self) -> str | None:
        try:
            return self._request.rendezvous.get_error(self._namespace, timeout=0)
        except Exception:
            logger.debug("cooperative rendezvous error check failed", exc_info=True)
            return None

    def _forward_local_error_if_needed(self, message: str | None) -> None:
        if message is None or message == self._local_error_forwarded:
            return
        if self._forward_node_error(message):
            self._local_error_forwarded = message

    def _forward_node_error(self, message: str) -> bool:
        message = _bounded_error_message(message)
        succeeded = False
        coordinator = self._control_coordinator
        if coordinator is not None:
            try:
                coordinator.publish_error(message)
                succeeded = True
            except Exception:
                logger.debug("failed to forward node error", exc_info=True)
        try:
            self._request.rendezvous.publish_error(self._namespace, message)
            succeeded = True
        except Exception:
            logger.debug("failed to publish rendezvous error", exc_info=True)
        return succeeded

    def _cache_remote_error(self, message: str) -> None:
        message = _bounded_error_message(message)
        with self._error_monitor_lock:
            if self._remote_error is None:
                self._remote_error = message
                first_error = True
            else:
                first_error = False
        if first_error and self._server is not None:
            try:
                self._server.publish_error(message)
            except Exception:
                logger.debug("failed to propagate error to local server", exc_info=True)

    def _raise_if_failed(self) -> None:
        with self._download_error_lock:
            download_error = self._download_error
        if download_error is not None:
            raise RuntimeError(
                "cooperative download pipeline failed"
            ) from download_error
        with self._error_monitor_lock:
            remote_error = self._remote_error
            monitor_failure = self._error_monitor_failure
        if remote_error is not None:
            _raise_remote_error(remote_error)
        if monitor_failure is not None:
            raise RuntimeError("cooperative error monitor failed") from monitor_failure
        if self._error_monitor_started:
            return
        for client in (self._local_control, self._control_coordinator):
            if client is None:
                continue
            remote = client.get_error(timeout=0)
            if remote is not None:
                self._cache_remote_error(remote)
                _raise_remote_error(remote)
        remote = self._request.rendezvous.get_error(self._namespace, timeout=0)
        if remote is not None:
            self._cache_remote_error(remote)
            _raise_remote_error(remote)

    def _publish_error(self, error: BaseException) -> None:
        message = _error_message(self._topology.global_rank, error)
        for client in (self._local_control, self._control_coordinator):
            if client is None:
                continue
            try:
                client.publish_error(message)
            except Exception:
                logger.debug("failed to publish cooperative error", exc_info=True)
        if self._server is not None:
            try:
                self._server.publish_error(message)
            except Exception:
                logger.debug("failed to publish local cooperative error", exc_info=True)
        try:
            self._request.rendezvous.publish_error(self._namespace, message)
        except Exception:
            logger.debug("failed to publish rendezvous error", exc_info=True)

    def _gather_local_rank_payloads(
        self, prefix: str, payload: bytes
    ) -> dict[int, bytes] | None:
        self._put_local(_local_rank_tag(prefix, self._topology.global_rank), payload)
        if not self._topology.is_node_leader:
            return None
        return {
            rank: self._take_local(_local_rank_tag(prefix, rank))
            for rank in self._topology.node_ranks
        }

    def _broadcast_bytes(self, prefix: str, payload: bytes | None) -> bytes:
        global_tag = _global_blob_tag(prefix)
        local_tag = _local_blob_tag(prefix)
        if self._topology.is_world_leader:
            if payload is None:
                raise ValueError("world leader must supply a broadcast payload")
            self._put_global(global_tag, payload)
        if self._topology.is_node_leader:
            node_payload = self._wait_global(global_tag)
            self._put_local(local_tag, node_payload)
        value = self._wait_local(local_tag)
        self._put_local(
            _local_rank_tag(f"broadcast-ack/{prefix}", self._topology.global_rank),
            b"1",
        )
        if self._topology.is_node_leader:
            for rank in self._topology.node_ranks:
                self._take_local(_local_rank_tag(f"broadcast-ack/{prefix}", rank))
            self._delete_local(local_tag)
            self._put_global(
                _global_node_tag(
                    f"broadcast-ack/{prefix}",
                    self._world_index(self._topology.node_id),
                ),
                b"1",
            )
        if self._topology.is_world_leader:
            for node_index in range(len(self._world_nodes)):
                self._take_global(
                    _global_node_tag(f"broadcast-ack/{prefix}", node_index)
                )
            self._delete_global(global_tag)
        return value

    def _scatter_node_payloads(
        self,
        prefix: str,
        payloads: Mapping[NodeId, bytes] | None,
    ) -> bytes:
        node_index = self._world_index(self._topology.node_id)
        if self._topology.is_world_leader:
            if payloads is None or set(payloads) != set(self._world_nodes):
                raise ValueError("world leader must supply one payload per node")
            for index, node_id in enumerate(self._world_nodes):
                self._put_global(
                    _global_node_tag(prefix, index),
                    payloads[node_id],
                )
        local_tag = _local_blob_tag(prefix)
        if self._topology.is_node_leader:
            node_payload = self._take_global(_global_node_tag(prefix, node_index))
            self._put_local(local_tag, node_payload)
        value = self._wait_local(local_tag)
        self._put_local(
            _local_rank_tag(f"scatter-ack/{prefix}", self._topology.global_rank),
            b"1",
        )
        if self._topology.is_node_leader:
            for rank in self._topology.node_ranks:
                self._take_local(_local_rank_tag(f"scatter-ack/{prefix}", rank))
            self._delete_local(local_tag)
        return value

    def _put_global(self, tag: str, value: bytes) -> None:
        self._require_control_coordinator().put_blob(tag, value)

    def _wait_global(self, tag: str) -> bytes:
        return self._wait_client_blob(self._require_control_coordinator(), tag)

    def _take_global(self, tag: str) -> bytes:
        value = self._wait_global(tag)
        self._delete_global(tag)
        return value

    def _delete_global(self, tag: str) -> None:
        self._delete_client_blob(self._require_control_coordinator(), tag)

    def _put_local(self, tag: str, value: bytes) -> None:
        self._require_local_control().put_blob(tag, value)

    def _wait_local(self, tag: str) -> bytes:
        return self._wait_client_blob(self._require_local_control(), tag)

    def _take_local(self, tag: str) -> bytes:
        value = self._wait_local(tag)
        self._delete_local(tag)
        return value

    def _delete_local(self, tag: str) -> None:
        self._delete_client_blob(self._require_local_control(), tag)

    def _wait_client_blob(self, client: NodeClient, tag: str) -> bytes:
        deadline = time.monotonic() + self._config.progress_timeout_seconds
        while True:
            value = client.get_blob(tag, timeout=1.0)
            if value is not None:
                return value
            self._raise_if_failed()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for control blob {tag!r}")

    def _delete_client_blob(self, client: NodeClient, tag: str) -> None:
        try:
            client.delete_blob(tag)
        except TransportError:
            logger.warning("failed to reclaim control blob %r", tag, exc_info=True)

    def _wait_bootstrap_blob(self, tag: str) -> bytes:
        deadline = time.monotonic() + self._config.progress_timeout_seconds
        while True:
            value = self._request.rendezvous.get_blob(self._namespace, tag, timeout=1.0)
            if value is not None:
                return value
            self._raise_if_failed()
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for bootstrap blob {tag!r}")

    def _cleanup(self) -> None:
        with self._metrics.latency(
            "cleanup.total",
            is_node_leader=self._topology.is_node_leader,
        ):
            self._cleanup_impl()

    def _cleanup_impl(self) -> None:
        self._download_stop.set()
        self._remove_visibility_probe_for_cleanup()
        for _ in range(self._config.max_inflight_batches):
            self._inflight.release()
        thread = self._download_thread
        thread_alive = self._join_download_thread_for_cleanup(thread)
        cleanup_error = self._close_control_resources()
        if thread_alive:
            assert thread is not None
            server, self._server = self._server, None
            pool, self._pool = self._pool, None
            lease, self._pool_lease = self._pool_lease, None
            threading.Thread(
                target=_cleanup_after_download,
                args=(
                    thread,
                    server,
                    pool,
                    lease,
                    self._config.progress_timeout_seconds,
                ),
                name="cooperative-cleanup-reaper",
                daemon=True,
            ).start()
            if cleanup_error is not None:
                raise cleanup_error
            return
        resource_error = self._close_session_resources(
            allow_cache_release=(
                self._completed_successfully and cleanup_error is None
            ),
        )
        combined = _combine_errors(cleanup_error, resource_error)
        if combined is not None:
            raise combined

    def _remove_visibility_probe_for_cleanup(self) -> None:
        with self._metrics.latency("cleanup.remove_visibility_probe"):
            try:
                self._remove_visibility_probe()
            except OSError:
                logger.warning("failed to remove visibility probe", exc_info=True)

    def _join_download_thread_for_cleanup(
        self,
        thread: threading.Thread | None,
    ) -> bool:
        with self._metrics.latency(
            "cleanup.download_thread_join",
            thread_present=thread is not None,
            thread_alive_before=thread is not None and thread.is_alive(),
        ) as fields:
            if thread is not None and thread.is_alive():
                thread.join(timeout=min(5.0, self._config.progress_timeout_seconds))
            fields["thread_alive_after"] = thread is not None and thread.is_alive()
        return thread is not None and thread.is_alive()

    def _close_control_resources(self) -> Exception | None:
        errors: list[Exception] = []
        try:
            with self._metrics.latency("cleanup.stop_error_monitor"):
                self._stop_error_monitor()
        except Exception as error:
            errors.append(error)
        try:
            with self._metrics.latency("cleanup.close_clients"):
                self._close_clients()
        except Exception as error:
            errors.append(error)
        return _combine_errors(*errors)

    def _close_session_resources(
        self, *, allow_cache_release: bool
    ) -> Exception | None:
        server, self._server = self._server, None
        pool, self._pool = self._pool, None
        lease, self._pool_lease = self._pool_lease, None
        server_error = _close_node_server(server, metrics=self._metrics)
        if pool is None:
            return server_error
        if (
            lease is not None
            and lease.retainable
            and allow_cache_release
            and server_error is None
        ):
            if not pool.reuse_supported:
                cleanup_error = _cleanup_chunk_pool(
                    pool,
                    self._config.progress_timeout_seconds,
                    metrics=self._metrics,
                )
                discard_error = _discard_pool_lease(lease)
                return _combine_errors(cleanup_error, discard_error)
            try:
                with self._metrics.latency(
                    "cleanup.pool_cache_release",
                    cache_reused=lease.reused,
                    capacity_bytes=pool.capacity_bytes,
                    chunk_count=pool.chunk_count,
                    active_reservation_count=pool.active_reservation_count,
                ):
                    pool.index.assert_quiescent()
                    pool.prepare_for_reuse()
                    lease.release()
                return None
            except Exception as error:
                cleanup_error = _cleanup_chunk_pool(
                    pool,
                    self._config.progress_timeout_seconds,
                    metrics=self._metrics,
                )
                discard_error = _discard_pool_lease(lease)
                return _combine_errors(error, cleanup_error, discard_error)
        cleanup_error = _cleanup_chunk_pool(
            pool,
            self._config.progress_timeout_seconds,
            metrics=self._metrics,
        )
        discard_error = None if lease is None else _discard_pool_lease(lease)
        return _combine_errors(server_error, cleanup_error, discard_error)

    def _stop_error_monitor(self) -> None:
        self._error_monitor_stop.set()
        with self._error_monitor_lock:
            client = self._error_monitor_client
        errors: list[Exception] = []
        if client is not None:
            try:
                client.close()
            except Exception as error:
                errors.append(error)
        thread = self._error_monitor_thread
        if thread is not None and thread.is_alive():
            thread.join(
                timeout=min(
                    _ERROR_MONITOR_MAX_REQUEST_SECONDS + 1,
                    max(1.0, self._config.progress_timeout_seconds),
                )
            )
            if thread.is_alive():
                errors.append(TimeoutError("cooperative error monitor did not stop"))
        if thread is None or not thread.is_alive():
            with self._error_monitor_lock:
                self._error_monitor_client = None
            self._error_monitor_thread = None
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise RuntimeError(
                "multiple cooperative error-monitor shutdown failures: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]

    def _close_clients(self) -> None:
        clients = [self._local_control, self._control_coordinator]
        self._local_control = None
        self._control_coordinator = None
        with self._data_clients_lock:
            for thread_clients in self._data_clients.values():
                clients.extend(thread_clients.values())
            self._data_clients.clear()
        seen: set[int] = set()
        errors: list[Exception] = []
        for client in clients:
            if client is None or id(client) in seen:
                continue
            seen.add(id(client))
            try:
                client.close()
            except Exception as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise RuntimeError(
                "multiple cooperative client shutdown failures: "
                + "; ".join(f"{type(error).__name__}: {error}" for error in errors)
            ) from errors[0]

    def _new_control_client(self, url: str) -> NodeClient:
        return self._new_transport_client(url)

    def _new_data_client(self, url: str) -> NodeClient:
        return self._new_transport_client(url)

    def _new_error_monitor_client(self, url: str) -> NodeClient:
        return NodeClient(
            url,
            protocol_version=_PROTOCOL_VERSION,
            load_token=self._namespace.load_token,
            request_timeout=min(
                _ERROR_MONITOR_MAX_REQUEST_SECONDS,
                self._config.progress_timeout_seconds,
            ),
            max_attempts=1,
            retry_delay=0,
            max_control_body_bytes=_CONTROL_BYTES_LIMIT,
        )

    def _new_transport_client(self, url: str) -> NodeClient:
        return NodeClient(
            url,
            protocol_version=_PROTOCOL_VERSION,
            load_token=self._namespace.load_token,
            request_timeout=min(30.0, self._config.progress_timeout_seconds),
            max_attempts=self._config.retry_attempts,
            retry_delay=self._config.retry_backoff_seconds,
            max_control_body_bytes=_CONTROL_BYTES_LIMIT,
        )

    def _world_index(self, node_id: NodeId) -> int:
        try:
            return self._world_nodes.index(node_id)
        except ValueError as error:
            raise ValueError(
                f"node {node_id!r} is outside this coordination world"
            ) from error

    def _node_members(self, node_id: NodeId) -> tuple[int, ...]:
        for node in self._topology.nodes:
            if node.node_id == node_id:
                return node.ranks
        raise ValueError(f"node {node_id!r} is absent from the topology")

    def _require_control_coordinator(self) -> NodeClient:
        if self._control_coordinator is None:
            raise RuntimeError("coordinator control client is unavailable")
        return self._control_coordinator

    def _require_local_control(self) -> NodeClient:
        if self._local_control is None:
            raise RuntimeError("local control client is unavailable")
        return self._local_control

    def _require_coordinator_control_url(self) -> str:
        if self._coordinator_control_url is None:
            raise RuntimeError("coordinator control URL is unavailable")
        return self._coordinator_control_url

    def _require_pool(self) -> ChunkPool:
        if self._pool is None:
            raise RuntimeError("local shared-memory pool is unavailable")
        return self._pool

    def _node_data_url(self, node_id: NodeId) -> str:
        try:
            return self._node_data_urls[node_id]
        except KeyError as error:
            raise ValueError(f"node {node_id!r} has no data endpoint") from error


def _shared_memory_pool_spec(
    request: CooperativeLoadRequest,
    config: CooperativeLoadConfig,
) -> _ChunkPoolSpec:
    requested_capacity = request.shared_memory_capacity_bytes
    if requested_capacity is None:
        requested_capacity = recommended_capacity_bytes(
            fraction=config.shared_memory_fraction,
            directory=request.shared_memory_directory,
        )
    capacity = min(
        requested_capacity,
        config.batch_target_bytes * config.max_inflight_batches,
    )
    if capacity <= 0:
        raise MemoryError("no shared-memory capacity is available")
    return _ChunkPoolSpec(
        directory=os.path.abspath(os.fspath(request.shared_memory_directory)),
        capacity_bytes=capacity,
        chunk_bytes=min(config.shared_memory_chunk_bytes, capacity),
    )


def _make_receive_pool(config: CooperativeLoadConfig) -> PinnedBufferPool:
    return PinnedBufferPool(
        slot_bytes=config.pinned_buffer_bytes,
        slot_count=config.pinned_buffer_count,
    )


def _plan_scatter_work(
    targets: Sequence[TensorReadTarget],
    batch_index: int,
    execution_plan: ProjectedExecutionPlan,
    *,
    slot_bytes: int,
    enable_fast_scatter: bool,
) -> _ScatterBatchPlan:
    if slot_bytes <= 0:
        raise ValueError("slot_bytes must be positive")

    open_groups: dict[
        tuple[NodeId, int, int],
        _OpenScatterGroup,
    ] = {}
    work_items: list[_ScatterWorkItem] = []

    def flush(key: tuple[NodeId, int, int]) -> None:
        group = open_groups.pop(key, None)
        if group is None or not group.indexed_targets:
            return
        grouped_targets = tuple(target for _, target in group.indexed_targets)
        work_items.append(
            _ScatterWorkItem(
                owner=key[0],
                source_rank=key[1],
                targets=grouped_targets,
                dense_nbytes=group.dense_nbytes,
                range_count=group.range_count,
                uses_grouped_fetch=True,
                first_target_index=group.indexed_targets[0][0],
                predicted_download_frontier_bytes=(
                    group.predicted_download_frontier_bytes
                ),
            )
        )

    candidates = _scatter_target_candidates(
        targets,
        batch_index,
        execution_plan,
    )
    for candidate in candidates:
        target = candidate.target
        dense_nbytes = target.source_pattern.dense_nbytes
        range_count = target.source_pattern.range_count
        can_group = (
            dense_nbytes <= slot_bytes
            and range_count <= _MAX_FETCH_RANGES
            and not (enable_fast_scatter and can_receive_directly_to_cpu(target))
        )
        if not can_group:
            work_items.append(
                _ScatterWorkItem(
                    owner=candidate.owner,
                    source_rank=target.source_rank,
                    targets=(target,),
                    dense_nbytes=dense_nbytes,
                    range_count=range_count,
                    uses_grouped_fetch=False,
                    first_target_index=candidate.target_index,
                    predicted_download_frontier_bytes=(
                        candidate.predicted_download_frontier_bytes
                    ),
                )
            )
            continue

        # Keep each target's source buffer naturally aligned for
        # torch.frombuffer. A group is source-local so all of its ranges become
        # ready along one download stream rather than at unrelated frontiers.
        key = (
            candidate.owner,
            target.source_rank,
            target.source_element_size_bytes,
        )
        group = open_groups.get(key)
        if (
            group is not None
            and group.indexed_targets
            and (
                group.dense_nbytes + dense_nbytes > slot_bytes
                or group.range_count + range_count > _MAX_FETCH_RANGES
            )
        ):
            flush(key)
            group = None
        if group is None:
            group = _OpenScatterGroup([])
            open_groups[key] = group
        group.indexed_targets.append((candidate.target_index, target))
        group.dense_nbytes += dense_nbytes
        group.range_count += range_count
        group.predicted_download_frontier_bytes = max(
            group.predicted_download_frontier_bytes,
            candidate.predicted_download_frontier_bytes,
        )

    for key in tuple(open_groups):
        flush(key)
    work_items.sort(
        key=lambda item: (
            item.predicted_download_frontier_bytes,
            item.source_rank,
            item.first_target_index,
        )
    )

    grouped_items = tuple(item for item in work_items if item.uses_grouped_fetch)
    coalesced_items = tuple(item for item in grouped_items if len(item.targets) > 1)
    return _ScatterBatchPlan(
        work_items=tuple(work_items),
        target_count=len(candidates),
        source_range_count=sum(
            candidate.target.source_pattern.range_count for candidate in candidates
        ),
        coalesced_group_count=len(coalesced_items),
        coalesced_target_count=sum(len(item.targets) for item in coalesced_items),
        buffered_fetch_count=len(grouped_items),
        fallback_target_count=sum(
            len(item.targets) for item in work_items if not item.uses_grouped_fetch
        ),
        max_group_bytes=max(
            (item.dense_nbytes for item in grouped_items),
            default=0,
        ),
        max_group_ranges=max(
            (item.range_count for item in grouped_items),
            default=0,
        ),
        max_group_targets=max(
            (len(item.targets) for item in grouped_items),
            default=0,
        ),
        max_predicted_download_frontier_bytes=max(
            (item.predicted_download_frontier_bytes for item in work_items),
            default=0,
        ),
        source_group_count=len({(item.owner, item.source_rank) for item in work_items}),
    )


def _scatter_target_candidates(
    targets: Sequence[TensorReadTarget],
    batch_index: int,
    execution_plan: ProjectedExecutionPlan,
) -> tuple[_ScatterTargetCandidate, ...]:
    candidates: list[_ScatterTargetCandidate] = []
    for target_index, target in enumerate(targets):
        if not target.source_pattern.dense_nbytes:
            continue
        schedule = execution_plan.schedule_for(batch_index, target.source_rank)
        candidates.append(
            _ScatterTargetCandidate(
                target_index=target_index,
                target=target,
                owner=execution_plan.owner_for(target.source_rank),
                predicted_download_frontier_bytes=_predicted_download_frontier(
                    target,
                    schedule,
                ),
            )
        )
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                item.predicted_download_frontier_bytes,
                item.target.source_rank,
                item.target_index,
            ),
        )
    )


def _predicted_download_frontier(
    target: TensorReadTarget,
    schedule: ProjectedSourceSchedule,
) -> int:
    frontier = 0
    for requested in target.source_pattern.iter_ranges():
        current = requested.offset
        requested_end = requested.offset + requested.length
        range_index = bisect_right(schedule.starts, current) - 1
        while current < requested_end:
            if range_index < 0 or range_index >= len(schedule.ranges):
                raise ValueError(
                    f"source range [{requested.offset}, {requested_end}) is not "
                    f"covered by source rank {target.source_rank}'s download schedule"
                )
            downloaded = schedule.ranges[range_index]
            if not downloaded.offset <= current < downloaded.end:
                raise ValueError(
                    f"source range [{requested.offset}, {requested_end}) is not "
                    f"covered by source rank {target.source_rank}'s download schedule"
                )
            current = min(requested_end, downloaded.end)
            bytes_before_range = (
                schedule.cumulative_bytes[range_index - 1] if range_index else 0
            )
            frontier = max(
                frontier,
                bytes_before_range + current - downloaded.offset,
            )
            range_index += 1
    return frontier


def _bounded_source_request_groups(
    target: TensorReadTarget,
    slot_bytes: int,
) -> Iterable[tuple[tuple[RangeRequest, ...], int]]:
    element_bytes = target.source_element_size_bytes
    payload_limit = slot_bytes - (slot_bytes % element_bytes)
    if payload_limit <= 0:
        raise MemoryError("receive slot cannot hold one source element")
    slab_bytes = prod(target.source_slice_shape[1:]) * element_bytes
    if slab_bytes <= payload_limit:
        payload_limit -= payload_limit % slab_bytes

    requests: list[RangeRequest] = []
    payload_bytes = 0
    for source_range in target.source_pattern.iter_ranges():
        if source_range.length % element_bytes:
            raise ValueError("source byte pattern is not element-aligned")
        consumed = 0
        while consumed < source_range.length:
            if requests and (
                payload_bytes == payload_limit or len(requests) == _MAX_FETCH_RANGES
            ):
                yield tuple(requests), payload_bytes
                requests = []
                payload_bytes = 0
            available = payload_limit - payload_bytes
            length = min(source_range.length - consumed, available)
            length -= length % element_bytes
            if length <= 0:
                raise MemoryError("receive slot cannot hold one source element")
            requests.append(
                RangeRequest(
                    str(target.source_rank),
                    source_range.offset + consumed,
                    length,
                )
            )
            payload_bytes += length
            consumed += length
    if requests:
        yield tuple(requests), payload_bytes


def _scatter_transposed_flat_chunk(
    target: TensorReadTarget,
    target_state_dict: Mapping[str, Any],
    source_buffer: object,
    *,
    source_element_offset: int,
    numel: int,
    non_blocking: bool,
) -> Any | None:
    destination = target_state_dict.get(target.target_fqn)
    if not isinstance(destination, torch.Tensor):
        raise TypeError(f"target {target.target_fqn!r} is not a torch.Tensor")
    if source_element_offset < 0 or numel <= 0:
        raise ValueError("transposed chunk offsets and lengths must be positive")
    if source_element_offset + numel > target.numel:
        raise ValueError("transposed source chunk exceeds its target")

    storage_offset = (
        target.destination_pattern.start_offset // target.target_element_size_bytes
    )
    destination_view = torch.as_strided(
        destination,
        size=target.target_slice_shape,
        stride=destination.stride(),
        storage_offset=storage_offset,
    )
    permuted_source_shape = tuple(
        target.source_slice_shape[index] for index in target.transpose_dims
    )
    if tuple(destination_view.shape) != permuted_source_shape:
        reshaped = destination_view.reshape(permuted_source_shape)
        if (
            reshaped.untyped_storage().data_ptr()
            != destination.untyped_storage().data_ptr()
        ):
            raise ValueError(
                "transposed target reshape would allocate temporary storage"
            )
        destination_view = reshaped
    inverse_transpose = tuple(
        target.transpose_dims.index(dimension)
        for dimension in range(len(target.transpose_dims))
    )
    source_order_destination = destination_view.permute(inverse_transpose)
    source = torch.frombuffer(
        source_buffer,
        dtype=_resolve_torch_dtype(target.source_dtype),
        count=numel,
    )
    _copy_flat_chunk_to_tensor_view(
        source_order_destination,
        source,
        source_element_offset,
        non_blocking=non_blocking,
    )
    if non_blocking and destination.is_cuda:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(destination.device))
        return event
    return None


def _copy_flat_chunk_to_tensor_view(
    destination: torch.Tensor,
    source: torch.Tensor,
    destination_offset: int,
    *,
    non_blocking: bool,
) -> None:
    if destination.is_contiguous():
        destination.view(-1).narrow(0, destination_offset, source.numel()).copy_(
            source,
            non_blocking=non_blocking,
        )
        return
    if destination.ndim == 0:
        destination.copy_(source.reshape(()), non_blocking=non_blocking)
        return

    slab_elements = prod(destination.shape[1:])
    if (
        slab_elements > 0
        and destination_offset % slab_elements == 0
        and source.numel() % slab_elements == 0
    ):
        slab_count = source.numel() // slab_elements
        destination.narrow(
            0,
            destination_offset // slab_elements,
            slab_count,
        ).copy_(
            source.reshape((slab_count, *destination.shape[1:])),
            non_blocking=non_blocking,
        )
        return

    row_length = int(destination.shape[-1])
    source_cursor = 0
    destination_cursor = destination_offset
    prefix_shape = tuple(int(size) for size in destination.shape[:-1])
    while source_cursor < source.numel():
        row_index, column = divmod(destination_cursor, row_length)
        count = min(source.numel() - source_cursor, row_length - column)
        prefix = _unravel_index(row_index, prefix_shape)
        row = destination[prefix] if prefix else destination
        row.narrow(0, column, count).copy_(
            source.narrow(0, source_cursor, count),
            non_blocking=non_blocking,
        )
        source_cursor += count
        destination_cursor += count


def _unravel_index(index: int, shape: Sequence[int]) -> tuple[int, ...]:
    coordinates = [0] * len(shape)
    for dimension in range(len(shape) - 1, -1, -1):
        index, coordinates[dimension] = divmod(index, shape[dimension])
    if index:
        raise ValueError("flat tensor index exceeds destination shape")
    return tuple(coordinates)


def _read_segment_slices(
    resolved: Sequence[Sequence[SegmentSlice]], destination: memoryview
) -> None:
    cursor = 0
    for logical_range in resolved:
        for segment in logical_range:
            view = destination[cursor : cursor + segment.length]
            try:
                with segment.path.open("rb", buffering=0) as source:
                    source.seek(segment.file_offset)
                    readinto_exact(source, view)
            finally:
                view.release()
            cursor += segment.length
    if cursor != len(destination):
        raise EOFError(
            f"resolved local ranges produced {cursor} bytes, expected {len(destination)}"
        )


def _remaining_retry_window(deadline: float) -> float:
    return min(30.0, max(0.0, deadline - time.monotonic()))


def _cleanup_after_download(
    thread: threading.Thread,
    server: NodeServer | None,
    pool: ChunkPool | None,
    lease: _ChunkPoolLease | None,
    timeout: float,
) -> None:
    thread.join()
    resource_error = _close_local_resources(server, pool, timeout)
    discard_error = None if lease is None else _discard_pool_lease(lease)
    error = _combine_errors(resource_error, discard_error)
    if error is not None:
        logger.error(
            "deferred cooperative cleanup failed",
            exc_info=(type(error), error, error.__traceback__),
        )


def _combine_errors(*errors: Exception | None) -> Exception | None:
    present = tuple(error for error in errors if error is not None)
    if not present:
        return None
    if len(present) == 1:
        return present[0]
    return RuntimeError(
        "multiple cooperative cleanup failures: "
        + "; ".join(f"{type(error).__name__}: {error}" for error in present)
    )


def _close_local_resources(
    server: NodeServer | None,
    pool: ChunkPool | None,
    timeout: float,
    *,
    metrics: _Metrics | None = None,
) -> Exception | None:
    return _combine_errors(
        _close_node_server(server, metrics=metrics),
        _cleanup_chunk_pool(pool, timeout, metrics=metrics),
    )


def _close_node_server(
    server: NodeServer | None,
    *,
    metrics: _Metrics | None = None,
) -> Exception | None:
    if server is None:
        return None
    try:
        if metrics is None:
            server.close()
        else:
            with metrics.latency("cleanup.server_close"):
                server.close()
    except Exception as error:
        return error
    return None


def _cleanup_chunk_pool(
    pool: ChunkPool | None,
    timeout: float,
    *,
    metrics: _Metrics | None = None,
) -> Exception | None:
    if pool is None:
        return None
    try:
        if metrics is None:
            pool.cleanup(timeout=timeout)
        else:
            with metrics.latency(
                "cleanup.pool_cleanup",
                capacity_bytes=pool.capacity_bytes,
                chunk_count=pool.chunk_count,
                active_reservation_count=pool.active_reservation_count,
            ):
                pool.cleanup(timeout=timeout)
    except Exception as error:
        return error
    return None


def _discard_pool_lease(lease: _ChunkPoolLease) -> Exception | None:
    try:
        lease.discard()
    except Exception as error:
        return error
    return None


def dedupe_aliased_targets(
    load_plan: Mapping[str, Sequence[LoadPlan]],
    target_state_dict: Mapping[str, Any],
) -> tuple[dict[str, Sequence[LoadPlan]], tuple[tuple[str, str], ...]]:
    """Keep the last planned name for tensors with identical destination views."""

    aliases = _tensor_view_aliases(
        tuple(fqn for fqn in target_state_dict if load_plan.get(fqn)),
        target_state_dict,
    )
    dropped = {alias for alias, _ in aliases}
    return (
        {fqn: plans for fqn, plans in load_plan.items() if fqn not in dropped},
        aliases,
    )


def _dedupe_resolved_targets(
    targets: Sequence[TensorReadTarget],
    target_state_dict: Mapping[str, Any],
) -> tuple[TensorReadTarget, ...]:
    aliases = _tensor_view_aliases(
        tuple(dict.fromkeys(target.target_fqn for target in targets)),
        target_state_dict,
    )
    dropped = {alias for alias, _ in aliases}
    return tuple(target for target in targets if target.target_fqn not in dropped)


def _tensor_view_aliases(
    fqns: Sequence[str],
    target_state_dict: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    canonical_by_view: dict[tuple[object, ...], str] = {}
    view_by_fqn: dict[str, tuple[object, ...]] = {}
    for fqn in fqns:
        value = target_state_dict.get(fqn)
        if not isinstance(value, torch.Tensor) or not value.numel():
            continue
        view = (
            value.device.type,
            value.device.index,
            value.untyped_storage().data_ptr(),
            value.storage_offset(),
            tuple(value.shape),
            tuple(value.stride()),
            value.dtype,
        )
        view_by_fqn[fqn] = view
        canonical_by_view[view] = fqn
    aliases = tuple(
        (fqn, canonical_by_view[view])
        for fqn, view in view_by_fqn.items()
        if canonical_by_view[view] != fqn
    )
    if aliases:
        logger.warning(
            "cooperative loading deduplicated %d exact destination aliases",
            len(aliases),
        )
    return aliases


def _source_demands_for_plan(
    load_plan: Mapping[str, Sequence[LoadPlan]],
) -> dict[int, frozenset[str]]:
    demands: dict[int, set[str]] = defaultdict(set)
    for plans in load_plan.values():
        for plan in plans:
            demands[int(plan.src_rank)].add(str(plan.src_fqn))
    return {rank: frozenset(fqns) for rank, fqns in sorted(demands.items())}


def _source_demands_to_wire(
    demands: Mapping[int, Collection[str]],
) -> dict[str, object]:
    canonical: dict[int, Collection[str]] = {}
    for source_rank, fqns in demands.items():
        if isinstance(source_rank, bool) or not isinstance(source_rank, int):
            raise ValueError("metadata demand source rank must be an integer")
        if source_rank < 0:
            raise ValueError("metadata demand source rank must be non-negative")
        if not isinstance(fqns, Collection) or isinstance(
            fqns, (str, bytes, bytearray)
        ):
            raise ValueError("metadata demand FQNs must be a collection")
        if not all(isinstance(fqn, str) for fqn in fqns):
            raise ValueError("metadata demand FQNs must be strings")
        canonical[source_rank] = fqns
    payload: dict[str, object] = {}
    for source_rank, fqns in sorted(canonical.items()):
        payload[str(source_rank)] = sorted(set(fqns))
    return payload


def _source_demands_from_wire(
    value: object,
) -> dict[int, frozenset[str]]:
    payload = _mapping(value, "source demands")
    result: dict[int, frozenset[str]] = {}
    for raw_rank, raw_fqns in payload.items():
        try:
            source_rank = int(raw_rank)
        except ValueError as error:
            raise ValueError(
                f"metadata demand source rank {raw_rank!r} is not an integer"
            ) from error
        if source_rank < 0 or str(source_rank) != raw_rank:
            raise ValueError(
                f"metadata demand source rank {raw_rank!r} is not canonical"
            )
        if source_rank in result:
            raise ValueError(
                f"source demands contain duplicate source rank {source_rank}"
            )
        raw_sequence = _sequence(
            raw_fqns,
            f"source demands for rank {raw_rank}",
        )
        if not isinstance(raw_sequence, list):
            raise ValueError("metadata demand FQNs must be strings in an array")
        fqns: list[str] = []
        for fqn in raw_sequence:
            if not isinstance(fqn, str):
                raise ValueError("metadata demand FQNs must be strings in an array")
            fqns.append(fqn)
        if fqns != sorted(set(fqns)):
            raise ValueError(f"source demands for rank {source_rank} are not canonical")
        result[source_rank] = frozenset(fqns)
    return result


def _merge_source_demands(
    demands: Iterable[Mapping[int, Collection[str]]],
) -> dict[int, frozenset[str]]:
    merged: dict[int, set[str]] = defaultdict(set)
    for rank_demands in demands:
        for source_rank, fqns in rank_demands.items():
            merged[source_rank].update(fqns)
    return {rank: frozenset(fqns) for rank, fqns in sorted(merged.items())}


def _partition_metadata_demands(
    demands: Mapping[int, Collection[str]],
    node_ids: Sequence[NodeId],
) -> dict[NodeId, dict[int, frozenset[str]]]:
    assigned: dict[NodeId, dict[int, frozenset[str]]] = {
        node_id: {} for node_id in node_ids
    }
    for index, (source_rank, fqns) in enumerate(sorted(demands.items())):
        assigned[node_ids[index % len(node_ids)]][source_rank] = frozenset(fqns)
    return assigned


def _validate_metadata_demands(
    metadata: Mapping[int, Mapping[str, SourceTensorMetadata]],
    demands: Mapping[int, Collection[str]],
) -> None:
    for source_rank, fqns in demands.items():
        available = metadata.get(source_rank)
        if available is None:
            raise ValueError(f"metadata is missing source rank {source_rank}")
        missing = set(fqns) - set(available)
        if missing:
            raise ValueError(
                f"metadata for source rank {source_rank} is missing {sorted(missing)!r}"
            )


def _metadata_for_demands(
    metadata: Mapping[int, Mapping[str, SourceTensorMetadata]],
    demands: Mapping[int, Collection[str]],
) -> dict[int, dict[str, SourceTensorMetadata]]:
    selected: dict[int, dict[str, SourceTensorMetadata]] = {}
    for source_rank, fqns in sorted(demands.items()):
        source = metadata.get(source_rank)
        if source is None:
            raise ValueError(f"metadata is missing source rank {source_rank}")
        selected[source_rank] = {}
        for fqn in sorted(fqns):
            try:
                selected[source_rank][fqn] = source[fqn]
            except KeyError as error:
                raise ValueError(
                    f"metadata for source rank {source_rank} is missing {fqn!r}"
                ) from error
    return selected


def _merge_metadata(
    destination: dict[int, dict[str, SourceTensorMetadata]],
    incoming: Mapping[int, Mapping[str, SourceTensorMetadata]],
) -> None:
    for source_rank, tensors in incoming.items():
        target = destination.setdefault(source_rank, {})
        for fqn, metadata in tensors.items():
            previous = target.get(fqn)
            if previous is not None and previous != metadata:
                raise ValueError(
                    f"conflicting metadata for source rank {source_rank}, {fqn!r}"
                )
            target[fqn] = metadata


def _metadata_to_wire(
    metadata: Mapping[int, Mapping[str, SourceTensorMetadata]],
) -> dict[str, object]:
    return {
        str(source_rank): {
            fqn: {
                "checkpoint_offset_bytes": item.checkpoint_offset_bytes,
                "dtype": item.dtype,
                "element_size_bytes": item.element_size_bytes,
                "fqn": item.fqn,
                "shape": list(item.shape),
                "storage_nbytes": item.storage_nbytes,
                "storage_offset_elements": item.storage_offset_elements,
                "stride": list(item.stride),
            }
            for fqn, item in sorted(tensors.items())
        }
        for source_rank, tensors in sorted(metadata.items())
    }


def _metadata_from_wire(
    value: object,
) -> dict[int, dict[str, SourceTensorMetadata]]:
    payload = _mapping(value, "source metadata")
    result: dict[int, dict[str, SourceTensorMetadata]] = {}
    for raw_rank, raw_tensors in payload.items():
        tensors = _mapping(raw_tensors, f"metadata for rank {raw_rank}")
        decoded: dict[str, SourceTensorMetadata] = {}
        for fqn, raw_item in tensors.items():
            item = _mapping(raw_item, f"metadata for {fqn}")
            metadata = SourceTensorMetadata(
                fqn=str(item["fqn"]),
                checkpoint_offset_bytes=int(item["checkpoint_offset_bytes"]),
                storage_offset_elements=int(item["storage_offset_elements"]),
                storage_nbytes=int(item["storage_nbytes"]),
                shape=tuple(
                    int(size) for size in _sequence(item["shape"], "metadata shape")
                ),
                stride=tuple(
                    int(stride)
                    for stride in _sequence(item["stride"], "metadata stride")
                ),
                dtype=str(item["dtype"]),
                element_size_bytes=int(item["element_size_bytes"]),
            )
            if metadata.fqn != fqn:
                raise ValueError("metadata key does not match its tensor name")
            decoded[fqn] = metadata
        result[int(raw_rank)] = decoded
    return result


def _target_consumer_bytes(
    targets: Iterable[TensorReadTarget],
) -> dict[int, int]:
    consumer_bytes: Counter[int] = Counter()
    for target in targets:
        if target.source_pattern.dense_nbytes:
            consumer_bytes[target.source_rank] += target.source_pattern.dense_nbytes
    return dict(sorted(consumer_bytes.items()))


def _byte_demand_payload_to_wire(
    demands: Iterable[FqnDemand],
    consumer_bytes: Mapping[int, int],
) -> dict[str, object]:
    return _byte_demand_payload_to_wire_from_canonical(
        [demand.to_dict() for demand in demands],
        consumer_bytes,
    )


def _byte_demand_payload_to_wire_from_canonical(
    demands: list[object],
    consumer_bytes: Mapping[int, int],
) -> dict[str, object]:
    encoded_consumer_bytes: dict[str, int] = {}
    for source_rank, byte_count in sorted(consumer_bytes.items()):
        if isinstance(source_rank, bool) or not isinstance(source_rank, int):
            raise ValueError("consumer source ranks must be integers")
        if source_rank < 0:
            raise ValueError("consumer source ranks must be non-negative")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise ValueError("consumer byte counts must be integers")
        if byte_count < 0:
            raise ValueError("consumer byte counts must be non-negative")
        encoded_consumer_bytes[str(source_rank)] = byte_count
    return {
        "consumer_bytes": encoded_consumer_bytes,
        "demands": demands,
        "version": _BYTE_DEMAND_PAYLOAD_VERSION,
    }


def _byte_demand_payload_from_wire(
    value: object,
) -> tuple[Sequence[object], dict[int, int]]:
    payload = _mapping(value, "byte demand payload")
    expected_keys = {"consumer_bytes", "demands", "version"}
    if set(payload) != expected_keys:
        raise ValueError(
            "byte demand payload keys differ: "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"unexpected={sorted(set(payload) - expected_keys)}"
        )
    version = payload["version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("byte demand payload version must be an integer")
    if version != _BYTE_DEMAND_PAYLOAD_VERSION:
        raise ValueError(f"unsupported byte demand payload version {version!r}")
    raw_consumer_bytes = _mapping(
        payload["consumer_bytes"], "byte demand consumer bytes"
    )
    consumer_bytes: dict[int, int] = {}
    for raw_source_rank, raw_byte_count in raw_consumer_bytes.items():
        try:
            source_rank = int(raw_source_rank)
        except ValueError as error:
            raise ValueError(
                f"consumer source rank {raw_source_rank!r} is not an integer"
            ) from error
        if str(source_rank) != raw_source_rank or source_rank < 0:
            raise ValueError(
                f"consumer source rank {raw_source_rank!r} is not canonical"
            )
        if isinstance(raw_byte_count, bool) or not isinstance(raw_byte_count, int):
            raise ValueError("consumer byte counts must be integers")
        if raw_byte_count < 0:
            raise ValueError("consumer byte counts must be non-negative")
        consumer_bytes[source_rank] = raw_byte_count
    demands = _sequence(payload["demands"], "byte demands")
    return demands, consumer_bytes


def _unique_demand_wire_payloads(
    payloads: Iterable[Sequence[object]],
) -> tuple[Sequence[object], ...]:
    unique: list[Sequence[object]] = []
    first_index_by_discriminator: dict[object, int] = {}
    encoded_indices_by_discriminator: dict[object, dict[bytes, int]] = {}
    for payload in payloads:
        discriminator = _demand_wire_payload_discriminator(payload)
        first_index = first_index_by_discriminator.get(discriminator)
        if first_index is None:
            first_index_by_discriminator[discriminator] = len(unique)
            unique.append(payload)
            continue
        encoded = _encode_demand_wire_payload_for_deduplication(payload)
        encoded_indices = encoded_indices_by_discriminator.get(discriminator)
        if encoded_indices is None:
            encoded_indices = {
                _encode_demand_wire_payload_for_deduplication(
                    unique[first_index]
                ): first_index
            }
            encoded_indices_by_discriminator[discriminator] = encoded_indices
        if encoded not in encoded_indices:
            encoded_indices[encoded] = len(unique)
            unique.append(payload)
    return tuple(unique)


def _demand_wire_payload_discriminator(
    payload: Sequence[object],
) -> tuple[object, ...] | None:
    if type(payload) is not list:
        return None
    if not payload:
        return (0,)
    sampled_demands: list[object] = []
    for demand_index in sorted({0, len(payload) // 2, len(payload) - 1}):
        raw_demand = payload[demand_index]
        if (
            type(raw_demand) is not dict
            or set(raw_demand) != {"fqn", "ranges"}
            or type(raw_demand["fqn"]) is not str
            or type(raw_demand["ranges"]) is not list
            or not raw_demand["ranges"]
        ):
            return None
        raw_ranges = raw_demand["ranges"]
        sampled_ranges: list[object] = []
        for range_index in sorted({0, len(raw_ranges) // 2, len(raw_ranges) - 1}):
            raw_range = raw_ranges[range_index]
            if (
                type(raw_range) is not dict
                or set(raw_range) != {"length", "offset", "source_rank"}
                or type(raw_range["length"]) is not int
                or type(raw_range["offset"]) is not int
                or type(raw_range["source_rank"]) is not int
            ):
                return None
            sampled_ranges.append(
                (
                    raw_range["source_rank"],
                    raw_range["offset"],
                    raw_range["length"],
                )
            )
        sampled_demands.append(
            (
                raw_demand["fqn"],
                len(raw_ranges),
                tuple(sampled_ranges),
            )
        )
    return (len(payload), tuple(sampled_demands))


def _encode_demand_wire_payload_for_deduplication(
    payload: Sequence[object],
) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _targets_to_demands(
    targets: Iterable[TensorReadTarget],
) -> tuple[FqnDemand, ...]:
    ranges: dict[str, list[SourceByteRange]] = defaultdict(list)
    for target in targets:
        for source_range in target.source_pattern.iter_ranges():
            if source_range.length:
                ranges[target.target_fqn].append(
                    SourceByteRange(
                        target.source_rank,
                        source_range.offset,
                        source_range.length,
                    )
                )
    return tuple(
        FqnDemand(fqn, tuple(byte_ranges))
        for fqn, byte_ranges in sorted(ranges.items())
    )


def _encode_control(value: object) -> bytes:
    return _encode_control_at_level(value, zlib.Z_DEFAULT_COMPRESSION)


def _encode_metadata_control(value: object) -> bytes:
    return _encode_control_at_level(value, _METADATA_CONTROL_COMPRESSION_LEVEL)


def _encode_control_at_level(value: object, compression_level: int) -> bytes:
    return zlib.compress(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(),
        compression_level,
    )


def _decode_control(value: bytes) -> object:
    if len(value) > _CONTROL_BYTES_LIMIT:
        raise ValueError("compressed cooperative control payload exceeds its limit")
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(
            value,
            _DECOMPRESSED_CONTROL_BYTES_LIMIT + 1,
        )
    except zlib.error as error:
        raise ValueError("invalid cooperative control payload") from error
    if len(decoded) > _DECOMPRESSED_CONTROL_BYTES_LIMIT or decoder.unconsumed_tail:
        raise ValueError("decompressed cooperative control payload exceeds its limit")
    try:
        decoded += decoder.flush()
    except zlib.error as error:
        raise ValueError("invalid cooperative control payload") from error
    if len(decoded) > _DECOMPRESSED_CONTROL_BYTES_LIMIT:
        raise ValueError("decompressed cooperative control payload exceeds its limit")
    if not decoder.eof or decoder.unused_data:
        raise ValueError("invalid cooperative control payload")
    try:
        return json.loads(decoded)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("invalid cooperative control payload") from error


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be an array")
    return value


def _node_id(value: object) -> NodeId:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("node_id must be a string or integer")
    return value


def _node_sort_key(node_id: NodeId) -> tuple[int, int | str]:
    return (0, node_id) if isinstance(node_id, int) else (1, node_id)


def _replace_url_host(base_url: str, host: str) -> str:
    parsed = urlsplit(base_url)
    if parsed.scheme != "http" or parsed.port is None:
        raise ValueError("server base URL must be HTTP and include a port")
    return f"http://{_format_url_host(host)}:{parsed.port}"


def _format_url_host(host: str) -> str:
    if host.startswith("[") or host.endswith("]"):
        if not (host.startswith("[") and host.endswith("]")):
            raise ValueError("advertise_host has mismatched IPv6 brackets")
        host = host[1:-1]
    if not host or any(character.isspace() for character in host):
        raise ValueError("advertise_host must be a nonempty host name or address")
    if any(character in host for character in "/?#@"):
        raise ValueError("advertise_host must not contain URL syntax")
    return f"[{host}]" if ":" in host else host


def _is_wildcard_host(host: str) -> bool:
    return host.removeprefix("[").removesuffix("]") in {"0.0.0.0", "::"}


def _bootstrap_node_tag(kind: str, node_index: int) -> str:
    return f"bootstrap/{kind}/{node_index}"


def _global_blob_tag(prefix: str) -> str:
    return f"global/{prefix}"


def _global_node_tag(prefix: str, node_index: int) -> str:
    return f"global/{prefix}/node/{node_index}"


def _local_blob_tag(prefix: str) -> str:
    return f"local/{prefix}"


def _local_rank_tag(prefix: str, rank: int) -> str:
    return f"local/{prefix}/rank/{rank}"


def _error_message(global_rank: int, error: BaseException) -> str:
    return _bounded_error_message(
        f"rank {global_rank}: {type(error).__name__}: {error}"
    )


def _raise_remote_error(message: str) -> None:
    cache_busy_marker = f": {_ChunkPoolCacheBusy.__name__}: "
    if cache_busy_marker in message:
        raise _ChunkPoolCacheBusy(message.split(cache_busy_marker, 1)[1])
    raise RuntimeError(message)


def _bounded_error_message(message: str) -> str:
    encoded = message.encode()
    if len(encoded) <= _ERROR_MESSAGE_BYTES_LIMIT:
        return message
    suffix = b"... [truncated]"
    prefix_bytes = max(0, _ERROR_MESSAGE_BYTES_LIMIT - len(suffix))
    prefix = encoded[:prefix_bytes].decode(errors="ignore")
    return prefix + suffix[: _ERROR_MESSAGE_BYTES_LIMIT - len(prefix.encode())].decode()


def _oversized_node_works(
    works: Iterable[BatchNodeWork],
    capacities: Mapping[NodeId, int],
) -> tuple[BatchNodeWork, ...]:
    return tuple(
        work for work in works if work.download_bytes > capacities[work.node_id]
    )
