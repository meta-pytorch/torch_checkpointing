# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import io
import json
import struct
import threading
import time
import zlib
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from unittest import mock

import pytest
import torch
from torch_checkpointing.experimental.cooperative_resharding import (
    loader as loader_module,
    transport as transport_module,
)
from torch_checkpointing.experimental.cooperative_resharding.config import (
    CooperativeLoadConfig,
)
from torch_checkpointing.experimental.cooperative_resharding.layout import (
    RepeatedStrideBytePattern,
    SourceTensorMetadata,
    TensorReadTarget,
)
from torch_checkpointing.experimental.cooperative_resharding.loader import (
    CooperativeLoadFailure,
    CooperativeLoadRequest,
    CooperativeLoadUnsupported,
    load_cooperatively,
    MetadataProvider,
    MetricCallback,
)
from torch_checkpointing.experimental.cooperative_resharding.planning import (
    ByteRange,
    FqnDemand,
    merge_fqn_demand_wire_payloads,
    NodeMembership,
    RankTopology,
)
from torch_checkpointing.experimental.cooperative_resharding.rendezvous import (
    InMemoryRendezvous,
)
from torch_checkpointing.experimental.cooperative_resharding.scatter import (
    BufferSlot,
    direct_cpu_destination_buffer,
    PinnedBufferPool,
)
from torch_checkpointing.experimental.cooperative_resharding.transport import (
    TransportError,
)
from torch_checkpointing.resharding import LoadPlan
from torch_checkpointing.storage.base_storage import ReadArgs, Storage


def _config() -> CooperativeLoadConfig:
    return CooperativeLoadConfig(
        batch_target_bytes=64,
        shared_memory_fraction=0.5,
        shared_memory_chunk_bytes=4,
        range_consolidation_gap_bytes=0,
        download_workers=2,
        fetch_workers=2,
        server_workers=4,
        max_inflight_batches=2,
        retry_attempts=20,
        retry_backoff_seconds=0.01,
        progress_timeout_seconds=10,
        pinned_buffer_bytes=16,
        pinned_buffer_count=4,
        enable_fast_scatter=True,
    )


class _MemoryStorage:
    def __init__(
        self,
        values: Mapping[Path, bytes],
        *,
        failure: Exception | None = None,
    ) -> None:
        self._values = dict(values)
        self._failure = failure
        self.reads: list[Path] = []
        self._lock = threading.Lock()

    @contextmanager
    def stream_read(
        self, path: Path, read_args: ReadArgs | None = None
    ) -> Iterator[io.BytesIO]:
        del read_args
        with self._lock:
            self.reads.append(path)
        if self._failure is not None:
            raise self._failure
        yield io.BytesIO(self._values[path])


class _StaticMetadataProvider(MetadataProvider):
    def __init__(
        self, metadata: Mapping[int, Mapping[str, SourceTensorMetadata]]
    ) -> None:
        self._metadata = metadata
        self.demands: list[dict[int, frozenset[str]]] = []

    def load_metadata(
        self,
        demands_by_rank: Mapping[int, Collection[str]],
        *,
        storage: Any,
        source_path_for_rank: Callable[[int], Path],
        max_workers: int,
        timeout_seconds: float,
    ) -> Mapping[int, Mapping[str, SourceTensorMetadata]]:
        del storage, source_path_for_rank, max_workers, timeout_seconds
        demands = {rank: frozenset(fqns) for rank, fqns in demands_by_rank.items()}
        self.demands.append(demands)
        return {
            rank: {fqn: self._metadata[rank][fqn] for fqn in fqns}
            for rank, fqns in demands.items()
        }


class _UnsupportedMetadataProvider(MetadataProvider):
    def __init__(self, detail: str) -> None:
        self._detail = detail

    def load_metadata(
        self,
        demands_by_rank: Mapping[int, Collection[str]],
        *,
        storage: Any,
        source_path_for_rank: Callable[[int], Path],
        max_workers: int,
        timeout_seconds: float,
    ) -> Mapping[int, Mapping[str, SourceTensorMetadata]]:
        del demands_by_rank
        del storage
        del source_path_for_rank
        del max_workers
        del timeout_seconds
        raise CooperativeLoadUnsupported(self._detail)


class _HungAndFailingStorage(_MemoryStorage):
    def __init__(self, values: Mapping[Path, bytes]) -> None:
        super().__init__(values)
        self.blocked = threading.Event()
        self.release = threading.Event()

    @contextmanager
    def stream_read(
        self, path: Path, read_args: ReadArgs | None = None
    ) -> Iterator[io.BytesIO]:
        del read_args
        if path == Path("source-0"):
            self.blocked.set()
            if not self.release.wait(timeout=10):
                raise TimeoutError("test did not release the hung source read")
            yield io.BytesIO(self._values[path])
            return
        if not self.blocked.wait(timeout=10):
            raise TimeoutError("the hung source read did not start")
        raise OSError("injected fast source failure")


class _ClosingClient:
    def __init__(self, url: str) -> None:
        self.url = url
        self.close_count = 0

    def close(self) -> None:
        self.close_count += 1


def _topology(
    rank: int,
    nodes: Sequence[tuple[str, Sequence[int]]],
    *,
    job_id: str,
) -> RankTopology:
    return RankTopology(
        global_rank=rank,
        nodes=tuple(NodeMembership(node_id, tuple(ranks)) for node_id, ranks in nodes),
        coordination_world_count=1,
        job_id=job_id,
    )


def _dense_target(
    tensor: torch.Tensor,
    *,
    target_fqn: str = "weight",
    source_rank: int = 0,
    source_offset: int = 0,
) -> TensorReadTarget:
    byte_count = tensor.numel() * tensor.element_size()
    return TensorReadTarget(
        target_fqn=target_fqn,
        source_rank=source_rank,
        source_fqn="source",
        source_pattern=RepeatedStrideBytePattern(source_offset, byte_count),
        destination_pattern=RepeatedStrideBytePattern(0, byte_count),
        source_tensor_shape=tuple(tensor.shape),
        source_slice_shape=tuple(tensor.shape),
        target_tensor_shape=tuple(tensor.shape),
        target_slice_shape=tuple(tensor.shape),
        source_dtype=str(tensor.dtype),
        target_dtype=str(tensor.dtype),
        source_element_size_bytes=tensor.element_size(),
        target_element_size_bytes=tensor.element_size(),
        transpose_dims=(),
        target_device=str(tensor.device),
    )


def test_byte_demand_payload_preserves_consumer_multiplicity() -> None:
    first = torch.zeros(2, dtype=torch.float32)
    second = torch.zeros(3, dtype=torch.float32)
    targets = (
        _dense_target(first, target_fqn="first", source_rank=7),
        _dense_target(second, target_fqn="second", source_rank=7),
    )
    demands = loader_module._targets_to_demands(targets)
    consumer_bytes = loader_module._target_consumer_bytes(targets)

    decoded_demands, decoded_consumer_bytes = (
        loader_module._byte_demand_payload_from_wire(
            loader_module._byte_demand_payload_to_wire(demands, consumer_bytes)
        )
    )

    assert decoded_demands == [demand.to_dict() for demand in demands]
    assert decoded_consumer_bytes == {7: 20}
    assert loader_module._byte_demand_payload_to_wire_from_canonical(
        [demand.to_dict() for demand in demands],
        consumer_bytes,
    ) == loader_module._byte_demand_payload_to_wire(demands, consumer_bytes)


def test_canonical_node_demand_wire_is_byte_compatible() -> None:
    payloads = (
        [
            FqnDemand(
                "weight",
                (
                    ByteRange(0, 0, 4),
                    ByteRange(1, 8, 4),
                ),
            ).to_dict()
        ],
        [
            FqnDemand(
                "weight",
                (
                    ByteRange(0, 4, 4),
                    ByteRange(1, 4, 4),
                ),
            ).to_dict()
        ],
    )
    canonical = loader_module._merge_fqn_demand_wire_payloads_to_canonical_wire(
        payloads
    )
    materialized = merge_fqn_demand_wire_payloads(payloads)
    consumer_bytes = {0: 8, 1: 8}

    canonical_wire = loader_module._byte_demand_payload_to_wire_from_canonical(
        canonical.payload,
        consumer_bytes,
    )
    materialized_wire = loader_module._byte_demand_payload_to_wire(
        materialized.demands,
        consumer_bytes,
    )
    assert canonical_wire == materialized_wire
    assert loader_module._encode_control(
        canonical_wire
    ) == loader_module._encode_control(materialized_wire)


@pytest.mark.parametrize(
    "payload",
    [
        {"consumer_bytes": {}, "demands": [], "version": 1},
        {"consumer_bytes": {}, "demands": [], "version": True},
        {"consumer_bytes": {"01": 1}, "demands": [], "version": 2},
        {"consumer_bytes": {"1": -1}, "demands": [], "version": 2},
        {
            "consumer_bytes": {},
            "demands": [],
            "unexpected": 1,
            "version": 2,
        },
    ],
)
def test_byte_demand_payload_rejects_malformed_values(
    payload: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError):
        loader_module._byte_demand_payload_from_wire(payload)


def test_unique_demand_payloads_ignore_consumer_counts() -> None:
    demand = loader_module.FqnDemand(
        "weight",
        (loader_module.SourceByteRange(0, 0, 4),),
    )
    first, _ = loader_module._byte_demand_payload_from_wire(
        loader_module._byte_demand_payload_to_wire((demand,), {0: 4})
    )
    second, _ = loader_module._byte_demand_payload_from_wire(
        loader_module._byte_demand_payload_to_wire((demand,), {0: 8})
    )

    assert loader_module._unique_demand_wire_payloads((first, second)) == (first,)


def test_unique_demand_payloads_skip_encoding_for_distinct_discriminators() -> None:
    first = [
        {
            "fqn": "weight",
            "ranges": [{"source_rank": 0, "offset": 0, "length": 4}],
        }
    ]
    second = [
        {
            "fqn": "weight",
            "ranges": [{"source_rank": 1, "offset": 0, "length": 4}],
        }
    ]

    with mock.patch.object(loader_module.json, "dumps", wraps=json.dumps) as dumps:
        unique = loader_module._unique_demand_wire_payloads((first, second))

    assert unique == (first, second)
    dumps.assert_not_called()


def test_unique_demand_payloads_fall_back_for_late_differences() -> None:
    first = [
        {
            "fqn": f"tensor.{index}",
            "ranges": [{"source_rank": 0, "offset": index, "length": 1}],
        }
        for index in range(9)
    ]
    second = json.loads(json.dumps(first))
    second[7]["ranges"][0]["offset"] = 100

    with mock.patch.object(loader_module.json, "dumps", wraps=json.dumps) as dumps:
        unique = loader_module._unique_demand_wire_payloads((first, second, first))

    assert unique == (first, second)
    assert dumps.call_count == 3


def test_unique_demand_payloads_hash_late_collision_buckets() -> None:
    first = [
        {
            "fqn": f"tensor.{index}",
            "ranges": [{"source_rank": 0, "offset": index, "length": 1}],
        }
        for index in range(9)
    ]
    second = json.loads(json.dumps(first))
    third = json.loads(json.dumps(first))
    second[7]["ranges"][0]["offset"] = 100
    third[7]["ranges"][0]["offset"] = 200

    with mock.patch.object(loader_module.json, "dumps", wraps=json.dumps) as dumps:
        unique = loader_module._unique_demand_wire_payloads(
            (first, second, third, second)
        )

    assert unique == (first, second, third)
    assert dumps.call_count == 4


def test_unique_demand_payloads_do_not_conflate_bools_and_integers() -> None:
    integer = [
        {
            "fqn": "weight",
            "ranges": [{"source_rank": 1, "offset": 0, "length": 4}],
        }
    ]
    boolean = json.loads(json.dumps(integer))
    boolean[0]["ranges"][0]["source_rank"] = True

    assert loader_module._unique_demand_wire_payloads((integer, boolean)) == (
        integer,
        boolean,
    )


def test_unique_demand_payloads_do_not_hide_unsampled_malformed_values() -> None:
    integer = [
        {
            "fqn": f"tensor.{index}",
            "ranges": [{"source_rank": 1, "offset": index, "length": 1}],
        }
        for index in range(9)
    ]
    boolean = json.loads(json.dumps(integer))
    boolean[7]["ranges"][0]["source_rank"] = True

    unique = loader_module._unique_demand_wire_payloads((integer, boolean))

    assert unique == (integer, boolean)
    with pytest.raises(ValueError):
        merge_fqn_demand_wire_payloads(unique)


def test_unique_demand_payloads_keep_json_equivalent_non_list_payloads() -> None:
    demand = {
        "fqn": "weight",
        "ranges": [{"source_rank": 0, "offset": 0, "length": 1}],
    }
    list_payload = [demand]
    tuple_payload = (demand,)

    unique = loader_module._unique_demand_wire_payloads((list_payload, tuple_payload))

    assert unique == (list_payload, tuple_payload)
    with pytest.raises(ValueError, match="rank byte demands must be a list"):
        merge_fqn_demand_wire_payloads(unique)


def test_unique_demand_payloads_deduplicate_empty_payloads() -> None:
    first: list[object] = []
    second: list[object] = []

    assert loader_module._unique_demand_wire_payloads((first, second)) == (first,)


def _request(
    *,
    rank: int,
    nodes: Sequence[tuple[str, Sequence[int]]],
    rendezvous: InMemoryRendezvous,
    session_token: str,
    storage: _MemoryStorage,
    target_state_dict: Mapping[str, Any],
    shared_memory_directory: Path,
    local_targets: Sequence[TensorReadTarget] | None = None,
    local_load_plan: Mapping[str, Sequence[LoadPlan]] | None = None,
    metadata_provider: MetadataProvider | None = None,
    metric_callback: MetricCallback | None = None,
) -> CooperativeLoadRequest:
    return CooperativeLoadRequest(
        topology=_topology(rank, nodes, job_id=f"job-{session_token}"),
        rendezvous=rendezvous,
        session_token=session_token,
        storage=cast(Storage, storage),
        source_path_for_rank=lambda source_rank: Path(f"source-{source_rank}"),
        target_state_dict=target_state_dict,
        local_load_plan=local_load_plan or {},
        local_targets=local_targets,
        metadata_provider=metadata_provider,
        metric_callback=metric_callback,
        bind_host="127.0.0.1",
        advertise_host="127.0.0.1" if len(nodes) > 1 else None,
        shared_memory_directory=shared_memory_directory,
        shared_memory_capacity_bytes=64,
    )


def _run_concurrently(
    requests: Sequence[CooperativeLoadRequest],
) -> tuple[list[object], list[BaseException]]:
    results: list[object] = []
    errors: list[BaseException] = []
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [
            executor.submit(load_cooperatively, request, config=_config())
            for request in requests
        ]
        for future in futures:
            try:
                results.append(future.result(timeout=20))
            except Exception as error:
                errors.append(error)
    return results, errors


def _cause_messages(error: BaseException) -> tuple[str, ...]:
    messages: list[str] = []
    current: BaseException | None = error
    while current is not None:
        messages.append(str(current))
        current = current.__cause__
    return tuple(messages)


def test_multi_node_request_requires_advertise_host(tmp_path: Path) -> None:
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)), ("node-b", (1,))),
        rendezvous=InMemoryRendezvous(),
        session_token="missing-advertise-host",
        storage=_MemoryStorage({}),
        target_state_dict={},
        shared_memory_directory=tmp_path,
        local_targets=(),
    )

    with pytest.raises(
        ValueError,
        match="advertise_host is required for multi-node loading",
    ):
        replace(request, advertise_host=None)


@pytest.mark.parametrize(
    ("bound_url", "advertise_host", "expected"),
    [
        (
            "http://0.0.0.0:12345",
            "worker.example",
            "http://worker.example:12345",
        ),
        (
            "http://[::]:54321",
            "2001:db8::7",
            "http://[2001:db8::7]:54321",
        ),
        (
            "http://[::1]:321",
            "[2001:db8::8]",
            "http://[2001:db8::8]:321",
        ),
    ],
)
def test_replace_url_host_preserves_port_and_formats_ipv6(
    bound_url: str,
    advertise_host: str,
    expected: str,
) -> None:
    assert loader_module._replace_url_host(bound_url, advertise_host) == expected


def test_wildcard_bind_uses_advertised_host(tmp_path: Path) -> None:
    source = struct.pack("f", 2.0)
    target = torch.zeros(1)
    request = replace(
        _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="wildcard-bind",
            storage=_MemoryStorage({Path("source-0"): source}),
            target_state_dict={"weight": target},
            shared_memory_directory=tmp_path,
            local_targets=(_dense_target(target),),
        ),
        bind_host="0.0.0.0",
        advertise_host="127.0.0.1",
    )

    result = load_cooperatively(request, config=_config())

    torch.testing.assert_close(target, torch.tensor([2.0]))
    assert result.target_count == 1
    assert list(tmp_path.iterdir()) == []


def test_decode_control_rejects_decompressed_amplification() -> None:
    payload = zlib.compress(json.dumps({"value": "x" * 512}).encode())

    with (
        mock.patch.object(loader_module, "_DECOMPRESSED_CONTROL_BYTES_LIMIT", 64),
        pytest.raises(ValueError, match="decompressed .* exceeds"),
    ):
        loader_module._decode_control(payload)


def test_metadata_control_uses_fast_compatible_compression() -> None:
    payload = {
        "0": {
            "": {
                "checkpoint_offset_bytes": 0,
                "dtype": "torch.float32",
                "element_size_bytes": 4,
                "fqn": "",
                "shape": [4],
                "storage_nbytes": 16,
                "storage_offset_elements": 0,
                "stride": [1],
            }
        }
    }
    compress = zlib.compress
    levels: list[int] = []

    def record_level(data: bytes, level: int = -1) -> bytes:
        levels.append(level)
        return compress(data, level)

    with mock.patch.object(loader_module.zlib, "compress", side_effect=record_level):
        encoded = loader_module._encode_metadata_control(payload)

    assert levels == [1]
    assert loader_module._decode_control(encoded) == payload


def test_error_messages_are_utf8_byte_bounded() -> None:
    with mock.patch.object(loader_module, "_ERROR_MESSAGE_BYTES_LIMIT", 64):
        message = loader_module._error_message(7, RuntimeError("é" * 100))

    assert len(message.encode()) <= 64
    assert message.startswith("rank 7: RuntimeError:")
    assert message.endswith("... [truncated]")


def test_local_error_forwarding_retries_until_one_publication_succeeds(
    tmp_path: Path,
) -> None:
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="forward-retry",
        storage=_MemoryStorage({}),
        target_state_dict={},
        shared_memory_directory=tmp_path,
        local_targets=(),
    )
    session = loader_module._CooperativeLoadSession(request, _config())
    coordinator = mock.Mock()
    coordinator.publish_error.side_effect = (
        RuntimeError("transient coordinator failure"),
        None,
    )
    session._control_coordinator = coordinator

    with mock.patch.object(
        request.rendezvous,
        "publish_error",
        side_effect=(RuntimeError("transient rendezvous failure"), None),
    ) as publish_rendezvous:
        session._forward_local_error_if_needed("injected node error")
        assert session._local_error_forwarded is None
        session._forward_local_error_if_needed("injected node error")
        assert session._local_error_forwarded == "injected node error"
        session._forward_local_error_if_needed("injected node error")

    assert coordinator.publish_error.call_count == 2
    assert publish_rendezvous.call_count == 2


@pytest.mark.parametrize("view_kind", ["conjugate", "negative"])
def test_direct_cpu_destination_buffer_preserves_logical_view_bits(
    view_kind: str,
) -> None:
    if view_kind == "conjugate":
        base = torch.zeros(1, dtype=torch.complex64)
        destination = base.conj()
        encoded = struct.pack("ff", 1.0, 2.0)
        expected = torch.tensor([1.0 + 2.0j], dtype=torch.complex64)
    else:
        base = torch.zeros(1, dtype=torch.float32)
        destination = base._neg_view()
        encoded = struct.pack("f", 3.0)
        expected = torch.tensor([3.0], dtype=torch.float32)

    target = _dense_target(destination)
    with direct_cpu_destination_buffer(target, {"weight": destination}) as buffer:
        buffer[:] = encoded

    torch.testing.assert_close(destination, expected)


def test_direct_cpu_destination_buffer_bumps_tensor_version() -> None:
    destination = torch.ones(1, dtype=torch.float32, requires_grad=True)
    saved_computation = destination.square()
    target = _dense_target(destination)
    version = destination._version

    with direct_cpu_destination_buffer(target, {"weight": destination}) as buffer:
        buffer[:] = struct.pack("f", 7.0)

    assert destination._version == version + 1
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        saved_computation.backward()


def test_direct_cpu_destination_buffer_validates_live_destination() -> None:
    destination = torch.zeros((2, 2), dtype=torch.float32)
    target = _dense_target(destination)

    with pytest.raises(ValueError, match="dtype changed"):
        with direct_cpu_destination_buffer(
            target,
            {"weight": torch.zeros((2, 2), dtype=torch.float64)},
        ):
            pass
    with pytest.raises(ValueError, match="device changed"):
        with direct_cpu_destination_buffer(
            replace(target, target_device="cuda:0"),
            {"weight": destination},
        ):
            pass
    with pytest.raises(ValueError, match="shape changed"):
        with direct_cpu_destination_buffer(
            target,
            {"weight": torch.zeros(4, dtype=torch.float32)},
        ):
            pass
    with pytest.raises(ValueError, match="stride .* incompatible"):
        with direct_cpu_destination_buffer(
            target,
            {"weight": torch.zeros((2, 2), dtype=torch.float32).t()},
        ):
            pass
    with pytest.raises(ValueError, match="exceed its 16-byte storage"):
        with direct_cpu_destination_buffer(
            replace(
                target,
                destination_pattern=RepeatedStrideBytePattern(4, 16),
            ),
            {"weight": destination},
        ):
            pass


def test_buffer_pool_rechecks_closed_after_wait() -> None:
    pool = PinnedBufferPool(
        slot_bytes=4,
        slot_count=1,
        use_pinned_memory=False,
    )

    class _CloseWhileWaitingCondition:
        def __enter__(self) -> "_CloseWhileWaitingCondition":
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def wait(self, timeout: float | None = None) -> None:
            del timeout
            pool._closed = True
            pool._free.append(0)

        def notify_all(self) -> None:
            pass

    pool._free.clear()
    pool._condition = cast(Any, _CloseWhileWaitingCondition())

    with pytest.raises(RuntimeError, match="buffer pool is closed"):
        with pool.acquire(1):
            pass
    assert pool._free == [0]
    assert pool._leased == set()


def test_buffer_pool_restores_slot_after_allocation_failure() -> None:
    pool = PinnedBufferPool(
        slot_bytes=4,
        slot_count=1,
        use_pinned_memory=False,
    )
    allocate = BufferSlot.allocate
    attempts = 0

    def allocate_once(size_bytes: int, *, pinned: bool) -> BufferSlot:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise MemoryError("injected allocation failure")
        return allocate(size_bytes, pinned=pinned)

    try:
        with mock.patch.object(BufferSlot, "allocate", side_effect=allocate_once):
            with pytest.raises(MemoryError, match="injected allocation failure"):
                with pool.acquire(1):
                    pass
            with pool.acquire(1, timeout=0.1) as slot:
                assert len(slot.view) == 4
    finally:
        pool.close()


def test_data_client_cache_is_bounded_lru_and_cleanup_closes_clients(
    tmp_path: Path,
) -> None:
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="data-client-lru",
        storage=_MemoryStorage({}),
        target_state_dict={},
        shared_memory_directory=tmp_path,
        local_targets=(),
    )
    session = loader_module._CooperativeLoadSession(request, _config())
    owners = tuple(f"owner-{index}" for index in range(6))
    session._node_data_urls = {
        owner: f"http://127.0.0.1:{10000 + index}" for index, owner in enumerate(owners)
    }
    created: dict[str, _ClosingClient] = {}

    def create_client(url: str) -> _ClosingClient:
        client = _ClosingClient(url)
        created[url] = client
        return client

    with mock.patch.object(session, "_new_data_client", side_effect=create_client):
        cached = [session._data_client(owner) for owner in owners[:4]]
        assert session._data_client(owners[0]) is cached[0]
        session._data_client(owners[4])

        thread_clients = session._data_clients[threading.get_ident()]
        assert tuple(thread_clients) == (owners[2], owners[3], owners[0], owners[4])
        assert len(thread_clients) == loader_module._MAX_DATA_CLIENTS_PER_THREAD
        assert cached[1].close_count == 1

        session._data_client(owners[5])

    assert cached[2].close_count == 1
    session._close_clients()
    assert session._data_clients == {}
    assert all(client.close_count == 1 for client in created.values())


def test_control_client_reconnects_immediately_after_stale_connection(
    tmp_path: Path,
) -> None:
    config = replace(_config(), retry_attempts=3, retry_backoff_seconds=0.5)
    session = loader_module._CooperativeLoadSession(
        _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="control-client-reconnect",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        config,
    )
    control = session._new_control_client("http://127.0.0.1:1")
    data = session._new_data_client("http://127.0.0.1:1")
    response = mock.MagicMock()
    response.getheader.return_value = "0"
    response.read.return_value = b""

    assert control._retry_delay == config.retry_backoff_seconds
    assert data._retry_delay == config.retry_backoff_seconds
    with (
        mock.patch.object(
            control,
            "_open_response",
            side_effect=(EOFError("stale connection"), (201, response)),
        ) as open_response,
        mock.patch.object(transport_module.time, "sleep") as sleep,
    ):
        control.put_blob("tag", b"value")

    assert open_response.call_count == 2
    sleep.assert_not_called()


def test_control_client_immediate_retries_remain_bounded(tmp_path: Path) -> None:
    config = replace(_config(), retry_attempts=3, retry_backoff_seconds=0.5)
    session = loader_module._CooperativeLoadSession(
        _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="control-client-bounded-retry",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        config,
    )
    control = session._new_control_client("http://127.0.0.1:1")

    with (
        mock.patch.object(
            control,
            "_open_response",
            side_effect=EOFError("persistent failure"),
        ) as open_response,
        mock.patch.object(transport_module.time, "sleep") as sleep,
        pytest.raises(TransportError, match="failed after retries"),
    ):
        control.put_blob("tag", b"value")

    assert open_response.call_count == config.retry_attempts
    assert sleep.call_args_list == [mock.call(1.0)]


def test_control_client_preserves_backoff_for_connection_failure(
    tmp_path: Path,
) -> None:
    config = replace(_config(), retry_attempts=3, retry_backoff_seconds=0.5)
    session = loader_module._CooperativeLoadSession(
        _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="control-client-transient-failure",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        config,
    )
    control = session._new_control_client("http://127.0.0.1:1")
    response = mock.MagicMock()
    response.getheader.return_value = "0"
    response.read.return_value = b""

    with (
        mock.patch.object(
            control,
            "_open_response",
            side_effect=(ConnectionRefusedError("not ready"), (201, response)),
        ) as open_response,
        mock.patch.object(transport_module.time, "sleep") as sleep,
    ):
        control.put_blob("tag", b"value")

    assert open_response.call_count == 2
    sleep.assert_called_once_with(config.retry_backoff_seconds)


def test_single_process_load_is_deterministic_and_cleans_up(tmp_path: Path) -> None:
    source = struct.pack("ff", 1.5, -2.0)
    observed_reads: list[tuple[Path, ...]] = []
    observed_results: list[object] = []

    for iteration in range(2):
        target = torch.zeros(2, dtype=torch.float32)
        storage = _MemoryStorage({Path("source-0"): source})
        request = _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token=f"single-{iteration}",
            storage=storage,
            target_state_dict={"weight": target},
            shared_memory_directory=tmp_path,
            local_targets=(_dense_target(target),),
        )

        result = load_cooperatively(request, config=_config())

        torch.testing.assert_close(target, torch.tensor([1.5, -2.0]))
        assert result.storage_bytes_read == len(source)
        assert result.network_bytes_received == 0
        assert result.target_count == 1
        assert result.batch_count == 1
        observed_reads.append(tuple(storage.reads))
        observed_results.append(
            replace(result, elapsed_seconds=0, slowest_rank_seconds=0)
        )
        assert list(tmp_path.iterdir()) == []

    assert observed_reads[0] == observed_reads[1] == (Path("source-0"),)
    assert observed_results[0] == observed_results[1]


def test_default_shared_memory_capacity_is_bounded_by_inflight_batches(
    tmp_path: Path,
) -> None:
    request = replace(
        _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="bounded-capacity",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        shared_memory_capacity_bytes=None,
    )
    config = CooperativeLoadConfig()

    with mock.patch.object(
        loader_module,
        "recommended_capacity_bytes",
        return_value=1024**5,
    ):
        spec = loader_module._shared_memory_pool_spec(request, config)

    assert spec.capacity_bytes == 512 * 1024**3
    assert spec.chunk_bytes == config.shared_memory_chunk_bytes


def test_large_synthetic_capacity_is_retained_and_plans_two_batches(
    tmp_path: Path,
) -> None:
    gib = 1024**3
    recommended_capacity = 400 * gib
    total_source_bytes = 3 * 1024**4
    source_count = 256
    node_ids = tuple(f"node-{index}" for index in range(8))
    request = replace(
        _request(
            rank=0,
            nodes=(("node-0", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="large-capacity",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        shared_memory_capacity_bytes=None,
    )
    config = CooperativeLoadConfig()

    with mock.patch.object(
        loader_module,
        "recommended_capacity_bytes",
        return_value=recommended_capacity,
    ):
        spec = loader_module._shared_memory_pool_spec(request, config)

    cache = loader_module._ChunkPoolCache()
    assert spec.capacity_bytes == recommended_capacity
    assert cache.can_retain(spec)

    source_bytes, remainder = divmod(total_source_bytes, source_count)
    demands = tuple(
        FqnDemand(
            fqn=f"weight-{source_rank}",
            ranges=(
                ByteRange(
                    source_rank=source_rank,
                    offset=0,
                    length=source_bytes + (source_rank < remainder),
                ),
            ),
        )
        for source_rank in range(source_count)
    )
    planning_result = loader_module._plan_cooperative_resharding_from_merged_demands(
        demands,
        node_ids,
        config.batch_target_bytes,
        config.range_consolidation_gap_bytes,
        source_consumer_bytes_by_node={node_id: {} for node_id in node_ids},
    )

    assert len(planning_result.batches) == 2


def test_default_pool_cache_treats_capacity_above_512_gib_as_one_shot() -> None:
    gib = 1024**3
    cache = loader_module._ChunkPoolCache()
    spec = loader_module._ChunkPoolSpec("/unused", 512 * gib + 1, gib)
    pool = mock.Mock()

    lease = cache.try_acquire(spec, lambda: pool, cleanup_timeout=1)

    assert lease is not None
    assert cache.max_retained_capacity_bytes == 512 * gib
    assert lease.retainable is False
    assert lease.reused is False
    with pytest.raises(RuntimeError, match="one-shot chunk-pool lease"):
        lease.release()
    lease.discard()
    assert cache._active_token is None


def test_shared_memory_capacity_clamps_to_batch_inflight_limit(
    tmp_path: Path,
) -> None:
    gib = 1024**3
    request = replace(
        _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="capacity-clamp",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        shared_memory_capacity_bytes=700 * gib,
    )

    spec = loader_module._shared_memory_pool_spec(
        request,
        CooperativeLoadConfig(),
    )

    assert spec.capacity_bytes == 512 * gib


def test_reusable_pool_keeps_backing_but_refreshes_load_state(
    tmp_path: Path,
) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)
    servers: list[tuple[Any, Any, str, str]] = []

    class RecordingNodeServer(loader_module.NodeServer):
        def start(self) -> Any:
            result = super().start()
            servers.append(
                (
                    self,
                    self._state.segment_index,
                    self._state.load_token,
                    self.control_base_url,
                )
            )
            return result

    metrics_by_load: list[list[tuple[str, Mapping[str, object]]]] = []
    cached_pool: Any | None = None
    cached_index: Any | None = None
    job_directory: Path | None = None
    backing_files: tuple[tuple[str, int], ...] | None = None
    try:
        with mock.patch.object(loader_module, "NodeServer", RecordingNodeServer):
            for iteration, value in enumerate((1.5, -2.0)):
                target = torch.zeros(1, dtype=torch.float32)
                metrics: list[tuple[str, Mapping[str, object]]] = []
                request = _request(
                    rank=0,
                    nodes=(("node-a", (0,)),),
                    rendezvous=InMemoryRendezvous(),
                    session_token=f"cached-{iteration}",
                    storage=_MemoryStorage(
                        {Path(f"source-{iteration}"): struct.pack("f", value)}
                    ),
                    target_state_dict={"weight": target},
                    shared_memory_directory=tmp_path,
                    local_targets=(_dense_target(target, source_rank=iteration),),
                    metric_callback=lambda event, fields, output=metrics: output.append(
                        (event, fields)
                    ),
                )

                load_cooperatively(request, config=_config(), _pool_cache=cache)

                torch.testing.assert_close(target, torch.tensor([value]))
                metrics_by_load.append(metrics)
                if iteration == 0:
                    cached_pool = cache._idle_pool
                    assert cached_pool is not None
                    cached_index = cached_pool.index
                    job_directory = cached_pool.job_directory
                    backing_files = tuple(
                        sorted(
                            (path.name, path.stat().st_ino)
                            for path in job_directory.iterdir()
                        )
                    )
                else:
                    assert cache._idle_pool is cached_pool
                    assert cached_pool is not None
                    assert cached_pool.index is not cached_index
                    assert cached_pool.job_directory == job_directory
                    assert backing_files == tuple(
                        sorted(
                            (path.name, path.stat().st_ino)
                            for path in cached_pool.job_directory.iterdir()
                        )
                    )

        assert len(servers) == 2
        assert servers[0][0] is not servers[1][0]
        assert servers[0][1] is not servers[1][1]
        assert servers[0][2] != servers[1][2]
        with pytest.raises(RuntimeError, match="sealed for pool reuse"):
            servers[0][1].assert_quiescent()
        assert any(
            event == "shared_memory_pool" and fields["cache_reused"] is False
            for event, fields in metrics_by_load[0]
        )
        assert any(
            event == "shared_memory_pool" and fields["cache_reused"] is True
            for event, fields in metrics_by_load[1]
        )
        for metrics in metrics_by_load:
            pool_metrics = next(
                fields for event, fields in metrics if event == "shared_memory_pool"
            )
            assert pool_metrics["cache_retain_ceiling_bytes"] == 128
            assert pool_metrics["cache_retained_capacity_bytes"] == 64
            latency_names = {
                fields["name"] for event, fields in metrics if event == "latency_ms"
            }
            assert "cleanup.pool_cache_release" in latency_names
            assert "cleanup.pool_cleanup" not in latency_names
    finally:
        cache.close(timeout=10)

    assert list(tmp_path.iterdir()) == []


def test_active_pool_cache_lease_is_rejected_without_second_pool(
    tmp_path: Path,
) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)
    spec = loader_module._ChunkPoolSpec(str(tmp_path), 64, 4)
    lease = cache.try_acquire(
        spec,
        lambda: loader_module.ChunkPool(
            capacity_bytes=64,
            chunk_bytes=4,
            job_token="active-lease",
            directory=tmp_path,
        ),
        cleanup_timeout=10,
    )
    assert lease is not None
    target = torch.zeros(1)
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="busy-cache",
        storage=_MemoryStorage({Path("source-0"): struct.pack("f", 3.0)}),
        target_state_dict={"weight": target},
        shared_memory_directory=tmp_path,
        local_targets=(_dense_target(target),),
    )
    try:
        with pytest.raises(
            CooperativeLoadUnsupported,
            match="already owns the reusable shared-memory pool",
        ):
            load_cooperatively(request, config=_config(), _pool_cache=cache)

        torch.testing.assert_close(target, torch.zeros(1))
        assert cache._idle_pool is None
        assert list(tmp_path.iterdir()) == [lease.pool.job_directory]
        lease.pool.prepare_for_reuse()
        lease.release()
    finally:
        cache.close(timeout=10)

    assert list(tmp_path.iterdir()) == []


def test_active_pool_cache_lease_is_collectively_unsupported(
    tmp_path: Path,
) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)
    spec = loader_module._ChunkPoolSpec(str(tmp_path), 64, 4)
    lease = cache.try_acquire(
        spec,
        lambda: loader_module.ChunkPool(
            capacity_bytes=64,
            chunk_bytes=4,
            job_token="collective-active-lease",
            directory=tmp_path,
        ),
        cleanup_timeout=10,
    )
    assert lease is not None
    rendezvous = InMemoryRendezvous()
    requests = tuple(
        _request(
            rank=rank,
            nodes=(("node-a", (0, 1)),),
            rendezvous=rendezvous,
            session_token="collective-busy-cache",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        )
        for rank in range(2)
    )
    errors: list[BaseException] = []
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = tuple(
                executor.submit(
                    load_cooperatively,
                    request,
                    config=_config(),
                    _pool_cache=cache,
                )
                for request in requests
            )
            for future in futures:
                with pytest.raises(CooperativeLoadUnsupported) as raised:
                    future.result(timeout=10)
                errors.append(raised.value)

        assert {str(error) for error in errors} == {
            "another cooperative load already owns the reusable shared-memory pool"
        }
        assert list(tmp_path.iterdir()) == [lease.pool.job_directory]
        lease.pool.prepare_for_reuse()
        lease.release()
    finally:
        cache.close(timeout=10)

    assert list(tmp_path.iterdir()) == []


def test_cached_pool_failure_is_destroyed(tmp_path: Path) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)
    target = torch.zeros(1)
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="cached-failure",
        storage=_MemoryStorage(
            {Path("source-0"): struct.pack("f", 1.0)},
            failure=OSError("injected cached read failure"),
        ),
        target_state_dict={"weight": target},
        shared_memory_directory=tmp_path,
        local_targets=(_dense_target(target),),
    )

    with pytest.raises(CooperativeLoadFailure) as raised:
        load_cooperatively(request, config=_config(), _pool_cache=cache)

    assert any(
        "injected cached read failure" in message
        for message in _cause_messages(raised.value)
    )
    assert cache._idle_pool is None
    assert cache._active_token is None
    assert list(tmp_path.iterdir()) == []


def test_pool_cache_replaces_mismatched_spec(tmp_path: Path) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)

    def run(capacity: int, token: str, value: float) -> None:
        target = torch.zeros(1)
        request = replace(
            _request(
                rank=0,
                nodes=(("node-a", (0,)),),
                rendezvous=InMemoryRendezvous(),
                session_token=token,
                storage=_MemoryStorage({Path("source-0"): struct.pack("f", value)}),
                target_state_dict={"weight": target},
                shared_memory_directory=tmp_path,
                local_targets=(_dense_target(target),),
            ),
            shared_memory_capacity_bytes=capacity,
        )
        load_cooperatively(request, config=_config(), _pool_cache=cache)
        torch.testing.assert_close(target, torch.tensor([value]))

    try:
        run(64, "first-spec", 4.0)
        first_pool = cache._idle_pool
        assert first_pool is not None
        first_directory = first_pool.job_directory

        run(32, "second-spec", 5.0)

        second_pool = cache._idle_pool
        assert second_pool is not None
        assert second_pool is not first_pool
        assert not first_directory.exists()
        assert second_pool.capacity_bytes == 32
    finally:
        cache.close(timeout=10)

    assert list(tmp_path.iterdir()) == []


def test_pool_cache_replaces_pool_with_missing_backing_file(tmp_path: Path) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)

    def run(token: str, value: float) -> torch.Tensor:
        target = torch.zeros(1)
        request = _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token=token,
            storage=_MemoryStorage({Path("source-0"): struct.pack("f", value)}),
            target_state_dict={"weight": target},
            shared_memory_directory=tmp_path,
            local_targets=(_dense_target(target),),
        )
        load_cooperatively(request, config=_config(), _pool_cache=cache)
        return target

    try:
        torch.testing.assert_close(run("intact-pool", 8.0), torch.tensor([8.0]))
        first_pool = cache._idle_pool
        assert first_pool is not None
        first_directory = first_pool.job_directory
        next(first_directory.iterdir()).unlink()

        torch.testing.assert_close(run("replaced-pool", 9.0), torch.tensor([9.0]))

        assert cache._idle_pool is not first_pool
        assert not first_directory.exists()
    finally:
        cache.close(timeout=10)

    assert list(tmp_path.iterdir()) == []


def test_pool_capacity_blocks_third_inflight_batch(tmp_path: Path) -> None:
    active_reservation_counts: list[int] = []
    pools: list[Any] = []

    class RecordingChunkPool(loader_module.ChunkPool):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            pools.append(self)

        def reserve(self, *args: Any, **kwargs: Any) -> Any:
            reservation = super().reserve(*args, **kwargs)
            active_reservation_counts.append(self.active_reservation_count)
            return reservation

    targets = {name: torch.zeros(1) for name in ("first", "second", "third")}
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="three-batches",
        storage=_MemoryStorage(
            {
                Path(f"source-{index}"): struct.pack("f", float(index + 1))
                for index in range(3)
            }
        ),
        target_state_dict=targets,
        shared_memory_directory=tmp_path,
        local_targets=tuple(
            _dense_target(target, target_fqn=name, source_rank=index)
            for index, (name, target) in enumerate(targets.items())
        ),
    )

    with mock.patch.object(loader_module, "ChunkPool", RecordingChunkPool):
        result = load_cooperatively(
            request,
            config=replace(_config(), batch_target_bytes=4),
        )

    assert result.batch_count == 3
    assert len(pools) == 1
    assert pools[0].capacity_bytes == 8
    assert max(active_reservation_counts) == 2
    assert list(tmp_path.iterdir()) == []


def test_pool_cache_does_not_retain_oversized_custom_pool(tmp_path: Path) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=32)
    metrics: list[tuple[str, Mapping[str, object]]] = []
    target = torch.zeros(1)
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="oversized-cache-pool",
        storage=_MemoryStorage({Path("source-0"): struct.pack("f", 6.0)}),
        target_state_dict={"weight": target},
        shared_memory_directory=tmp_path,
        local_targets=(_dense_target(target),),
        metric_callback=lambda event, fields: metrics.append((event, fields)),
    )

    load_cooperatively(request, config=_config(), _pool_cache=cache)

    torch.testing.assert_close(target, torch.tensor([6.0]))
    assert cache._idle_pool is None
    assert any(
        event == "shared_memory_pool"
        and fields["capacity_bytes"] == 64
        and fields["cache_retained"] is False
        and fields["cache_retain_ceiling_bytes"] == 32
        and fields["cache_retained_capacity_bytes"] == 0
        for event, fields in metrics
    )
    assert list(tmp_path.iterdir()) == []


def test_server_close_failure_destroys_cached_pool(tmp_path: Path) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)
    target = torch.zeros(1)
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="server-close-failure",
        storage=_MemoryStorage({Path("source-0"): struct.pack("f", 7.0)}),
        target_state_dict={"weight": target},
        shared_memory_directory=tmp_path,
        local_targets=(_dense_target(target),),
    )
    original_close = loader_module.NodeServer.close

    def fail_after_close(server: Any) -> None:
        original_close(server)
        raise OSError("injected server close failure")

    with (
        mock.patch.object(loader_module.NodeServer, "close", fail_after_close),
        pytest.raises(CooperativeLoadFailure) as raised,
    ):
        load_cooperatively(request, config=_config(), _pool_cache=cache)

    assert any(
        "injected server close failure" in message
        for message in _cause_messages(raised.value)
    )
    assert cache._idle_pool is None
    assert cache._active_token is None
    assert list(tmp_path.iterdir()) == []


def test_compact_execution_plan_runs_multiple_batches_with_zero_target(
    tmp_path: Path,
) -> None:
    empty = torch.zeros(0, dtype=torch.float32)
    first = torch.zeros(1, dtype=torch.float32)
    second = torch.zeros(1, dtype=torch.float32)
    storage = _MemoryStorage(
        {
            Path("source-0"): struct.pack("f", 3.0),
            Path("source-1"): struct.pack("f", 4.0),
        }
    )
    metrics: list[tuple[str, Mapping[str, object]]] = []
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="compact-multi-batch",
        storage=storage,
        target_state_dict={"empty": empty, "first": first, "second": second},
        shared_memory_directory=tmp_path,
        local_targets=(
            _dense_target(empty, target_fqn="empty"),
            _dense_target(first, target_fqn="first", source_rank=0),
            _dense_target(second, target_fqn="second", source_rank=1),
        ),
        metric_callback=lambda event, fields: metrics.append((event, fields)),
    )

    result = load_cooperatively(
        request,
        config=replace(_config(), batch_target_bytes=4),
    )

    assert empty.numel() == 0
    torch.testing.assert_close(first, torch.tensor([3.0]))
    torch.testing.assert_close(second, torch.tensor([4.0]))
    assert result.target_count == 3
    assert result.batch_count == 2
    assert tuple(storage.reads) == (Path("source-0"), Path("source-1"))
    projection = next(
        fields
        for event, fields in metrics
        if event == "latency_ms" and fields["name"] == "exchange_plan.project"
    )
    assert projection["batch_count"] == 2
    assert projection["local_target_count"] == 2
    assert projection["local_download_range_count"] == 2


def test_batch_barriers_wait_for_active_download_nodes_and_all_done_nodes(
    tmp_path: Path,
) -> None:
    session = loader_module._CooperativeLoadSession(
        _request(
            rank=0,
            nodes=(("node-a", (0,)), ("node-b", (1,))),
            rendezvous=InMemoryRendezvous(),
            session_token="projected-barriers",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        _config(),
    )

    with (
        mock.patch.object(session, "_take_global", return_value=b"1") as take,
        mock.patch.object(
            session,
            "_broadcast_bytes",
            return_value=b"1",
        ) as broadcast,
    ):
        session._wait_batch_registered(3, ("node-b",))

    take.assert_called_once_with(loader_module._global_node_tag("registered/3", 1))
    broadcast.assert_called_once_with("scatter-start/3", b"1")

    with (
        mock.patch.object(
            session,
            "_gather_local_rank_payloads",
            return_value={0: b"1"},
        ),
        mock.patch.object(session, "_put_global") as put,
        mock.patch.object(session, "_take_global", return_value=b"1") as take,
        mock.patch.object(
            session,
            "_broadcast_bytes",
            return_value=b"1",
        ) as broadcast,
    ):
        session._complete_and_retire_batch(3, ("node-b",))

    put.assert_called_once_with(loader_module._global_node_tag("done/3", 0), b"1")
    assert take.call_args_list == [
        mock.call(loader_module._global_node_tag("downloaded/3", 1)),
        mock.call(loader_module._global_node_tag("done/3", 0)),
        mock.call(loader_module._global_node_tag("done/3", 1)),
    ]
    broadcast.assert_called_once_with("retire/3", b"1")


def test_fused_plan_preserves_gap_zero_capacity_fallback(tmp_path: Path) -> None:
    metrics: list[tuple[str, Mapping[str, object]]] = []
    session = loader_module._CooperativeLoadSession(
        _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="fused-plan-capacity-fallback",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
            metric_callback=lambda event, fields: metrics.append((event, fields)),
        ),
        replace(_config(), range_consolidation_gap_bytes=4),
    )
    session._node_capacities = {"node-a": 8}
    demand_plan = loader_module._GlobalByteDemandPlan(
        demands=(
            loader_module.FqnDemand(
                "weight",
                (
                    loader_module.SourceByteRange(0, 0, 4),
                    loader_module.SourceByteRange(0, 8, 4),
                ),
            ),
        ),
        source_consumer_bytes_by_node={"node-a": {0: 8}},
    )

    with mock.patch.object(
        session,
        "_broadcast_bytes",
        side_effect=lambda _prefix, payload: cast(bytes, payload),
    ):
        execution_plan = session._exchange_plan(demand_plan, ())

    assert len(execution_plan.local_downloads) == 1
    assert execution_plan.local_downloads[0].download_bytes == 8
    assert execution_plan.local_downloads[0].download_ranges == (
        loader_module.SourceByteRange(0, 0, 4),
        loader_module.SourceByteRange(0, 8, 4),
    )
    build_metric = next(
        fields
        for event, fields in metrics
        if event == "latency_ms" and fields["name"] == "exchange_plan.build_works"
    )
    assert build_metric["consolidate_gap_bytes"] == 4
    assert build_metric["effective_consolidate_gap_bytes"] == 0


@pytest.mark.parametrize(
    ("nodes", "expected_reader_rank"),
    [
        ((("node-a", (0, 1)),), 0),
        ((("node-a", (0,)), ("node-b", (1,))), 1),
    ],
)
def test_empty_leader_assigns_source_to_consuming_node(
    tmp_path: Path,
    nodes: Sequence[tuple[str, Sequence[int]]],
    expected_reader_rank: int,
) -> None:
    rendezvous = InMemoryRendezvous()
    source = struct.pack("ff", 3.0, 4.0)
    storages = [
        _MemoryStorage({Path("source-0"): source}),
        _MemoryStorage({Path("source-0"): source}),
    ]
    targets = [torch.zeros(0), torch.zeros(2, dtype=torch.float32)]
    requests = [
        _request(
            rank=rank,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token=f"fanout-{len(nodes)}",
            storage=storages[rank],
            target_state_dict=({} if rank == 0 else {"weight": targets[rank]}),
            shared_memory_directory=tmp_path,
            local_targets=(() if rank == 0 else (_dense_target(targets[rank]),)),
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert errors == []
    assert len(results) == 2
    torch.testing.assert_close(targets[1], torch.tensor([3.0, 4.0]))
    assert {result.network_bytes_received for result in results} == {0}
    assert storages[expected_reader_rank].reads == [Path("source-0")]
    assert storages[1 - expected_reader_rank].reads == []
    assert list(tmp_path.iterdir()) == []


def test_assignment_affinity_counts_identical_rank_consumers(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    source = struct.pack("ff", 3.0, 4.0)
    storages = [
        _MemoryStorage({Path("source-0"): source}),
        _MemoryStorage({Path("source-0"): source}),
    ]
    targets = [torch.zeros(2), torch.zeros(2)]
    metrics: list[list[tuple[str, Mapping[str, object]]]] = [[], []]
    requests = [
        _request(
            rank=rank,
            nodes=(("node-a", (0, 1)),),
            rendezvous=rendezvous,
            session_token="consumer-multiplicity",
            storage=storages[rank],
            target_state_dict={"weight": targets[rank]},
            shared_memory_directory=tmp_path,
            local_targets=(_dense_target(targets[rank]),),
            metric_callback=lambda event, fields, rank=rank: metrics[rank].append(
                (event, fields)
            ),
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert errors == []
    assert len(results) == 2
    for target in targets:
        torch.testing.assert_close(target, torch.tensor([3.0, 4.0]))
    locality = [
        fields
        for event, fields in metrics[0]
        if event == "exchange_plan.assignment_locality"
    ]
    assert len(locality) == 1
    assert locality[0]["rank"] == 0
    assert locality[0]["total_consumer_bytes"] == 16
    assert locality[0]["local_consumer_bytes"] == 16
    assert locality[0]["remote_consumer_bytes"] == 0
    assert locality[0]["baseline_remote_consumer_bytes"] == 0
    assert locality[0]["theoretical_max_local_consumer_bytes"] == 16
    assert locality[0]["local_fraction"] == 1.0
    assert storages[0].reads == [Path("source-0")]
    assert storages[1].reads == []
    assert list(tmp_path.iterdir()) == []


def test_metadata_path_deduplicates_alias_and_loads_strided_dtype_conversion(
    tmp_path: Path,
) -> None:
    source = struct.pack("ffff", 1.0, 99.0, 2.0, 99.0)
    target = torch.zeros(2, dtype=torch.float16)
    plan = LoadPlan(
        offsets=(0,),
        sizes=(2,),
        src_rank=0,
        src_fqn="source",
        src_offsets=(0, 0),
        src_sizes=(2, 1),
        src_elem_size=4,
        src_dtype="torch.float32",
    )
    provider = _StaticMetadataProvider(
        {
            0: {
                "source": SourceTensorMetadata(
                    fqn="source",
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=len(source),
                    shape=(2, 2),
                    stride=(2, 1),
                    dtype="torch.float32",
                    element_size_bytes=4,
                )
            }
        }
    )
    metrics: list[tuple[str, Mapping[str, object]]] = []
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="metadata",
        storage=_MemoryStorage({Path("source-0"): source}),
        target_state_dict={"alias": target, "canonical": target},
        shared_memory_directory=tmp_path,
        local_load_plan={"alias": (plan,), "canonical": (plan,)},
        metadata_provider=provider,
        metric_callback=lambda event, fields: metrics.append((event, fields)),
    )

    result = load_cooperatively(request, config=_config())

    torch.testing.assert_close(target, torch.tensor([1.0, 2.0], dtype=torch.float16))
    assert provider.demands == [{0: frozenset({"source"})}]
    assert result.target_count == 1
    assert any(
        event == "aliases" and fields["alias_count"] == 1 for event, fields in metrics
    )
    config_metrics = [fields for event, fields in metrics if event == "config"]
    assert len(config_metrics) == 1
    assert config_metrics[0]["download_workers"] == _config().download_workers
    phase_metrics = [fields for event, fields in metrics if event == "phase"]
    assert phase_metrics
    assert all(float(fields["latency_ms"]) >= 0 for fields in phase_metrics)
    latency_names = {
        fields["name"] for event, fields in metrics if event == "latency_ms"
    }
    assert {
        "prepare_targets.target_mode",
        "prepare_targets.dedupe_aliases",
        "prepare_targets.build_source_demands",
        "prepare_targets.inspect_archives",
        "prepare_targets.resolve_targets",
        "exchange_demands.decode_local",
        "exchange_demands.deduplicate_local",
        "exchange_demands.decode_nodes",
        "exchange_demands.deduplicate_nodes",
        "exchange_demands.merge_world",
        "exchange_plan.fused_build",
        "exchange_plan.assign_sources",
        "exchange_plan.plan_batches",
        "exchange_plan.build_works",
        "exchange_plan.broadcast",
        "exchange_plan.decode",
        "exchange_plan.project",
        "cleanup.server_close",
        "cleanup.pool_cleanup",
        "cleanup.total",
    }.issubset(latency_names)
    demand_decode_metrics = [
        fields
        for event, fields in metrics
        if event == "latency_ms"
        and fields["name"]
        in {"exchange_demands.decode_local", "exchange_demands.decode_nodes"}
    ]
    assert len(demand_decode_metrics) == 2
    assert all(fields["input_payload_count"] == 1 for fields in demand_decode_metrics)
    assert all(
        int(fields["compressed_input_bytes"]) > 0 for fields in demand_decode_metrics
    )
    demand_deduplicate_metrics = [
        fields
        for event, fields in metrics
        if event == "latency_ms"
        and fields["name"]
        in {
            "exchange_demands.deduplicate_local",
            "exchange_demands.deduplicate_nodes",
        }
    ]
    assert len(demand_deduplicate_metrics) == 2
    assert all(
        fields["input_payload_count"] == fields["unique_payload_count"] == 1
        for fields in demand_deduplicate_metrics
    )
    demand_merge_metric = next(
        fields
        for event, fields in metrics
        if event == "latency_ms" and fields["name"] == "exchange_demands.merge_world"
    )
    assert float(demand_merge_metric["merge_decode_ms"]) >= 0
    assert float(demand_merge_metric["merge_input_iteration_ms"]) >= 0
    stream_metrics = [fields for event, fields in metrics if event == "stream_read"]
    assert len(stream_metrics) == 1
    assert stream_metrics[0]["batch_index"] == 0
    assert stream_metrics[0]["source_rank"] == 0
    assert float(stream_metrics[0]["queue_latency_ms"]) >= 0
    assert float(stream_metrics[0]["latency_ms"]) >= 0
    batch_summaries = [fields for event, fields in metrics if event == "batch_summary"]
    assert len(batch_summaries) == 1
    assert "readiness_retry_operation_count" in batch_summaries[0]
    load_total = [fields for event, fields in metrics if event == "load_total"]
    assert len(load_total) == 1
    assert load_total[0]["succeeded"] is True
    assert float(load_total[0]["latency_ms"]) >= 0
    assert list(tmp_path.iterdir()) == []


def test_metadata_path_accepts_empty_root_fqn(tmp_path: Path) -> None:
    source = struct.pack("f", 7.0)
    target = torch.zeros(1)
    plan = LoadPlan(
        offsets=(0,),
        sizes=(1,),
        src_rank=0,
        src_fqn="",
        src_offsets=(0,),
        src_sizes=(1,),
        src_elem_size=4,
        src_dtype="torch.float32",
    )
    provider = _StaticMetadataProvider(
        {
            0: {
                "": SourceTensorMetadata(
                    fqn="",
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=len(source),
                    shape=(1,),
                    stride=(1,),
                    dtype="torch.float32",
                    element_size_bytes=4,
                )
            }
        }
    )
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="root-fqn",
        storage=_MemoryStorage({Path("source-0"): source}),
        target_state_dict={"": target},
        shared_memory_directory=tmp_path,
        local_load_plan={"": (plan,)},
        metadata_provider=provider,
    )

    result = load_cooperatively(request, config=_config())

    torch.testing.assert_close(target, torch.tensor([7.0]))
    assert provider.demands == [{0: frozenset({""})}]
    assert result.target_count == 1
    assert list(tmp_path.iterdir()) == []


def test_empty_load_plan_rank_participates_in_metadata_collectives(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0, 1)),)
    source = struct.pack("f", 11.0)
    target = torch.zeros(1)
    plan = LoadPlan(
        offsets=(0,),
        sizes=(1,),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=(1,),
        src_elem_size=4,
        src_dtype="torch.float32",
    )
    provider = _StaticMetadataProvider(
        {
            0: {
                "weight": SourceTensorMetadata(
                    fqn="weight",
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=len(source),
                    shape=(1,),
                    stride=(1,),
                    dtype="torch.float32",
                    element_size_bytes=4,
                )
            }
        }
    )
    storages = [
        _MemoryStorage({Path("source-0"): source}),
        _MemoryStorage({Path("source-0"): source}),
    ]
    requests = [
        _request(
            rank=rank,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token="empty-plan",
            storage=storages[rank],
            target_state_dict=({} if rank == 0 else {"weight": target}),
            shared_memory_directory=tmp_path,
            local_load_plan=({} if rank == 0 else {"weight": (plan,)}),
            metadata_provider=provider,
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert errors == []
    assert len(results) == 2
    torch.testing.assert_close(target, torch.tensor([11.0]))
    assert provider.demands == [{0: frozenset({"weight"})}]
    assert storages[0].reads == [Path("source-0")]
    assert storages[1].reads == []
    assert list(tmp_path.iterdir()) == []


def test_metadata_fanout_sends_one_union_record_per_node(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    source = struct.pack("f", 11.0)
    targets = [torch.zeros(1), torch.zeros(1)]
    plan = LoadPlan(
        offsets=(0,),
        sizes=(1,),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=(1,),
        src_elem_size=4,
        src_dtype="torch.float32",
    )
    provider = _StaticMetadataProvider(
        {
            0: {
                "weight": SourceTensorMetadata(
                    fqn="weight",
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=len(source),
                    shape=(1,),
                    stride=(1,),
                    dtype="torch.float32",
                    element_size_bytes=4,
                )
            }
        }
    )
    metrics: list[list[tuple[str, Mapping[str, object]]]] = [[], []]
    requests = [
        _request(
            rank=rank,
            nodes=(("node-a", (0, 1)),),
            rendezvous=rendezvous,
            session_token="metadata-node-union",
            storage=_MemoryStorage({Path("source-0"): source}),
            target_state_dict={"weight": targets[rank]},
            shared_memory_directory=tmp_path,
            local_load_plan={"weight": (plan,)},
            metadata_provider=provider,
            metric_callback=lambda event, fields, rank=rank: metrics[rank].append(
                (event, fields)
            ),
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert errors == []
    assert len(results) == 2
    for target in targets:
        torch.testing.assert_close(target, torch.tensor([11.0]))
    encoded = [
        fields
        for event, fields in metrics[0]
        if event == "latency_ms"
        and fields["name"] == "prepare_targets.encode_node_metadata"
    ]
    assert len(encoded) == 1
    assert encoded[0]["globally_unique_tensor_count"] == 1
    assert encoded[0]["emitted_tensor_count"] == 1
    for rank_metrics in metrics:
        decoded = [
            fields
            for event, fields in rank_metrics
            if event == "latency_ms"
            and fields["name"] == "prepare_targets.decode_metadata"
        ]
        assert len(decoded) == 1
        assert decoded[0]["node_union_tensor_count"] == 1
        assert decoded[0]["tensor_count"] == 1
    assert provider.demands == [{0: frozenset({"weight"})}]
    assert list(tmp_path.iterdir()) == []


def test_metadata_publisher_trims_unassigned_provider_records(tmp_path: Path) -> None:
    source = struct.pack("f", 13.0)
    target = torch.zeros(1)
    plan = LoadPlan(
        offsets=(0,),
        sizes=(1,),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=(1,),
        src_elem_size=4,
        src_dtype="torch.float32",
    )
    provider = _StaticMetadataProvider(
        {
            0: {
                "weight": SourceTensorMetadata(
                    fqn="weight",
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=len(source),
                    shape=(1,),
                    stride=(1,),
                    dtype="torch.float32",
                    element_size_bytes=4,
                ),
                "unused": SourceTensorMetadata(
                    fqn="unused",
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=len(source),
                    shape=(1,),
                    stride=(1,),
                    dtype="torch.float32",
                    element_size_bytes=4,
                ),
            },
            9: {
                "unassigned": SourceTensorMetadata(
                    fqn="unassigned",
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=len(source),
                    shape=(1,),
                    stride=(1,),
                    dtype="torch.float32",
                    element_size_bytes=4,
                )
            },
        }
    )
    metrics: list[tuple[str, Mapping[str, object]]] = []
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="metadata-trim-extras",
        storage=_MemoryStorage({Path("source-0"): source}),
        target_state_dict={"weight": target},
        shared_memory_directory=tmp_path,
        local_load_plan={"weight": (plan,)},
        metadata_provider=provider,
        metric_callback=lambda event, fields: metrics.append((event, fields)),
    )

    result = load_cooperatively(request, config=_config())

    torch.testing.assert_close(target, torch.tensor([13.0]))
    assert result.target_count == 1
    merge_metric = next(
        fields
        for event, fields in metrics
        if event == "latency_ms"
        and fields["name"] == "prepare_targets.merge_metadata_outcomes"
    )
    assert merge_metric["source_rank_count"] == 1
    assert merge_metric["tensor_count"] == 1


def test_metadata_publisher_rejects_malformed_record_before_writes(
    tmp_path: Path,
) -> None:
    source = struct.pack("f", 13.0)
    target = torch.full((1,), -1.0)
    item = SourceTensorMetadata(
        fqn="weight",
        checkpoint_offset_bytes=0,
        storage_offset_elements=0,
        storage_nbytes=len(source),
        shape=(1,),
        stride=(1,),
        dtype="torch.float32",
        element_size_bytes=4,
    )
    object.__setattr__(item, "storage_nbytes", 1)
    plan = LoadPlan(
        offsets=(0,),
        sizes=(1,),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=(1,),
        src_elem_size=4,
        src_dtype="torch.float32",
    )
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="metadata-malformed-prewrite",
        storage=_MemoryStorage({Path("source-0"): source}),
        target_state_dict={"weight": target},
        shared_memory_directory=tmp_path,
        local_load_plan={"weight": (plan,)},
        metadata_provider=_StaticMetadataProvider({0: {"weight": item}}),
    )

    with pytest.raises(CooperativeLoadFailure) as raised:
        load_cooperatively(request, config=_config())

    assert raised.value.target_writes_started is False
    assert any(
        "storage contains" in message for message in _cause_messages(raised.value)
    )
    torch.testing.assert_close(target, torch.full((1,), -1.0))
    assert list(tmp_path.iterdir()) == []


def test_metadata_partitioned_wire_succeeds_across_two_nodes(tmp_path: Path) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0,)), ("node-b", (1,)))
    source_values = {
        Path("source-0"): struct.pack("f", 17.0),
        Path("source-1"): struct.pack("f", 19.0),
    }
    metadata = {
        0: {
            "left": SourceTensorMetadata(
                fqn="left",
                checkpoint_offset_bytes=0,
                storage_offset_elements=0,
                storage_nbytes=4,
                shape=(1,),
                stride=(1,),
                dtype="torch.float32",
                element_size_bytes=4,
            )
        },
        1: {
            "right": SourceTensorMetadata(
                fqn="right",
                checkpoint_offset_bytes=0,
                storage_offset_elements=0,
                storage_nbytes=4,
                shape=(1,),
                stride=(1,),
                dtype="torch.float32",
                element_size_bytes=4,
            )
        },
    }
    providers = [_StaticMetadataProvider(metadata), _StaticMetadataProvider(metadata)]
    targets = [{"left": torch.zeros(1), "right": torch.zeros(1)} for _ in range(2)]
    plans = {
        "left": (
            LoadPlan(
                offsets=(0,),
                sizes=(1,),
                src_rank=0,
                src_fqn="left",
                src_offsets=(0,),
                src_sizes=(1,),
                src_elem_size=4,
                src_dtype="torch.float32",
            ),
        ),
        "right": (
            LoadPlan(
                offsets=(0,),
                sizes=(1,),
                src_rank=1,
                src_fqn="right",
                src_offsets=(0,),
                src_sizes=(1,),
                src_elem_size=4,
                src_dtype="torch.float32",
            ),
        ),
    }
    metrics: list[list[tuple[str, Mapping[str, object]]]] = [[], []]
    requests = [
        _request(
            rank=rank,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token="partitioned-metadata-two-node",
            storage=_MemoryStorage(source_values),
            target_state_dict=targets[rank],
            shared_memory_directory=tmp_path,
            local_load_plan=plans,
            metadata_provider=providers[rank],
            metric_callback=lambda event, fields, rank=rank: metrics[rank].append(
                (event, fields)
            ),
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert errors == []
    assert len(results) == 2
    for target in targets:
        torch.testing.assert_close(target["left"], torch.tensor([17.0]))
        torch.testing.assert_close(target["right"], torch.tensor([19.0]))
    assert providers[0].demands == [{0: frozenset({"left"})}]
    assert providers[1].demands == [{1: frozenset({"right"})}]
    merge_metric = next(
        fields
        for event, fields in metrics[0]
        if event == "latency_ms"
        and fields["name"] == "prepare_targets.merge_metadata_outcomes"
    )
    assert merge_metric["source_rank_count"] == 2
    assert merge_metric["tensor_count"] == 2
    assert merge_metric["duplicate_tensor_count"] == 0
    for rank_metrics in metrics:
        decode_metric = next(
            fields
            for event, fields in rank_metrics
            if event == "latency_ms"
            and fields["name"] == "prepare_targets.decode_metadata"
        )
        assert decode_metric["node_union_source_rank_count"] == 2
        assert decode_metric["node_union_tensor_count"] == 2
        assert decode_metric["source_rank_count"] == 2
        assert decode_metric["tensor_count"] == 2
    assert list(tmp_path.iterdir()) == []


def test_metadata_partitioned_wire_failure_is_collective_and_prewrite_across_two_nodes(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0,)), ("node-b", (1,)))
    source = struct.pack("f", 23.0)
    malformed = SourceTensorMetadata(
        fqn="weight",
        checkpoint_offset_bytes=0,
        storage_offset_elements=0,
        storage_nbytes=4,
        shape=(1,),
        stride=(1,),
        dtype="torch.float32",
        element_size_bytes=4,
    )
    object.__setattr__(malformed, "storage_nbytes", 1)
    provider = _StaticMetadataProvider({0: {"weight": malformed}})
    targets = [torch.full((1,), -1.0), torch.full((1,), -1.0)]
    plan = LoadPlan(
        offsets=(0,),
        sizes=(1,),
        src_rank=0,
        src_fqn="weight",
        src_offsets=(0,),
        src_sizes=(1,),
        src_elem_size=4,
        src_dtype="torch.float32",
    )
    requests = [
        _request(
            rank=rank,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token="partitioned-metadata-invalid-two-node",
            storage=_MemoryStorage({Path("source-0"): source}),
            target_state_dict={"weight": targets[rank]},
            shared_memory_directory=tmp_path,
            local_load_plan={"weight": (plan,)},
            metadata_provider=provider,
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert results == []
    assert len(errors) == 2
    assert all(isinstance(error, CooperativeLoadFailure) for error in errors)
    assert all(
        cast(CooperativeLoadFailure, error).target_writes_started is False
        for error in errors
    )
    assert all(
        any("storage contains" in message for message in _cause_messages(error))
        for error in errors
    )
    for target in targets:
        torch.testing.assert_close(target, torch.full((1,), -1.0))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"rank": []}, "not an integer"),
        ({"-1": []}, "not canonical"),
        ({"01": []}, "not canonical"),
        ({"1": [], "01": []}, "not canonical"),
        ({"0": "weight"}, "must be an array"),
        ({"0": [1]}, "strings in an array"),
        ({"0": ["bias", "weight", "bias"]}, "not canonical"),
        ({"0": ["weight", "bias"]}, "not canonical"),
    ],
)
def test_source_metadata_demands_reject_noncanonical_wire(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        loader_module._source_demands_from_wire(payload)


def test_source_metadata_demands_wire_preserves_empty_root_fqn() -> None:
    payload = loader_module._source_demands_to_wire({3: {"", "weight"}, 7: ()})

    assert payload == {"3": ["", "weight"], "7": []}
    assert loader_module._source_demands_from_wire(payload) == {
        3: frozenset({"", "weight"}),
        7: frozenset(),
    }


def test_mixed_target_preparation_modes_are_rejected_collectively(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0, 1)),)
    requests = [
        _request(
            rank=0,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token="mixed-target-modes",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        ),
        _request(
            rank=1,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token="mixed-target-modes",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
        ),
    ]

    started = time.monotonic()
    results, errors = _run_concurrently(requests)

    assert results == []
    assert len(errors) == 2
    assert time.monotonic() - started < 2
    assert all(isinstance(error, CooperativeLoadFailure) for error in errors)
    assert all(
        not error.target_writes_started
        for error in errors
        if isinstance(error, CooperativeLoadFailure)
    )
    assert all(
        any(
            "planned ranks [1], resolved ranks [0]" in message
            for message in _cause_messages(error)
        )
        for error in errors
    )
    assert list(tmp_path.iterdir()) == []


def test_nonshared_local_directory_falls_back_to_data_transport(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0, 1)),)
    source = struct.pack("ff", 7.0, 8.0)
    target = torch.zeros(2, dtype=torch.float32)
    requests = [
        _request(
            rank=rank,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token="nonshared",
            storage=_MemoryStorage({Path("source-0"): source}),
            target_state_dict=({} if rank == 0 else {"weight": target}),
            shared_memory_directory=tmp_path,
            local_targets=(() if rank == 0 else (_dense_target(target),)),
        )
        for rank in range(2)
    ]
    requests[1] = replace(
        requests[1],
        shared_memory_visibility_probe=lambda path, expected: False,
    )

    with mock.patch(
        "torch_checkpointing.experimental.cooperative_resharding.loader.NodeClient.resolve_ranges",
        side_effect=AssertionError(
            "resolve must not be used without shared visibility"
        ),
    ):
        results, errors = _run_concurrently(requests)

    assert errors == []
    assert len(results) == 2
    torch.testing.assert_close(target, torch.tensor([7.0, 8.0]))
    assert {result.network_bytes_received for result in results} == {0}
    assert list(tmp_path.iterdir()) == []


def test_metadata_unsupported_is_collective_deterministic_and_prewrite(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0,)), ("node-b", (1,)))
    requests: list[CooperativeLoadRequest] = []
    for rank in range(2):
        target = torch.zeros(1, dtype=torch.float32)
        plan = LoadPlan(
            offsets=(0,),
            sizes=(1,),
            src_rank=rank,
            src_fqn=f"source-{rank}",
            src_offsets=(0,),
            src_sizes=(1,),
        )
        requests.append(
            _request(
                rank=rank,
                nodes=nodes,
                rendezvous=rendezvous,
                session_token="unsupported",
                storage=_MemoryStorage({Path(f"source-{rank}"): b""}),
                target_state_dict={"weight": target},
                shared_memory_directory=tmp_path,
                local_load_plan={"weight": (plan,)},
                metadata_provider=_UnsupportedMetadataProvider(
                    f"unsupported-on-node-{rank}"
                ),
            )
        )

    results, errors = _run_concurrently(requests)

    assert results == []
    assert len(errors) == 2
    assert all(isinstance(error, CooperativeLoadUnsupported) for error in errors)
    assert {str(error) for error in errors} == {"node 'node-a': unsupported-on-node-0"}
    assert list(tmp_path.iterdir()) == []


def test_oversized_pipeline_plan_is_collectively_unsupported_before_writes(
    tmp_path: Path,
) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0, 1)),)
    source = struct.pack("ff", 1.0, 2.0)
    target = torch.full((2,), -1.0)
    storages = [
        _MemoryStorage({Path("source-0"): source}),
        _MemoryStorage({Path("source-0"): source}),
    ]
    requests = [
        replace(
            _request(
                rank=rank,
                nodes=nodes,
                rendezvous=rendezvous,
                session_token="oversized-plan",
                storage=storages[rank],
                target_state_dict=({} if rank == 0 else {"weight": target}),
                shared_memory_directory=tmp_path,
                local_targets=(() if rank == 0 else (_dense_target(target),)),
            ),
            shared_memory_capacity_bytes=4,
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert results == []
    assert len(errors) == 2
    assert all(isinstance(error, CooperativeLoadUnsupported) for error in errors)
    assert {str(error) for error in errors} == {
        "batch 0 needs 8 bytes on node 'node-a', whose capacity is 4 bytes"
    }
    torch.testing.assert_close(target, torch.full((2,), -1.0))
    assert storages[0].reads == []
    assert storages[1].reads == []
    assert list(tmp_path.iterdir()) == []


def test_download_error_propagates_and_marks_execution_terminal(tmp_path: Path) -> None:
    rendezvous = InMemoryRendezvous()
    nodes = (("node-a", (0,)), ("node-b", (1,)))
    targets = [torch.zeros(0), torch.zeros(2, dtype=torch.float32)]
    requests = [
        _request(
            rank=rank,
            nodes=nodes,
            rendezvous=rendezvous,
            session_token="failure",
            storage=_MemoryStorage(
                {Path("source-0"): b""},
                failure=OSError("injected read failure"),
            ),
            target_state_dict=({} if rank == 0 else {"weight": targets[rank]}),
            shared_memory_directory=tmp_path,
            local_targets=(() if rank == 0 else (_dense_target(targets[rank]),)),
        )
        for rank in range(2)
    ]

    results, errors = _run_concurrently(requests)

    assert results == []
    assert len(errors) == 2
    assert all(isinstance(error, CooperativeLoadFailure) for error in errors)
    assert all(
        error.target_writes_started
        for error in errors
        if isinstance(error, CooperativeLoadFailure)
    )
    assert all(
        any("injected read failure" in message for message in _cause_messages(error))
        for error in errors
    )
    assert list(tmp_path.iterdir()) == []


def test_download_client_setup_failure_is_reported_without_registration_timeout(
    tmp_path: Path,
) -> None:
    target = torch.zeros(1)
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="download-client-failure",
        storage=_MemoryStorage({Path("source-0"): struct.pack("f", 1.0)}),
        target_state_dict={"weight": target},
        shared_memory_directory=tmp_path,
        local_targets=(_dense_target(target),),
    )
    original = loader_module._CooperativeLoadSession._new_control_client

    def fail_download_client(
        session: Any,
        url: str,
    ) -> Any:
        if threading.current_thread().name == "cooperative-download":
            raise RuntimeError("injected download client failure")
        return original(session, url)

    with (
        mock.patch.object(
            loader_module._CooperativeLoadSession,
            "_new_control_client",
            fail_download_client,
        ),
        pytest.raises(CooperativeLoadFailure) as raised,
    ):
        load_cooperatively(request, config=_config())

    assert any(
        "injected download client failure" in message
        for message in _cause_messages(raised.value)
    )
    assert raised.value.target_writes_started
    assert list(tmp_path.iterdir()) == []


def test_fast_source_failure_is_reported_while_another_read_is_hung(
    tmp_path: Path,
) -> None:
    source = struct.pack("f", 1.0)
    storage = _HungAndFailingStorage(
        {
            Path("source-0"): source,
            Path("source-1"): source,
        }
    )
    targets = {
        "first": torch.zeros(1),
        "second": torch.zeros(1),
    }
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="fast-download-failure",
        storage=storage,
        target_state_dict=targets,
        shared_memory_directory=tmp_path,
        local_targets=(
            _dense_target(targets["first"], target_fqn="first", source_rank=0),
            _dense_target(targets["second"], target_fqn="second", source_rank=1),
        ),
    )
    config = replace(
        _config(),
        progress_timeout_seconds=0.25,
        retry_attempts=1,
    )

    started = time.monotonic()
    try:
        with pytest.raises(CooperativeLoadFailure) as raised:
            load_cooperatively(request, config=config)
    finally:
        storage.release.set()

    assert time.monotonic() - started < 2
    assert any(
        "injected fast source failure" in message
        for message in _cause_messages(raised.value)
    )
    deadline = time.monotonic() + 5
    while list(tmp_path.iterdir()) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert list(tmp_path.iterdir()) == []


def test_cached_failure_holds_lease_until_hung_read_cleanup(
    tmp_path: Path,
) -> None:
    cache = loader_module._ChunkPoolCache(max_retained_capacity_bytes=128)
    source = struct.pack("f", 1.0)
    storage = _HungAndFailingStorage(
        {
            Path("source-0"): source,
            Path("source-1"): source,
        }
    )
    targets = {
        "first": torch.zeros(1),
        "second": torch.zeros(1),
    }
    request = _request(
        rank=0,
        nodes=(("node-a", (0,)),),
        rendezvous=InMemoryRendezvous(),
        session_token="cached-fast-download-failure",
        storage=storage,
        target_state_dict=targets,
        shared_memory_directory=tmp_path,
        local_targets=(
            _dense_target(targets["first"], target_fqn="first", source_rank=0),
            _dense_target(targets["second"], target_fqn="second", source_rank=1),
        ),
    )
    config = replace(
        _config(),
        progress_timeout_seconds=0.25,
        retry_attempts=1,
    )

    started = time.monotonic()
    try:
        with pytest.raises(CooperativeLoadFailure) as raised:
            load_cooperatively(request, config=config, _pool_cache=cache)

        assert time.monotonic() - started < 2
        assert any(
            "injected fast source failure" in message
            for message in _cause_messages(raised.value)
        )
        assert cache._active_token is not None
        retained_paths = tuple(tmp_path.iterdir())
        assert len(retained_paths) == 1

        blocked_request = _request(
            rank=0,
            nodes=(("node-a", (0,)),),
            rendezvous=InMemoryRendezvous(),
            session_token="blocked-during-reaping",
            storage=_MemoryStorage({}),
            target_state_dict={},
            shared_memory_directory=tmp_path,
            local_targets=(),
        )
        with pytest.raises(CooperativeLoadUnsupported):
            load_cooperatively(
                blocked_request,
                config=_config(),
                _pool_cache=cache,
            )
        assert tuple(tmp_path.iterdir()) == retained_paths
    finally:
        storage.release.set()

    deadline = time.monotonic() + 5
    while cache._active_token is not None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert cache._active_token is None
    assert cache._idle_pool is None
    assert list(tmp_path.iterdir()) == []
