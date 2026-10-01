# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import math
import random
import threading
import warnings
import zipfile
from collections import OrderedDict
from collections.abc import Collection, Mapping
from pathlib import Path
from time import monotonic
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.utils.serialization import config as serialization_config
from torch_checkpointing.experimental.cooperative_resharding import (
    loader as loader_module,
    metadata as metadata_module,
)
from torch_checkpointing.experimental.cooperative_resharding.metadata import (
    ArchiveMetadataInspectionResult,
    ArchiveMetadataPreflightError,
    merge_source_tensor_metadata_wire,
    MetadataIneligibilityReason,
    MetadataPreflightErrorKind,
    MetadataPreparationEligible,
    MetadataPreparationIneligible,
    prepare_source_tensor_metadata,
    select_source_tensor_metadata_wire,
)
from torch_checkpointing.storage.base_storage import ReadArgs


class _BytesStorage:
    def __init__(
        self,
        data_by_path: Mapping[Path, bytes],
        *,
        seekable: bool = True,
    ) -> None:
        self._data_by_path = data_by_path
        self._seekable = seekable
        self.reads: list[tuple[Path, ReadArgs | None]] = []

    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> io.BytesIO:
        self.reads.append((path, read_args))
        data = self._data_by_path[path]
        if not self._seekable:
            return _NonSeekableBytes(data)
        return io.BytesIO(data)


class _CountingBytes(io.BytesIO):
    def __init__(self, data: bytes) -> None:
        super().__init__(data)
        self.read_count = 0
        self.readinto_count = 0
        self.seek_count = 0

    def read(self, *args: Any, **kwargs: Any) -> bytes:
        self.read_count += 1
        return super().read(*args, **kwargs)

    def readinto(self, buffer: Any, /) -> int:
        self.readinto_count += 1
        return super().readinto(buffer)

    def seek(self, *args: Any, **kwargs: Any) -> int:
        self.seek_count += 1
        return super().seek(*args, **kwargs)


class _CountingBytesStorage:
    def __init__(self, data_by_path: Mapping[Path, bytes]) -> None:
        self._data_by_path = data_by_path
        self.streams: list[_CountingBytes] = []

    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> _CountingBytes:
        stream = _CountingBytes(self._data_by_path[path])
        self.streams.append(stream)
        return stream


class _NonSeekableBytes(io.BytesIO):
    def seekable(self) -> bool:
        return False

    def seek(self, *args: Any, **kwargs: Any) -> int:
        raise io.UnsupportedOperation("stream is not seekable")


class _FailingStorage:
    def stream_read(
        self,
        path: Path,
        read_args: ReadArgs | None = None,
    ) -> io.BytesIO:
        raise OSError("read failed")


class _BoundedConcurrencyAdapter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._two_active = threading.Event()
        self.active = 0
        self.max_active = 0
        self.calls: list[Path] = []

    def inspect(
        self,
        storage: Any,
        path: Path,
        *,
        item_key: str,
        demanded_fqns: Collection[str],
        timeout_seconds: float,
    ) -> ArchiveMetadataInspectionResult:
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append(path)
            if self.active == 2:
                self._two_active.set()
        if not self._two_active.wait(timeout=1):
            raise AssertionError("metadata inspections did not run concurrently")
        try:
            return {
                fqn: metadata_module.SourceTensorMetadata(
                    fqn=fqn,
                    checkpoint_offset_bytes=0,
                    storage_offset_elements=0,
                    storage_nbytes=4,
                    shape=(1,),
                    stride=(1,),
                    dtype="torch.float32",
                    element_size_bytes=4,
                )
                for fqn in demanded_fqns
            }
        finally:
            with self._lock:
                self.active -= 1


class _MixedFailureAdapter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.completed_ranks: set[int] = set()

    def inspect(
        self,
        storage: Any,
        path: Path,
        *,
        item_key: str,
        demanded_fqns: Collection[str],
        timeout_seconds: float,
    ) -> ArchiveMetadataInspectionResult:
        source_rank = int(path.stem.removeprefix("rank-"))
        try:
            if source_rank == 0:
                return MetadataPreparationIneligible(
                    reason=MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
                    detail="unsupported archive",
                    path=path,
                )
            raise ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
                path,
                "corrupt archive",
            )
        finally:
            with self._lock:
                self.completed_ranks.add(source_rank)


class _BlockingAdapter:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.timeout_seconds: float | None = None

    def inspect(
        self,
        storage: Any,
        path: Path,
        *,
        item_key: str,
        demanded_fqns: Collection[str],
        timeout_seconds: float,
    ) -> ArchiveMetadataInspectionResult:
        self.timeout_seconds = timeout_seconds
        self.started.set()
        try:
            self.release.wait(timeout=2)
            return {}
        finally:
            self.finished.set()


class _UnusedCustomValue:
    pass


class _LegacyUnsupportedValue:
    def __init__(self) -> None:
        self.state = 1

    def __setstate__(self, state: object) -> None:
        raise NotImplementedError("unsupported legacy value")


class _DirectUntypedStorageReduce:
    def __reduce__(self) -> tuple[object, tuple[int]]:
        return torch.UntypedStorage, (1024,)


class _TensorSubclass(torch.Tensor):
    pass


def _torch_archive(value: Any, *, pickle_protocol: int = 2) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer, pickle_protocol=pickle_protocol)
    return buffer.getvalue()


def _rewrite_torch_archive(
    data: bytes,
    *,
    drop_suffix: str | None = None,
    replacement_by_suffix: Mapping[str, bytes] | None = None,
) -> bytes:
    source = io.BytesIO(data)
    destination = io.BytesIO()
    replacements = {} if replacement_by_suffix is None else replacement_by_suffix
    with (
        zipfile.ZipFile(source) as reader,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED) as writer,
    ):
        for entry in reader.infolist():
            if drop_suffix is not None and entry.filename.endswith(drop_suffix):
                continue
            payload = reader.read(entry.filename)
            for suffix, replacement in replacements.items():
                if entry.filename.endswith(suffix):
                    payload = replacement
                    break
            writer.writestr(entry.filename, payload)
    return destination.getvalue()


def _archive_with_duplicate_record(data: bytes) -> bytes:
    buffer = io.BytesIO(data)
    with zipfile.ZipFile(buffer, "a") as archive:
        name = next(
            entry.filename
            for entry in archive.infolist()
            if entry.filename.endswith("/data.pkl")
        )
        payload = archive.read(name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            archive.writestr(name, payload)
    return buffer.getvalue()


def _compressed_archive(value: Any) -> bytes:
    source = io.BytesIO(_torch_archive(value))
    destination = io.BytesIO()
    with (
        zipfile.ZipFile(source) as reader,
        zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as writer,
    ):
        for entry in reader.infolist():
            writer.writestr(entry.filename, reader.read(entry.filename))
    return destination.getvalue()


def _prepare(
    storage: Any,
    path: Path,
    *fqns: str,
) -> MetadataPreparationEligible | MetadataPreparationIneligible:
    return prepare_source_tensor_metadata(
        storage,
        {7: path},
        {7: fqns},
        item_key="model",
        timeout_seconds=5.0,
    )


def _prepare_with_metrics(
    storage: Any,
    path: Path,
    *fqns: str,
) -> tuple[
    MetadataPreparationEligible | MetadataPreparationIneligible,
    list[tuple[str, Mapping[str, object]]],
]:
    metrics: list[tuple[str, Mapping[str, object]]] = []
    result = prepare_source_tensor_metadata(
        storage,
        {7: path},
        {7: fqns},
        item_key="model",
        timeout_seconds=5.0,
        _metric_callback=lambda event, fields: metrics.append((event, fields)),
    )
    return result, metrics


def _archive_decode_metric(
    metrics: Collection[tuple[str, Mapping[str, object]]],
) -> Mapping[str, object]:
    matches = [
        fields for event, fields in metrics if event == "metadata_archive_decode"
    ]
    assert len(matches) == 1
    return matches[0]


def _force_legacy_metadata_decode(stream: Any, path: Path) -> Any:
    raise metadata_module._FastMetadataUnsupported("forced legacy decode")


def test_nested_paths_are_deterministic_and_only_demands_are_retained() -> None:
    path = Path("custom/layout/model-rank-seven.bin")
    storage = _BytesStorage(
        {
            path: _torch_archive(
                {
                    "z": torch.arange(3),
                    "encoder": {
                        "layers": [{"weight": torch.arange(6, dtype=torch.float32)}]
                    },
                    "unused_non_tensor": "allowed when it is not demanded",
                }
            )
        }
    )

    result = _prepare(storage, path, "z", "encoder.layers.0.weight")

    assert isinstance(result, MetadataPreparationEligible)
    assert list(result.metadata_by_rank) == [7]
    assert list(result.metadata_by_rank[7]) == ["encoder.layers.0.weight", "z"]
    assert result.metadata_by_rank[7]["encoder.layers.0.weight"].shape == (6,)
    assert [read_path for read_path, _ in storage.reads] == [path]
    read_args = storage.reads[0][1]
    assert isinstance(read_args, ReadArgs)
    assert read_args.pre_read_full_file is False
    assert read_args.timeout_us == 5_000_000


def test_root_tensor_uses_empty_fqn() -> None:
    path = Path("root.pt")

    result = _prepare(
        _BytesStorage({path: _torch_archive(torch.arange(4, dtype=torch.float32))}),
        path,
        "",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert result.metadata_by_rank[7][""].shape == (4,)


def test_lightweight_decode_bounds_archive_reads_and_emits_phase_metrics() -> None:
    path = Path("many-tensors.pt")
    values = {
        f"weight_{index}": torch.arange(index + 1, dtype=torch.float32)
        for index in range(128)
    }
    storage = _CountingBytesStorage({path: _torch_archive(values)})

    result, metrics = _prepare_with_metrics(storage, path, *values)

    assert isinstance(result, MetadataPreparationEligible)
    assert len(result.metadata_by_rank[7]) == 128
    assert len(storage.streams) == 1
    assert storage.streams[0].read_count + storage.streams[0].readinto_count <= 48
    assert storage.streams[0].seek_count <= 80
    decode = _archive_decode_metric(metrics)
    assert decode["mode"] == "fast"
    assert decode["fallback_reason"] == ""
    assert decode["storage_index_mode"] == "calculated"
    assert decode["storage_record_count"] == 128
    assert decode["archive_entry_count"] == 134
    assert int(cast(Any, decode["data_pickle_bytes"])) > 0
    for field in (
        "extraction_latency_ms",
        "latency_ms",
        "metadata_load_latency_ms",
        "storage_index_latency_ms",
        "zip_validation_latency_ms",
    ):
        assert float(cast(Any, decode[field])) >= 0
    assert decode["succeeded"] is True


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint16,
        torch.uint32,
        torch.uint64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.complex32,
        torch.complex64,
    ],
)
def test_lightweight_decode_preserves_dtype_scalar_and_empty_shapes(
    dtype: torch.dtype,
) -> None:
    path = Path(f"dtype-{str(dtype).removeprefix('torch.')}.pt")
    values = {
        "scalar": torch.zeros((), dtype=dtype),
        "empty": torch.empty((0, 3), dtype=dtype),
    }

    result, metrics = _prepare_with_metrics(
        storage := _BytesStorage({path: _torch_archive(values)}),
        path,
        "scalar",
        "empty",
    )

    assert storage.reads
    assert isinstance(result, MetadataPreparationEligible)
    scalar = result.metadata_by_rank[7]["scalar"]
    empty = result.metadata_by_rank[7]["empty"]
    assert scalar.dtype == str(dtype)
    assert scalar.shape == ()
    assert scalar.stride == ()
    assert empty.dtype == str(dtype)
    assert empty.shape == (0, 3)
    assert _archive_decode_metric(metrics)["mode"] == "fast"


@pytest.mark.parametrize("pickle_protocol", [2, 4, 5])
def test_lightweight_decode_supports_pickle_protocols(pickle_protocol: int) -> None:
    path = Path(f"protocol-{pickle_protocol}.pt")

    result, metrics = _prepare_with_metrics(
        _BytesStorage(
            {
                path: _torch_archive(
                    {"weight": torch.arange(4, dtype=torch.float32)},
                    pickle_protocol=pickle_protocol,
                )
            }
        ),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert result.metadata_by_rank[7]["weight"].shape == (4,)
    assert _archive_decode_metric(metrics)["mode"] == "fast"


def test_nested_ordered_containers_and_parameters_use_fast_decode() -> None:
    path = Path("ordered-parameter.pt")
    parameter = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    parameter.metadata_note = "ignored"
    value: OrderedDict[str, object] = OrderedDict()
    value["blocks"] = [{"weight": parameter}]
    value["pair"] = (torch.arange(2, dtype=torch.int64), "unused")

    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: _torch_archive(value)}),
        path,
        "blocks.0.weight",
        "pair.0",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert result.metadata_by_rank[7]["blocks.0.weight"].shape == (2, 3)
    assert result.metadata_by_rank[7]["pair.0"].dtype == "torch.int64"
    assert _archive_decode_metric(metrics)["mode"] == "fast"


def test_archive_inspection_is_concurrent_bounded_and_rank_ordered() -> None:
    adapter = _BoundedConcurrencyAdapter()
    metrics: list[tuple[str, Mapping[str, object]]] = []
    source_paths = {rank: Path(f"rank-{rank}.pt") for rank in range(5)}
    demands = {
        rank: (() if rank == 2 else (f"weight.{rank}",)) for rank in reversed(range(5))
    }

    result = prepare_source_tensor_metadata(
        _BytesStorage({}),
        source_paths,
        demands,
        item_key="model",
        timeout_seconds=5.0,
        adapter=adapter,
        _max_workers=2,
        _metric_callback=lambda event, fields: metrics.append((event, fields)),
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert list(result.metadata_by_rank) == list(range(5))
    assert adapter.max_active == 2
    assert sorted(adapter.calls) == [
        path for rank, path in source_paths.items() if rank != 2
    ]
    archive_metrics = [
        fields for event, fields in metrics if event == "metadata_archive"
    ]
    assert len(archive_metrics) == 4
    assert {fields["source_rank"] for fields in archive_metrics} == {0, 1, 3, 4}
    assert (
        max(int(cast(Any, fields["active_worker_count"])) for fields in archive_metrics)
        == 2
    )
    assert all(float(fields["latency_ms"]) >= 0 for fields in archive_metrics)
    summaries = [
        fields for event, fields in metrics if event == "metadata_archive_summary"
    ]
    assert len(summaries) == 1
    assert summaries[0]["archive_count"] == 4
    assert summaries[0]["completed_count"] == 4
    assert summaries[0]["peak_worker_count"] == 2
    assert summaries[0]["worker_count"] == 2
    assert float(summaries[0]["latency_ms"]) >= 0

    with pytest.raises(ValueError, match="_max_workers must be positive"):
        prepare_source_tensor_metadata(
            _BytesStorage({}),
            source_paths,
            {0: {"weight.0"}},
            item_key="model",
            timeout_seconds=5.0,
            adapter=adapter,
            _max_workers=0,
        )


def test_fatal_preflight_error_precedes_lower_rank_ineligible_result() -> None:
    adapter = _MixedFailureAdapter()

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        prepare_source_tensor_metadata(
            _BytesStorage({}),
            {0: Path("rank-0.pt"), 1: Path("rank-1.pt")},
            {0: {"weight"}, 1: {"weight"}},
            item_key="model",
            timeout_seconds=5.0,
            adapter=adapter,
            _max_workers=2,
        )

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE
    assert raised.value.source_rank == 1
    assert adapter.completed_ranks == {0, 1}


def test_stuck_archive_inspection_times_out_without_waiting_for_worker() -> None:
    adapter = _BlockingAdapter()
    safety_release = threading.Timer(1.0, adapter.release.set)
    safety_release.daemon = True
    safety_release.start()
    started = monotonic()
    try:
        with pytest.raises(ArchiveMetadataPreflightError) as raised:
            prepare_source_tensor_metadata(
                _BytesStorage({}),
                {7: Path("blocked.pt")},
                {7: {"weight"}},
                item_key="model",
                timeout_seconds=0.05,
                adapter=adapter,
            )
        elapsed = monotonic() - started
    finally:
        adapter.release.set()
        safety_release.cancel()

    assert adapter.started.is_set()
    assert adapter.finished.wait(timeout=1)
    assert adapter.timeout_seconds == 0.05
    assert elapsed < 0.5
    assert raised.value.kind is MetadataPreflightErrorKind.IO
    assert raised.value.source_rank == 7
    assert "timed out" in raised.value.detail


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, math.inf, math.nan])
def test_timeout_must_be_positive_and_finite(timeout_seconds: float) -> None:
    with pytest.raises(ValueError, match="timeout_seconds"):
        prepare_source_tensor_metadata(
            _BytesStorage({}),
            {},
            {},
            item_key="model",
            timeout_seconds=timeout_seconds,
        )


def test_alias_offsets_and_noncontiguous_strides_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    path = Path("archive.pt")
    storage = _BytesStorage(
        {
            path: _torch_archive(
                {
                    "view": base[1:, 1:4],
                    "transpose": base.t(),
                }
            )
        }
    )

    result, metrics = _prepare_with_metrics(storage, path, "transpose", "view")

    assert isinstance(result, MetadataPreparationEligible)
    view = result.metadata_by_rank[7]["view"]
    transpose = result.metadata_by_rank[7]["transpose"]
    assert view.checkpoint_offset_bytes == transpose.checkpoint_offset_bytes
    assert view.storage_nbytes == transpose.storage_nbytes == 80
    assert view.dtype == transpose.dtype == "torch.float32"
    assert view.storage_offset_elements == 6
    assert view.shape == (3, 3)
    assert view.stride == (5, 1)
    assert transpose.storage_offset_elements == 0
    assert transpose.shape == (5, 4)
    assert transpose.stride == (1, 5)
    assert _archive_decode_metric(metrics)["mode"] == "fast"

    monkeypatch.setattr(
        metadata_module,
        "_load_lightweight_meta_archive",
        _force_legacy_metadata_decode,
    )
    legacy = _prepare(storage, path, "transpose", "view")
    assert legacy == result


def test_sparse_and_nested_tensors_are_recoverably_unsupported() -> None:
    values = {
        "sparse": torch.sparse_coo_tensor(
            torch.tensor([[0, 1], [1, 0]]),
            torch.tensor([1.0, 2.0]),
            (2, 2),
        ),
        "nested": torch.nested.nested_tensor(
            [torch.arange(2, dtype=torch.float32), torch.arange(3, dtype=torch.float32)]
        ),
    }

    for name, value in values.items():
        path = Path(f"{name}.pt")
        result, metrics = _prepare_with_metrics(
            _BytesStorage({path: _torch_archive({"weight": value})}),
            path,
            "weight",
        )

        assert isinstance(result, MetadataPreparationIneligible)
        assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_VALUE
        assert _archive_decode_metric(metrics)["mode"] == "legacy"


def test_conjugate_and_negative_view_bits_are_recoverably_unsupported() -> None:
    values = {
        "conjugate": torch.tensor([1 + 2j, 3 + 4j]).conj(),
        "negative": torch.arange(3, dtype=torch.float32)._neg_view(),
        "both": torch.tensor([1 + 2j, 3 + 4j]).conj()._neg_view(),
    }

    for name, value in values.items():
        path = Path(f"{name}.pt")
        result, metrics = _prepare_with_metrics(
            _BytesStorage({path: _torch_archive({"weight": value})}),
            path,
            "weight",
        )

        assert isinstance(result, MetadataPreparationIneligible)
        assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_VALUE
        assert "view bits" in result.detail
        assert _archive_decode_metric(metrics)["mode"] == "fast"


def test_demanded_non_tensor_is_preparation_ineligible() -> None:
    path = Path("non-tensor.pt")
    result = _prepare(
        _BytesStorage({path: _torch_archive({"step": 12})}),
        path,
        "step",
    )

    assert isinstance(result, MetadataPreparationIneligible)
    assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_VALUE
    assert result.source_rank == 7
    assert result.path == path


def test_quantized_tensor_is_preparation_ineligible() -> None:
    path = Path("quantized.pt")
    quantized = torch.quantize_per_tensor(
        torch.arange(8, dtype=torch.float32),
        scale=0.1,
        zero_point=3,
        dtype=torch.quint8,
    )

    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: _torch_archive({"weight": quantized})}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationIneligible)
    assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_VALUE
    assert _archive_decode_metric(metrics)["mode"] == "legacy"


def test_missing_checkpoint_offset_is_preparation_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("missing-offset.pt")
    original_load = torch.load

    def load_without_offset(*args: Any, **kwargs: Any) -> Any:
        loaded = original_load(*args, **kwargs)
        delattr(loaded["weight"].untyped_storage(), "_checkpoint_offset")
        return loaded

    monkeypatch.setattr(
        metadata_module,
        "_load_lightweight_meta_archive",
        _force_legacy_metadata_decode,
    )
    monkeypatch.setattr(metadata_module.torch, "load", load_without_offset)
    result = _prepare(
        _BytesStorage({path: _torch_archive({"weight": torch.arange(4)})}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationIneligible)
    assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_VALUE
    assert "checkpoint offset" in result.detail


def test_checkpoint_offset_must_match_exact_storage_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("wrong-record-offset.pt")
    original_load = torch.load

    def load_with_shifted_offset(*args: Any, **kwargs: Any) -> Any:
        loaded = original_load(*args, **kwargs)
        storage = loaded["weight"].untyped_storage()
        storage._checkpoint_offset += 1
        return loaded

    monkeypatch.setattr(
        metadata_module,
        "_load_lightweight_meta_archive",
        _force_legacy_metadata_decode,
    )
    monkeypatch.setattr(metadata_module.torch, "load", load_with_shifted_offset)
    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(
            _BytesStorage({path: _torch_archive({"weight": torch.arange(4)})}),
            path,
            "weight",
        )

    assert raised.value.kind is MetadataPreflightErrorKind.INVALID_METADATA
    assert "storage record" in raised.value.detail


def test_dtensor_metadata_uses_its_local_tensor_without_legacy_rebuild(
    tmp_path: Path,
) -> None:
    if dist.is_initialized():
        pytest.skip("test requires ownership of the default process group")
    rendezvous = tmp_path / "dtensor-rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        local = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        mesh = DeviceMesh("cpu", [0])
        value = DTensor.from_local(
            local,
            mesh,
            [Shard(0)],
            run_check=False,
            shape=local.shape,
            stride=local.stride(),
        )
        archive = _torch_archive({"weight": value})
    finally:
        dist.destroy_process_group()

    path = Path("dtensor.pt")
    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: archive}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    metadata = result.metadata_by_rank[7]["weight"]
    assert metadata.shape == (3, 4)
    assert metadata.stride == (4, 1)
    assert metadata.dtype == "torch.float32"
    assert _archive_decode_metric(metrics)["mode"] == "fast"


def test_unknown_global_falls_back_to_legacy_decode() -> None:
    path = Path("unknown-global.pt")
    value = {
        "weight": torch.arange(4, dtype=torch.float32),
        "unused": _UnusedCustomValue(),
    }

    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: _torch_archive(value)}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    decode = _archive_decode_metric(metrics)
    assert decode["mode"] == "legacy"
    assert "unsupported serialized global" in str(decode["fallback_reason"])
    assert float(cast(Any, decode["legacy_load_latency_ms"])) >= 0


def test_fallback_reason_metric_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    path = Path("bounded-fallback-metric.pt")

    def reject_fast_decode(stream: Any, archive_path: Path) -> Any:
        raise metadata_module._FastMetadataUnsupported("x" * 1024)

    monkeypatch.setattr(
        metadata_module,
        "_load_lightweight_meta_archive",
        reject_fast_decode,
    )
    result, metrics = _prepare_with_metrics(
        _BytesStorage(
            {path: _torch_archive({"weight": torch.arange(4, dtype=torch.float32)})}
        ),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert len(str(_archive_decode_metric(metrics)["fallback_reason"])) == 512


@pytest.mark.parametrize("device", [torch.device("cpu"), torch.device("cuda", 0)])
def test_serialized_torch_device_does_not_force_legacy_decode(
    device: torch.device,
) -> None:
    path = Path("device-metadata.pt")
    value = {
        "weight": torch.arange(4, dtype=torch.float32),
        "unused_device": device,
    }

    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: _torch_archive(value)}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert _archive_decode_metric(metrics)["mode"] == "fast"


def test_tensor_subclass_falls_back_to_legacy_decode() -> None:
    path = Path("tensor-subclass.pt")
    tensor = torch.arange(4, dtype=torch.float32).as_subclass(_TensorSubclass)

    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: _torch_archive({"weight": tensor})}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert result.metadata_by_rank[7]["weight"].shape == (4,)
    assert _archive_decode_metric(metrics)["mode"] == "legacy"


def test_lightweight_decode_restores_serialization_thread_local_state() -> None:
    path = Path("unknown-global.pt")
    stream = io.BytesIO(_torch_archive({"unused": _UnusedCustomValue()}))
    serialization_tls = torch.serialization._serialization_tls
    sentinel = object()
    previous = serialization_tls.map_location
    serialization_tls.map_location = sentinel
    try:
        with pytest.raises(metadata_module._FastMetadataUnsupported):
            metadata_module._load_lightweight_meta_archive(stream, path)
        assert serialization_tls.map_location is sentinel
    finally:
        serialization_tls.map_location = previous


def test_legacy_decode_restores_serialization_thread_local_state() -> None:
    path = Path("legacy-unsupported.pt")
    stream = io.BytesIO(_torch_archive({"unused": _LegacyUnsupportedValue()}))
    serialization_tls = torch.serialization._serialization_tls
    sentinel = object()
    previous = serialization_tls.map_location
    serialization_tls.map_location = sentinel
    try:
        result = metadata_module._load_meta_archive(stream, path)
        assert isinstance(result, MetadataPreparationIneligible)
        assert serialization_tls.map_location is sentinel
    finally:
        serialization_tls.map_location = previous


def test_known_rebuilds_reject_malformed_arity_and_metadata_flags() -> None:
    with pytest.raises(TypeError):
        cast(Any, metadata_module._rebuild_tensor_v2_descriptor)(
            object(),
            0,
            (1,),
            (1,),
            False,
            {},
            None,
            "unexpected",
        )
    with pytest.raises(TypeError):
        cast(Any, metadata_module._rebuild_parameter_descriptor)(
            object(), False, {}, {}
        )
    with pytest.raises(metadata_module._FastMetadataUnsupported):
        metadata_module._rebuild_tensor_v2_descriptor(
            object(),
            0,
            (1,),
            (1,),
            1,
            {},
        )
    with pytest.raises(metadata_module._FastMetadataUnsupported):
        metadata_module._rebuild_from_type_descriptor(
            metadata_module._rebuild_tensor_subclass_descriptor,
            metadata_module._DTensorMarker,
            (),
            {},
        )
    with pytest.raises(metadata_module._FastMetadataUnsupported):
        metadata_module._tensor_metadata_flags({"conj": 1})
    with pytest.raises(metadata_module._FastMetadataUnsupported):
        metadata_module._tensor_metadata_flags({"unknown": False})
    assert (
        metadata_module._tensor_metadata_flags({"conj": False, "neg": False}) is False
    )
    assert metadata_module._tensor_metadata_flags({"conj": True, "neg": False}) is True


def test_untyped_storage_global_cannot_be_called_by_pickle_reduce() -> None:
    path = Path("direct-storage-reduce.pt")
    stream = io.BytesIO(_torch_archive({"value": _DirectUntypedStorageReduce()}))

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        metadata_module._load_lightweight_meta_archive(stream, path)

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE
    assert "not callable" in raised.value.detail


@pytest.mark.parametrize(
    ("second_nbytes", "second_dtype"),
    [(32, torch.float32), (16, torch.int32)],
)
def test_aliases_require_consistent_storage_size_and_dtype(
    second_nbytes: int,
    second_dtype: torch.dtype,
) -> None:
    first = metadata_module._TensorDescriptor(
        storage=metadata_module._TensorStorageDescriptor(64, 16),
        storage_offset_elements=0,
        shape=(4,),
        stride=(1,),
        dtype=torch.float32,
        has_view_bits=False,
    )
    second = metadata_module._TensorDescriptor(
        storage=metadata_module._TensorStorageDescriptor(64, second_nbytes),
        storage_offset_elements=0,
        shape=(4,),
        stride=(1,),
        dtype=second_dtype,
        has_view_bits=False,
    )

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        metadata_module._extract_demanded_metadata(
            {"first": first, "second": second},
            item_key="model",
            demanded_fqns={"first", "second"},
            storage_records={64: 16},
            path=Path("inconsistent-alias.pt"),
        )

    assert raised.value.kind is MetadataPreflightErrorKind.INVALID_METADATA
    assert "inconsistent size or dtype" in raised.value.detail


def test_nonseekable_stream_is_preparation_ineligible() -> None:
    path = Path("remote.pt")
    storage = _BytesStorage(
        {path: _torch_archive({"weight": torch.arange(4)})},
        seekable=False,
    )

    result = _prepare(storage, path, "weight")

    assert isinstance(result, MetadataPreparationIneligible)
    assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_STORAGE


def test_compressed_torch_archive_is_preparation_ineligible() -> None:
    path = Path("compressed.pt")

    result = _prepare(
        _BytesStorage({path: _compressed_archive({"weight": torch.arange(4)})}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationIneligible)
    assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE


def test_torchscript_archive_is_rejected_before_torch_load_dispatch() -> None:
    path = Path("scripted.pt")
    module = torch.jit.trace(torch.nn.Linear(2, 2), torch.ones(1, 2))
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)

    result = _prepare(_BytesStorage({path: buffer.getvalue()}), path, "weight")

    assert isinstance(result, MetadataPreparationIneligible)
    assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE
    assert "TorchScript" in result.detail


@pytest.mark.parametrize("format_version", [None, b"0"])
def test_missing_or_old_format_marker_uses_header_index(
    format_version: bytes | None,
) -> None:
    path = Path("old-format.pt")
    raw = _torch_archive({"weight": torch.arange(4, dtype=torch.float32)})
    if format_version is None:
        archive = _rewrite_torch_archive(raw, drop_suffix="/.format_version")
    else:
        archive = _rewrite_torch_archive(
            raw,
            replacement_by_suffix={"/.format_version": format_version},
        )

    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: archive}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert result.metadata_by_rank[7]["weight"].shape == (4,)
    decode = _archive_decode_metric(metrics)
    assert decode["mode"] == "fast"
    assert decode["storage_index_mode"] == "headers"


def test_missing_calculated_offset_api_uses_header_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = Path("no-private-api.pt")
    monkeypatch.setattr(
        metadata_module,
        "_collect_calculated_storage_records",
        lambda *args, **kwargs: None,
    )

    result, metrics = _prepare_with_metrics(
        _BytesStorage(
            {path: _torch_archive({"weight": torch.arange(4, dtype=torch.float32)})}
        ),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert _archive_decode_metric(metrics)["storage_index_mode"] == "headers"


def test_duplicate_zip_records_are_fatal_corruption() -> None:
    path = Path("duplicate.pt")
    archive = _archive_with_duplicate_record(
        _torch_archive({"weight": torch.arange(4, dtype=torch.float32)})
    )

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(_BytesStorage({path: archive}), path, "weight")

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE
    assert "duplicate record" in raised.value.detail


def test_calculated_offset_must_match_aligned_record_boundary() -> None:
    path = Path("misaligned.pt")
    archive = _rewrite_torch_archive(
        _torch_archive({"weight": torch.arange(4, dtype=torch.float32)}),
        replacement_by_suffix={"/.storage_alignment": b"32"},
    )

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(_BytesStorage({path: archive}), path, "weight")

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE
    assert "padding or record boundary" in raised.value.detail


def test_nonpositive_storage_alignment_is_fatal_corruption() -> None:
    path = Path("invalid-alignment.pt")
    archive = _rewrite_torch_archive(
        _torch_archive({"weight": torch.arange(4, dtype=torch.float32)}),
        replacement_by_suffix={"/.storage_alignment": b"0"},
    )

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(_BytesStorage({path: archive}), path, "weight")

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE
    assert "alignment must be positive" in raised.value.detail


@pytest.mark.parametrize(
    ("header_offset", "record_size", "expected_data_offset"),
    [(2**32, 1, 2**32 + 64), (16, 2**32 + 16, 128)],
)
def test_zip64_calculated_offsets_validate_against_record_boundaries(
    header_offset: int,
    record_size: int,
    expected_data_offset: int,
) -> None:
    archive = io.BytesIO(_torch_archive({"weight": torch.arange(1)}))
    reader = torch._C.PyTorchFileReader(archive)
    entry = zipfile.ZipInfo("archive/data/0")
    entry.header_offset = header_offset
    entry.file_size = record_size
    data_offset = reader.get_record_offset_no_read(
        header_offset,
        "data/0",
        record_size,
        64,
    )
    assert data_offset == expected_data_offset

    metadata_module._validate_calculated_storage_boundary(
        entry,
        data_offset=data_offset,
        next_header_offset=data_offset + record_size + 24,
        alignment=64,
        path=Path("zip64.pt"),
    )


def test_zip64_calculated_offsets_remain_valid_past_four_gibibytes() -> None:
    archive = io.BytesIO(_torch_archive({"weight": torch.arange(1)}))
    reader = torch._C.PyTorchFileReader(archive)
    record_size = 512 * 1024 * 1024
    next_header_offset = 0

    for index in range(10):
        name = f"data/{index}"
        entry = zipfile.ZipInfo(f"archive/{name}")
        entry.header_offset = next_header_offset
        entry.file_size = record_size
        data_offset = reader.get_record_offset_no_read(
            next_header_offset,
            name,
            record_size,
            64,
        )
        descriptor_bytes = 24 if next_header_offset >= 0xFFFFFFFF else 16
        next_header_offset = data_offset + record_size + descriptor_bytes
        metadata_module._validate_calculated_storage_boundary(
            entry,
            data_offset=data_offset,
            next_header_offset=next_header_offset,
            alignment=64,
            path=Path("large-checkpoint.pt"),
        )

    assert next_header_offset > 2**32


def test_malformed_data_pickle_is_fatal_corruption() -> None:
    path = Path("malformed-data-pickle.pt")
    archive = _rewrite_torch_archive(
        _torch_archive({"weight": torch.arange(4, dtype=torch.float32)}),
        drop_suffix="/.format_version",
        replacement_by_suffix={"/data.pkl": b"not a pickle"},
    )

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(_BytesStorage({path: archive}), path, "weight")

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE


def test_data_pickle_crc_mismatch_is_fatal_corruption() -> None:
    path = Path("bad-data-pickle-crc.pt")
    archive = bytearray(
        _torch_archive({"weight": torch.arange(4, dtype=torch.float32)})
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as reader:
        entry = next(
            entry for entry in reader.infolist() if entry.filename.endswith("/data.pkl")
        )
    stream = io.BytesIO(archive)
    data_offset = metadata_module._zip_entry_data_offset(
        stream,
        entry,
        archive_size=len(archive),
        path=path,
    )
    archive[data_offset + 17] ^= 1

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(_BytesStorage({path: bytes(archive)}), path, "weight")

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE
    assert "CRC" in raised.value.detail


def test_disabled_data_pickle_crc_remains_supported() -> None:
    path = Path("crc-disabled.pt")
    previous = serialization_config.save.compute_crc32
    serialization_config.save.compute_crc32 = False
    try:
        archive = _torch_archive({"weight": torch.arange(4, dtype=torch.float32)})
    finally:
        serialization_config.save.compute_crc32 = previous
    with zipfile.ZipFile(io.BytesIO(archive)) as reader:
        data_pickle = next(
            entry for entry in reader.infolist() if entry.filename.endswith("/data.pkl")
        )
        assert data_pickle.CRC == 0

    result, metrics = _prepare_with_metrics(
        _BytesStorage({path: archive}),
        path,
        "weight",
    )

    assert isinstance(result, MetadataPreparationEligible)
    assert _archive_decode_metric(metrics)["mode"] == "fast"


def test_missing_demanded_fqn_is_invalid_metadata() -> None:
    path = Path("missing-fqn.pt")

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(
            _BytesStorage({path: _torch_archive({"weight": torch.arange(4)})}),
            path,
            "missing",
        )

    assert raised.value.kind is MetadataPreflightErrorKind.INVALID_METADATA
    assert raised.value.source_rank == 7


def test_cyclic_checkpoint_container_is_invalid_metadata() -> None:
    path = Path("cyclic.pt")
    value: list[object] = []
    value.append(value)

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(_BytesStorage({path: _torch_archive(value)}), path, "0")

    assert raised.value.kind is MetadataPreflightErrorKind.INVALID_METADATA
    assert "contains a cycle" in raised.value.detail


def test_excessive_checkpoint_container_depth_is_invalid_metadata() -> None:
    value: object = torch.tensor(1)
    for _ in range(metadata_module._MAX_CHECKPOINT_CONTAINER_DEPTH + 1):
        value = [value]

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        metadata_module._extract_demanded_metadata(
            value,
            item_key="model",
            demanded_fqns={"missing"},
            storage_records={},
            path=Path("too-deep.pt"),
        )

    assert raised.value.kind is MetadataPreflightErrorKind.INVALID_METADATA
    assert "nesting exceeds" in raised.value.detail


def test_corruption_and_io_are_distinct_from_unsupported_archives() -> None:
    corrupt_path = Path("corrupt.pt")
    with pytest.raises(ArchiveMetadataPreflightError) as corrupt:
        _prepare(
            _BytesStorage({corrupt_path: b"PK\x03\x04broken"}),
            corrupt_path,
            "weight",
        )

    assert corrupt.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE
    assert corrupt.value.source_rank == 7

    io_path = Path("io.pt")
    with pytest.raises(ArchiveMetadataPreflightError) as io_failure:
        _prepare(_FailingStorage(), io_path, "weight")

    assert io_failure.value.kind is MetadataPreflightErrorKind.IO
    assert io_failure.value.source_rank == 7


def test_invalid_utf8_in_zip_central_directory_is_fatal_corruption() -> None:
    path = Path("invalid-filename.pt")
    archive = bytearray(
        _torch_archive({"weight": torch.arange(4, dtype=torch.float32)})
    )
    filename = b"archive/data.pkl"
    central_filename_offset = archive.rfind(filename)
    assert central_filename_offset >= 0
    archive[central_filename_offset] = 0xFF

    with pytest.raises(ArchiveMetadataPreflightError) as raised:
        _prepare(_BytesStorage({path: bytes(archive)}), path, "weight")

    assert raised.value.kind is MetadataPreflightErrorKind.CORRUPT_ARCHIVE


def test_plain_non_torch_archive_is_recoverably_unsupported() -> None:
    path = Path("legacy.bin")

    result = _prepare(_BytesStorage({path: b"not a torch archive"}), path, "weight")

    assert isinstance(result, MetadataPreparationIneligible)
    assert result.reason is MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE


def _source_metadata(
    fqn: str,
    *,
    checkpoint_offset_bytes: int = 0,
    size: int = 4,
) -> metadata_module.SourceTensorMetadata:
    return metadata_module.SourceTensorMetadata(
        fqn=fqn,
        checkpoint_offset_bytes=checkpoint_offset_bytes,
        storage_offset_elements=0,
        storage_nbytes=size * 4,
        shape=(size,),
        stride=(1,),
        dtype="torch.float32",
        element_size_bytes=4,
    )


def _metadata_wire_item(fqn: str) -> dict[str, object]:
    return {
        "checkpoint_offset_bytes": 0,
        "dtype": "torch.float32",
        "element_size_bytes": 4,
        "fqn": fqn,
        "shape": [4],
        "storage_nbytes": 16,
        "storage_offset_elements": 0,
        "stride": [1],
    }


def test_wire_merge_matches_materialized_merge_and_preserves_root_fqn() -> None:
    root = _source_metadata("")
    weight = _source_metadata("weight", checkpoint_offset_bytes=64)
    bias = _source_metadata("bias", checkpoint_offset_bytes=128)
    payloads = [
        loader_module._metadata_to_wire({2: {"weight": weight}, 0: {"": root}}),
        loader_module._metadata_to_wire({2: {"bias": bias, "weight": weight}}),
    ]

    merged = merge_source_tensor_metadata_wire(payloads)
    legacy: dict[int, dict[str, metadata_module.SourceTensorMetadata]] = {}
    for payload in payloads:
        loader_module._merge_metadata(
            legacy,
            loader_module._metadata_from_wire(payload),
        )

    assert merged.payload == loader_module._metadata_to_wire(legacy)
    assert list(merged.payload) == ["0", "2"]
    root_payload = merged.payload["0"]
    rank_two_payload = merged.payload["2"]
    assert isinstance(root_payload, Mapping)
    assert isinstance(rank_two_payload, Mapping)
    assert list(root_payload) == [""]
    assert list(rank_two_payload) == ["bias", "weight"]
    assert merged.source_rank_count == 2
    assert merged.tensor_count == 3
    assert merged.duplicate_tensor_count == 1


def test_wire_selection_unions_rank_demands_before_selecting() -> None:
    metadata = {
        0: {"": _source_metadata(""), "weight": _source_metadata("weight")},
        3: {
            "bias": _source_metadata("bias"),
            "unused": _source_metadata("unused"),
        },
    }
    merged = merge_source_tensor_metadata_wire(
        [loader_module._metadata_to_wire(metadata)]
    )
    rank_demands = [
        {0: {"", "weight"}, 3: {"bias"}},
        {0: {"weight"}, 3: {"bias"}},
    ]

    selected = select_source_tensor_metadata_wire(merged, rank_demands)
    union = loader_module._merge_source_demands(rank_demands)
    expected = loader_module._metadata_to_wire(
        loader_module._metadata_for_demands(metadata, union)
    )

    assert selected.payload == expected
    assert selected.source_rank_count == 2
    assert selected.tensor_count == 3
    assert selected.duplicate_tensor_count == 0


def test_wire_merge_and_selection_allow_empty_inputs() -> None:
    merged = merge_source_tensor_metadata_wire([])

    assert merged.payload == {}
    assert merged.source_rank_count == 0
    assert merged.tensor_count == 0
    assert select_source_tensor_metadata_wire(merged, []).payload == {}

    with pytest.raises(ValueError, match="missing source rank 7"):
        select_source_tensor_metadata_wire(merged, [{7: ()}])


def test_wire_merge_rejects_conflicting_duplicates() -> None:
    item = _metadata_wire_item("weight")
    conflicting = {**item, "checkpoint_offset_bytes": 64}

    with pytest.raises(ValueError, match="conflicting metadata"):
        merge_source_tensor_metadata_wire(
            [
                {"0": {"weight": item}},
                {"0": {"weight": conflicting}},
            ]
        )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "source metadata"),
        ({"rank": {}}, "not an integer"),
        ({"-1": {}}, "non-negative"),
        ({"01": {}}, "not canonical"),
        ({"0": []}, "metadata for rank"),
        ({"0": {"weight": []}}, "metadata for weight"),
        (
            {"0": {"weight": {**_metadata_wire_item("weight"), "fqn": "bias"}}},
            "tensor name",
        ),
        (
            {
                "0": {
                    "weight": {
                        key: value
                        for key, value in _metadata_wire_item("weight").items()
                        if key != "stride"
                    }
                }
            },
            "keys differ",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "unexpected": 1,
                    }
                }
            },
            "keys differ",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "checkpoint_offset_bytes": True,
                    }
                }
            },
            "expected an integer",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "shape": [1.9],
                    }
                }
            },
            "only integers",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "fqn": 7,
                    }
                }
            },
            "fqn must be a string",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "dtype": 7,
                    }
                }
            },
            "dtype must be a string",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "checkpoint_offset_bytes": -1,
                    }
                }
            },
            "non-negative",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "stride": [1, 1],
                    }
                }
            },
            "same rank",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "element_size_bytes": 8,
                    }
                }
            },
            "does not use",
        ),
        (
            {
                "0": {
                    "weight": {
                        **_metadata_wire_item("weight"),
                        "storage_nbytes": 4,
                    }
                }
            },
            "storage contains",
        ),
    ],
)
def test_wire_merge_rejects_malformed_metadata(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        merge_source_tensor_metadata_wire([payload])


def test_wire_merge_does_not_materialize_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = loader_module._metadata_to_wire(
        {0: {"weight": _source_metadata("weight")}}
    )

    class _ForbiddenSourceTensorMetadata:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError(f"unexpected materialization: {kwargs}")

    monkeypatch.setattr(
        metadata_module,
        "SourceTensorMetadata",
        _ForbiddenSourceTensorMetadata,
    )

    merged = merge_source_tensor_metadata_wire([payload])

    assert merged.tensor_count == 1


def test_wire_merge_and_selection_match_materialized_reference_randomized() -> None:
    generator = random.Random(42)
    for _ in range(40):
        metadata: dict[int, dict[str, metadata_module.SourceTensorMetadata]] = {}
        payload_metadata: list[
            dict[int, dict[str, metadata_module.SourceTensorMetadata]]
        ] = [{}, {}, {}]
        expected_duplicate_count = 0
        for source_rank in range(generator.randint(1, 5)):
            tensors: dict[str, metadata_module.SourceTensorMetadata] = {}
            for tensor_index in range(generator.randint(1, 6)):
                fqn = (
                    ""
                    if source_rank == 0 and tensor_index == 0
                    else (f"layers.{source_rank}.{tensor_index}.weight")
                )
                item = _source_metadata(
                    fqn,
                    checkpoint_offset_bytes=64 * (source_rank + tensor_index),
                    size=generator.randint(1, 8),
                )
                tensors[fqn] = item
                owner = generator.randrange(len(payload_metadata))
                payload_metadata[owner].setdefault(source_rank, {})[fqn] = item
                if generator.random() < 0.35:
                    duplicate = (owner + 1) % len(payload_metadata)
                    payload_metadata[duplicate].setdefault(source_rank, {})[fqn] = item
                    expected_duplicate_count += 1
            metadata[source_rank] = tensors
        payloads = [
            loader_module._metadata_to_wire(payload) for payload in payload_metadata
        ]

        merged = merge_source_tensor_metadata_wire(payloads)
        legacy: dict[int, dict[str, metadata_module.SourceTensorMetadata]] = {}
        for payload in payloads:
            loader_module._merge_metadata(
                legacy,
                loader_module._metadata_from_wire(payload),
            )
        assert merged.payload == loader_module._metadata_to_wire(legacy)
        assert merged.duplicate_tensor_count == expected_duplicate_count

        rank_demands: list[dict[int, set[str]]] = []
        for _target_rank in range(generator.randint(1, 6)):
            target_demands: dict[int, set[str]] = {}
            for source_rank, tensors in metadata.items():
                selected = {fqn for fqn in tensors if generator.random() < 0.45}
                if selected:
                    target_demands[source_rank] = selected
            rank_demands.append(target_demands)
        union = loader_module._merge_source_demands(rank_demands)
        selected = select_source_tensor_metadata_wire(merged, rank_demands)
        expected = loader_module._metadata_to_wire(
            loader_module._metadata_for_demands(metadata, union)
        )
        assert selected.payload == expected


def test_partitioned_wire_trims_provider_extras_and_preserves_empty_ranks() -> None:
    metadata = {
        0: {
            "": _source_metadata(""),
            "weight": _source_metadata("weight"),
            "unused": _source_metadata("unused"),
        },
        7: {},
        9: {"extra": _source_metadata("extra")},
    }

    wire = metadata_module._build_partitioned_source_tensor_metadata_wire(
        metadata,
        {0: {"", "weight"}, 7: ()},
    )

    assert wire.payload == [
        "TCM",
        1,
        [
            [
                ["", "weight"],
                [[[4], [1], "torch.float32", 4]],
                [
                    [0, [0, 1], [0, 0], [0, 0], [16, 16], [0, 0]],
                    [7, [], [], [], [], []],
                ],
            ]
        ],
    ]
    assert wire.source_rank_count == 2
    assert wire.tensor_count == 2
    materialized = metadata_module._materialize_trusted_source_tensor_metadata_wire(
        wire
    )
    assert loader_module._metadata_to_wire(
        materialized
    ) == loader_module._metadata_to_wire(
        {0: {"": metadata[0][""], "weight": metadata[0]["weight"]}, 7: {}}
    )
    selected_empty = metadata_module._select_trusted_source_tensor_metadata_wire(
        wire,
        ({7: ()},),
    )
    assert metadata_module._materialize_trusted_source_tensor_metadata_wire(
        selected_empty
    ) == {7: {}}


def test_partitioned_wire_is_deterministic_across_mapping_insertion_order() -> None:
    first = {
        2: {
            "z": _source_metadata("z", checkpoint_offset_bytes=128),
            "a": _source_metadata("a", checkpoint_offset_bytes=64),
        },
        0: {"": _source_metadata("")},
    }
    second = {
        0: {"": first[0][""]},
        2: {"a": first[2]["a"], "z": first[2]["z"]},
    }

    first_wire = metadata_module._build_partitioned_source_tensor_metadata_wire(
        first,
        {2: {"z", "a"}, 0: {""}},
    )
    second_wire = metadata_module._build_partitioned_source_tensor_metadata_wire(
        second,
        {0: {""}, 2: {"a", "z"}},
    )

    assert first_wire.payload == second_wire.payload


@pytest.mark.parametrize(
    ("metadata", "demands", "message"),
    [
        ({}, {0: {"weight"}}, "missing source rank 0"),
        ({0: {}}, {0: {"weight"}}, "missing 'weight'"),
        ({0: {"weight": object()}}, {0: {"weight"}}, "SourceTensorMetadata"),
        ({0: {}}, {True: ()}, "must be an integer"),
        ({0: {}}, {-1: ()}, "non-negative"),
        ({0: {}}, {0: ("weight", 1)}, "must be strings"),
    ],
)
def test_partitioned_wire_rejects_malformed_producer_metadata(
    metadata: object,
    demands: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metadata_module._build_partitioned_source_tensor_metadata_wire(
            cast(Any, metadata),
            cast(Any, demands),
        )


def test_partitioned_wire_revalidates_mutated_producer_record() -> None:
    item = _source_metadata("weight")
    object.__setattr__(item, "storage_nbytes", 1)

    with pytest.raises(ValueError, match="storage contains"):
        metadata_module._build_partitioned_source_tensor_metadata_wire(
            {0: {"weight": item}},
            {0: {"weight"}},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("fqn", 7, "fqn must be a string"),
        ("checkpoint_offset_bytes", True, "expected an integer"),
        ("storage_offset_elements", 1.5, "expected an integer"),
        ("storage_nbytes", False, "expected an integer"),
        ("shape", (1.5,), "must contain only integers"),
        ("stride", (False,), "must contain only integers"),
        ("dtype", 7, "dtype must be a string"),
        ("element_size_bytes", True, "expected an integer"),
    ],
)
def test_partitioned_wire_preserves_exact_producer_type_validation(
    field: str,
    value: object,
    message: str,
) -> None:
    item = _source_metadata("weight")
    object.__setattr__(item, field, value)

    with pytest.raises(ValueError, match=message):
        metadata_module._build_partitioned_source_tensor_metadata_wire(
            {0: {"weight": item}},
            {0: {"weight"}},
        )


def test_partitioned_wire_merge_checks_ownership_without_revalidating_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {0: {"": _source_metadata("")}},
        {0: {""}},
    )
    second = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {2: {"weight": _source_metadata("weight")}, 4: {}},
        {2: {"weight"}, 4: ()},
    )

    monkeypatch.setattr(
        metadata_module,
        "_canonical_source_tensor_metadata_item",
        lambda *_args, **_kwargs: pytest.fail("coordinator revalidated a record"),
    )
    merged = metadata_module._merge_partitioned_source_tensor_metadata_wire(
        [(first.payload, {0: {""}}), (second.payload, {2: {"weight"}, 4: ()})]
    )

    assert merged.payload == ["TCM", 1, [first.payload[2][0], second.payload[2][0]]]
    assert merged.sections[0].fqn_table is first.sections[0].fqn_table
    assert merged.sections[1].layout_table is second.sections[0].layout_table
    assert merged.source_rank_count == 3
    assert merged.tensor_count == 2


@pytest.mark.parametrize(
    ("partitions", "_message"),
    [
        ([([], {})], "must be an object"),
        ([({"0": []}, {0: ()})], "must be an object"),
        ([({"01": {}}, {1: ()})], "not canonical"),
        ([({"1": {}}, {0: ()})], "source ranks differ"),
        ([({"0": {}}, {0: {"weight"}})], "differs from its assignment"),
        (
            [({"0": {"weight": _metadata_wire_item("weight")}}, {0: ()})],
            "unexpected",
        ),
        (
            [
                ({"0": {"weight": _metadata_wire_item("weight")}}, {0: {"weight"}}),
                ({"0": {"weight": _metadata_wire_item("weight")}}, {0: {"weight"}}),
            ],
            "duplicate source ranks",
        ),
        (
            [
                ({"0": {"weight": _metadata_wire_item("weight")}}, {0: {"weight"}}),
                (
                    {
                        "0": {
                            "weight": {
                                **_metadata_wire_item("weight"),
                                "checkpoint_offset_bytes": 64,
                            }
                        }
                    },
                    {0: {"weight"}},
                ),
            ],
            "duplicate source ranks",
        ),
    ],
)
def test_partitioned_wire_rejects_legacy_and_noncompact_partitions(
    partitions: object,
    _message: str,
) -> None:
    with pytest.raises(ValueError):
        metadata_module._merge_partitioned_source_tensor_metadata_wire(
            cast(Any, partitions)
        )


def test_trusted_wire_materialization_performs_full_local_validation() -> None:
    trusted = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {0: {"weight": _source_metadata("weight")}},
        {0: {"weight"}},
    )
    block = trusted.sections[0].rank_blocks[0]
    cast(list[int], block[4])[0] = 1
    selected = metadata_module._select_trusted_source_tensor_metadata_wire(
        trusted,
        ({0: {"weight"}},),
    )

    with pytest.raises(ValueError, match="storage contains"):
        metadata_module._materialize_trusted_source_tensor_metadata_wire(selected)


def _compact_metadata_wire() -> list[object]:
    return [
        "TCM",
        1,
        [
            [
                ["weight"],
                [[[1], [1], "torch.float32", 4]],
                [[0, [0], [0], [0], [4], [0]]],
            ]
        ],
    ]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "exactly three fields"),
        (["BAD", 1, []], "invalid magic"),
        (["TCM", 2, []], "unsupported version"),
        (["TCM", True, []], "unsupported version"),
        (["TCM", 1, {}], "sections must be an array"),
        (["TCM", 1, [[]]], "exactly three fields"),
        (["TCM", 1, [[{}, [], []]]], "FQN table must be an array"),
        (["TCM", 1, [[["z", "a"], [], []]]], "FQN table is not canonical"),
        (["TCM", 1, [[["a", "a"], [], []]]], "FQN table is not canonical"),
        (
            ["TCM", 1, [[["weight"], [[[1], [1], "torch.float32"]], []]]],
            "must contain four fields",
        ),
        (
            ["TCM", 1, [[["weight"], [[[1.0], [1], "torch.float32", 4]], []]]],
            "must contain only integers",
        ),
        (
            ["TCM", 1, [[["weight"], [[[1], [1], 7, 4]], []]]],
            "dtype must be a string",
        ),
        (
            ["TCM", 1, [[["weight"], [[[1], [1], "torch.float32", True]], []]]],
            "element size must be an integer",
        ),
        (
            [
                "TCM",
                1,
                [
                    [
                        ["weight"],
                        [
                            [[2], [1], "torch.float32", 4],
                            [[1], [1], "torch.float32", 4],
                        ],
                        [],
                    ]
                ],
            ],
            "layout table is not canonical",
        ),
        (
            [
                "TCM",
                1,
                [[["weight"], [[[1], [1], "torch.float32", 4]], [[0]]]],
            ],
            "must contain six fields",
        ),
        (
            [
                "TCM",
                1,
                [
                    [
                        ["weight"],
                        [[[1], [1], "torch.float32", 4]],
                        [[0, [0], [0], [], [4], [0]]],
                    ]
                ],
            ],
            "columns have different lengths",
        ),
        (
            [
                "TCM",
                1,
                [
                    [
                        ["weight"],
                        [[[1], [1], "torch.float32", 4]],
                        [[0, [1], [0], [0], [4], [0]]],
                    ]
                ],
            ],
            "invalid FQN ID",
        ),
        (
            [
                "TCM",
                1,
                [
                    [
                        ["weight"],
                        [[[1], [1], "torch.float32", 4]],
                        [[0, [0], [0], [0], [4], [1]]],
                    ]
                ],
            ],
            "invalid layout ID",
        ),
        (
            [
                "TCM",
                1,
                [
                    [
                        ["weight"],
                        [[[1], [1], "torch.float32", 4]],
                        [[0, [0], [False], [0], [4], [0]]],
                    ]
                ],
            ],
            "must contain only integers",
        ),
        (
            [
                "TCM",
                1,
                [
                    [
                        ["weight"],
                        [[[1], [1], "torch.float32", 4]],
                        [
                            [1, [0], [0], [0], [4], [0]],
                            [0, [0], [0], [0], [4], [0]],
                        ],
                    ]
                ],
            ],
            "rank blocks are not canonical",
        ),
    ],
)
def test_compact_wire_rejects_malformed_structure(
    payload: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metadata_module._decode_trusted_source_tensor_metadata_wire(payload)


def test_partitioned_wire_merge_requires_exact_tables_and_ownership() -> None:
    empty = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {0: {}},
        {0: ()},
    )
    one = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {0: {"weight": _source_metadata("weight")}},
        {0: {"weight"}},
    )
    unreferenced_fqn = _compact_metadata_wire()
    section = cast(list[object], cast(list[object], unreferenced_fqn[2])[0])
    cast(list[str], section[0]).append("z")

    with pytest.raises(ValueError, match="unreferenced entries"):
        metadata_module._merge_partitioned_source_tensor_metadata_wire(
            [(unreferenced_fqn, {0: {"weight"}})]
        )
    with pytest.raises(ValueError, match="source ranks differ"):
        metadata_module._merge_partitioned_source_tensor_metadata_wire(
            [(empty.payload, {1: ()})]
        )
    with pytest.raises(ValueError, match="differs from its assignment"):
        metadata_module._merge_partitioned_source_tensor_metadata_wire(
            [(empty.payload, {0: {"weight"}})]
        )
    with pytest.raises(ValueError, match="unexpected"):
        metadata_module._merge_partitioned_source_tensor_metadata_wire(
            [(one.payload, {0: ()})]
        )
    with pytest.raises(ValueError, match="duplicate source ranks"):
        metadata_module._merge_partitioned_source_tensor_metadata_wire(
            [(one.payload, {0: {"weight"}}), (one.payload, {0: {"weight"}})]
        )

    overlapping_sections = _compact_metadata_wire()
    sections = cast(list[object], overlapping_sections[2])
    sections.append(sections[0])
    with pytest.raises(ValueError, match="duplicate source rank 0"):
        metadata_module._decode_trusted_source_tensor_metadata_wire(
            overlapping_sections
        )

    second = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {1: {}},
        {1: ()},
    )
    multi_section_producer = [
        "TCM",
        1,
        [one.payload[2][0], second.payload[2][0]],
    ]
    with pytest.raises(ValueError, match="exactly one section"):
        metadata_module._merge_partitioned_source_tensor_metadata_wire(
            [(multi_section_producer, {0: {"weight"}, 1: ()})]
        )


def test_trusted_wire_preserves_tables_and_validates_only_selected_scalars() -> None:
    wire = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {
            0: {"bad": _source_metadata("bad")},
            1: {"good": _source_metadata("good")},
        },
        {0: {"bad"}, 1: {"good"}},
    )
    block = wire.sections[0].rank_blocks[0]
    cast(list[int], block[4])[0] = 1
    merged = metadata_module._merge_partitioned_source_tensor_metadata_wire(
        [(wire.payload, {0: {"bad"}, 1: {"good"}})]
    )

    selected_good = metadata_module._select_trusted_source_tensor_metadata_wire(
        merged,
        ({1: {"good"}},),
    )
    assert selected_good.sections[0].fqn_table is merged.sections[0].fqn_table
    assert selected_good.sections[0].layout_table is merged.sections[0].layout_table
    materialized = metadata_module._materialize_trusted_source_tensor_metadata_wire(
        selected_good
    )
    assert set(materialized) == {1}
    assert set(materialized[1]) == {"good"}

    selected_bad = metadata_module._select_trusted_source_tensor_metadata_wire(
        merged,
        ({0: {"bad"}},),
    )
    with pytest.raises(ValueError, match="storage contains"):
        metadata_module._materialize_trusted_source_tensor_metadata_wire(selected_bad)


def test_trusted_wire_validates_only_selected_layouts() -> None:
    wire = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {
            0: {"bad": _source_metadata("bad")},
            1: {"good": _source_metadata("good", size=2)},
        },
        {0: {"bad"}, 1: {"good"}},
    )
    bad_layout_id = cast(list[int], wire.sections[0].rank_blocks[0][5])[0]
    wire.sections[0].layout_table[bad_layout_id][2] = "not_a_dtype"
    merged = metadata_module._merge_partitioned_source_tensor_metadata_wire(
        [(wire.payload, {0: {"bad"}, 1: {"good"}})]
    )

    selected_good = metadata_module._select_trusted_source_tensor_metadata_wire(
        merged,
        ({1: {"good"}},),
    )
    materialized = metadata_module._materialize_trusted_source_tensor_metadata_wire(
        selected_good
    )
    assert set(materialized[1]) == {"good"}

    selected_bad = metadata_module._select_trusted_source_tensor_metadata_wire(
        merged,
        ({0: {"bad"}},),
    )
    with pytest.raises(ValueError, match="unsupported tensor dtype"):
        metadata_module._materialize_trusted_source_tensor_metadata_wire(selected_bad)


def test_trusted_wire_decode_allows_unreferenced_tables_after_selection() -> None:
    wire = metadata_module._build_partitioned_source_tensor_metadata_wire(
        {
            0: {"left": _source_metadata("left")},
            1: {"right": _source_metadata("right", checkpoint_offset_bytes=64, size=2)},
        },
        {0: {"left"}, 1: {"right"}},
    )
    selected = metadata_module._select_trusted_source_tensor_metadata_wire(
        wire,
        ({0: {"left"}},),
    )
    decoded = metadata_module._decode_trusted_source_tensor_metadata_wire(
        selected.payload
    )

    assert decoded.tensor_count == 1
    assert decoded.sections[0].fqn_table == ["left", "right"]
    assert len(decoded.sections[0].layout_table) == 2


def test_partitioned_wire_matches_untrusted_reference_randomized() -> None:
    generator = random.Random(3905)
    for _ in range(40):
        metadata: dict[int, dict[str, metadata_module.SourceTensorMetadata]] = {}
        demands: dict[int, frozenset[str]] = {}
        partitions: list[dict[int, dict[str, metadata_module.SourceTensorMetadata]]] = [
            {},
            {},
            {},
        ]
        partition_demands: list[dict[int, frozenset[str]]] = [{}, {}, {}]
        for source_rank in range(generator.randint(1, 8)):
            tensors: dict[str, metadata_module.SourceTensorMetadata] = {}
            selected_fqns: set[str] = set()
            for tensor_index in range(generator.randint(0, 8)):
                fqn = (
                    ""
                    if source_rank == 0 and tensor_index == 0
                    else f"layers.{source_rank}.{tensor_index}.weight"
                )
                tensors[fqn] = _source_metadata(
                    fqn,
                    checkpoint_offset_bytes=64 * (source_rank + tensor_index),
                    size=generator.randint(1, 8),
                )
                if generator.random() < 0.8:
                    selected_fqns.add(fqn)
            metadata[source_rank] = tensors
            demands[source_rank] = frozenset(selected_fqns)
            owner = source_rank % len(partitions)
            partitions[owner][source_rank] = {
                **tensors,
                f"extra.{source_rank}": _source_metadata(f"extra.{source_rank}"),
            }
            partition_demands[owner][source_rank] = frozenset(selected_fqns)

        encoded = [
            metadata_module._build_partitioned_source_tensor_metadata_wire(
                partition,
                assigned,
            )
            for partition, assigned in zip(partitions, partition_demands)
        ]
        merged = metadata_module._merge_partitioned_source_tensor_metadata_wire(
            [
                (wire.payload, assigned)
                for wire, assigned in zip(encoded, partition_demands)
            ]
        )
        rank_demands = [
            {
                source_rank: {fqn for fqn in fqns if generator.random() < 0.6}
                for source_rank, fqns in demands.items()
            }
            for _ in range(generator.randint(1, 5))
        ]
        selected = metadata_module._select_trusted_source_tensor_metadata_wire(
            merged,
            rank_demands,
        )
        materialized = metadata_module._materialize_trusted_source_tensor_metadata_wire(
            selected
        )
        union = loader_module._merge_source_demands(rank_demands)
        expected = loader_module._metadata_for_demands(metadata, union)

        assert loader_module._metadata_to_wire(materialized) == (
            loader_module._metadata_to_wire(expected)
        )
