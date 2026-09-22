# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Archive-neutral discovery of checkpoint tensor byte metadata."""

from __future__ import annotations

import io
import logging
import math
import operator
import pickle
import threading
import zipfile
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Set,
)
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from functools import cache
from pathlib import Path
from time import monotonic, monotonic_ns
from typing import Any, cast, NoReturn, Protocol, TypeAlias

import torch
from torch.distributed.tensor import DTensor

from ...resharding_utils import get_fqn_from_nested_path
from ...storage.base_storage import ReadArgs, Storage
from ...types import NestedPath
from .layout import SourceTensorMetadata

logger = logging.getLogger(__name__)

_DEFAULT_METADATA_MAX_WORKERS = 32
_MAX_STORAGE_TIMEOUT_US = 900_000_000
_ZIP_LOCAL_FILE_HEADER_BYTES = 30
_ZIP_DATA_DESCRIPTOR_BYTES = 16
_ZIP64_DATA_DESCRIPTOR_BYTES = 24
_ZIP32_MAX = 0xFFFFFFFF
_MAX_CHECKPOINT_CONTAINER_DEPTH = 1024
_MAX_METRIC_DETAIL_CHARS = 512


class MetadataIneligibilityReason(str, Enum):
    """A recoverable reason cooperative archive reads cannot be prepared."""

    UNSUPPORTED_STORAGE = "unsupported_storage"
    UNSUPPORTED_ARCHIVE = "unsupported_archive"
    UNSUPPORTED_VALUE = "unsupported_value"


class MetadataPreflightErrorKind(str, Enum):
    """Classification for non-recoverable metadata preparation failures."""

    IO = "io"
    CORRUPT_ARCHIVE = "corrupt_archive"
    INVALID_METADATA = "invalid_metadata"


class ArchiveMetadataPreflightError(RuntimeError):
    """An I/O, corruption, or metadata-integrity failure during preflight."""

    def __init__(
        self,
        kind: MetadataPreflightErrorKind,
        path: Path,
        detail: str,
        *,
        source_rank: int | None = None,
    ) -> None:
        self.kind = kind
        self.path = path
        self.detail = detail
        self.source_rank = source_rank
        rank_detail = "" if source_rank is None else f" for source rank {source_rank}"
        super().__init__(f"{kind.value}{rank_detail} at {path}: {detail}")


@dataclass(frozen=True, slots=True)
class MetadataPreparationIneligible:
    """A recoverable preflight result that asks the caller to use its fallback."""

    reason: MetadataIneligibilityReason
    detail: str
    source_rank: int | None = None
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class MetadataPreparationEligible:
    """The demanded tensor metadata, grouped by checkpoint source rank."""

    metadata_by_rank: Mapping[int, Mapping[str, SourceTensorMetadata]]


@dataclass(frozen=True, slots=True)
class SourceTensorMetadataWire:
    """Validated, canonical tensor metadata in its JSON-ready wire form."""

    payload: Mapping[str, object]
    source_rank_count: int
    tensor_count: int
    duplicate_tensor_count: int = 0


@dataclass(frozen=True, slots=True)
class _TrustedSourceTensorMetadataWire:
    """Producer-validated metadata carried by the private cooperative wire."""

    payload: list[object]
    source_rank_count: int
    tensor_count: int
    sections: tuple[_TrustedSourceTensorMetadataSection, ...]


@dataclass(frozen=True, slots=True)
class _TrustedSourceTensorMetadataSection:
    """One producer's structurally validated compact metadata section."""

    payload: list[object]
    fqn_table: list[str]
    layout_table: list[list[object]]
    rank_blocks: list[list[object]]


ArchiveMetadataInspectionResult: TypeAlias = (
    Mapping[str, SourceTensorMetadata] | MetadataPreparationIneligible
)
MetadataPreparationResult: TypeAlias = (
    MetadataPreparationEligible | MetadataPreparationIneligible
)
_InspectionOutcome: TypeAlias = ArchiveMetadataInspectionResult | Exception
_StorageRecords: TypeAlias = Mapping[int, int]
_MetricValue: TypeAlias = bool | int | float | str
_MetricCallback: TypeAlias = Callable[[str, Mapping[str, _MetricValue]], None]
_CanonicalMetadataItem: TypeAlias = dict[str, object]
_CanonicalMetadataByRank: TypeAlias = dict[
    int,
    dict[str, _CanonicalMetadataItem],
]

_TRUSTED_METADATA_WIRE_MAGIC = "TCM"
_TRUSTED_METADATA_WIRE_VERSION = 1


class _FastMetadataUnsupported(RuntimeError):
    """A valid pickle form that the lightweight decoder does not model."""


class _InvalidCheckpointStructure(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class _TensorStorageDescriptor:
    checkpoint_offset_bytes: int
    nbytes: int


@dataclass(frozen=True, slots=True)
class _TensorDescriptor:
    storage: _TensorStorageDescriptor
    storage_offset_elements: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    has_view_bits: bool


@dataclass(slots=True)
class _ArchiveDecodeMetrics:
    archive_name: str
    mode: str = "fast"
    fallback_reason: str = ""
    zip_validation_latency_ms: float = 0.0
    storage_index_latency_ms: float = 0.0
    metadata_load_latency_ms: float = 0.0
    legacy_load_latency_ms: float = 0.0
    extraction_latency_ms: float = 0.0
    archive_entry_count: int = 0
    storage_record_count: int = 0
    data_pickle_bytes: int = 0
    storage_index_mode: str = ""

    def fields(
        self, *, latency_ms: float, succeeded: bool
    ) -> Mapping[str, _MetricValue]:
        return {
            "archive_entry_count": self.archive_entry_count,
            "archive_name": self.archive_name,
            "data_pickle_bytes": self.data_pickle_bytes,
            "extraction_latency_ms": self.extraction_latency_ms,
            "fallback_reason": self.fallback_reason,
            "latency_ms": latency_ms,
            "legacy_load_latency_ms": self.legacy_load_latency_ms,
            "metadata_load_latency_ms": self.metadata_load_latency_ms,
            "mode": self.mode,
            "storage_index_latency_ms": self.storage_index_latency_ms,
            "storage_index_mode": self.storage_index_mode,
            "storage_record_count": self.storage_record_count,
            "succeeded": succeeded,
            "zip_validation_latency_ms": self.zip_validation_latency_ms,
        }


class _OpaqueMetadataObject:
    """Inert stand-in for DTensor placement metadata ignored by this reader."""

    def __new__(cls, *args: object, **kwargs: object) -> _OpaqueMetadataObject:
        value = super().__new__(cls)
        value.args = args
        value.state = None
        return value

    def __setstate__(self, state: object) -> None:
        self.state = state


class _DTensorMarker:
    pass


@dataclass(frozen=True, slots=True)
class _StorageTypeMarker:
    dtype: torch.dtype


_UNTYPED_STORAGE_MARKER = _StorageTypeMarker(torch.uint8)


def _build_partitioned_source_tensor_metadata_wire(
    metadata: Mapping[int, Mapping[str, SourceTensorMetadata]],
    demands: Mapping[int, Collection[str]],
) -> _TrustedSourceTensorMetadataWire:
    """Validate and encode exactly one producer's assigned source-rank partition."""

    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    normalized_demands = _canonical_metadata_demands(demands)
    fqn_table = sorted({fqn for fqns in normalized_demands.values() for fqn in fqns})
    fqn_ids = {fqn: index for index, fqn in enumerate(fqn_table)}
    layout_keys: list[tuple[tuple[int, ...], tuple[int, ...], str, int]] = []
    provisional_layout_ids: dict[
        tuple[tuple[int, ...], tuple[int, ...], str, int], int
    ] = {}
    pending_rank_blocks: list[
        tuple[
            int,
            list[int],
            list[int],
            list[int],
            list[int],
            list[int],
        ]
    ] = []
    for source_rank, fqns in normalized_demands.items():
        tensors = metadata.get(source_rank)
        if tensors is None:
            raise ValueError(f"metadata is missing source rank {source_rank}")
        if not isinstance(tensors, Mapping):
            raise ValueError(
                f"metadata for source rank {source_rank} must be a mapping"
            )
        block_fqn_ids: list[int] = []
        checkpoint_offsets: list[int] = []
        storage_offsets: list[int] = []
        storage_nbytes: list[int] = []
        block_layout_ids: list[int] = []
        for fqn in sorted(fqns):
            try:
                item = tensors[fqn]
            except KeyError as error:
                raise ValueError(
                    f"metadata for source rank {source_rank} is missing {fqn!r}"
                ) from error
            (
                checkpoint_offset_bytes,
                storage_offset_elements,
                tensor_storage_nbytes,
                layout_key,
            ) = _validated_source_tensor_metadata_fields(
                fqn,
                item,
            )
            block_fqn_ids.append(fqn_ids[fqn])
            checkpoint_offsets.append(checkpoint_offset_bytes)
            storage_offsets.append(storage_offset_elements)
            storage_nbytes.append(tensor_storage_nbytes)
            layout_id = provisional_layout_ids.get(layout_key)
            if layout_id is None:
                layout_id = len(layout_keys)
                provisional_layout_ids[layout_key] = layout_id
                layout_keys.append(layout_key)
            block_layout_ids.append(layout_id)
        pending_rank_blocks.append(
            (
                source_rank,
                block_fqn_ids,
                checkpoint_offsets,
                storage_offsets,
                storage_nbytes,
                block_layout_ids,
            )
        )

    sorted_layouts = sorted(layout_keys)
    layout_ids = {layout: index for index, layout in enumerate(sorted_layouts)}
    layout_id_remap = [layout_ids[layout] for layout in layout_keys]
    layout_table: list[list[object]] = [
        [list(shape), list(stride), dtype, element_size_bytes]
        for shape, stride, dtype, element_size_bytes in sorted_layouts
    ]
    rank_blocks: list[list[object]] = [
        [
            source_rank,
            block_fqn_ids,
            checkpoint_offsets,
            storage_offsets,
            storage_nbytes,
            [layout_id_remap[layout_id] for layout_id in block_layout_ids],
        ]
        for (
            source_rank,
            block_fqn_ids,
            checkpoint_offsets,
            storage_offsets,
            storage_nbytes,
            block_layout_ids,
        ) in pending_rank_blocks
    ]
    section_payload: list[object] = [fqn_table, layout_table, rank_blocks]
    section = _TrustedSourceTensorMetadataSection(
        payload=section_payload,
        fqn_table=fqn_table,
        layout_table=layout_table,
        rank_blocks=rank_blocks,
    )
    return _TrustedSourceTensorMetadataWire(
        payload=_trusted_metadata_payload([section_payload]),
        source_rank_count=len(normalized_demands),
        tensor_count=sum(len(fqns) for fqns in normalized_demands.values()),
        sections=(section,),
    )


def _validated_source_tensor_metadata_fields(
    fqn: str,
    metadata: SourceTensorMetadata,
) -> tuple[
    int,
    int,
    int,
    tuple[tuple[int, ...], tuple[int, ...], str, int],
]:
    if not isinstance(metadata, SourceTensorMetadata):
        raise ValueError("metadata records must be SourceTensorMetadata instances")
    try:
        if not isinstance(metadata.fqn, str):
            raise TypeError("fqn must be a string")
        checkpoint_offset_bytes = _wire_int(metadata.checkpoint_offset_bytes)
        storage_offset_elements = _wire_int(metadata.storage_offset_elements)
        storage_nbytes = _wire_int(metadata.storage_nbytes)
        shape = _validated_metadata_dimensions(metadata.shape, "metadata shape")
        stride = _validated_metadata_dimensions(metadata.stride, "metadata stride")
        if not isinstance(metadata.dtype, str):
            raise TypeError("dtype must be a string")
        element_size_bytes = _wire_int(metadata.element_size_bytes)
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata for {fqn!r} is invalid: {error}") from error
    if metadata.fqn != fqn:
        raise ValueError("metadata key does not match its tensor name")
    _validate_canonical_metadata_item(
        fqn=fqn,
        checkpoint_offset_bytes=checkpoint_offset_bytes,
        storage_offset_elements=storage_offset_elements,
        storage_nbytes=storage_nbytes,
        shape=shape,
        stride=stride,
        dtype=metadata.dtype,
        element_size_bytes=element_size_bytes,
    )
    return (
        checkpoint_offset_bytes,
        storage_offset_elements,
        storage_nbytes,
        (
            shape,
            stride,
            metadata.dtype,
            element_size_bytes,
        ),
    )


def _validated_metadata_dimensions(value: object, name: str) -> tuple[int, ...]:
    if type(value) is tuple and all(type(item) is int for item in value):
        return cast(tuple[int, ...], value)
    return tuple(_wire_int_array(list(cast(Iterable[object], value)), name))


def _merge_partitioned_source_tensor_metadata_wire(
    partitions: Iterable[tuple[object, Mapping[int, Collection[str]]]],
) -> _TrustedSourceTensorMetadataWire:
    """Combine disjoint producer-validated partitions without item revalidation."""

    merged_sections: list[_TrustedSourceTensorMetadataSection] = []
    claimed_source_ranks: set[int] = set()
    tensor_count = 0
    for partition_index, (payload, demands) in enumerate(partitions):
        expected = _canonical_metadata_demands(demands)
        duplicate_ranks = claimed_source_ranks.intersection(expected)
        if duplicate_ranks:
            raise ValueError(
                "metadata producer partitions contain duplicate source ranks "
                f"{sorted(duplicate_ranks)!r}"
            )
        incoming = _decode_trusted_source_tensor_metadata_wire_impl(
            payload,
            f"metadata producer partition {partition_index}",
            require_table_coverage=True,
        )
        if len(incoming.sections) != 1:
            raise ValueError(
                f"metadata producer partition {partition_index} must contain "
                "exactly one section"
            )
        blocks_by_rank = {
            cast(int, block[0]): (section, block)
            for section in incoming.sections
            for block in section.rank_blocks
        }
        actual_source_ranks = set(blocks_by_rank)
        expected_source_ranks = set(expected)
        if actual_source_ranks != expected_source_ranks:
            raise ValueError(
                f"metadata producer partition {partition_index} source ranks differ: "
                f"missing={sorted(expected_source_ranks - actual_source_ranks)!r}, "
                f"unexpected={sorted(actual_source_ranks - expected_source_ranks)!r}"
            )
        for source_rank, fqns in expected.items():
            section, block = blocks_by_rank[source_rank]
            actual_fqns = {
                section.fqn_table[fqn_id] for fqn_id in cast(list[int], block[1])
            }
            if actual_fqns != fqns:
                raise ValueError(
                    f"metadata producer partition {partition_index} for source rank "
                    f"{source_rank} differs from its assignment: "
                    f"missing={sorted(fqns - actual_fqns)!r}, "
                    f"unexpected={sorted(actual_fqns - fqns)!r}"
                )
            tensor_count += len(actual_fqns)
        merged_sections.extend(incoming.sections)
        claimed_source_ranks.update(expected)
    return _TrustedSourceTensorMetadataWire(
        payload=_trusted_metadata_payload(
            [section.payload for section in merged_sections]
        ),
        source_rank_count=len(claimed_source_ranks),
        tensor_count=tensor_count,
        sections=tuple(merged_sections),
    )


def _decode_trusted_source_tensor_metadata_wire(
    payload: object,
) -> _TrustedSourceTensorMetadataWire:
    """Structurally decode trusted metadata without inspecting tensor records."""

    return _decode_trusted_source_tensor_metadata_wire_impl(
        payload,
        "trusted source metadata",
        require_table_coverage=False,
    )


def _select_trusted_source_tensor_metadata_wire(
    metadata: _TrustedSourceTensorMetadataWire,
    demands: Iterable[Mapping[int, Collection[str]]],
) -> _TrustedSourceTensorMetadataWire:
    """Select demanded records by reference from producer-validated metadata."""

    demanded_by_source_rank: dict[int, set[str]] = {}
    for rank_demands in demands:
        normalized = _canonical_metadata_demands(rank_demands)
        for source_rank, fqns in normalized.items():
            demanded_by_source_rank.setdefault(source_rank, set()).update(fqns)

    selected_sections: list[_TrustedSourceTensorMetadataSection] = []
    found_source_ranks: set[int] = set()
    tensor_count = 0
    for section in metadata.sections:
        selected_blocks: list[list[object]] = []
        for block in section.rank_blocks:
            source_rank = cast(int, block[0])
            wanted = demanded_by_source_rank.get(source_rank)
            if wanted is None:
                continue
            found_source_ranks.add(source_rank)
            block_fqn_ids = cast(list[int], block[1])
            present = {
                section.fqn_table[fqn_id]: index
                for index, fqn_id in enumerate(block_fqn_ids)
            }
            missing = wanted - present.keys()
            if missing:
                missing_fqn = min(missing)
                raise ValueError(
                    f"metadata for source rank {source_rank} is missing {missing_fqn!r}"
                )
            selected_indices = [present[fqn] for fqn in sorted(wanted)]
            selected_blocks.append(
                [
                    source_rank,
                    [block_fqn_ids[index] for index in selected_indices],
                    [cast(list[int], block[2])[index] for index in selected_indices],
                    [cast(list[int], block[3])[index] for index in selected_indices],
                    [cast(list[int], block[4])[index] for index in selected_indices],
                    [cast(list[int], block[5])[index] for index in selected_indices],
                ]
            )
            tensor_count += len(selected_indices)
        if selected_blocks:
            section_payload: list[object] = [
                section.fqn_table,
                section.layout_table,
                selected_blocks,
            ]
            selected_sections.append(
                _TrustedSourceTensorMetadataSection(
                    payload=section_payload,
                    fqn_table=section.fqn_table,
                    layout_table=section.layout_table,
                    rank_blocks=selected_blocks,
                )
            )
    missing_source_ranks = demanded_by_source_rank.keys() - found_source_ranks
    if missing_source_ranks:
        raise ValueError(f"metadata is missing source rank {min(missing_source_ranks)}")
    return _TrustedSourceTensorMetadataWire(
        payload=_trusted_metadata_payload(
            [section.payload for section in selected_sections]
        ),
        source_rank_count=len(found_source_ranks),
        tensor_count=tensor_count,
        sections=tuple(selected_sections),
    )


def _materialize_trusted_source_tensor_metadata_wire(
    metadata: _TrustedSourceTensorMetadataWire,
) -> dict[int, dict[str, SourceTensorMetadata]]:
    """Fully validate locally selected records and construct source metadata."""

    result: dict[int, dict[str, SourceTensorMetadata]] = {}
    for section in metadata.sections:
        for block in section.rank_blocks:
            source_rank = cast(int, block[0])
            decoded: dict[str, SourceTensorMetadata] = {}
            for (
                fqn_id,
                checkpoint_offset_bytes,
                storage_offset_elements,
                storage_nbytes,
                layout_id,
            ) in zip(
                cast(list[int], block[1]),
                cast(list[int], block[2]),
                cast(list[int], block[3]),
                cast(list[int], block[4]),
                cast(list[int], block[5]),
            ):
                fqn = section.fqn_table[fqn_id]
                layout = section.layout_table[layout_id]
                shape = cast(list[int], layout[0])
                stride = cast(list[int], layout[1])
                dtype = cast(str, layout[2])
                element_size_bytes = cast(int, layout[3])
                decoded[fqn] = SourceTensorMetadata(
                    fqn=fqn,
                    checkpoint_offset_bytes=checkpoint_offset_bytes,
                    storage_offset_elements=storage_offset_elements,
                    storage_nbytes=storage_nbytes,
                    shape=tuple(shape),
                    stride=tuple(stride),
                    dtype=dtype,
                    element_size_bytes=element_size_bytes,
                )
            result[source_rank] = decoded
    return result


def _trusted_metadata_payload(sections: list[list[object]]) -> list[object]:
    return [
        _TRUSTED_METADATA_WIRE_MAGIC,
        _TRUSTED_METADATA_WIRE_VERSION,
        sections,
    ]


def _decode_trusted_source_tensor_metadata_wire_impl(
    payload: object,
    name: str,
    *,
    require_table_coverage: bool,
) -> _TrustedSourceTensorMetadataWire:
    wire = _compact_wire_list(payload, name)
    if len(wire) != 3:
        raise ValueError(f"{name} must contain exactly three fields")
    if wire[0] != _TRUSTED_METADATA_WIRE_MAGIC:
        raise ValueError(f"{name} has invalid magic")
    if type(wire[1]) is not int or wire[1] != _TRUSTED_METADATA_WIRE_VERSION:
        raise ValueError(f"{name} has unsupported version {wire[1]!r}")
    raw_sections = _compact_wire_list(wire[2], f"{name} sections")
    sections: list[_TrustedSourceTensorMetadataSection] = []
    claimed_source_ranks: set[int] = set()
    tensor_count = 0
    for section_index, raw_section in enumerate(raw_sections):
        section = _validate_trusted_metadata_section(
            raw_section,
            f"{name} section {section_index}",
            require_table_coverage=require_table_coverage,
        )
        for block in section.rank_blocks:
            source_rank = cast(int, block[0])
            if source_rank in claimed_source_ranks:
                raise ValueError(f"{name} contains duplicate source rank {source_rank}")
            claimed_source_ranks.add(source_rank)
            tensor_count += len(cast(list[int], block[1]))
        sections.append(section)
    return _TrustedSourceTensorMetadataWire(
        payload=wire,
        source_rank_count=len(claimed_source_ranks),
        tensor_count=tensor_count,
        sections=tuple(sections),
    )


def _validate_trusted_metadata_section(
    payload: object,
    name: str,
    *,
    require_table_coverage: bool,
) -> _TrustedSourceTensorMetadataSection:
    section = _compact_wire_list(payload, name)
    if len(section) != 3:
        raise ValueError(f"{name} must contain exactly three fields")
    fqn_table = _validate_trusted_metadata_fqn_table(section[0], name)
    layout_table = _validate_trusted_metadata_layout_table(section[1], name)
    rank_blocks, used_fqn_ids, used_layout_ids = _validate_trusted_metadata_rank_blocks(
        section[2],
        name,
        fqn_count=len(fqn_table),
        layout_count=len(layout_table),
    )
    if require_table_coverage and used_fqn_ids != set(range(len(fqn_table))):
        raise ValueError(f"{name} FQN table contains unreferenced entries")
    if require_table_coverage and used_layout_ids != set(range(len(layout_table))):
        raise ValueError(f"{name} layout table contains unreferenced entries")
    return _TrustedSourceTensorMetadataSection(
        payload=section,
        fqn_table=fqn_table,
        layout_table=layout_table,
        rank_blocks=rank_blocks,
    )


def _validate_trusted_metadata_fqn_table(
    payload: object,
    section_name: str,
) -> list[str]:
    raw_fqn_table = _compact_wire_list(payload, f"{section_name} FQN table")
    if not all(isinstance(fqn, str) for fqn in raw_fqn_table):
        raise ValueError(f"{section_name} FQN table must contain only strings")
    fqn_table = cast(list[str], raw_fqn_table)
    if fqn_table != sorted(set(fqn_table)):
        raise ValueError(f"{section_name} FQN table is not canonical")
    return fqn_table


def _validate_trusted_metadata_layout_table(
    payload: object,
    section_name: str,
) -> list[list[object]]:
    raw_layout_table = _compact_wire_list(
        payload,
        f"{section_name} layout table",
    )
    layout_keys: list[tuple[tuple[int, ...], tuple[int, ...], str, int]] = []
    for layout_index, raw_layout in enumerate(raw_layout_table):
        layout = _compact_wire_list(
            raw_layout,
            f"{section_name} layout {layout_index}",
        )
        if len(layout) != 4:
            raise ValueError(
                f"{section_name} layout {layout_index} must contain four fields"
            )
        shape = _compact_wire_int_list(
            layout[0],
            f"{section_name} layout {layout_index} shape",
        )
        stride = _compact_wire_int_list(
            layout[1],
            f"{section_name} layout {layout_index} stride",
        )
        if not isinstance(layout[2], str):
            raise ValueError(
                f"{section_name} layout {layout_index} dtype must be a string"
            )
        if type(layout[3]) is not int:
            raise ValueError(
                f"{section_name} layout {layout_index} element size must be an integer"
            )
        layout_keys.append(
            (tuple(shape), tuple(stride), layout[2], cast(int, layout[3]))
        )
    if layout_keys != sorted(set(layout_keys)):
        raise ValueError(f"{section_name} layout table is not canonical")
    return cast(list[list[object]], raw_layout_table)


def _validate_trusted_metadata_rank_blocks(
    payload: object,
    section_name: str,
    *,
    fqn_count: int,
    layout_count: int,
) -> tuple[list[list[object]], set[int], set[int]]:
    raw_rank_blocks = _compact_wire_list(
        payload,
        f"{section_name} rank blocks",
    )
    previous_source_rank = -1
    used_fqn_ids: set[int] = set()
    used_layout_ids: set[int] = set()
    for block_index, raw_block in enumerate(raw_rank_blocks):
        source_rank, block_fqn_ids, block_layout_ids = (
            _validate_trusted_metadata_rank_block(
                raw_block,
                section_name,
                block_index,
                previous_source_rank=previous_source_rank,
                fqn_count=fqn_count,
                layout_count=layout_count,
            )
        )
        previous_source_rank = source_rank
        used_fqn_ids.update(block_fqn_ids)
        used_layout_ids.update(block_layout_ids)
    return (
        cast(list[list[object]], raw_rank_blocks),
        used_fqn_ids,
        used_layout_ids,
    )


def _validate_trusted_metadata_rank_block(
    payload: object,
    section_name: str,
    block_index: int,
    *,
    previous_source_rank: int,
    fqn_count: int,
    layout_count: int,
) -> tuple[int, list[int], list[int]]:
    name = f"{section_name} rank block {block_index}"
    block = _compact_wire_list(payload, name)
    if len(block) != 6:
        raise ValueError(f"{name} must contain six fields")
    source_rank = block[0]
    if type(source_rank) is not int or source_rank < 0:
        raise ValueError(f"{name} source rank must be a non-negative integer")
    if source_rank <= previous_source_rank:
        raise ValueError(f"{section_name} rank blocks are not canonical")
    columns = [
        _compact_wire_int_list(
            block[column_index],
            f"{name} column {column_index}",
        )
        for column_index in range(1, 6)
    ]
    if len({len(column) for column in columns}) != 1:
        raise ValueError(f"{name} columns have different lengths")
    block_fqn_ids = columns[0]
    if block_fqn_ids != sorted(set(block_fqn_ids)):
        raise ValueError(f"{name} FQN IDs are not canonical")
    if any(fqn_id < 0 or fqn_id >= fqn_count for fqn_id in block_fqn_ids):
        raise ValueError(f"{name} has invalid FQN ID")
    block_layout_ids = columns[4]
    if any(
        layout_id < 0 or layout_id >= layout_count for layout_id in block_layout_ids
    ):
        raise ValueError(f"{name} has invalid layout ID")
    return cast(int, source_rank), block_fqn_ids, block_layout_ids


def _compact_wire_list(value: object, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _compact_wire_int_list(value: object, name: str) -> list[int]:
    result = _compact_wire_list(value, name)
    if not all(type(item) is int for item in result):
        raise ValueError(f"{name} must contain only integers")
    return cast(list[int], result)


def _canonical_metadata_demands(
    demands: Mapping[int, Collection[str]],
) -> dict[int, frozenset[str]]:
    if not isinstance(demands, Mapping):
        raise ValueError("metadata demands must be a mapping")
    normalized: dict[int, frozenset[str]] = {}
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
        normalized[source_rank] = frozenset(fqns)
    return dict(sorted(normalized.items()))


def _source_tensor_metadata_item_to_wire(
    metadata: SourceTensorMetadata,
) -> dict[str, object]:
    if not isinstance(metadata, SourceTensorMetadata):
        raise ValueError("metadata records must be SourceTensorMetadata instances")
    return {
        "checkpoint_offset_bytes": metadata.checkpoint_offset_bytes,
        "dtype": metadata.dtype,
        "element_size_bytes": metadata.element_size_bytes,
        "fqn": metadata.fqn,
        "shape": list(metadata.shape),
        "storage_nbytes": metadata.storage_nbytes,
        "storage_offset_elements": metadata.storage_offset_elements,
        "stride": list(metadata.stride),
    }


def merge_source_tensor_metadata_wire(
    payloads: Iterable[object],
) -> SourceTensorMetadataWire:
    """Validate and merge metadata payloads without constructing tensor objects."""

    merged: _CanonicalMetadataByRank = {}
    duplicate_tensor_count = 0
    for payload in payloads:
        incoming = _wire_mapping(payload, "source metadata")
        seen_source_ranks: set[int] = set()
        for raw_rank, raw_tensors in incoming.items():
            source_rank = _canonical_source_rank(raw_rank)
            if source_rank in seen_source_ranks:
                raise ValueError(
                    f"source metadata contains duplicate source rank {source_rank}"
                )
            seen_source_ranks.add(source_rank)
            tensors = _wire_mapping(
                raw_tensors,
                f"metadata for rank {raw_rank}",
            )
            destination = merged.setdefault(source_rank, {})
            for fqn, raw_item in tensors.items():
                item = _canonical_source_tensor_metadata_item(fqn, raw_item)
                previous = destination.get(fqn)
                if previous is not None:
                    if previous != item:
                        raise ValueError(
                            "conflicting metadata for source rank "
                            f"{source_rank}, {fqn!r}"
                        )
                    duplicate_tensor_count += 1
                    continue
                destination[fqn] = item
    return _source_tensor_metadata_wire(
        merged,
        duplicate_tensor_count=duplicate_tensor_count,
    )


def select_source_tensor_metadata_wire(
    metadata: SourceTensorMetadataWire,
    demands: Iterable[Mapping[int, Collection[str]]],
) -> SourceTensorMetadataWire:
    """Select the union of several ranks' demands once for one destination node."""

    demanded_by_source_rank: dict[int, set[str]] = {}
    for rank_demands in demands:
        for source_rank, fqns in rank_demands.items():
            if isinstance(source_rank, bool) or not isinstance(source_rank, int):
                raise ValueError("metadata demand source rank must be an integer")
            if source_rank < 0:
                raise ValueError("metadata demand source rank must be non-negative")
            demanded = demanded_by_source_rank.setdefault(source_rank, set())
            for fqn in fqns:
                if not isinstance(fqn, str):
                    raise ValueError("metadata demand FQNs must be strings")
                demanded.add(fqn)

    selected: _CanonicalMetadataByRank = {}
    for source_rank, fqns in sorted(demanded_by_source_rank.items()):
        raw_tensors = metadata.payload.get(str(source_rank))
        if raw_tensors is None:
            raise ValueError(f"metadata is missing source rank {source_rank}")
        tensors = _wire_mapping(raw_tensors, f"metadata for rank {source_rank}")
        selected[source_rank] = {}
        for fqn in sorted(fqns):
            try:
                raw_item = tensors[fqn]
            except KeyError as error:
                raise ValueError(
                    f"metadata for source rank {source_rank} is missing {fqn!r}"
                ) from error
            selected[source_rank][fqn] = _wire_mapping(
                raw_item,
                f"metadata for {fqn}",
            )
    return _source_tensor_metadata_wire(selected)


def _canonical_source_rank(value: str) -> int:
    try:
        source_rank = int(value)
    except ValueError as error:
        raise ValueError(f"metadata source rank {value!r} is not an integer") from error
    if source_rank < 0:
        raise ValueError("metadata source rank must be non-negative")
    if str(source_rank) != value:
        raise ValueError(f"metadata source rank {value!r} is not canonical")
    return source_rank


def _canonical_source_tensor_metadata_item(
    fqn: str,
    value: object,
    *,
    validate_semantics: bool = True,
) -> _CanonicalMetadataItem:
    item = _wire_mapping(value, f"metadata for {fqn}")
    expected_keys = {
        "checkpoint_offset_bytes",
        "dtype",
        "element_size_bytes",
        "fqn",
        "shape",
        "storage_nbytes",
        "storage_offset_elements",
        "stride",
    }
    if set(item) != expected_keys:
        raise ValueError(
            f"metadata for {fqn!r} keys differ: "
            f"missing={sorted(expected_keys - set(item))}, "
            f"unexpected={sorted(set(item) - expected_keys)}"
        )
    try:
        raw_metadata_fqn = item["fqn"]
        raw_checkpoint_offset_bytes = item["checkpoint_offset_bytes"]
        raw_storage_offset_elements = item["storage_offset_elements"]
        raw_storage_nbytes = item["storage_nbytes"]
        raw_shape = item["shape"]
        raw_stride = item["stride"]
        raw_dtype = item["dtype"]
        raw_element_size_bytes = item["element_size_bytes"]
        if not isinstance(raw_metadata_fqn, str):
            raise TypeError("fqn must be a string")
        metadata_fqn = raw_metadata_fqn
        checkpoint_offset_bytes = _wire_int(raw_checkpoint_offset_bytes)
        storage_offset_elements = _wire_int(raw_storage_offset_elements)
        storage_nbytes = _wire_int(raw_storage_nbytes)
        shape = _wire_int_array(
            raw_shape,
            "metadata shape",
        )
        stride = _wire_int_array(
            raw_stride,
            "metadata stride",
        )
        if not isinstance(raw_dtype, str):
            raise TypeError("dtype must be a string")
        dtype = raw_dtype
        element_size_bytes = _wire_int(raw_element_size_bytes)
    except KeyError as error:
        raise ValueError(
            f"metadata for {fqn!r} is missing field {error.args[0]!r}"
        ) from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"metadata for {fqn!r} is invalid: {error}") from error

    if metadata_fqn != fqn:
        raise ValueError("metadata key does not match its tensor name")
    if validate_semantics:
        _validate_canonical_metadata_item(
            fqn=fqn,
            checkpoint_offset_bytes=checkpoint_offset_bytes,
            storage_offset_elements=storage_offset_elements,
            storage_nbytes=storage_nbytes,
            shape=shape,
            stride=stride,
            dtype=dtype,
            element_size_bytes=element_size_bytes,
        )
    if (
        isinstance(value, dict)
        and len(value) == 8
        and isinstance(raw_metadata_fqn, str)
        and type(raw_checkpoint_offset_bytes) is int
        and type(raw_storage_offset_elements) is int
        and type(raw_storage_nbytes) is int
        and isinstance(raw_shape, list)
        and shape is raw_shape
        and isinstance(raw_stride, list)
        and stride is raw_stride
        and isinstance(raw_dtype, str)
        and type(raw_element_size_bytes) is int
    ):
        return value
    return {
        "checkpoint_offset_bytes": checkpoint_offset_bytes,
        "dtype": dtype,
        "element_size_bytes": element_size_bytes,
        "fqn": metadata_fqn,
        "shape": list(shape),
        "storage_nbytes": storage_nbytes,
        "storage_offset_elements": storage_offset_elements,
        "stride": list(stride),
    }


def _validate_canonical_metadata_item(
    *,
    fqn: str,
    checkpoint_offset_bytes: int,
    storage_offset_elements: int,
    storage_nbytes: int,
    shape: Sequence[int],
    stride: Sequence[int],
    dtype: str,
    element_size_bytes: int,
) -> None:
    if checkpoint_offset_bytes < 0:
        raise ValueError("checkpoint_offset_bytes must be non-negative")
    if storage_offset_elements < 0:
        raise ValueError("storage_offset_elements must be non-negative")
    if storage_nbytes < 0:
        raise ValueError("storage_nbytes must be non-negative")
    if len(shape) != len(stride):
        raise ValueError("shape and stride must have the same rank")
    if any(size < 0 for size in shape):
        raise ValueError("shape dimensions must be non-negative")
    if any(step < 0 for step in stride):
        raise ValueError("negative source strides are unsupported")
    if element_size_bytes <= 0:
        raise ValueError("element_size_bytes must be positive")
    if _wire_dtype_element_size(dtype) != element_size_bytes:
        raise ValueError(f"dtype {dtype!r} does not use {element_size_bytes} bytes")
    if math.prod(shape) == 0:
        return
    last_element = storage_offset_elements + sum(
        (size - 1) * step for size, step in zip(shape, stride)
    )
    required_nbytes = (last_element + 1) * element_size_bytes
    if required_nbytes > storage_nbytes:
        raise ValueError(
            f"metadata for {fqn!r} addresses {required_nbytes} bytes, "
            f"but its storage contains {storage_nbytes} bytes"
        )


@cache
def _wire_dtype_element_size(dtype_name: str) -> int:
    if not dtype_name:
        raise ValueError("dtype must be a non-empty string")
    aliases = {
        "byte": "uint8",
        "char": "int8",
        "double": "float64",
        "float": "float32",
        "half": "float16",
        "int": "int32",
        "long": "int64",
        "short": "int16",
    }
    name = dtype_name.removeprefix("torch.")
    torch_dtype = getattr(torch, aliases.get(name, name), None)
    if not isinstance(torch_dtype, torch.dtype):
        raise ValueError(f"unsupported tensor dtype {dtype_name!r}")
    return torch.empty((), dtype=torch_dtype).element_size()


def _source_tensor_metadata_wire(
    metadata: _CanonicalMetadataByRank,
    *,
    duplicate_tensor_count: int = 0,
) -> SourceTensorMetadataWire:
    payload = {
        str(source_rank): dict(sorted(tensors.items()))
        for source_rank, tensors in sorted(metadata.items())
    }
    return SourceTensorMetadataWire(
        payload=payload,
        source_rank_count=len(payload),
        tensor_count=sum(len(tensors) for tensors in metadata.values()),
        duplicate_tensor_count=duplicate_tensor_count,
    )


def _wire_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return value


def _wire_sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be an array")
    return value


def _wire_int(value: Any) -> int:
    if type(value) is not int:
        raise TypeError("expected an integer")
    return value


def _wire_int_array(value: object, name: str) -> Sequence[int]:
    sequence = _wire_sequence(value, name)
    if not isinstance(sequence, list) or not all(
        type(item) is int for item in sequence
    ):
        raise TypeError(f"{name} must contain only integers")
    return sequence


def _tensor_metadata_flags(metadata: object) -> bool:
    if metadata is None:
        return False
    if not isinstance(metadata, dict) or any(
        key not in {"conj", "neg"} for key in metadata
    ):
        raise _FastMetadataUnsupported("unsupported serialized tensor metadata")
    if any(type(value) is not bool for value in metadata.values()):
        raise _FastMetadataUnsupported(
            "serialized tensor metadata flags are not booleans"
        )
    return bool(metadata.get("conj") or metadata.get("neg"))


def _tensor_descriptor(
    storage: Any,
    storage_offset: object,
    shape: Iterable[object],
    stride: Iterable[object],
    dtype: torch.dtype,
    metadata: object,
) -> _TensorDescriptor:
    if not isinstance(dtype, torch.dtype):
        raise _FastMetadataUnsupported("serialized tensor dtype is unavailable")
    try:
        untyped_storage = storage._untyped_storage
        checkpoint_offset = operator.index(untyped_storage._checkpoint_offset)
        storage_nbytes = operator.index(untyped_storage.nbytes())
        normalized_offset = operator.index(storage_offset)
        normalized_shape = tuple(operator.index(size) for size in shape)
        normalized_stride = tuple(operator.index(step) for step in stride)
    except (AttributeError, TypeError, ValueError) as error:
        raise _FastMetadataUnsupported(
            f"serialized tensor storage metadata is unavailable: {error}"
        ) from error
    return _TensorDescriptor(
        storage=_TensorStorageDescriptor(
            checkpoint_offset_bytes=checkpoint_offset,
            nbytes=storage_nbytes,
        ),
        storage_offset_elements=normalized_offset,
        shape=normalized_shape,
        stride=normalized_stride,
        dtype=dtype,
        has_view_bits=_tensor_metadata_flags(metadata),
    )


def _rebuild_tensor_descriptor(
    storage: Any,
    storage_offset: object,
    shape: Iterable[object],
    stride: Iterable[object],
) -> _TensorDescriptor:
    return _tensor_descriptor(
        storage,
        storage_offset,
        shape,
        stride,
        storage.dtype,
        None,
    )


def _rebuild_tensor_v2_descriptor(
    storage: Any,
    storage_offset: object,
    shape: Iterable[object],
    stride: Iterable[object],
    requires_grad: object,
    backward_hooks: object,
    metadata: object = None,
) -> _TensorDescriptor:
    _validate_tensor_autograd_state(requires_grad, backward_hooks)
    return _tensor_descriptor(
        storage,
        storage_offset,
        shape,
        stride,
        storage.dtype,
        metadata,
    )


def _rebuild_tensor_v3_descriptor(
    storage: Any,
    storage_offset: object,
    shape: Iterable[object],
    stride: Iterable[object],
    requires_grad: object,
    backward_hooks: object,
    dtype: torch.dtype,
    metadata: object = None,
) -> _TensorDescriptor:
    _validate_tensor_autograd_state(requires_grad, backward_hooks)
    return _tensor_descriptor(
        storage,
        storage_offset,
        shape,
        stride,
        dtype,
        metadata,
    )


def _rebuild_parameter_descriptor(
    data: object,
    requires_grad: object,
    backward_hooks: object,
) -> object:
    _validate_parameter_rebuild(data, requires_grad, backward_hooks)
    return data


def _rebuild_parameter_with_state_descriptor(
    data: object,
    requires_grad: object,
    backward_hooks: object,
    state: object,
) -> object:
    _validate_parameter_rebuild(data, requires_grad, backward_hooks)
    if not isinstance(state, Mapping):
        raise _FastMetadataUnsupported("unsupported serialized Parameter state")
    return data


def _validate_tensor_autograd_state(
    requires_grad: object,
    backward_hooks: object,
) -> None:
    if type(requires_grad) is not bool or not isinstance(backward_hooks, Mapping):
        raise _FastMetadataUnsupported(
            "unsupported serialized tensor autograd metadata"
        )


def _validate_parameter_rebuild(
    data: object,
    requires_grad: object,
    backward_hooks: object,
) -> None:
    if not isinstance(data, _TensorDescriptor):
        raise _FastMetadataUnsupported("serialized Parameter data is not a tensor")
    _validate_tensor_autograd_state(requires_grad, backward_hooks)


def _rebuild_tensor_subclass_descriptor(
    tensor_type: object,
    dtype: object,
    shape: object,
    stride: object,
    storage_offset: object,
    layout: object,
    device: object,
    requires_grad: object,
) -> NoReturn:
    raise _FastMetadataUnsupported(
        f"unsupported serialized wrapper tensor subclass {tensor_type!r}"
    )


def _dtensor_local_descriptor(state: object) -> _TensorDescriptor:
    normalized_state = state
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[1], dict):
        normalized_state = state[1]
    if not isinstance(normalized_state, dict):
        raise _FastMetadataUnsupported("serialized DTensor state is not a mapping")
    local_tensor = normalized_state.get("_local_tensor")
    if not isinstance(local_tensor, _TensorDescriptor):
        raise _FastMetadataUnsupported(
            "serialized DTensor state has no plain local tensor"
        )
    return local_tensor


def _rebuild_from_type_descriptor(
    rebuild: object,
    tensor_type: object,
    args: object,
    state: object,
) -> _TensorDescriptor:
    if tensor_type is not _DTensorMarker:
        raise _FastMetadataUnsupported(
            f"unsupported serialized tensor subclass {tensor_type!r}"
        )
    _validate_dtensor_wrapper_rebuild(rebuild, args)
    return _dtensor_local_descriptor(state)


def _validate_dtensor_wrapper_rebuild(rebuild: object, args: object) -> None:
    if (
        rebuild is not _rebuild_tensor_subclass_descriptor
        or not isinstance(args, tuple)
        or len(args) != 8
    ):
        raise _FastMetadataUnsupported("unsupported serialized DTensor rebuild")
    (
        tensor_type,
        dtype,
        shape,
        stride,
        storage_offset,
        layout,
        device,
        requires_grad,
    ) = args
    try:
        normalized_shape = tuple(operator.index(size) for size in shape)
        normalized_stride = tuple(operator.index(step) for step in stride)
        normalized_storage_offset = operator.index(storage_offset)
    except (TypeError, ValueError) as error:
        raise _FastMetadataUnsupported(
            f"unsupported serialized DTensor layout: {error}"
        ) from error
    if (
        tensor_type is not _DTensorMarker
        or not isinstance(dtype, torch.dtype)
        or len(normalized_shape) != len(normalized_stride)
        or any(size < 0 for size in normalized_shape)
        or any(step < 0 for step in normalized_stride)
        or normalized_storage_offset < 0
        or layout != "torch.strided"
        or not isinstance(device, torch.device)
        or type(requires_grad) is not bool
    ):
        raise _FastMetadataUnsupported("unsupported serialized DTensor layout")


def _identity(value: object) -> object:
    return value


def _serialized_device(
    device_type: object,
    index: object = None,
) -> torch.device:
    if not isinstance(device_type, str) or (
        index is not None and type(index) is not int
    ):
        raise _FastMetadataUnsupported("invalid serialized torch.device")
    try:
        if index is None:
            return torch.device(device_type)
        return torch.device(device_type, index)
    except (RuntimeError, TypeError, ValueError) as error:
        raise _FastMetadataUnsupported(
            f"invalid serialized torch.device: {error}"
        ) from error


class _MetadataUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        handler = _METADATA_PICKLE_GLOBALS.get((module, name))
        if handler is not None:
            return handler
        if module == "torch":
            value = getattr(torch, name, None)
            if isinstance(value, torch.dtype):
                return value
        if (module, name) in _DTENSOR_GLOBALS:
            return _DTensorMarker
        if module in _OPAQUE_DTENSOR_MODULES:
            return _OpaqueMetadataObject
        raise _FastMetadataUnsupported(f"unsupported serialized global {module}.{name}")


class _MetadataPickleModule:
    __name__ = "pickle"
    Unpickler = _MetadataUnpickler
    load = staticmethod(pickle.load)


_METADATA_PICKLE_GLOBALS: Mapping[tuple[str, str], object] = {
    ("builtins", "complex"): complex,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "set"): set,
    ("builtins", "slice"): slice,
    ("collections", "OrderedDict"): dict,
    ("torch", "Size"): tuple,
    ("torch", "device"): _serialized_device,
    ("torch._tensor", "_rebuild_from_type"): _rebuild_from_type_descriptor,
    ("torch._tensor", "_rebuild_from_type_v2"): _rebuild_from_type_descriptor,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter_descriptor,
    (
        "torch._utils",
        "_rebuild_parameter_with_state",
    ): _rebuild_parameter_with_state_descriptor,
    ("torch._utils", "_rebuild_tensor"): _rebuild_tensor_descriptor,
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2_descriptor,
    ("torch._utils", "_rebuild_tensor_v3"): _rebuild_tensor_v3_descriptor,
    (
        "torch._utils",
        "_rebuild_wrapper_subclass",
    ): _rebuild_tensor_subclass_descriptor,
    ("torch.serialization", "_get_layout"): _identity,
    ("torch.storage", "UntypedStorage"): _UNTYPED_STORAGE_MARKER,
}

_DTENSOR_GLOBALS = {
    ("torch.distributed._tensor", "DTensor"),
    ("torch.distributed.tensor", "DTensor"),
    ("torch.distributed.tensor._api", "DTensor"),
}

_OPAQUE_DTENSOR_MODULES = {
    "torch.distributed._mesh_layout",
    "torch.distributed.device_mesh",
    "torch.distributed.tensor._dtensor_spec",
    "torch.distributed.tensor.placement_types",
}


class ArchiveMetadataAdapter(Protocol):
    """Archive-format adapter used by cooperative metadata preparation."""

    def inspect(
        self,
        storage: Storage,
        path: Path,
        *,
        item_key: str,
        demanded_fqns: Collection[str],
        timeout_seconds: float,
    ) -> ArchiveMetadataInspectionResult:
        """Inspect one archive and return metadata only for demanded FQNs."""
        ...


class TorchSerializationMetadataAdapter:
    """Discover tensor storage addresses in a ``torch.save`` ZIP archive."""

    def __init__(self, metric_callback: _MetricCallback | None = None) -> None:
        self._metric_callback = metric_callback

    def inspect(
        self,
        storage: Storage,
        path: Path,
        *,
        item_key: str,
        demanded_fqns: Collection[str],
        timeout_seconds: float,
    ) -> ArchiveMetadataInspectionResult:
        """Inspect a TorchSerialization archive without reading tensor payloads."""

        timeout_seconds = _validate_timeout_seconds(timeout_seconds)
        started_ns = monotonic_ns()
        metrics = _ArchiveDecodeMetrics(archive_name=path.name)
        open_latency_ms = 0.0
        succeeded = False
        try:
            open_started_ns = monotonic_ns()
            stream_context = storage.stream_read(
                path,
                ReadArgs(
                    pre_read_full_file=False,
                    timeout_us=_storage_timeout_us(timeout_seconds),
                ),
            )
            with stream_context as stream:
                open_latency_ms = (monotonic_ns() - open_started_ns) / 1_000_000
                archive_size = _seekable_archive_size(stream)
                if archive_size is None:
                    succeeded = True
                    return _ineligible(
                        MetadataIneligibilityReason.UNSUPPORTED_STORAGE,
                        "storage stream does not provide reliable seek and tell",
                        path,
                    )
                archive_result = _validate_torch_zip_archive(
                    stream,
                    path,
                    archive_size,
                    metrics=metrics,
                )
                if isinstance(archive_result, MetadataPreparationIneligible):
                    succeeded = True
                    return archive_result
                load_started_ns = monotonic_ns()
                try:
                    loaded = _load_lightweight_meta_archive(stream, path)
                except _FastMetadataUnsupported as error:
                    metrics.mode = "legacy"
                    fallback_reason = str(error)
                    metrics.fallback_reason = fallback_reason[:_MAX_METRIC_DETAIL_CHARS]
                    metrics.metadata_load_latency_ms = (
                        monotonic_ns() - load_started_ns
                    ) / 1_000_000
                    legacy_started_ns = monotonic_ns()
                    loaded = _load_meta_archive(stream, path)
                    metrics.legacy_load_latency_ms = (
                        monotonic_ns() - legacy_started_ns
                    ) / 1_000_000
                else:
                    metrics.metadata_load_latency_ms = (
                        monotonic_ns() - load_started_ns
                    ) / 1_000_000
                if isinstance(loaded, MetadataPreparationIneligible):
                    succeeded = True
                    return loaded
                extraction_started_ns = monotonic_ns()
                result = _extract_demanded_metadata(
                    loaded,
                    item_key=item_key,
                    demanded_fqns=demanded_fqns,
                    storage_records=archive_result,
                    path=path,
                )
                metrics.extraction_latency_ms = (
                    monotonic_ns() - extraction_started_ns
                ) / 1_000_000
                succeeded = True
                return result
        except ArchiveMetadataPreflightError:
            raise
        except (io.UnsupportedOperation, NotImplementedError) as error:
            return _ineligible(
                MetadataIneligibilityReason.UNSUPPORTED_STORAGE,
                f"storage stream does not support offset reads: {error}",
                path,
            )
        except OSError as error:
            raise ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.IO,
                path,
                str(error),
            ) from error
        finally:
            _emit_metric(
                self._metric_callback,
                "metadata_archive_decode",
                {
                    **metrics.fields(
                        latency_ms=(monotonic_ns() - started_ns) / 1_000_000,
                        succeeded=succeeded,
                    ),
                    "open_latency_ms": open_latency_ms,
                },
            )


@dataclass(frozen=True, slots=True)
class _InspectionRequest:
    source_rank: int
    path: Path
    demanded_fqns: frozenset[str]


def prepare_source_tensor_metadata(
    storage: Storage,
    source_paths: Mapping[int, Path],
    demands_by_rank: Mapping[int, Collection[str]],
    *,
    item_key: str,
    timeout_seconds: float,
    adapter: ArchiveMetadataAdapter | None = None,
    _max_workers: int = _DEFAULT_METADATA_MAX_WORKERS,
    _metric_callback: _MetricCallback | None = None,
) -> MetadataPreparationResult:
    """Prepare demanded source tensor metadata in deterministic rank/FQN order."""

    if _max_workers <= 0:
        raise ValueError("_max_workers must be positive")
    timeout_seconds = _validate_timeout_seconds(timeout_seconds)
    archive_adapter = (
        TorchSerializationMetadataAdapter(_metric_callback)
        if adapter is None
        else adapter
    )
    requests, outcomes = _build_inspection_requests(
        source_paths,
        demands_by_rank,
    )
    outcomes.update(
        _inspect_source_archives(
            storage,
            archive_adapter,
            requests,
            item_key=item_key,
            max_workers=_max_workers,
            timeout_seconds=timeout_seconds,
            metric_callback=_metric_callback,
        )
    )
    return _resolve_inspection_outcomes(demands_by_rank, requests, outcomes)


def _build_inspection_requests(
    source_paths: Mapping[int, Path],
    demands_by_rank: Mapping[int, Collection[str]],
) -> tuple[list[_InspectionRequest], dict[int, _InspectionOutcome]]:
    requests: list[_InspectionRequest] = []
    outcomes: dict[int, _InspectionOutcome] = {}
    for source_rank in sorted(demands_by_rank):
        if source_rank < 0:
            outcomes[source_rank] = ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.INVALID_METADATA,
                Path("."),
                "source rank must be non-negative",
                source_rank=source_rank,
            )
            continue
        try:
            path = Path(source_paths[source_rank])
        except KeyError:
            outcomes[source_rank] = ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.INVALID_METADATA,
                Path("."),
                "no checkpoint path was provided",
                source_rank=source_rank,
            )
            continue

        raw_demands = tuple(demands_by_rank[source_rank])
        if any(not isinstance(fqn, str) for fqn in raw_demands):
            outcomes[source_rank] = ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.INVALID_METADATA,
                path,
                "demanded FQNs must be strings",
                source_rank=source_rank,
            )
            continue
        demanded_fqns = frozenset(raw_demands)
        if not demanded_fqns:
            outcomes[source_rank] = {}
            continue
        requests.append(
            _InspectionRequest(
                source_rank=source_rank,
                path=path,
                demanded_fqns=demanded_fqns,
            )
        )
    return requests, outcomes


def _inspect_source_archives(
    storage: Storage,
    adapter: ArchiveMetadataAdapter,
    requests: Sequence[_InspectionRequest],
    *,
    item_key: str,
    max_workers: int,
    timeout_seconds: float,
    metric_callback: _MetricCallback | None,
) -> dict[int, _InspectionOutcome]:
    if not requests:
        return {}
    started_ns = monotonic_ns()
    active_lock = threading.Lock()
    active_workers = 0
    peak_workers = 0

    def inspect(
        request: _InspectionRequest,
        submitted_ns: int,
    ) -> ArchiveMetadataInspectionResult:
        nonlocal active_workers, peak_workers
        worker_started_ns = monotonic_ns()
        with active_lock:
            active_workers += 1
            active_worker_count = active_workers
            peak_workers = max(peak_workers, active_workers)
        succeeded = False
        try:
            result = adapter.inspect(
                storage,
                request.path,
                item_key=item_key,
                demanded_fqns=request.demanded_fqns,
                timeout_seconds=timeout_seconds,
            )
            succeeded = True
            return result
        finally:
            finished_ns = monotonic_ns()
            with active_lock:
                active_workers -= 1
            _emit_metric(
                metric_callback,
                "metadata_archive",
                {
                    "active_worker_count": active_worker_count,
                    "archive_name": request.path.name,
                    "demanded_fqn_count": len(request.demanded_fqns),
                    "latency_ms": (finished_ns - worker_started_ns) / 1_000_000,
                    "queue_latency_ms": (worker_started_ns - submitted_ns) / 1_000_000,
                    "source_rank": request.source_rank,
                    "succeeded": succeeded,
                },
            )

    futures: dict[int, Future[ArchiveMetadataInspectionResult]] = {}
    outcomes: dict[int, _InspectionOutcome] = {}
    executor = ThreadPoolExecutor(
        max_workers=min(max_workers, len(requests)),
        thread_name_prefix="checkpoint-metadata",
    )
    wait_for_shutdown = False
    deadline = monotonic() + timeout_seconds
    try:
        for request in requests:
            submitted_ns = monotonic_ns()
            futures[request.source_rank] = executor.submit(
                inspect,
                request,
                submitted_ns,
            )
        completed, pending = wait(
            futures.values(),
            timeout=max(0.0, deadline - monotonic()),
        )
        wait_for_shutdown = not pending
        for request in requests:
            future = futures[request.source_rank]
            if future not in completed:
                outcomes[request.source_rank] = ArchiveMetadataPreflightError(
                    MetadataPreflightErrorKind.IO,
                    request.path,
                    f"metadata inspection timed out after {timeout_seconds} seconds",
                    source_rank=request.source_rank,
                )
                continue
            try:
                outcomes[request.source_rank] = future.result()
            except Exception as error:
                outcomes[request.source_rank] = error
    finally:
        if not wait_for_shutdown:
            for future in futures.values():
                future.cancel()
        executor.shutdown(
            wait=wait_for_shutdown,
            cancel_futures=not wait_for_shutdown,
        )
        _emit_metric(
            metric_callback,
            "metadata_archive_summary",
            {
                "archive_count": len(requests),
                "completed_count": sum(
                    future.done() and not future.cancelled()
                    for future in futures.values()
                ),
                "latency_ms": (monotonic_ns() - started_ns) / 1_000_000,
                "peak_worker_count": peak_workers,
                "worker_count": min(max_workers, len(requests)),
            },
        )
    return outcomes


def _emit_metric(
    callback: _MetricCallback | None,
    event: str,
    fields: Mapping[str, _MetricValue],
) -> None:
    if callback is None:
        return
    try:
        callback(event, fields)
    except Exception:
        logger.warning("cooperative metadata metric callback failed", exc_info=True)


def _resolve_inspection_outcomes(
    demands_by_rank: Mapping[int, Collection[str]],
    requests: Sequence[_InspectionRequest],
    outcomes: Mapping[int, _InspectionOutcome],
) -> MetadataPreparationResult:
    requests_by_rank = {request.source_rank: request for request in requests}
    for source_rank in sorted(demands_by_rank):
        outcome = outcomes[source_rank]
        if isinstance(outcome, Exception):
            _raise_inspection_error(source_rank, outcome)
        assert not isinstance(outcome, Exception)
        request = requests_by_rank.get(source_rank)
        if request is None or isinstance(outcome, MetadataPreparationIneligible):
            continue
        missing = request.demanded_fqns.difference(outcome)
        if missing:
            raise ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.INVALID_METADATA,
                request.path,
                f"archive adapter omitted demanded FQNs: {sorted(missing)!r}",
                source_rank=source_rank,
            )

    for source_rank in sorted(demands_by_rank):
        outcome = outcomes[source_rank]
        if isinstance(outcome, MetadataPreparationIneligible):
            request = requests_by_rank[source_rank]
            return _contextualize_ineligible(source_rank, request.path, outcome)

    metadata_by_rank: dict[int, Mapping[str, SourceTensorMetadata]] = {}
    for source_rank in sorted(demands_by_rank):
        request = requests_by_rank.get(source_rank)
        if request is None:
            metadata_by_rank[source_rank] = {}
            continue
        inspected = outcomes[source_rank]
        assert not isinstance(inspected, (Exception, MetadataPreparationIneligible))
        metadata_by_rank[source_rank] = {
            fqn: inspected[fqn] for fqn in sorted(request.demanded_fqns)
        }

    return MetadataPreparationEligible(metadata_by_rank=metadata_by_rank)


def _raise_inspection_error(source_rank: int, error: Exception) -> NoReturn:
    if not isinstance(error, ArchiveMetadataPreflightError):
        raise error
    if error.source_rank is not None:
        raise error
    raise ArchiveMetadataPreflightError(
        error.kind,
        error.path,
        error.detail,
        source_rank=source_rank,
    ) from error


def _contextualize_ineligible(
    source_rank: int,
    path: Path,
    result: MetadataPreparationIneligible,
) -> MetadataPreparationIneligible:
    return MetadataPreparationIneligible(
        reason=result.reason,
        detail=result.detail,
        source_rank=(source_rank if result.source_rank is None else result.source_rank),
        path=path if result.path is None else result.path,
    )


def _validate_timeout_seconds(timeout_seconds: float) -> float:
    try:
        normalized = float(timeout_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError("timeout_seconds must be a positive finite number") from error
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError("timeout_seconds must be a positive finite number")
    return normalized


def _storage_timeout_us(timeout_seconds: float) -> int:
    return min(
        _MAX_STORAGE_TIMEOUT_US,
        max(1, math.ceil(timeout_seconds * 1_000_000)),
    )


def _seekable_archive_size(stream: Any) -> int | None:
    seekable = getattr(stream, "seekable", None)
    if callable(seekable) and not seekable():
        return None
    try:
        stream.seek(0, io.SEEK_END)
        archive_size = operator.index(stream.tell())
        stream.seek(0, io.SEEK_SET)
        if stream.tell() != 0 or archive_size < 0:
            return None
        return archive_size
    except (AttributeError, TypeError):
        return None


def _validate_torch_zip_archive(
    stream: Any,
    path: Path,
    archive_size: int,
    *,
    metrics: _ArchiveDecodeMetrics | None = None,
) -> _StorageRecords | MetadataPreparationIneligible:
    validation_started_ns = monotonic_ns()
    stream.seek(0)
    signature = stream.read(4)
    stream.seek(0)
    if not isinstance(signature, bytes):
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_STORAGE,
            "storage stream read did not return bytes",
            path,
        )
    if signature != b"PK\x03\x04":
        if signature.startswith(b"PK"):
            raise ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
                path,
                "truncated ZIP archive header",
            )
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
            "archive is not the ZIP-based TorchSerialization format",
            path,
        )

    entries = _read_zip_entries(stream, path)
    if metrics is not None:
        metrics.archive_entry_count = len(entries)
        data_pickle_entries = [
            entry
            for entry in entries
            if entry.filename == "data.pkl" or entry.filename.endswith("/data.pkl")
        ]
        if len(data_pickle_entries) == 1:
            metrics.data_pickle_bytes = data_pickle_entries[0].file_size
    unsupported = _unsupported_zip_layout(entries, path)
    if unsupported is not None:
        return unsupported
    _validate_data_pickle_crc(stream, entries, path)
    if metrics is not None:
        metrics.zip_validation_latency_ms = (
            monotonic_ns() - validation_started_ns
        ) / 1_000_000
    index_started_ns = monotonic_ns()
    storage_records, index_mode = _collect_storage_records(
        stream,
        entries,
        archive_size=archive_size,
        path=path,
    )
    if metrics is not None:
        metrics.storage_index_latency_ms = (
            monotonic_ns() - index_started_ns
        ) / 1_000_000
        metrics.storage_index_mode = index_mode
        metrics.storage_record_count = len(storage_records)
    return storage_records


def _read_zip_entries(stream: Any, path: Path) -> list[zipfile.ZipInfo]:
    try:
        with zipfile.ZipFile(stream) as archive:
            return archive.infolist()
    except (
        ValueError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ) as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            str(error),
        ) from error
    finally:
        stream.seek(0)


def _unsupported_zip_layout(
    entries: Sequence[zipfile.ZipInfo],
    path: Path,
) -> MetadataPreparationIneligible | None:
    names = [entry.filename.rstrip("/") for entry in entries]
    if len(names) != len(set(names)):
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            "ZIP archive contains duplicate record names",
        )
    if any(
        name == "constants.pkl" or name.endswith("/constants.pkl") for name in names
    ):
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
            "archive is a TorchScript package rather than a TorchSerialization "
            "checkpoint",
            path,
        )
    data_pickle_names = [
        name for name in names if name == "data.pkl" or name.endswith("/data.pkl")
    ]
    if not data_pickle_names:
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
            "ZIP archive has no TorchSerialization data.pkl record",
            path,
        )
    if len(data_pickle_names) != 1:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            "ZIP archive contains multiple data.pkl records",
        )
    if not any(name == "version" or name.endswith("/version") for name in names):
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
            "ZIP archive has no TorchSerialization version record",
            path,
        )
    if any(
        not entry.is_dir() and entry.compress_type != zipfile.ZIP_STORED
        for entry in entries
    ):
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
            "compressed TorchSerialization records do not have direct byte addresses",
            path,
        )
    if any(entry.flag_bits & 0x1 for entry in entries):
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_ARCHIVE,
            "encrypted TorchSerialization records do not have readable byte addresses",
            path,
        )
    return None


def _validate_data_pickle_crc(
    stream: Any,
    entries: Sequence[zipfile.ZipInfo],
    path: Path,
) -> None:
    entry = next(
        entry
        for entry in entries
        if entry.filename == "data.pkl" or entry.filename.endswith("/data.pkl")
    )
    if entry.CRC == 0:
        return
    try:
        with zipfile.ZipFile(stream) as archive:
            with archive.open(entry) as data_pickle:
                while data_pickle.read(1024 * 1024):
                    pass
    except (ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            str(error),
        ) from error
    finally:
        stream.seek(0)


def _collect_storage_records(
    stream: Any,
    entries: Sequence[zipfile.ZipInfo],
    *,
    archive_size: int,
    path: Path,
) -> tuple[_StorageRecords, str]:
    calculated = _collect_calculated_storage_records(
        stream,
        entries,
        archive_size=archive_size,
        path=path,
    )
    if calculated is not None:
        return calculated, "calculated"
    return (
        _collect_storage_records_from_headers(
            stream,
            entries,
            archive_size=archive_size,
            path=path,
        ),
        "headers",
    )


def _collect_calculated_storage_records(
    stream: Any,
    entries: Sequence[zipfile.ZipInfo],
    *,
    archive_size: int,
    path: Path,
) -> _StorageRecords | None:
    reader_type = getattr(torch._C, "PyTorchFileReader", None)
    offset_without_read = getattr(reader_type, "get_record_offset_no_read", None)
    if reader_type is None or offset_without_read is None:
        return None

    prefix = _torch_archive_prefix(entries)
    if prefix is None:
        return None
    try:
        stream.seek(0)
        reader = reader_type(stream)
        if not reader.has_record(".format_version"):
            return None
        if reader.get_record(".format_version") < b"1":
            return None
        alignment = 64
        if reader.has_record(".storage_alignment"):
            alignment = int(reader.get_record(".storage_alignment"))
        if alignment <= 0:
            raise ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
                path,
                "TorchSerialization storage alignment must be positive",
            )
        return _calculated_storage_records_from_entries(
            reader,
            entries,
            prefix=prefix,
            alignment=alignment,
            archive_size=archive_size,
            path=path,
        )
    except ArchiveMetadataPreflightError:
        raise
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return None
    finally:
        stream.seek(0)


def _calculated_storage_records_from_entries(
    reader: Any,
    entries: Sequence[zipfile.ZipInfo],
    *,
    prefix: str,
    alignment: int,
    archive_size: int,
    path: Path,
) -> _StorageRecords | None:
    reader_records = set(reader.get_all_records())
    records: dict[int, int] = {}
    storage_names: set[str] = set()
    ordered_entries = sorted(
        (entry for entry in entries if not entry.is_dir()),
        key=lambda entry: entry.header_offset,
    )
    next_header_by_name = {
        entry.filename: operator.index(next_entry.header_offset)
        for entry, next_entry in zip(ordered_entries, ordered_entries[1:])
    }
    for entry in ordered_entries:
        if not _is_storage_record(entry):
            continue
        name = entry.filename.removeprefix(prefix)
        if name == entry.filename or name not in reader_records:
            return None
        if name in storage_names:
            raise ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
                path,
                f"duplicate TorchSerialization record {name!r}",
            )
        storage_names.add(name)
        record_size = operator.index(entry.file_size)
        data_offset = reader.get_record_offset_no_read(
            operator.index(entry.header_offset),
            name,
            record_size,
            alignment,
        )
        _validate_storage_record_bounds(
            data_offset,
            record_size,
            archive_size=archive_size,
            record_name=entry.filename,
            path=path,
        )
        _validate_calculated_storage_boundary(
            entry,
            data_offset=data_offset,
            next_header_offset=next_header_by_name.get(entry.filename),
            alignment=alignment,
            path=path,
        )
        if data_offset in records:
            raise ArchiveMetadataPreflightError(
                MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
                path,
                f"multiple storage records start at byte {data_offset}",
            )
        records[data_offset] = record_size
    return records


def _torch_archive_prefix(entries: Sequence[zipfile.ZipInfo]) -> str | None:
    data_pickle_names = [
        entry.filename
        for entry in entries
        if entry.filename == "data.pkl" or entry.filename.endswith("/data.pkl")
    ]
    if len(data_pickle_names) != 1:
        return None
    return data_pickle_names[0][: -len("data.pkl")]


def _collect_storage_records_from_headers(
    stream: Any,
    entries: Sequence[zipfile.ZipInfo],
    *,
    archive_size: int,
    path: Path,
) -> _StorageRecords:
    records: dict[int, int] = {}
    storage_names: set[str] = set()
    try:
        for entry in entries:
            if not _is_storage_record(entry):
                continue
            if entry.filename in storage_names:
                raise ArchiveMetadataPreflightError(
                    MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
                    path,
                    f"duplicate TorchSerialization record {entry.filename!r}",
                )
            storage_names.add(entry.filename)
            data_offset = _zip_entry_data_offset(
                stream,
                entry,
                archive_size=archive_size,
                path=path,
            )
            if data_offset in records:
                raise ArchiveMetadataPreflightError(
                    MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
                    path,
                    f"multiple storage records start at byte {data_offset}",
                )
            records[data_offset] = entry.file_size
    finally:
        stream.seek(0)
    return records


def _validate_storage_record_bounds(
    data_offset: int,
    record_size: int,
    *,
    archive_size: int,
    record_name: str,
    path: Path,
) -> None:
    data_end = data_offset + record_size
    if data_offset < 0 or data_end > archive_size:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {record_name!r} extends beyond the archive",
        )


def _validate_calculated_storage_boundary(
    entry: zipfile.ZipInfo,
    *,
    data_offset: int,
    next_header_offset: int | None,
    alignment: int,
    path: Path,
) -> None:
    header_offset = operator.index(entry.header_offset)
    encoded_name_bytes = len(entry.filename.encode("utf-8"))
    uses_zip64 = header_offset >= _ZIP32_MAX or entry.file_size >= _ZIP32_MAX
    minimum_data_offset = (
        header_offset + _ZIP_LOCAL_FILE_HEADER_BYTES + encoded_name_bytes + 4
    )
    if data_offset < minimum_data_offset or data_offset % alignment != 0:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {entry.filename!r} has inconsistent alignment padding",
        )
    if next_header_offset is None:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {entry.filename!r} has no following ZIP record",
        )
    descriptor_bytes = 0
    if entry.file_size > 0:
        descriptor_bytes = (
            _ZIP64_DATA_DESCRIPTOR_BYTES if uses_zip64 else _ZIP_DATA_DESCRIPTOR_BYTES
        )
    expected_next_header = data_offset + entry.file_size + descriptor_bytes
    if expected_next_header != next_header_offset:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {entry.filename!r} has inconsistent local-header "
            "padding or record boundary",
        )


def _is_storage_record(entry: zipfile.ZipInfo) -> bool:
    components = entry.filename.rstrip("/").split("/")
    return (
        not entry.is_dir()
        and len(components) >= 2
        and components[-2] == "data"
        and components[-1].isdigit()
    )


def _zip_entry_data_offset(
    stream: Any,
    entry: zipfile.ZipInfo,
    *,
    archive_size: int,
    path: Path,
) -> int:
    header_offset = operator.index(entry.header_offset)
    if header_offset < 0 or header_offset + _ZIP_LOCAL_FILE_HEADER_BYTES > archive_size:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {entry.filename!r} has an invalid local header offset",
        )
    stream.seek(header_offset)
    try:
        header = _read_exact_bytes(stream, _ZIP_LOCAL_FILE_HEADER_BYTES)
    except EOFError as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {entry.filename!r} has a truncated local header",
        ) from error
    if header[:4] != b"PK\x03\x04":
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {entry.filename!r} has an invalid local header",
        )
    filename_bytes = int.from_bytes(header[26:28], "little")
    extra_bytes = int.from_bytes(header[28:30], "little")
    data_offset = (
        header_offset + _ZIP_LOCAL_FILE_HEADER_BYTES + filename_bytes + extra_bytes
    )
    data_end = data_offset + operator.index(entry.compress_size)
    if data_offset < 0 or data_end > archive_size:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"storage record {entry.filename!r} extends beyond the archive",
        )
    return data_offset


def _read_exact_bytes(stream: Any, length: int) -> bytes:
    result = bytearray()
    while len(result) < length:
        chunk = stream.read(length - len(result))
        if not isinstance(chunk, bytes) or not chunk:
            raise EOFError(f"expected {length} bytes, read {len(result)}")
        result.extend(chunk)
    return bytes(result)


def _load_lightweight_meta_archive(stream: Any, path: Path) -> Any:
    stream.seek(0)
    serialization_tls = getattr(torch.serialization, "_serialization_tls", None)
    previous_map_location = getattr(serialization_tls, "map_location", None)
    try:
        return torch.load(
            stream,
            map_location="meta",
            pickle_module=_MetadataPickleModule,
            weights_only=False,
        )
    except _FastMetadataUnsupported:
        raise
    except (io.UnsupportedOperation, OSError):
        raise
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            str(error),
        ) from error
    except Exception as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"TorchSerialization metadata decode failed: {error}",
        ) from error
    finally:
        if serialization_tls is not None:
            serialization_tls.map_location = previous_map_location
        stream.seek(0)


def _load_meta_archive(
    stream: Any,
    path: Path,
) -> Any | MetadataPreparationIneligible:
    stream.seek(0)
    serialization_tls = getattr(torch.serialization, "_serialization_tls", None)
    previous_map_location = getattr(serialization_tls, "map_location", None)
    try:
        return torch.load(stream, map_location="meta", weights_only=False)
    except NotImplementedError as error:
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_VALUE,
            f"TorchSerialization value is unsupported on the meta device: {error}",
            path,
        )
    except (io.UnsupportedOperation, OSError):
        raise
    except (EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            str(error),
        ) from error
    except Exception as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.CORRUPT_ARCHIVE,
            path,
            f"TorchSerialization load failed: {error}",
        ) from error
    finally:
        if serialization_tls is not None:
            serialization_tls.map_location = previous_map_location
        stream.seek(0)


def _extract_demanded_metadata(
    value: Any,
    *,
    item_key: str,
    demanded_fqns: Collection[str],
    storage_records: _StorageRecords,
    path: Path,
) -> ArchiveMetadataInspectionResult:
    demanded = set(demanded_fqns)
    leaves = _collect_demanded_checkpoint_leaves(
        value,
        demanded=demanded,
        item_key=item_key,
        path=path,
    )

    result: dict[str, SourceTensorMetadata] = {}
    storage_contracts: dict[int, tuple[int, torch.dtype]] = {}
    for fqn in sorted(demanded):
        tensor = _unwrap_dtensor(leaves[fqn][1])
        if isinstance(tensor, _TensorDescriptor):
            storage_contract = (tensor.storage.nbytes, tensor.dtype)
            previous_contract = storage_contracts.setdefault(
                tensor.storage.checkpoint_offset_bytes,
                storage_contract,
            )
            if previous_contract != storage_contract:
                raise ArchiveMetadataPreflightError(
                    MetadataPreflightErrorKind.INVALID_METADATA,
                    path,
                    "serialized tensors alias one storage record with "
                    "inconsistent size or dtype",
                )
            tensor_metadata = _metadata_for_tensor_descriptor(
                tensor,
                fqn=fqn,
                storage_records=storage_records,
                path=path,
            )
            if isinstance(tensor_metadata, MetadataPreparationIneligible):
                return tensor_metadata
            result[fqn] = tensor_metadata
            continue
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "meta":
            return _ineligible(
                MetadataIneligibilityReason.UNSUPPORTED_VALUE,
                f"demanded value {fqn!r} is not a plain serialized tensor",
                path,
            )
        tensor_metadata = _metadata_for_tensor(
            tensor,
            fqn=fqn,
            storage_records=storage_records,
            path=path,
        )
        if isinstance(tensor_metadata, MetadataPreparationIneligible):
            return tensor_metadata
        result[fqn] = tensor_metadata
    return result


def _collect_demanded_checkpoint_leaves(
    value: Any,
    *,
    demanded: Set[str],
    item_key: str,
    path: Path,
) -> dict[str, tuple[NestedPath, Any]]:
    leaves: dict[str, tuple[NestedPath, Any]] = {}
    try:
        for nested_path, leaf in _iter_checkpoint_leaves(value):
            fqn = get_fqn_from_nested_path(nested_path)
            if fqn not in demanded:
                continue
            previous = leaves.get(fqn)
            if previous is not None:
                raise ArchiveMetadataPreflightError(
                    MetadataPreflightErrorKind.INVALID_METADATA,
                    path,
                    f"item {item_key!r} has an FQN collision between "
                    f"{previous[0]!r} and {nested_path!r}",
                )
            leaves[fqn] = (nested_path, leaf)
    except _InvalidCheckpointStructure as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.INVALID_METADATA,
            path,
            f"item {item_key!r} has an invalid container structure: {error}",
        ) from error

    missing = demanded.difference(leaves)
    if missing:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.INVALID_METADATA,
            path,
            f"item {item_key!r} is missing demanded FQNs: {sorted(missing)!r}",
        )
    return leaves


def _metadata_for_tensor_descriptor(
    tensor: _TensorDescriptor,
    *,
    fqn: str,
    storage_records: _StorageRecords,
    path: Path,
) -> SourceTensorMetadata | MetadataPreparationIneligible:
    if tensor.has_view_bits:
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_VALUE,
            f"demanded tensor {fqn!r} has conjugate or negative view bits",
            path,
        )
    try:
        metadata = SourceTensorMetadata(
            fqn=fqn,
            checkpoint_offset_bytes=tensor.storage.checkpoint_offset_bytes,
            storage_offset_elements=tensor.storage_offset_elements,
            storage_nbytes=tensor.storage.nbytes,
            shape=tensor.shape,
            stride=tensor.stride,
            dtype=str(tensor.dtype),
            element_size_bytes=torch.empty((), dtype=tensor.dtype).element_size(),
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.INVALID_METADATA,
            path,
            f"invalid metadata for demanded tensor {fqn!r}: {error}",
        ) from error
    _validate_metadata_storage_record(
        metadata,
        storage_records=storage_records,
        path=path,
    )
    return metadata


def _iter_checkpoint_leaves(
    value: Any,
    nested_path: NestedPath = (),
) -> Iterator[tuple[NestedPath, Any]]:
    stack: list[tuple[bool, NestedPath, Any]] = [(False, nested_path, value)]
    active_containers: set[int] = set()
    while stack:
        exiting, current_path, current = stack.pop()
        if exiting:
            active_containers.remove(id(current))
            continue
        children: list[tuple[object, Any]] | None = None
        if isinstance(current, Mapping):
            children = list(current.items())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            children = list(enumerate(current))
        if children is None or isinstance(current, Set):
            yield current_path, current
            continue
        if len(current_path) >= _MAX_CHECKPOINT_CONTAINER_DEPTH:
            raise _InvalidCheckpointStructure(
                "container nesting exceeds the supported depth"
            )
        container_id = id(current)
        if container_id in active_containers:
            raise _InvalidCheckpointStructure("container graph contains a cycle")
        active_containers.add(container_id)
        stack.append((True, current_path, current))
        for key, child in reversed(children):
            stack.append((False, (*current_path, key), child))


def _unwrap_dtensor(value: Any) -> Any:
    return value._local_tensor if isinstance(value, DTensor) else value


def _metadata_for_tensor(
    tensor: torch.Tensor,
    *,
    fqn: str,
    storage_records: _StorageRecords,
    path: Path,
) -> SourceTensorMetadata | MetadataPreparationIneligible:
    if tensor.layout is not torch.strided or tensor.is_quantized or tensor.is_nested:
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_VALUE,
            f"demanded tensor {fqn!r} does not use a plain strided storage",
            path,
        )
    if tensor.is_conj() or tensor.is_neg():
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_VALUE,
            f"demanded tensor {fqn!r} has conjugate or negative view bits",
            path,
        )

    storage = tensor.untyped_storage()
    checkpoint_offset_value = getattr(storage, "_checkpoint_offset", None)
    if checkpoint_offset_value is None:
        return _ineligible(
            MetadataIneligibilityReason.UNSUPPORTED_VALUE,
            f"demanded tensor {fqn!r} has no resolved checkpoint offset",
            path,
        )

    try:
        checkpoint_offset = operator.index(checkpoint_offset_value)
        storage_offset = operator.index(tensor.storage_offset())
        storage_nbytes = operator.index(storage.nbytes())
        shape = tuple(operator.index(size) for size in tensor.shape)
        stride = tuple(operator.index(step) for step in tensor.stride())
        element_size = operator.index(tensor.element_size())
        metadata = SourceTensorMetadata(
            fqn=fqn,
            checkpoint_offset_bytes=checkpoint_offset,
            storage_offset_elements=storage_offset,
            storage_nbytes=storage_nbytes,
            shape=shape,
            stride=stride,
            dtype=str(tensor.dtype),
            element_size_bytes=element_size,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.INVALID_METADATA,
            path,
            f"invalid metadata for demanded tensor {fqn!r}: {error}",
        ) from error

    _validate_metadata_storage_record(
        metadata,
        storage_records=storage_records,
        path=path,
    )
    return metadata


def _validate_metadata_storage_record(
    metadata: SourceTensorMetadata,
    *,
    storage_records: _StorageRecords,
    path: Path,
) -> None:
    record_nbytes = storage_records.get(metadata.checkpoint_offset_bytes)
    if record_nbytes is None:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.INVALID_METADATA,
            path,
            f"demanded tensor {metadata.fqn!r} checkpoint offset does not identify a "
            "TorchSerialization storage record",
        )
    if metadata.storage_nbytes != record_nbytes:
        raise ArchiveMetadataPreflightError(
            MetadataPreflightErrorKind.INVALID_METADATA,
            path,
            f"demanded tensor {metadata.fqn!r} reports "
            f"{metadata.storage_nbytes} storage bytes, "
            f"but its archive record contains {record_nbytes}",
        )


def _ineligible(
    reason: MetadataIneligibilityReason,
    detail: str,
    path: Path,
) -> MetadataPreparationIneligible:
    return MetadataPreparationIneligible(reason=reason, detail=detail, path=path)
