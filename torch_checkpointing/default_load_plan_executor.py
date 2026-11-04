# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The default ``LoadPlanExecutor``: reads source checkpoint files and copies the
planned chunks into their target tensors.
"""

import io
import logging
import os
import zipfile
from collections.abc import Generator, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from enum import auto, Enum
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from typing_extensions import override

from .checkpoint_layout import (
    LayoutInfo,
    SafetensorsSerialization,
    TorchSerialization,
)
from .distributed_metadata import DistributedItemMetadata
from .hf.metadata import HuggingFaceSafetensorsItemMetadata
from .load_plan_executor import LoadExecutionContext, LoadPlanExecutor
from .logging_utils import EventLogger, EventType
from .resharding import LoadPlan, ReshardingInfo
from .resharding_utils import (
    convert_nested_path_dict_to_fqn,
    dedupe_aliased_targets,
    get_fqn_from_nested_path,
)
from .safetensors_metadata import SafetensorsFileMetadata
from .storage.base_storage import ReadArgs, Storage
from .storage.torch_serialization import load_torch_serialized_from_storage
from .types import CheckpointPath, ItemKey, NestedPath
from .walk_utils import walk_checkpoint_structure

logger: logging.Logger = logging.getLogger(__name__)

# Source files read concurrently per host when the executor does not set
# ``file_read_workers``. Ranks on a host share its storage client and network
# link, so each local rank reads an equal share. Too few reads in flight leave
# the link idle while each waits on latency: filling it takes about host read
# bandwidth / one read's throughput. Clients that read ahead into a fixed cache
# start evicting each other past cache size / read-ahead window.
DEFAULT_READS_IN_FLIGHT_PER_HOST: int = 24
# Source files each rank reads concurrently when ``LOCAL_WORLD_SIZE`` is unset or
# not a positive integer, so the number of ranks sharing the host is unknown.
DEFAULT_FILE_READ_WORKERS: int = 8

__all__ = ["DefaultLoadPlanExecutor", "ReshardingReadStrategy"]


def _default_file_read_workers() -> int:
    # Assumes the host runs only this job and every local rank is a trainer that
    # loads at the same time; set ``file_read_workers`` on the executor otherwise.
    local_world_size = os.environ.get("LOCAL_WORLD_SIZE", "")
    if not local_world_size.isdigit() or int(local_world_size) < 1:
        return DEFAULT_FILE_READ_WORKERS
    return max(1, DEFAULT_READS_IN_FLIGHT_PER_HOST // int(local_world_size))


class ReshardingReadStrategy(Enum):
    """How DefaultLoadPlanExecutor reads source checkpoint files."""

    AUTO = auto()
    OFFSET = auto()
    FULL_FILE = auto()


def _read_exact(stream: io.RawIOBase, offset: int, buffer: memoryview) -> None:
    stream.seek(offset)
    if stream.tell() != offset:
        raise OSError(f"Failed to seek to checkpoint offset {offset}")

    bytes_read = 0
    while bytes_read < len(buffer):
        count = stream.readinto(buffer[bytes_read:])
        if not count:
            raise EOFError(
                f"Expected {len(buffer)} bytes at checkpoint offset {offset}, "
                f"but read {bytes_read}"
            )
        bytes_read += count


def _validate_offset_read_archive(stream: io.RawIOBase) -> None:
    """Require a seekable torch archive whose records are stored uncompressed."""
    try:
        if not stream.seekable():
            raise NotImplementedError("Checkpoint stream is not seekable")
        with zipfile.ZipFile(stream) as archive:
            if any(
                member.compress_type != zipfile.ZIP_STORED
                for member in archive.infolist()
            ):
                raise NotImplementedError("Checkpoint archive contains compressed data")
        stream.seek(0)
    except (io.UnsupportedOperation, zipfile.BadZipFile) as error:
        raise NotImplementedError(
            "Checkpoint is not a seekable, uncompressed torch archive"
        ) from error


def _collect_leaf_values(item_key: str, value: Any) -> dict[NestedPath, Any]:
    """Collect checkpoint leaves keyed by their NestedPath.

    Args:
        item_key: The checkpoint item key.
        value: Nested checkpoint value to walk.

    Returns:
        Dictionary mapping NestedPath to leaf values.
    """
    result: dict[NestedPath, Any] = {}

    def _collect(path: CheckpointPath, obj: Any, _: Any) -> Any:
        result[path.nested_path] = obj
        return obj

    walk_checkpoint_structure(
        item_key=item_key,
        source=value,
        target=None,
        leaf_fn=_collect,
    )
    return result


def _flatten_state_dict(item_key: str, value: Any) -> dict[str, Any]:
    """Flatten a checkpoint value to dot-separated keys.

    Args:
        item_key: The checkpoint item key.
        value: Nested checkpoint value to flatten.

    Returns:
        Flat dictionary with dot-separated keys.
    """
    path_to_value = _collect_leaf_values(item_key, value)
    flattened, _ = convert_nested_path_dict_to_fqn(path_to_value)
    return flattened


def _unwrap_dtensor(value: torch.Tensor) -> torch.Tensor:
    return value._local_tensor if isinstance(value, DTensor) else value


def _slice_source_tensor(source: torch.Tensor, load_plan: LoadPlan) -> torch.Tensor:
    # The full-file fallback does not pass through the offset-read validation.
    # Validate here because Python slicing silently clamps out-of-bounds ranges.
    _validate_source_slice_bounds(tuple(source.shape), load_plan)
    source_slice = tuple(
        slice(offset, offset + size)
        for offset, size in zip(load_plan.src_offsets, load_plan.src_sizes)
    )
    return source[source_slice]


def _validate_source_slice_bounds(
    source_shape: tuple[int, ...],
    load_plan: LoadPlan,
) -> None:
    if not (
        len(load_plan.src_offsets) == len(load_plan.src_sizes) == len(source_shape)
    ):
        raise ValueError(
            f"Load plan rank does not match source tensor {load_plan.src_fqn!r}"
        )

    for dimension, (source_size, offset, size) in enumerate(
        zip(source_shape, load_plan.src_offsets, load_plan.src_sizes)
    ):
        if source_size < 0 or offset < 0 or size < 0 or offset + size > source_size:
            raise ValueError(
                f"Source slice for {load_plan.src_fqn!r} is out of bounds in "
                f"dimension {dimension}: offset={offset}, size={size}, "
                f"source_size={source_size}"
            )


def _read_source_tensor_slice(
    stream: io.RawIOBase,
    source: torch.Tensor,
    load_plan: LoadPlan,
) -> torch.Tensor:
    source = _unwrap_dtensor(source)
    if type(source) not in (torch.Tensor, torch.nn.Parameter) or not source.is_meta:
        raise NotImplementedError(f"Source {load_plan.src_fqn!r} is not a plain tensor")
    if source.layout is not torch.strided:
        raise NotImplementedError(
            f"Source {load_plan.src_fqn!r} does not use a strided storage"
        )
    if source.is_quantized:
        raise NotImplementedError(f"Source {load_plan.src_fqn!r} is quantized")
    _validate_source_slice_bounds(tuple(source.shape), load_plan)
    strides = tuple(source.stride())

    if any(size == 0 for size in load_plan.src_sizes):
        result = torch.empty(load_plan.src_sizes, dtype=source.dtype)
    else:
        checkpoint_offset = getattr(
            source.untyped_storage(), "_checkpoint_offset", None
        )
        if checkpoint_offset is None:
            raise NotImplementedError(
                f"Source {load_plan.src_fqn!r} has no checkpoint offset"
            )

        element_size = source.element_size()
        first_element = source.storage_offset() + sum(
            offset * stride for offset, stride in zip(load_plan.src_offsets, strides)
        )
        # Span from the first wanted element through the last, read in one go.
        # For a contiguous source this never exceeds the tensor itself, so the
        # worst case matches reading the whole tensor.
        span = 1 + sum(
            (size - 1) * stride for size, stride in zip(load_plan.src_sizes, strides)
        )
        # Uninitialized, unlike a bytearray, which would zero-fill memory that
        # the read overwrites anyway.
        packed = torch.empty(span * element_size, dtype=torch.uint8)
        _read_exact(
            stream,
            checkpoint_offset + first_element * element_size,
            memoryview(packed.numpy()),
        )
        result = packed.view(source.dtype).as_strided(load_plan.src_sizes, strides)

    if source.is_conj():
        result = result.conj()
    if source.is_neg():
        result = result._neg_view()

    return result


class DefaultLoadPlanExecutor(LoadPlanExecutor):
    """Reads each item's source files and copies the planned chunks into its targets.

    Items are loaded one after another, in the order given. Within an item,
    source files are read concurrently.

    Args:
        read_strategy: Whether to use offset reads, full-file reads, or try
            offset reads and fall back to full-file reads when unsupported.
        file_read_workers: Number of source files read concurrently. None
            splits a per-host default across ``LOCAL_WORLD_SIZE`` ranks, or uses
            ``DEFAULT_FILE_READ_WORKERS`` if it is unset or invalid. Set 1 to read one file
            at a time.
    """

    def __init__(
        self,
        *,
        read_strategy: ReshardingReadStrategy = ReshardingReadStrategy.AUTO,
        file_read_workers: int | None = None,
    ) -> None:
        if file_read_workers is not None and file_read_workers < 1:
            raise ValueError(
                f"file_read_workers must be at least 1, got {file_read_workers}"
            )
        self._read_strategy = read_strategy
        self._file_read_workers = (
            _default_file_read_workers()
            if file_read_workers is None
            else file_read_workers
        )

    @override
    def execute(
        self,
        items: Mapping[ItemKey, ReshardingInfo],
        context: LoadExecutionContext,
    ) -> None:
        event_logger = EventLogger()
        for item_key, resharding_info in items.items():
            if not resharding_info.nested_path_to_load_plans:
                logger.warning("No load plans generated for item %r", item_key)
                continue
            self._execute_item(
                context.checkpoint_path,
                context.source_metadata[item_key],
                item_key,
                resharding_info,
                context.storage,
            )
            logger.info(
                f"Done reading planned data for {item_key}.",
                extra=event_logger(
                    EventType.LOG_METRIC,
                    metric_name=f"train.checkpoint_read.execute.filesystem.{item_key}.read.latency_ms",
                ),
            )

    def _execute_item(
        self,
        source_path: Path,
        source_metadata: DistributedItemMetadata,
        item_key: str,
        resharding_info: ReshardingInfo,
        storage: Storage,
    ) -> None:
        """Execute load plans by reading source files and copying data into target.

        Groups load plans by source rank, then reads and copies the planned
        source data into the bound target tensors.

        Args:
            source_path: Base path to the source checkpoint directory.
            source_metadata: Source checkpoint metadata for this item.
            item_key: The checkpoint item key being loaded.
            resharding_info: The load plans and the target tensors they fill.
            storage: Storage backend for reading checkpoint files.
        """
        target_tensors = {
            path: resharding_info.nested_path_to_target_tensors[path]
            for path in resharding_info.nested_path_to_load_plans
        }
        _, nested_path_to_load_plans, aliased_paths = dedupe_aliased_targets(
            target_tensors,
            resharding_info.nested_path_to_load_plans,
        )
        # Aliased targets share memory, so concurrent reads could race on it.
        # Read one file at a time whenever any are found.
        file_read_workers = 1 if aliased_paths else self._file_read_workers
        if aliased_paths:
            logger.warning(
                "Loading %d aliased target(s) once, from the last path that "
                "shares each tensor, reading one file at a time: %s",
                len(aliased_paths),
                ", ".join(
                    f"{get_fqn_from_nested_path(dropped)} -> "
                    f"{get_fqn_from_nested_path(kept)}"
                    for dropped, kept in aliased_paths
                ),
            )

        # Group load plans by source rank
        plans_by_rank: dict[int, list[tuple[NestedPath, LoadPlan]]] = {}
        for nested_path, load_plans in nested_path_to_load_plans.items():
            for lp in load_plans:
                if lp.src_rank not in plans_by_rank:
                    plans_by_rank[lp.src_rank] = []
                plans_by_rank[lp.src_rank].append((nested_path, lp))

        source_layouts_by_rank = {
            src_rank: source_metadata.get_layout_info(src_rank, item_key)
            for src_rank in plans_by_rank
        }
        # A Hugging Face export's metadata already parsed every shard header.
        shard_headers = (
            source_metadata.shard_headers
            if isinstance(source_metadata, HuggingFaceSafetensorsItemMetadata)
            else {}
        )
        self._execute_load_plans_with_read_strategy(
            self._read_strategy,
            source_path,
            source_layouts_by_rank,
            item_key,
            plans_by_rank,
            target_tensors,
            storage,
            file_read_workers,
            shard_headers,
        )

    def _execute_load_plans_with_read_strategy(
        self,
        read_strategy: ReshardingReadStrategy,
        source_path: Path,
        source_layouts_by_rank: dict[int, LayoutInfo],
        item_key: str,
        plans_by_rank: dict[int, list[tuple[NestedPath, LoadPlan]]],
        target_tensors: dict[NestedPath, torch.Tensor],
        storage: Storage,
        file_read_workers: int,
        shard_headers: dict[int, SafetensorsFileMetadata],
    ) -> None:
        # Worker threads start on the default stream. Copy on the caller's
        # current stream so the writes are ordered after its pending work on the
        # targets. Assumes accelerator targets are on the current device.
        on_accelerator = any(
            target_tensors[path].device.type != "cpu"
            for rank_plans in plans_by_rank.values()
            for path, _ in rank_plans
        )
        caller_stream = torch.accelerator.current_stream() if on_accelerator else None

        def copy(staged: Iterable[tuple[NestedPath, LoadPlan, torch.Tensor]]) -> None:
            if caller_stream is not None:
                torch.accelerator.set_stream(caller_stream)
            for nested_path, load_plan, src_data in staged:
                target_tensor = target_tensors[nested_path]
                tgt_slice = tuple(
                    slice(o, o + s) for o, s in zip(load_plan.offsets, load_plan.sizes)
                )
                target_tensor[tgt_slice].copy_(src_data)

        def read_with_offsets(src_rank: int) -> bool:
            """Return False if the file must be read in full instead."""
            layout_info = source_layouts_by_rank[src_rank]
            file_path = source_path / layout_info.file_path
            try:
                # Close the generator even if copy() raises, so its stream is
                # closed then rather than whenever the generator is collected.
                with closing(
                    self._read_source_slices_with_offset_reads(
                        file_path,
                        layout_info,
                        src_rank,
                        item_key,
                        plans_by_rank[src_rank],
                        storage,
                        shard_headers.get(src_rank),
                    )
                ) as staged:
                    copy(staged)
                return True
            except NotImplementedError as error:
                if read_strategy is ReshardingReadStrategy.OFFSET:
                    raise
                logger.warning(
                    "Offset reads unavailable for %s; reading it in full: %s",
                    file_path,
                    error,
                )
                return False
            except Exception:
                # Only the first failure is raised to the caller; log every one.
                logger.exception("Reading %s failed", file_path)
                raise

        full_file_ranks = list(plans_by_rank)
        if read_strategy is not ReshardingReadStrategy.FULL_FILE:
            with ThreadPoolExecutor(
                max_workers=file_read_workers,
                thread_name_prefix="ckpt-read",
            ) as pool:
                try:
                    read = list(pool.map(read_with_offsets, plans_by_rank))
                except BaseException:
                    # The load has failed, so skip the files not yet started.
                    pool.shutdown(cancel_futures=True)
                    raise
            full_file_ranks = [
                src_rank for src_rank, done in zip(plans_by_rank, read) if not done
            ]

        # A full-file read holds the whole file in memory, so read these one at a
        # time to bound memory regardless of file_read_workers.
        for src_rank in full_file_ranks:
            layout_info = source_layouts_by_rank[src_rank]
            copy(
                self._read_source_slices_with_full_file_read(
                    source_path / layout_info.file_path,
                    layout_info,
                    item_key,
                    plans_by_rank[src_rank],
                    storage,
                )
            )

    def _read_source_slices_with_offset_reads(
        self,
        file_path: Path,
        layout_info: LayoutInfo,
        source_rank: int,
        item_key: str,
        rank_plans: list[tuple[NestedPath, LoadPlan]],
        storage: Storage,
        safetensors_header: SafetensorsFileMetadata | None = None,
    ) -> Generator[tuple[NestedPath, LoadPlan, torch.Tensor], None, None]:
        """Yield the source data each plan needs, one slice at a time.

        Each slice's buffer is released after the caller copies it, rather
        than every slice in the file being held until the last one is read.
        """
        # O_DIRECT bypasses the page cache. On FUSE mounts, page-cache reads
        # share a small per-mount limit on in-flight requests, which caps how
        # fast concurrent readers on one host can go.
        with storage.stream_read(
            file_path,
            ReadArgs(pre_read_full_file=False, direct_io=True),
        ) as stream:
            serialization_format = layout_info.serialization_format
            source_fqns = {load_plan.src_fqn for _, load_plan in rank_plans}
            if isinstance(serialization_format, TorchSerialization):
                _validate_offset_read_archive(stream)
                metadata = torch.load(
                    stream,  # type: ignore[arg-type]
                    map_location="meta",
                    weights_only=False,
                )
                flattened = _flatten_state_dict(item_key, metadata)
                source_tensors = {
                    source_fqn: _unwrap_dtensor(flattened[source_fqn])
                    for source_fqn in source_fqns
                }
            elif isinstance(serialization_format, SafetensorsSerialization):
                if safetensors_header is None:
                    safetensors_header = SafetensorsFileMetadata.from_stream(
                        stream,
                        layout_info.file_path,
                        source_rank,
                    )
                source_tensors = safetensors_header.as_meta_tensors(source_fqns)
            else:
                raise ValueError(
                    "Unsupported serialization format "
                    f"{type(serialization_format).__name__}"
                )
            for path, plan in rank_plans:
                yield (
                    path,
                    plan,
                    _read_source_tensor_slice(
                        stream,
                        source_tensors[plan.src_fqn],
                        plan,
                    ),
                )

    def _read_source_slices_with_full_file_read(
        self,
        file_path: Path,
        layout_info: LayoutInfo,
        item_key: str,
        rank_plans: list[tuple[NestedPath, LoadPlan]],
        storage: Storage,
    ) -> list[tuple[NestedPath, LoadPlan, torch.Tensor]]:
        """Read the source data every plan needs by loading the full file."""
        serialization_format = layout_info.serialization_format
        if isinstance(serialization_format, TorchSerialization):
            loaded_data = load_torch_serialized_from_storage(
                file_path,
                storage,
                map_location="cpu",
            )
        elif isinstance(serialization_format, SafetensorsSerialization):
            from safetensors.torch import load as safetensors_load

            loaded_data = safetensors_load(
                storage.read(file_path, ReadArgs(pre_read_full_file=False))
            )
        else:
            raise ValueError(
                "Unsupported serialization format "
                f"{type(serialization_format).__name__}"
            )

        flattened = _flatten_state_dict(item_key, loaded_data)
        return [
            (
                path,
                plan,
                _slice_source_tensor(_unwrap_dtensor(flattened[plan.src_fqn]), plan),
            )
            for path, plan in rank_plans
        ]
