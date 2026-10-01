# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor receive buffers and pointer-free cooperative scatter operations."""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

from .layout import _resolve_torch_dtype, TensorReadTarget


@dataclass(slots=True)
class BufferSlot:
    """One reusable CPU byte tensor and its writable buffer-protocol view."""

    tensor: torch.Tensor
    _ctypes_array: Any
    view: memoryview
    pending_cuda_events: list[Any]

    @classmethod
    def allocate(cls, size_bytes: int, *, pinned: bool) -> "BufferSlot":
        if size_bytes <= 0:
            raise ValueError("buffer size must be positive")
        tensor = torch.empty(size_bytes, dtype=torch.uint8, pin_memory=pinned)
        ctypes_array = (ctypes.c_ubyte * size_bytes).from_address(tensor.data_ptr())
        return cls(
            tensor=tensor,
            _ctypes_array=ctypes_array,
            view=memoryview(ctypes_array).cast("B"),
            pending_cuda_events=[],
        )


class PinnedBufferPool:
    """Bounded reusable receive buffers with explicit lifetime management."""

    def __init__(
        self,
        *,
        slot_bytes: int,
        slot_count: int,
        use_pinned_memory: bool | None = None,
    ) -> None:
        if slot_bytes <= 0 or slot_count <= 0:
            raise ValueError("slot_bytes and slot_count must be positive")
        self._slot_bytes = slot_bytes
        self._use_pinned_memory = (
            torch.cuda.is_available()
            if use_pinned_memory is None
            else use_pinned_memory
        )
        self._slots: list[BufferSlot | None] = [None] * slot_count
        self._free = list(range(slot_count - 1, -1, -1))
        self._leased: set[int] = set()
        self._pending: set[int] = set()
        self._condition = threading.Condition()
        self._closed = False

    @property
    def slot_bytes(self) -> int:
        return self._slot_bytes

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    @contextmanager
    def acquire(
        self,
        required_bytes: int,
        *,
        timeout: float | None = None,
    ) -> Iterator[BufferSlot]:
        """Acquire one slot and return it after the caller finishes copying."""

        if required_bytes < 0:
            raise ValueError("required_bytes must be non-negative")
        if required_bytes > self._slot_bytes:
            raise ValueError(
                f"request needs {required_bytes} bytes but slot size is "
                f"{self._slot_bytes}"
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            if self._closed:
                raise RuntimeError("buffer pool is closed")
            while not self._free:
                self._reap_completed_locked()
                if self._free:
                    break
                if self._closed:
                    raise RuntimeError("buffer pool is closed")
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError("timed out waiting for a receive buffer")
                self._condition.wait(
                    0.01 if remaining is None else min(remaining, 0.01)
                )
            if self._closed:
                raise RuntimeError("buffer pool is closed")
            slot_index, slot = self._lease_free_slot_locked()
        try:
            yield slot
        finally:
            with self._condition:
                self._leased.remove(slot_index)
                if slot.pending_cuda_events:
                    self._pending.add(slot_index)
                else:
                    self._free.append(slot_index)
                self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                while self._slots:
                    self._condition.wait()
                return
            self._closed = True
            self._condition.notify_all()
            while self._leased:
                self._condition.wait()
            pending_slots = [
                self._slots[index]
                for index in self._pending
                if self._slots[index] is not None
            ]
        try:
            for slot in pending_slots:
                assert slot is not None
                for event in slot.pending_cuda_events:
                    event.synchronize()
                slot.pending_cuda_events.clear()
        finally:
            with self._condition:
                self._pending.clear()
                self._free.clear()
                self._slots.clear()
                self._condition.notify_all()

    def _reap_completed_locked(self) -> None:
        completed: list[int] = []
        for slot_index in self._pending:
            slot = self._slots[slot_index]
            if slot is not None and all(
                event.query() for event in slot.pending_cuda_events
            ):
                slot.pending_cuda_events.clear()
                completed.append(slot_index)
        for slot_index in completed:
            self._pending.remove(slot_index)
            self._free.append(slot_index)

    def _lease_free_slot_locked(self) -> tuple[int, BufferSlot]:
        slot_index = self._free.pop()
        slot = self._slots[slot_index]
        if slot is None:
            try:
                slot = BufferSlot.allocate(
                    self._slot_bytes,
                    pinned=self._use_pinned_memory,
                )
            except BaseException:
                self._free.append(slot_index)
                self._condition.notify_all()
                raise
            self._slots[slot_index] = slot
        self._leased.add(slot_index)
        return slot_index, slot


def scatter_dense_buffer(
    target: TensorReadTarget,
    target_state_dict: Mapping[str, Any],
    source_buffer: object,
    *,
    non_blocking: bool = False,
) -> Any | None:
    """Copy one densely packed logical source slice into its target slice."""

    destination = target_state_dict.get(target.target_fqn)
    if not isinstance(destination, torch.Tensor):
        raise TypeError(f"target {target.target_fqn!r} is not a torch.Tensor")
    if target.numel == 0:
        return None
    source_dtype = _resolve_torch_dtype(target.source_dtype)
    source = torch.frombuffer(
        source_buffer,
        dtype=source_dtype,
        count=target.numel,
    ).reshape(target.source_slice_shape)
    if target.transpose_dims:
        source = source.permute(target.transpose_dims)
    if tuple(source.shape) != target.target_slice_shape:
        source = source.reshape(target.target_slice_shape)

    destination_storage_offset = (
        target.destination_pattern.start_offset // target.target_element_size_bytes
    )
    destination_view = torch.as_strided(
        destination,
        size=target.target_slice_shape,
        stride=destination.stride(),
        storage_offset=destination_storage_offset,
    )
    destination_view.copy_(source, non_blocking=non_blocking)
    if non_blocking and destination.is_cuda:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(destination.device))
        return event
    return None


def scatter_buffer_slice(
    target: TensorReadTarget,
    target_state_dict: Mapping[str, Any],
    slot: BufferSlot,
    *,
    offset_bytes: int = 0,
    non_blocking: bool = False,
) -> Any | None:
    """Scatter one target from a byte range inside a receive slot."""

    end = offset_bytes + target.source_pattern.dense_nbytes
    if offset_bytes < 0 or end > len(slot.view):
        raise ValueError("target source bytes fall outside the receive slot")
    return scatter_dense_buffer(
        target,
        target_state_dict,
        slot.view[offset_bytes:end],
        non_blocking=non_blocking,
    )


def scatter_flat_buffer_chunk(
    target: TensorReadTarget,
    target_state_dict: Mapping[str, Any],
    source_buffer: object,
    *,
    destination_offset_bytes: int,
    numel: int,
    non_blocking: bool = False,
) -> Any | None:
    """Copy one logical non-transposed chunk into contiguous target storage."""

    if target.transpose_dims:
        raise ValueError("flat chunk scatter does not support transposed targets")
    destination = target_state_dict.get(target.target_fqn)
    if not isinstance(destination, torch.Tensor):
        raise TypeError(f"target {target.target_fqn!r} is not a torch.Tensor")
    if destination_offset_bytes % target.target_element_size_bytes:
        raise ValueError("destination chunk is not element-aligned")
    source = torch.frombuffer(
        source_buffer,
        dtype=_resolve_torch_dtype(target.source_dtype),
        count=numel,
    )
    destination_view = torch.as_strided(
        destination,
        size=(numel,),
        stride=(1,),
        storage_offset=destination_offset_bytes // target.target_element_size_bytes,
    )
    destination_view.copy_(source, non_blocking=non_blocking)
    if non_blocking and destination.is_cuda:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(destination.device))
        return event
    return None


@contextmanager
def direct_cpu_destination_buffer(
    target: TensorReadTarget,
    target_state_dict: Mapping[str, Any],
) -> Iterator[memoryview]:
    """Expose a writable view for a contiguous same-dtype CPU destination."""

    destination = target_state_dict.get(target.target_fqn)
    if not isinstance(destination, torch.Tensor):
        raise TypeError(f"target {target.target_fqn!r} is not a torch.Tensor")
    if destination.device.type != "cpu":
        raise ValueError("direct destination receive requires a CPU tensor")
    if target.requires_transform or target.destination_pattern.range_count != 1:
        raise ValueError("target is not eligible for direct destination receive")
    _validate_direct_cpu_destination(target, destination)
    length = target.destination_pattern.dense_nbytes
    if destination.is_conj() or destination.is_neg():
        staging = bytearray(length)
        view = memoryview(staging)
        try:
            yield view
        except BaseException:
            raise
        else:
            scatter_dense_buffer(target, {target.target_fqn: destination}, view)
        finally:
            view.release()
        return
    address = (
        destination.untyped_storage().data_ptr()
        + target.destination_pattern.start_offset
    )
    ctypes_array = (ctypes.c_ubyte * length).from_address(address)
    view = memoryview(ctypes_array).cast("B")
    try:
        yield view
    finally:
        view.release()
        torch.autograd.graph.increment_version(destination)


def can_receive_directly_to_cpu(target: TensorReadTarget) -> bool:
    return (
        target.target_device == "cpu"
        and not target.requires_transform
        and target.destination_pattern.range_count == 1
        and target.destination_pattern.dense_nbytes
        == target.source_pattern.dense_nbytes
    )


def _validate_direct_cpu_destination(
    target: TensorReadTarget,
    destination: torch.Tensor,
) -> None:
    expected_dtype = _resolve_torch_dtype(target.target_dtype)
    if destination.dtype != expected_dtype:
        raise ValueError(
            f"target {target.target_fqn!r} dtype changed from "
            f"{target.target_dtype} to {destination.dtype}"
        )
    if str(destination.device) != target.target_device:
        raise ValueError(
            f"target {target.target_fqn!r} device changed from "
            f"{target.target_device} to {destination.device}"
        )
    live_shape = tuple(int(size) for size in destination.shape)
    if live_shape != target.target_tensor_shape:
        raise ValueError(
            f"target {target.target_fqn!r} shape changed from "
            f"{target.target_tensor_shape} to {live_shape}"
        )
    if destination.element_size() != target.target_element_size_bytes:
        raise ValueError(
            f"target {target.target_fqn!r} element size changed from "
            f"{target.target_element_size_bytes} to {destination.element_size()}"
        )
    live_strides = tuple(int(stride) for stride in destination.stride())
    if not _is_dense_slice(target.target_slice_shape, live_strides):
        raise ValueError(
            f"target {target.target_fqn!r} stride {live_strides} is incompatible "
            "with direct destination receive"
        )

    start = target.destination_pattern.start_offset
    length = target.destination_pattern.dense_nbytes
    element_size = target.target_element_size_bytes
    if start % element_size or length % element_size:
        raise ValueError(
            f"target {target.target_fqn!r} destination bytes are not element-aligned"
        )
    storage_nbytes = int(destination.untyped_storage().nbytes())
    if start + length > storage_nbytes:
        raise ValueError(
            f"target {target.target_fqn!r} destination bytes [{start}, "
            f"{start + length}) exceed its {storage_nbytes}-byte storage"
        )

    tensor_start = int(destination.storage_offset()) * element_size
    tensor_span_elements = 0
    if destination.numel():
        tensor_span_elements = 1 + sum(
            (size - 1) * stride for size, stride in zip(live_shape, live_strides)
        )
    tensor_end = tensor_start + tensor_span_elements * element_size
    if start < tensor_start or start + length > tensor_end:
        raise ValueError(
            f"target {target.target_fqn!r} destination bytes [{start}, "
            f"{start + length}) fall outside its live tensor view "
            f"[{tensor_start}, {tensor_end})"
        )


def _is_dense_slice(shape: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    if len(shape) != len(strides):
        return False
    expected_stride = 1
    for size, stride in reversed(tuple(zip(shape, strides))):
        if size <= 1:
            continue
        if stride != expected_stride:
            return False
        expected_stride *= size
    return True
