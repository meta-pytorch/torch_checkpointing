# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mypy: allow-untyped-defs
import logging
import types
import warnings
import weakref
from collections import Counter, defaultdict
from copyreg import dispatch_table
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torch.storage import UntypedStorage

from . import _pin_memory_utils as pin_memory_utils

logger = logging.getLogger(__name__)


class StorageManager:
    """
    Persistent holder for CPU UntypedStorage objects.

    The only distinguishing feature of an UntypedStorage is its size in bytes; storages
    of the same size are interchangeable. Thus, when we need a storage of a given size,
    we first check if we aready have a storage of that size available. If so, we can
    reuse it. If not, we create a new storage and add it to the pool.

    Attributes:
        pin_memory (bool): Whether to pin CPU memory for faster CPU-GPU transfers
        share_memory (bool): Whether to share memory across processes
        pin_memory_min_bytes (int): Minimum tensor size in bytes to pin memory
    """

    def __init__(self, pin_memory: bool, share_memory: bool, pin_memory_min_bytes: int):
        self._pin_memory = pin_memory
        self._share_memory = share_memory
        self._pin_memory_min_bytes = pin_memory_min_bytes
        # Size in bytes -> list of storages currently allocated
        self._size_to_storages: defaultdict[int, list[UntypedStorage]] = defaultdict(
            list
        )
        # Size in bytes -> index of first unused storage
        self._size_to_storages_next_index: Counter[int] = Counter()

    def _new_storage(self, size: int) -> UntypedStorage:
        """Create a new UntypedStorage of the given size."""
        if self._share_memory:
            # type: ignore
            storage: UntypedStorage = UntypedStorage._new_shared(size, device="cpu")
        else:
            storage = UntypedStorage(size, device="cpu")

        # Skip pinning for tensors below the minimum size threshold
        # Small tensors (e.g., optimizer step counters, scalars) have negligible
        # transfer time improvement from pinning, but pinning overhead is significant
        if self._pin_memory and storage.nbytes() >= self._pin_memory_min_bytes:
            pin_memory_utils.pin_memory(storage.data_ptr(), storage.nbytes())

        return storage

    def get(self, size: int) -> UntypedStorage:
        """
        Get a storage of the given size from the pool, or create a new one if necessary.

        Args:
            size (int): The size of the storage to get.

        Returns:
            UntypedStorage: A storage of the given size.
        """
        storages = self._size_to_storages[size]
        index = self._size_to_storages_next_index[size]
        if index < len(storages):
            # Reuse existing storage
            storage = storages[index]
        elif index == len(storages):
            # Create new storage
            storage = self._new_storage(size)
            storages.append(storage)
        else:
            # Should not get here
            raise RuntimeError(f"Bug in StorageManager: {index=} > {len(storages)=}")

        self._size_to_storages_next_index[size] += 1
        return storage

    def reset(self) -> None:
        """Mark all storages as unused, allowing them to be reused."""
        self._size_to_storages_next_index.clear()

    def delete_unused(self) -> None:
        """Unpin and delete all storages that are not in use."""
        for size, storages in self._size_to_storages.items():
            offset = self._size_to_storages_next_index[size]
            while len(storages) > offset:
                storage = storages.pop()
                if storage.is_pinned():
                    # NOTE: Previously we unpinned Storage as a weakref.Finalizer, but
                    # this can cause race conditions. So instead we explicitly unpin
                    # before deleting the storage.
                    pin_memory_utils.unpin_memory(storage.data_ptr())
                del storage

    def total_num_bytes(self) -> int:
        """Return the total bytes held by all storages in the pool."""
        # atomic shallow copy in CPython to avoid RuntimeError: dictionary changed size
        # during iteration
        size_to_storages = self._size_to_storages.copy()
        return sum(size * len(storages) for size, storages in size_to_storages.items())

    def pinned_num_bytes(self) -> int:
        """Return the total bytes held by pinned storages in the pool."""
        if not self._pin_memory:
            return 0

        # atomic shallow copy in CPython to avoid RuntimeError: dictionary changed size
        # during iteration
        size_to_storages = self._size_to_storages.copy()
        return sum(
            size * len(storages)
            for size, storages in size_to_storages.items()
            if size >= self._pin_memory_min_bytes
        )

    def close(self) -> None:
        """Unpin and delete all storages in the pool."""
        self.reset()
        self.delete_unused()

    def __del__(self) -> None:
        self.close()


class StateDictStager:
    """
    A class for optimizing storage objects during staging for async checkpointing.

    StateDictStager stages the state_dict to CPU DRAM while applying optimizations
    like memory sharing and pinning to improve performance. It caches storage objects
    to avoid redundant copies and can be configured to automatically share memory
    (for multi-process usage) and pin memory (for faster CPU-GPU transfers).

    Attributes:
        pin_memory (bool): Whether to pin CPU memory for faster CPU-GPU transfers
        share_memory (bool): Whether to share memory across processes
        pin_memory_min_bytes (int): Minimum tensor size in bytes to pin memory (default: 5)
        use_non_blocking_copy (bool): Schedule d2h copies without blocking CPU
    """

    def __init__(
        self,
        pin_memory: bool = False,
        share_memory: bool = False,
        pin_memory_min_bytes: int = 5,
        use_non_blocking_copy: bool = False,
    ):
        if pin_memory and not torch.cuda.is_available():
            warnings.warn(
                "Ignoring pin_memory flag for checkpoint staging as pinning memory"
                "requires CUDA, but CUDA is not available. ",
                stacklevel=2,
            )
            pin_memory = False
        self._storage_manager = StorageManager(
            pin_memory=pin_memory,
            share_memory=share_memory,
            pin_memory_min_bytes=pin_memory_min_bytes,
        )
        self._use_non_blocking_copy = use_non_blocking_copy

        def _deepcopy_atomic(x, _):
            return x

        def _deepcopy_list(x, memo):
            y: list = []
            memo[id(x)] = y
            append = y.append
            for a in x:
                append(self.deepcopy_with_tensor_offload(a, memo))
            return y

        def _deepcopy_tuple(x, memo):
            y = [self.deepcopy_with_tensor_offload(a, memo) for idx, a in enumerate(x)]
            # We're not going to put the tuple in the memo, but it's still important we
            # check for it, in case the tuple contains recursive mutable structures.
            try:
                return memo[id(x)]
            except KeyError:
                pass

            # Check if any elements changed during deepcopy
            for k, j in zip(x, y):
                if k is not j:
                    # At least one element changed, create new tuple
                    return tuple(y)

            # No elements changed, return original tuple
            return x

        def _deepcopy_dict(x, memo):
            y: dict = {}
            memo[id(x)] = y
            for key, value in x.items():
                y[self.deepcopy_with_tensor_offload(key, memo)] = (
                    self.deepcopy_with_tensor_offload(value, memo)
                )
            return y

        def _deepcopy_method(x, memo):
            return type(x)(
                x.__func__,
                self.deepcopy_with_tensor_offload(x.__self__, memo),
            )

        d: dict[Any, Any] = {}
        self._deepcopy_dispatch = d
        d[type(None)] = _deepcopy_atomic
        d[int] = _deepcopy_atomic
        d[float] = _deepcopy_atomic
        d[bool] = _deepcopy_atomic
        d[complex] = _deepcopy_atomic
        d[bytes] = _deepcopy_atomic
        d[str] = _deepcopy_atomic
        d[types.CodeType] = _deepcopy_atomic
        d[type] = _deepcopy_atomic
        d[range] = _deepcopy_atomic
        d[types.BuiltinFunctionType] = _deepcopy_atomic
        d[types.FunctionType] = _deepcopy_atomic
        d[weakref.ref] = _deepcopy_atomic
        d[property] = _deepcopy_atomic
        d[types.MethodType] = _deepcopy_method
        d[dict] = _deepcopy_dict
        d[tuple] = _deepcopy_tuple
        d[list] = _deepcopy_list

    def _stage_untyped_storage(self, storage: UntypedStorage) -> UntypedStorage:
        """
        Called from the hooked storage_deepcopy function in torch.Tensor.__deepcopy__.

        Copies the storage to a CPU storage, reusing previously allocated storage if
        possible.

        Args:
            storage: The storage to optimize

        Returns:
            The optimized storage
        """
        cpu_storage = self._storage_manager.get(storage.nbytes())
        cpu_storage.copy_(
            storage,
            non_blocking=cpu_storage.is_pinned() and self._use_non_blocking_copy,
        )
        return cpu_storage

    @torch.no_grad()
    def stage(self, state_dict: Any) -> Any:
        self._storage_manager.reset()
        staged = self.deepcopy_with_tensor_offload(state_dict, None, [])
        self._storage_manager.delete_unused()
        return staged

    def _offload_tensor(self, x, memo):
        """
        Deep copy a PyTorch tensor with optimized storage handling.

        This method creates a CPU copy of a tensor while applying memory optimizations
        like sharing and pinning based on the StateDictStager configuration.

        Args:
            x: The tensor to copy
            memo: Memo dictionary for tracking already copied objects
            fqn: Fully qualified name tuple for the tensor

        Returns:
            A CPU copy of the tensor with optimized storage
        """
        if isinstance(x, DTensor) and any(
            placement.is_partial() for placement in x.placements
        ):
            raise ValueError(
                "Checkpointing DTensor with partial placements is not supported."
            )

        # if data_ptr is not 0, we allocate a new storage below. so we can skip
        # memory allocation by using [] for size.
        y = x.new_empty([] if x.data_ptr() != 0 else x.size(), device="cpu")

        # Store in memo dict early to handle recursive references
        d = id(x)
        memo[d] = y

        if type(x) is torch.Tensor or x.data_ptr() != 0:
            # Get the untyped storage
            untyped_storage = x.untyped_storage()
            storage_id = id(untyped_storage)

            # Check if this storage has already been staged in this deepcopy operation
            # This handles the case where different tensors share the same storage
            # (e.g., FSDP state_dict where norm.weight and norm_weight reference same storage)
            # PyTorch caches untyped_storage() calls, so same storage -> same id
            if storage_id in memo:
                copied_storage = memo[storage_id]
            else:
                # Storage not seen before in this operation, stage it
                copied_storage = self._stage_untyped_storage(untyped_storage)
                # Add to memo to avoid re-staging if we see this storage again
                memo[storage_id] = copied_storage

            # Set the tensor data using the staged storage
            y.set_(copied_storage, x.storage_offset(), x.size(), x.stride())

        # Copy any attributes the tensor might have
        if hasattr(x, "__dict__"):
            for attr_name, attr_value in x.__dict__.items():
                setattr(
                    y,
                    attr_name,
                    self.deepcopy_with_tensor_offload(attr_value, memo),
                )

        if hasattr(x, "__slots__"):
            for slot in x.__slots__:
                if hasattr(x, slot):
                    setattr(
                        y,
                        slot,
                        self.deepcopy_with_tensor_offload(getattr(x, slot), memo),
                    )

        return y

    def close(self):
        """
        Clean up all cached storages and release associated resources.

        This method clears the internal storage cache, allowing garbage collection
        of cached CPU storages. Any pinned memory associated with cached storages
        will be automatically unpinned through weak reference finalizers.
        """
        self._storage_manager.close()

    @torch.no_grad()
    def deepcopy_with_tensor_offload(
        self,
        x,
        memo=None,
        _nil=[],  # noqa: B006
    ):
        """Deep copy operation on arbitrary Python objects with special handling for PyTorch tensors.

        This implementation extends the standard deepcopy functionality to handle PyTorch tensors
        and their storages in a way that optimizes memory usage and performance, similar to the
        stage method. It applies memory sharing and pinning optimizations based on the StateDictStager
        configuration.

        Args:
            x: The object to deep copy
            memo: Memo dictionary for tracking already copied objects
            _nil: Sentinel value for memo dictionary
            fqn: Fully qualified name tuple for tracking nested keys

        Returns:
            A deep copy of the input object with optimized tensor storage handling
        """
        if memo is None:
            memo = {}

        d = id(x)
        y = memo.get(d, _nil)
        if y is not _nil:
            return y

        cls = type(x)

        # tensors and subclasses of tensors are handled separately
        if isinstance(x, torch.Tensor):
            y = self._offload_tensor(x, memo)
        else:
            # Use the dispatch table for standard types
            copier = self._deepcopy_dispatch.get(cls)
            if copier is not None:
                # Check if this is an atomic copier (only accepts x and memo)
                if copier.__name__ == "_deepcopy_atomic":
                    y = copier(x, memo)
                else:
                    y = copier(x, memo)
            else:
                if issubclass(cls, type):
                    # type copier is also atomic
                    y = self._deepcopy_dispatch[type](x, memo)
                else:
                    copier = getattr(x, "__deepcopy__", None)
                    if copier is not None:
                        y = copier(memo)
                    else:
                        reductor = dispatch_table.get(cls)
                        if reductor:
                            rv = reductor(x)
                        else:
                            reductor = getattr(x, "__reduce_ex__", None)
                            if reductor is not None:
                                rv = reductor(4)
                            else:
                                reductor = getattr(x, "__reduce__", None)
                                if reductor:
                                    rv = reductor()
                                else:
                                    raise RuntimeError(
                                        f"un(deep)copyable object of type {cls}"
                                    )
                        if isinstance(rv, str):
                            y = x
                        else:
                            y = self._reconstruct(x, memo, *rv)

        # If is its own copy, don't memoize.
        if y is not x:
            memo[d] = y
            self._keep_alive(x, memo)  # Make sure x lives at least as long as d
        return y

    def _keep_alive(self, x, memo):
        """Keeps a reference to the object x in the memo.

        Because we remember objects by their id, we have
        to assure that possibly temporary objects are kept
        alive by referencing them.
        We store a reference at the id of the memo, which should
        normally not be used unless someone tries to deepcopy
        the memo itself...
        """
        try:
            memo[id(memo)].append(x)
        except KeyError:
            # aha, this is the first one :-)
            memo[id(memo)] = [x]

    def _reconstruct(
        self,
        x,
        memo,
        func,
        args,
        state=None,
        listiter=None,
        dictiter=None,
    ):
        deep = memo is not None
        if deep and args:
            args = tuple(self.deepcopy_with_tensor_offload(arg, memo) for arg in args)
        y = func(*args)
        if deep:
            memo[id(x)] = y

        if state is not None:
            if deep:
                state = self.deepcopy_with_tensor_offload(state, memo)
            if hasattr(y, "__setstate__"):
                y.__setstate__(state)
            else:
                if isinstance(state, tuple) and len(state) == 2:
                    state, slotstate = state
                else:
                    slotstate = None
                if state is not None:
                    y.__dict__.update(state)
                if slotstate is not None:
                    for key, value in slotstate.items():
                        setattr(y, key, value)

        if listiter is not None:
            if deep:
                for item in listiter:
                    item = self.deepcopy_with_tensor_offload(item, memo)
                    y.append(item)
            else:
                for item in listiter:
                    y.append(item)
        if dictiter is not None:
            if deep:
                for key, value in dictiter:
                    key = self.deepcopy_with_tensor_offload(key, memo)
                    value = self.deepcopy_with_tensor_offload(value, memo)
                    y[key] = value
            else:
                for key, value in dictiter:
                    y[key] = value
        return y
