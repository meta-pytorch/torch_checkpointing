"""
Utility functions for the experimental checkpoint module.

This module contains helper functions and utilities used across the experimental
checkpoint functionality.
"""

import enum
import logging
from collections import deque, OrderedDict
from collections.abc import Mapping, MutableMapping, Sequence, Set
from concurrent.futures import Future
from dataclasses import asdict, fields, is_dataclass, MISSING
from itertools import zip_longest
from typing import Any, Callable, get_args, get_origin, List, TypeVar, Union

import numpy as np
import torch
from torch.distributed.tensor import DTensor

from .types import CheckpointPath, STATE_DICT

logger = logging.getLogger(__name__)


_T = TypeVar("_T")
_S = TypeVar("_S")


def ensure_future(fut_or_val: Future[_T] | _T) -> Future[_T]:
    if isinstance(fut_or_val, Future):
        return fut_or_val
    else:
        completed_fut = Future()
        completed_fut.set_result(fut_or_val)
        return completed_fut


def fut_then(
    fut: Future[_T],
    then_func: Callable[[_T], _S],
    err_func: Callable[[BaseException], _S] | None = None,
) -> Future[_S]:
    """
    Returns a Future that completes with the result of `then_func` when `fut`
    completes successfully, or the result of `err_func` when `fut` completes with an
    error. Analogous to JavaScript's `Promise.then`.

    If `fut` is already complete, `then_func` or `err_func` will be called immediately.
    If `fut` is cancelled, neither function will run and the returned Future will also
    be cancelled.

    Args:
        fut: The Future to wait on.
        then_func: The function to apply to the result of `fut`. It takes fut.result()
            as its only argument.
        err_func: The function to apply to the exception of `fut`. It takes
            fut.exception() as its only argument. If None, the original exception will
            be propagated.

    Returns:
        A Future representing the result of then_func(fut) or err_func(fut).
    """
    new_fut = Future()

    def on_complete(f: Future[_T]) -> None:
        if f.cancelled():
            new_fut.cancel()
            return

        exc = f.exception()
        if exc is not None:
            if err_func is not None:
                try:
                    new_fut.set_result(err_func(exc))
                except Exception as e:
                    new_fut.set_exception(e)
            else:
                new_fut.set_exception(exc)
        else:
            try:
                new_fut.set_result(then_func(f.result()))
            except Exception as e:
                new_fut.set_exception(e)

    fut.add_done_callback(on_complete)
    return new_fut


def wrap_future(original_result: Any) -> Future[None]:
    """
    Wraps a result (Future or not) to return a Future with None result.

    If the input is a Future, returns a new Future that completes with None when
    the original Future completes successfully, or propagates any exception.
    If the input is not a Future, returns a completed Future with None result.

    Args:
        original_result: The result to wrap (Future or any other value).

    Returns:
        A Future that completes with None on success or propagates exceptions.
    """
    return fut_then(ensure_future(original_result), lambda _: None)


def from_dict(cls, data):
    if not is_dataclass(cls):
        return data  # base case for primitive types

    kwargs = {}
    for f in fields(cls):
        f_name = f.name
        f_type = f.type

        if f_name not in data:
            if f.default is not MISSING:
                kwargs[f_name] = f.default
                continue
            elif f.default_factory is not MISSING:  # for fields with default_factory
                kwargs[f_name] = f.default_factory()
                continue
            elif get_origin(f_type) is Union and type(None) in get_args(f_type):
                kwargs[f_name] = None  # optional field missing from data
                continue
            else:
                raise ValueError(f"Missing required field: {f_name}")

        value = data[f_name]
        origin = get_origin(f_type)

        # handle optional fields
        if origin is Union and type(None) in get_args(f_type):
            actual_type = [t for t in get_args(f_type) if t is not type(None)][0]
            if is_dataclass(actual_type):
                value = from_dict(actual_type, value)
        # handle lists of dataclasses
        elif origin in (list, List):
            elem_type = get_args(f_type)[0]
            value = [from_dict(elem_type, v) for v in value]
        # handle nested dataclass
        elif is_dataclass(f_type):
            value = from_dict(f_type, value)

        kwargs[f_name] = value

    return cls(**kwargs)


def set_thread_name_safe(name: str) -> None:
    torch._C._set_thread_name(name)


class CompareStateDictsResult(enum.Enum):
    """
    Enum for the result of comparing two state dicts.
    """

    # Present in one state dict but not the other.
    LEFT_ONLY = "LEFT_ONLY"
    RIGHT_ONLY = "RIGHT_ONLY"

    # Present in both but with different values.
    TYPE_NOT_EQUAL = "TYPE_NOT_EQUAL"

    # Comparisons for tensors, numpy arrays, etc
    SHAPE_NOT_EQUAL = "SHAPE_NOT_EQUAL"
    DTYPE_NOT_EQUAL = "DTYPE_NOT_EQUAL"
    PLACEMENT_NOT_EQUAL = "PLACEMENT_NOT_EQUAL"  # For DTensors
    ELEMENTS_NOT_EQUAL = "ELEMENTS_NOT_EQUAL"

    # Comparisons for other objects
    SEQ_LENGTH_NOT_EQUAL = "SEQ_LENGTH_NOT_EQUAL"
    OBJ_NOT_EQUAL = "OBJ_NOT_EQUAL"
    OBJ_NOT_COMPARABLE = "OBJ_NOT_COMPARABLE"  # __eq__ throws or is not a bool


def compare_state_dicts(
    left: STATE_DICT, right: STATE_DICT
) -> dict[tuple[str | int, ...], CompareStateDictsResult]:
    """
    Compares two state dicts recursively and returns a dict describing the differences.
    If the returned dict is empty, the two state dicts are equal.

    Args:
        left: The first state dict.
        right: The second state dict.

    Returns:
        A dict mapping keys to their differences. The keys are paths from the root of
        the state dict to the mismatched value; the values describe how they differ.

    Example:
        >>> compare_state_dicts({'a': {'b': 1}, 'c': 2}, {'a': {'b': 2}, 'd': 3})
        {
            ('a', 'b'): 'NOT_EQUAL',
            ('c'): 'LEFT_ONLY',
            ('d'): 'RIGHT_ONLY',
        }
    """
    differences: dict[tuple[str | int, ...], CompareStateDictsResult] = {}

    def _compare_recursive(
        left_val: Any,
        right_val: Any,
        path: tuple[str | int, ...],
    ) -> None:
        """Recursively compare two values and record differences."""
        if type(left_val) is not type(right_val):
            differences[path] = CompareStateDictsResult.TYPE_NOT_EQUAL
        elif isinstance(left_val, (torch.Tensor, np.ndarray)):
            _compare_arraylike(left_val, right_val, path)
        elif isinstance(left_val, dict):
            _compare_dicts(left_val, right_val, path)
        elif is_dataclass(left_val):
            _compare_dicts(asdict(left_val), asdict(right_val), path)
        elif isinstance(left_val, (list, tuple, deque)):
            _compare_sequences(left_val, right_val, path)
        else:
            _compare_objs(left_val, right_val, path)

    def _compare_arraylike(
        left_arr: torch.Tensor | np.ndarray,
        right_arr: torch.Tensor | np.ndarray,
        path: tuple[str | int, ...],
    ) -> None:
        """Compare two tensors, handling regular Tensor and DTensor."""
        if left_arr.shape != right_arr.shape:
            differences[path] = CompareStateDictsResult.SHAPE_NOT_EQUAL
        elif left_arr.dtype != right_arr.dtype:
            differences[path] = CompareStateDictsResult.DTYPE_NOT_EQUAL
        elif isinstance(left_arr, np.ndarray):
            if not np.array_equal(left_arr, right_arr):
                differences[path] = CompareStateDictsResult.ELEMENTS_NOT_EQUAL
        else:
            # Handle Tensor and DTensor
            if isinstance(left_arr, DTensor):
                assert isinstance(right_arr, DTensor)
                if left_arr.placements != right_arr.placements:
                    differences[path] = CompareStateDictsResult.PLACEMENT_NOT_EQUAL
                    return
                # Continue comparing the local shards
                left_arr = left_arr.to_local()
                right_arr = right_arr.to_local()
            # Tensors need to be on the same device to compare
            assert isinstance(right_arr, torch.Tensor)
            if not torch.equal(left_arr.to("cpu"), right_arr.to("cpu")):
                differences[path] = CompareStateDictsResult.ELEMENTS_NOT_EQUAL

    def _compare_dicts(
        left_dict: dict,
        right_dict: dict,
        path: tuple[str | int, ...],
    ) -> None:
        """Compare two dictionaries recursively."""
        left_keys = set(left_dict.keys())
        right_keys = set(right_dict.keys())

        # Check for keys only in left
        for key in left_keys - right_keys:
            differences[path + (key,)] = CompareStateDictsResult.LEFT_ONLY

        # Check for keys only in right
        for key in right_keys - left_keys:
            differences[path + (key,)] = CompareStateDictsResult.RIGHT_ONLY

        # Recursively compare common keys
        for key in left_keys & right_keys:
            _compare_recursive(left_dict[key], right_dict[key], path + (key,))

    def _compare_sequences(
        left_seq: list | tuple | deque,
        right_seq: list | tuple | deque,
        path: tuple[str | int, ...],
    ) -> None:
        """Compare two sequences (lists or tuples) recursively."""
        if len(left_seq) != len(right_seq):
            # Record length mismatch
            differences[path] = CompareStateDictsResult.SEQ_LENGTH_NOT_EQUAL
            return

        # Recursively compare elements
        for i, (left_elem, right_elem) in enumerate(zip(left_seq, right_seq)):
            _compare_recursive(left_elem, right_elem, path + (i,))

    def _compare_objs(
        left_obj: Any, right_obj: Any, path: tuple[str | int, ...]
    ) -> None:
        """Compare two non-container objects."""
        try:
            if left_obj != right_obj:
                differences[path] = CompareStateDictsResult.OBJ_NOT_EQUAL
        except Exception:
            differences[path] = CompareStateDictsResult.OBJ_NOT_COMPARABLE

    # Start comparison from the root
    _compare_recursive(left, right, ())

    return differences
