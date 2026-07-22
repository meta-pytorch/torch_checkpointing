# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pytest
import torch
from torch_checkpointing.utils import (
    compare_state_dicts,
    CompareStateDictsResult,
    ensure_future,
    from_dict,
    fut_then,
    wrap_future,
)


class TestEnsureFuture:
    """Tests for ensure_future function."""

    def test_ensure_future_with_future_returns_same_future(self):
        """Test that passing a Future returns the same Future object."""
        original_future: Future[int] = Future()
        result = ensure_future(original_future)

        assert result is original_future

    def test_ensure_future_with_value_returns_completed_future(self):
        """Test that passing a value returns a completed Future with that value."""
        value = 42
        result = ensure_future(value)

        assert isinstance(result, Future)
        assert result.done()
        assert result.result() == value


class TestFutThen:
    """Tests for fut_then function."""

    def test_fut_then_success_case(self):
        """Test that fut_then applies then_func on successful Future completion."""
        fut: Future[int] = Future()
        result_fut = fut_then(fut, lambda x: x * 2)

        fut.set_result(5)

        assert result_fut.result() == 10

    def test_fut_then_with_already_completed_future(self):
        """Test that fut_then works with already completed Future."""
        fut: Future[int] = Future()
        fut.set_result(10)

        result_fut = fut_then(fut, lambda x: x + 5)

        assert result_fut.result() == 15

    def test_fut_then_propagates_exception_without_err_func(self):
        """Test that fut_then propagates exception when no err_func is provided."""
        fut: Future[int] = Future()
        result_fut = fut_then(fut, lambda x: x * 2)

        test_exception = ValueError("test error")
        fut.set_exception(test_exception)

        with pytest.raises(ValueError, match="test error"):
            result_fut.result()

    def test_fut_then_calls_err_func_on_exception(self):
        """Test that fut_then calls err_func when Future completes with exception."""
        fut: Future[int] = Future()
        result_fut = fut_then(
            fut,
            then_func=lambda x: x * 2,
            err_func=lambda e: -1,  # error recovery
        )

        fut.set_exception(ValueError("test error"))

        assert result_fut.result() == -1

    def test_fut_then_err_func_exception_propagates(self):
        """Test that exception in err_func is propagated."""
        fut: Future[int] = Future()

        def raising_err_func(e):
            raise RuntimeError("err_func failed")

        result_fut = fut_then(fut, lambda x: x, err_func=raising_err_func)
        fut.set_exception(ValueError("original error"))

        with pytest.raises(RuntimeError, match="err_func failed"):
            result_fut.result()

    def test_fut_then_then_func_exception_propagates(self):
        """Test that exception in then_func is propagated."""
        fut: Future[int] = Future()

        def raising_then_func(x):
            raise RuntimeError("then_func failed")

        result_fut = fut_then(fut, raising_then_func)
        fut.set_result(5)

        with pytest.raises(RuntimeError, match="then_func failed"):
            result_fut.result()

    def test_fut_then_cancelled_future(self):
        """Test that fut_then handles cancelled futures."""
        fut: Future[int] = Future()
        result_fut = fut_then(fut, lambda x: x * 2)

        fut.cancel()

        assert result_fut.cancelled()

    def test_fut_then_chaining(self):
        """Test that fut_then can be chained multiple times."""
        fut: Future[int] = Future()
        result_fut = fut_then(fut_then(fut, lambda x: x + 1), lambda x: x * 2)

        fut.set_result(5)

        # (5 + 1) * 2 = 12
        assert result_fut.result() == 12

    def test_fut_then_with_type_transformation(self):
        """Test that fut_then correctly transforms types."""
        fut: Future[int] = Future()
        result_fut = fut_then(fut, lambda x: str(x))

        fut.set_result(42)

        result = result_fut.result()
        assert result == "42"
        assert isinstance(result, str)


class TestWrapFuture:
    """Tests for wrap_future function."""

    def test_wrap_future_with_value_returns_none(self):
        """Test that wrap_future returns None result for non-Future input."""
        result = wrap_future("any value")

        assert result.done()
        assert result.result() is None

    def test_wrap_future_with_future_returns_none_on_success(self):
        """Test that wrap_future returns None when input Future succeeds."""
        original_future: Future[str] = Future()
        result = wrap_future(original_future)

        original_future.set_result("success value")

        assert result.result() is None

    def test_wrap_future_propagates_exception(self):
        """Test that wrap_future propagates exception from input Future."""
        original_future: Future[str] = Future()
        result = wrap_future(original_future)

        original_future.set_exception(ValueError("test error"))

        with pytest.raises(ValueError, match="test error"):
            result.result()


class TestFromDict:
    """Tests for from_dict function."""

    def test_from_dict_with_primitive_type(self):
        """Test that from_dict returns primitive data unchanged."""
        result = from_dict(str, "simple_string")

        assert result == "simple_string"

    def test_from_dict_with_simple_dataclass(self):
        """Test that from_dict correctly deserializes a simple dataclass."""

        @dataclass
        class SimpleData:
            name: str
            value: int

        data = {"name": "test", "value": 42}
        result = from_dict(SimpleData, data)

        assert isinstance(result, SimpleData)
        assert result.name == "test"
        assert result.value == 42

    def test_from_dict_with_default_values(self):
        """Test that from_dict uses default values when field is missing."""

        @dataclass
        class DataWithDefaults:
            required: str
            optional: int = 100

        data = {"required": "value"}
        result = from_dict(DataWithDefaults, data)

        assert result.required == "value"
        assert result.optional == 100

    def test_from_dict_with_default_factory(self):
        """Test that from_dict uses default_factory when field is missing."""

        @dataclass
        class DataWithFactory:
            name: str
            items: List[str] = field(default_factory=list)

        data = {"name": "test"}
        result = from_dict(DataWithFactory, data)

        assert result.name == "test"
        assert result.items == []

    def test_from_dict_with_optional_field(self):
        """Test that from_dict handles Optional fields correctly."""

        @dataclass
        class DataWithOptional:
            name: str
            nickname: Optional[str] = None

        data = {"name": "test"}
        result = from_dict(DataWithOptional, data)

        assert result.name == "test"
        assert result.nickname is None

    def test_from_dict_with_optional_field_provided(self):
        """Test that from_dict correctly assigns Optional field when provided."""

        @dataclass
        class DataWithOptional:
            name: str
            nickname: Optional[str] = None

        data = {"name": "test", "nickname": "testy"}
        result = from_dict(DataWithOptional, data)

        assert result.name == "test"
        assert result.nickname == "testy"

    def test_from_dict_with_nested_dataclass(self):
        """Test that from_dict correctly deserializes nested dataclasses."""

        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            name: str
            inner: Inner

        data = {"name": "test", "inner": {"value": 42}}
        result = from_dict(Outer, data)

        assert result.name == "test"
        assert isinstance(result.inner, Inner)
        assert result.inner.value == 42

    def test_from_dict_with_optional_nested_dataclass(self):
        """Test that from_dict handles Optional nested dataclass correctly."""

        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            name: str
            inner: Optional[Inner] = None

        data = {"name": "test", "inner": {"value": 42}}
        result = from_dict(Outer, data)

        assert result.name == "test"
        assert isinstance(result.inner, Inner)
        assert result.inner.value == 42

    def test_from_dict_with_list_of_dataclasses(self):
        """Test that from_dict correctly deserializes lists of dataclasses."""

        @dataclass
        class Item:
            id: int
            name: str

        @dataclass
        class Container:
            items: List[Item]

        data = {"items": [{"id": 1, "name": "first"}, {"id": 2, "name": "second"}]}
        result = from_dict(Container, data)

        assert len(result.items) == 2
        assert all(isinstance(item, Item) for item in result.items)
        assert result.items[0].id == 1
        assert result.items[0].name == "first"
        assert result.items[1].id == 2
        assert result.items[1].name == "second"

    def test_from_dict_raises_on_missing_required_field(self):
        """Test that from_dict raises ValueError for missing required field."""

        @dataclass
        class RequiredFields:
            required_field: str
            another_required: int

        data = {"required_field": "value"}  # missing 'another_required'

        with pytest.raises(
            ValueError, match="Missing required field: another_required"
        ):
            from_dict(RequiredFields, data)

    def test_from_dict_with_empty_list(self):
        """Test that from_dict handles empty lists correctly."""

        @dataclass
        class Item:
            id: int

        @dataclass
        class Container:
            items: List[Item]

        data = {"items": []}
        result = from_dict(Container, data)

        assert result.items == []


class TestCompareStateDicts:
    """Tests for compare_state_dicts function."""

    # ==================== Equal state dicts tests ====================

    def test_equal_empty_dicts(self):
        """Test that two empty dicts are considered equal."""
        result = compare_state_dicts({}, {})
        assert result == {}

    def test_equal_nested_dicts(self):
        """Test that two equal nested dicts return no differences."""
        left = {"a": {"b": {"c": 1}}, "d": [1, 2, 3]}
        right = {"a": {"b": {"c": 1}}, "d": [1, 2, 3]}
        result = compare_state_dicts(left, right)
        assert result == {}

    def test_equal_with_tensors(self):
        """Test that dicts with equal tensors return no differences."""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        left = {"weight": tensor.clone()}
        right = {"weight": tensor.clone()}
        result = compare_state_dicts(left, right)
        assert result == {}

    def test_equal_with_numpy_arrays(self):
        """Test that dicts with equal numpy arrays return no differences."""
        arr = np.array([1.0, 2.0, 3.0])
        left = {"data": arr.copy()}
        right = {"data": arr.copy()}
        result = compare_state_dicts(left, right)
        assert result == {}

    # ==================== LEFT_ONLY and RIGHT_ONLY tests ====================

    def test_left_and_right_only_keys(self):
        """Test detection of keys unique to each dict."""
        left = {"a": 1, "b": 2}
        right = {"a": 1, "c": 3}
        result = compare_state_dicts(left, right)
        assert result == {
            ("b",): CompareStateDictsResult.LEFT_ONLY,
            ("c",): CompareStateDictsResult.RIGHT_ONLY,
        }

    def test_nested_left_only_key(self):
        """Test detection of nested keys only in left dict."""
        left = {"a": {"b": 1, "c": 2}}
        right = {"a": {"b": 1}}
        result = compare_state_dicts(left, right)
        assert result == {("a", "c"): CompareStateDictsResult.LEFT_ONLY}

    def test_nested_right_only_key(self):
        """Test detection of nested keys only in right dict."""
        left = {"a": {"b": 1}}
        right = {"a": {"b": 1, "c": 2}}
        result = compare_state_dicts(left, right)
        assert result == {("a", "c"): CompareStateDictsResult.RIGHT_ONLY}

    # ==================== TYPE_NOT_EQUAL tests ====================

    def test_type_mismatch_int_vs_str(self):
        """Test detection of type mismatch between int and string."""
        left = {"a": 1}
        right = {"a": "1"}
        result = compare_state_dicts(left, right)
        assert result == {("a",): CompareStateDictsResult.TYPE_NOT_EQUAL}

    def test_type_mismatch_list_vs_tuple(self):
        """Test detection of type mismatch between list and tuple."""
        left = {"a": [1, 2, 3]}
        right = {"a": (1, 2, 3)}
        result = compare_state_dicts(left, right)
        assert result == {("a",): CompareStateDictsResult.TYPE_NOT_EQUAL}

    def test_type_mismatch_dict_vs_list(self):
        """Test detection of type mismatch between dict and list."""
        left = {"a": {"b": 1}}
        right = {"a": [1]}
        result = compare_state_dicts(left, right)
        assert result == {("a",): CompareStateDictsResult.TYPE_NOT_EQUAL}

    def test_type_mismatch_tensor_vs_ndarray(self):
        """Test detection of type mismatch between torch.Tensor and np.ndarray."""
        left = {"a": torch.tensor([1.0, 2.0])}
        right = {"a": np.array([1.0, 2.0])}
        result = compare_state_dicts(left, right)
        assert result == {("a",): CompareStateDictsResult.TYPE_NOT_EQUAL}

    def test_type_mismatch_nested(self):
        """Test detection of type mismatch in nested structure."""
        left = {"a": {"b": {"c": 1}}}
        right = {"a": {"b": {"c": "1"}}}
        result = compare_state_dicts(left, right)
        assert result == {("a", "b", "c"): CompareStateDictsResult.TYPE_NOT_EQUAL}

    # ==================== SHAPE_NOT_EQUAL tests ====================

    def test_tensor_shape_mismatch(self):
        """Test detection of tensor shape mismatch."""
        left = {"weight": torch.randn(3, 4)}
        right = {"weight": torch.randn(4, 3)}
        result = compare_state_dicts(left, right)
        assert result == {("weight",): CompareStateDictsResult.SHAPE_NOT_EQUAL}

    def test_numpy_shape_mismatch(self):
        """Test detection of numpy array shape mismatch."""
        left = {"data": np.zeros((2, 3))}
        right = {"data": np.zeros((3, 2))}
        result = compare_state_dicts(left, right)
        assert result == {("data",): CompareStateDictsResult.SHAPE_NOT_EQUAL}

    def test_tensor_shape_mismatch_different_dims(self):
        """Test detection of tensor shape mismatch with different dimensions."""
        left = {"weight": torch.randn(3, 4)}
        right = {"weight": torch.randn(3, 4, 5)}
        result = compare_state_dicts(left, right)
        assert result == {("weight",): CompareStateDictsResult.SHAPE_NOT_EQUAL}

    # ==================== DTYPE_NOT_EQUAL tests ====================

    def test_tensor_dtype_mismatch(self):
        """Test detection of tensor dtype mismatch."""
        left = {"weight": torch.zeros(3, 4, dtype=torch.float32)}
        right = {"weight": torch.zeros(3, 4, dtype=torch.float64)}
        result = compare_state_dicts(left, right)
        assert result == {("weight",): CompareStateDictsResult.DTYPE_NOT_EQUAL}

    def test_numpy_dtype_mismatch(self):
        """Test detection of numpy array dtype mismatch."""
        left = {"data": np.zeros((2, 3), dtype=np.float32)}
        right = {"data": np.zeros((2, 3), dtype=np.int32)}
        result = compare_state_dicts(left, right)
        assert result == {("data",): CompareStateDictsResult.DTYPE_NOT_EQUAL}

    # ==================== ELEMENTS_NOT_EQUAL tests ====================

    def test_tensor_elements_not_equal(self):
        """Test detection of tensor element differences."""
        left = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        right = {"weight": torch.tensor([1.0, 2.0, 4.0])}
        result = compare_state_dicts(left, right)
        assert result == {("weight",): CompareStateDictsResult.ELEMENTS_NOT_EQUAL}

    def test_numpy_elements_not_equal(self):
        """Test detection of numpy array element differences."""
        left = {"data": np.array([1.0, 2.0, 3.0])}
        right = {"data": np.array([1.0, 2.5, 3.0])}
        result = compare_state_dicts(left, right)
        assert result == {("data",): CompareStateDictsResult.ELEMENTS_NOT_EQUAL}

    # ==================== SEQ_LENGTH_NOT_EQUAL tests ====================

    def test_list_length_mismatch(self):
        """Test detection of list length mismatch."""
        left = {"items": [1, 2, 3]}
        right = {"items": [1, 2]}
        result = compare_state_dicts(left, right)
        assert result == {("items",): CompareStateDictsResult.SEQ_LENGTH_NOT_EQUAL}

    def test_tuple_length_mismatch(self):
        """Test detection of tuple length mismatch."""
        left = {"items": (1, 2, 3, 4)}
        right = {"items": (1, 2)}
        result = compare_state_dicts(left, right)
        assert result == {("items",): CompareStateDictsResult.SEQ_LENGTH_NOT_EQUAL}

    def test_nested_list_length_mismatch(self):
        """Test detection of nested list length mismatch."""
        left = {"data": {"items": [1, 2, 3]}}
        right = {"data": {"items": [1, 2, 3, 4]}}
        result = compare_state_dicts(left, right)
        assert result == {
            ("data", "items"): CompareStateDictsResult.SEQ_LENGTH_NOT_EQUAL
        }

    def test_empty_vs_nonempty_list(self):
        """Test detection of empty vs non-empty list."""
        left = {"items": []}
        right = {"items": [1]}
        result = compare_state_dicts(left, right)
        assert result == {("items",): CompareStateDictsResult.SEQ_LENGTH_NOT_EQUAL}

    # ==================== OBJ_NOT_EQUAL tests ====================

    def test_obj_not_equal_strings(self):
        """Test detection of string value differences."""
        left = {"name": "hello"}
        right = {"name": "world"}
        result = compare_state_dicts(left, right)
        assert result == {("name",): CompareStateDictsResult.OBJ_NOT_EQUAL}

    def test_obj_not_equal_floats(self):
        """Test detection of float value differences."""
        left = {"value": 3.14}
        right = {"value": 2.71}
        result = compare_state_dicts(left, right)
        assert result == {("value",): CompareStateDictsResult.OBJ_NOT_EQUAL}

    # ==================== OBJ_NOT_COMPARABLE tests ====================

    def test_obj_not_comparable(self):
        """Test detection of objects that cannot be compared."""

        class NonComparable:
            def __eq__(self, other):
                raise TypeError("Cannot compare")

        left = {"obj": NonComparable()}
        right = {"obj": NonComparable()}
        result = compare_state_dicts(left, right)
        assert result == {("obj",): CompareStateDictsResult.OBJ_NOT_COMPARABLE}

    # ==================== deque support tests (new feature) ====================

    def test_deque_equal(self):
        """Test that equal deques return no differences."""
        left = {"queue": deque([1, 2, 3])}
        right = {"queue": deque([1, 2, 3])}
        result = compare_state_dicts(left, right)
        assert result == {}

    def test_deque_length_mismatch(self):
        """Test detection of deque length mismatch."""
        left = {"queue": deque([1, 2, 3])}
        right = {"queue": deque([1, 2])}
        result = compare_state_dicts(left, right)
        assert result == {("queue",): CompareStateDictsResult.SEQ_LENGTH_NOT_EQUAL}

    def test_deque_element_difference(self):
        """Test detection of deque element differences."""
        left = {"queue": deque([1, 2, 3])}
        right = {"queue": deque([1, 2, 4])}
        result = compare_state_dicts(left, right)
        assert result == {("queue", 2): CompareStateDictsResult.OBJ_NOT_EQUAL}

    def test_deque_with_complex_elements(self):
        """Test deque with nested dicts as elements."""
        left = {"queue": deque([{"a": 1}, {"b": 2}])}
        right = {"queue": deque([{"a": 1}, {"b": 3}])}
        result = compare_state_dicts(left, right)
        assert result == {("queue", 1, "b"): CompareStateDictsResult.OBJ_NOT_EQUAL}

    def test_deque_empty_equal(self):
        """Test that empty deques are equal."""
        left = {"queue": deque()}
        right = {"queue": deque()}
        result = compare_state_dicts(left, right)
        assert result == {}

    def test_deque_vs_list_type_mismatch(self):
        """Test type mismatch between deque and list."""
        left = {"seq": deque([1, 2, 3])}
        right = {"seq": [1, 2, 3]}
        result = compare_state_dicts(left, right)
        assert result == {("seq",): CompareStateDictsResult.TYPE_NOT_EQUAL}

    # ==================== Dataclass tests ====================

    def test_dataclass_equal(self):
        """Test that equal dataclasses return no differences."""

        @dataclass
        class Config:
            lr: float
            epochs: int

        left = {"config": Config(lr=0.001, epochs=10)}
        right = {"config": Config(lr=0.001, epochs=10)}
        result = compare_state_dicts(left, right)
        assert result == {}

    def test_dataclass_field_difference(self):
        """Test detection of dataclass field differences."""

        @dataclass
        class Config:
            lr: float
            epochs: int

        left = {"config": Config(lr=0.001, epochs=10)}
        right = {"config": Config(lr=0.001, epochs=20)}
        result = compare_state_dicts(left, right)
        assert result == {("config", "epochs"): CompareStateDictsResult.OBJ_NOT_EQUAL}

    def test_dataclass_multiple_field_differences(self):
        """Test detection of multiple dataclass field differences."""

        @dataclass
        class Config:
            lr: float
            epochs: int
            batch_size: int

        left = {"config": Config(lr=0.001, epochs=10, batch_size=32)}
        right = {"config": Config(lr=0.01, epochs=20, batch_size=32)}
        result = compare_state_dicts(left, right)
        assert ("config", "lr") in result
        assert ("config", "epochs") in result
        assert result[("config", "lr")] == CompareStateDictsResult.OBJ_NOT_EQUAL
        assert result[("config", "epochs")] == CompareStateDictsResult.OBJ_NOT_EQUAL

    def test_nested_dataclass(self):
        """Test comparison of nested dataclasses."""

        @dataclass
        class Inner:
            value: int

        @dataclass
        class Outer:
            inner: Inner
            name: str

        left = {"obj": Outer(inner=Inner(value=1), name="test")}
        right = {"obj": Outer(inner=Inner(value=2), name="test")}
        result = compare_state_dicts(left, right)
        assert result == {
            ("obj", "inner", "value"): CompareStateDictsResult.OBJ_NOT_EQUAL
        }

    # ==================== Complex nested structure tests ====================

    def test_deeply_nested_dict(self):
        """Test comparison of deeply nested dicts."""
        left = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        right = {"a": {"b": {"c": {"d": {"e": 2}}}}}
        result = compare_state_dicts(left, right)
        assert result == {
            ("a", "b", "c", "d", "e"): CompareStateDictsResult.OBJ_NOT_EQUAL
        }

    def test_mixed_nested_structures(self):
        """Test comparison of mixed nested structures (dict, list, tuple)."""
        left = {"data": {"items": [{"values": (1, 2, 3)}]}}
        right = {"data": {"items": [{"values": (1, 2, 4)}]}}
        result = compare_state_dicts(left, right)
        assert result == {
            ("data", "items", 0, "values", 2): CompareStateDictsResult.OBJ_NOT_EQUAL
        }

    def test_multiple_differences_at_various_levels(self):
        """Test detection of multiple differences at various nesting levels."""
        left = {
            "a": 1,
            "b": {"c": 2, "d": [3, 4]},
            "e": "hello",
        }
        right = {
            "a": 10,
            "b": {"c": 2, "d": [3, 5]},
            "e": "world",
        }
        result = compare_state_dicts(left, right)
        assert ("a",) in result
        assert ("b", "d", 1) in result
        assert ("e",) in result
        assert result[("a",)] == CompareStateDictsResult.OBJ_NOT_EQUAL
        assert result[("b", "d", 1)] == CompareStateDictsResult.OBJ_NOT_EQUAL
        assert result[("e",)] == CompareStateDictsResult.OBJ_NOT_EQUAL

    def test_list_of_tensors(self):
        """Test comparison of list of tensors."""
        left = {"weights": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]}
        right = {"weights": [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 5.0])]}
        result = compare_state_dicts(left, right)
        assert result == {("weights", 1): CompareStateDictsResult.ELEMENTS_NOT_EQUAL}

    def test_dict_of_lists_of_dicts(self):
        """Test complex structure: dict of lists of dicts."""
        left = {"layers": [{"weight": 1, "bias": 2}, {"weight": 3, "bias": 4}]}
        right = {"layers": [{"weight": 1, "bias": 2}, {"weight": 3, "bias": 5}]}
        result = compare_state_dicts(left, right)
        assert result == {("layers", 1, "bias"): CompareStateDictsResult.OBJ_NOT_EQUAL}

    # ==================== Edge cases ====================

    def test_list_with_element_differences_in_sequence(self):
        """Test detection of multiple element differences in a list."""
        left = {"items": [1, 2, 3, 4]}
        right = {"items": [1, 5, 3, 6]}
        result = compare_state_dicts(left, right)
        assert ("items", 1) in result
        assert ("items", 3) in result
        assert result[("items", 1)] == CompareStateDictsResult.OBJ_NOT_EQUAL
        assert result[("items", 3)] == CompareStateDictsResult.OBJ_NOT_EQUAL

    def test_same_values_different_keys(self):
        """Test that same values under different keys are detected."""
        left = {"key_a": 1}
        right = {"key_b": 1}
        result = compare_state_dicts(left, right)
        assert result == {
            ("key_a",): CompareStateDictsResult.LEFT_ONLY,
            ("key_b",): CompareStateDictsResult.RIGHT_ONLY,
        }

    def test_tuple_with_tensors(self):
        """Test tuple containing tensors."""
        left = {"data": (torch.tensor([1.0]), torch.tensor([2.0]))}
        right = {"data": (torch.tensor([1.0]), torch.tensor([3.0]))}
        result = compare_state_dicts(left, right)
        assert result == {("data", 1): CompareStateDictsResult.ELEMENTS_NOT_EQUAL}

    def test_single_element_tensor_equal(self):
        """Test comparison of single-element tensors."""
        left = {"scalar": torch.tensor(3.14)}
        right = {"scalar": torch.tensor(3.14)}
        result = compare_state_dicts(left, right)
        assert result == {}

    def test_single_element_tensor_not_equal(self):
        """Test detection of single-element tensor difference."""
        left = {"scalar": torch.tensor(3.14)}
        right = {"scalar": torch.tensor(2.71)}
        result = compare_state_dicts(left, right)
        assert result == {("scalar",): CompareStateDictsResult.ELEMENTS_NOT_EQUAL}

    def test_complex_real_world_state_dict(self):
        """Test comparison of a realistic model state dict."""
        left = {
            "model.layer1.weight": torch.randn(10, 5),
            "model.layer1.bias": torch.zeros(10),
            "model.layer2.weight": torch.randn(5, 10),
            "optimizer.step": 100,
            "scheduler.last_epoch": 50,
            "metadata": {"version": "1.0", "name": "test_model"},
        }
        right_weight = torch.randn(10, 5)
        right = {
            "model.layer1.weight": right_weight,
            "model.layer1.bias": torch.zeros(10),
            "model.layer2.weight": left["model.layer2.weight"].clone(),
            "optimizer.step": 100,
            "scheduler.last_epoch": 50,
            "metadata": {"version": "1.0", "name": "test_model"},
        }
        result = compare_state_dicts(left, right)
        assert ("model.layer1.weight",) in result
        assert (
            result[("model.layer1.weight",)]
            == CompareStateDictsResult.ELEMENTS_NOT_EQUAL
        )
