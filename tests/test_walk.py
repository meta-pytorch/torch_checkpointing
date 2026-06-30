# Owner(s): ["oncall: distributed checkpointing"]
"""Tests for walk_checkpoint_structure function.

These tests verify the walk_checkpoint_structure function works correctly with:
- All container types (dict, list, tuple, deque, set, string)
- Custom leaf_fn callbacks and default tensor copy behavior
- Complex nested structures mixing multiple container types
- Missing path tracking
- In-place modification behavior for mutable containers
"""

from collections import deque, OrderedDict as OrderedDictType

import pytest
import torch
from torch_checkpointing.types import CheckpointPath
from torch_checkpointing.walk_utils import walk_checkpoint_structure


class TestWalkCheckpointStructure:
    """Tests for walk_checkpoint_structure unified function."""

    # ========== Container type tests ==========

    @pytest.mark.parametrize(
        "source,target,expected_structure",
        [
            # Dict with nested structure
            pytest.param(
                {"a": {"x": 1, "y": 2}, "b": 3},
                {"a": {"x": None}, "b": None},
                {"a": {"x": 1}, "b": 3},
                id="dict_nested",
            ),
            # List filtering
            pytest.param(
                [1, 2, 3, 4, 5],
                [None, None, None],
                [1, 2, 3],
                id="list_filter",
            ),
            # Tuple filtering
            pytest.param(
                (10, 20, 30, 40),
                (None, None),
                (10, 20),
                id="tuple_filter",
            ),
            # Deque filtering
            pytest.param(
                deque([{"item": 1}, {"item": 2}, {"item": 3}]),
                deque([None, None]),
                deque([{"item": 1}, {"item": 2}]),
                id="deque_filter",
            ),
            # Empty dict
            pytest.param(
                {},
                {},
                {},
                id="empty_dict",
            ),
            # Empty list
            pytest.param(
                [],
                [],
                [],
                id="empty_list",
            ),
            # Nested lists
            pytest.param(
                [[1, 2], [3, 4]],
                [[None, None]],
                [[1, 2]],
                id="nested_lists",
            ),
        ],
    )
    def test_container_types(self, source, target, expected_structure):
        """Test all container types work correctly."""
        result, missing_paths = walk_checkpoint_structure(
            item_key="item",
            source=source,
            target=target,
            leaf_fn=None,
        )
        assert result == expected_structure
        assert missing_paths == []

    # ========== Leaf function tests ==========

    def test_custom_leaf_fn(self):
        """Test custom leaf_fn is called with correct arguments."""
        source = {"a": 1, "b": {"c": 2}}
        target = None
        visited = []

        def leaf_fn(path: CheckpointPath, src_val, tgt_val):
            visited.append((str(path), src_val, tgt_val))
            return src_val * 10

        result, missing_paths = walk_checkpoint_structure(
            item_key="data",
            source=source,
            target=target,
            leaf_fn=leaf_fn,
        )

        # Verify leaf_fn was called and values were transformed
        assert result == {"a": 10, "b": {"c": 20}}
        assert len(visited) == 2
        # Check paths were correctly constructed
        paths = [p for p, _, _ in visited]
        assert "data::a" in paths
        assert "data::b.c" in paths

    # ========== In-place modification tests ==========

    def test_dict_in_place_modification(self):
        """Test that dicts are modified in-place."""
        source = {"a": 1, "b": 2}
        target = {"a": None, "b": None}

        result, missing_paths = walk_checkpoint_structure(
            item_key="data",
            source=source,
            target=target,
        )

        # Result should be the same dict object as target (in-place modification)
        assert result is target
        # Values should be updated
        assert result == {"a": 1, "b": 2}

    def test_tensor_in_place_copy_with_target(self):
        """Test default tensor copy_() behavior when leaf_fn is None."""
        src_tensor = torch.tensor([1.0, 2.0, 3.0])
        tgt_tensor = torch.zeros(3)

        source = {"weight": src_tensor}
        target = {"weight": tgt_tensor}

        result, missing_paths = walk_checkpoint_structure(
            item_key="model",
            source=source,
            target=target,
            leaf_fn=None,
        )

        # Target tensor should be modified in place
        assert torch.equal(tgt_tensor, src_tensor)
        # Result should reference the target tensor
        assert result["weight"] is tgt_tensor

    def test_list_in_place_modification(self):
        """Test that lists are modified in-place like dicts."""
        src_tensor = torch.tensor([1.0, 2.0, 3.0])
        tgt_tensor = torch.zeros(3)

        source = [src_tensor, "value"]
        target = [tgt_tensor, None]

        result, missing_paths = walk_checkpoint_structure(
            item_key="model",
            source=source,
            target=target,
            leaf_fn=None,
        )

        # Result should be the same list object as target (in-place modification)
        assert result is target
        # Target tensor should be modified in place
        assert torch.equal(tgt_tensor, src_tensor)
        # Elements should reference target list's elements
        assert result[0] is tgt_tensor

    def test_deque_in_place_modification(self):
        """Test that deques are modified in-place like lists."""
        source = deque([1, 2, 3])
        target = deque([0, 0])

        result, missing_paths = walk_checkpoint_structure(
            item_key="data",
            source=source,
            target=target,
        )

        # Result should be the same deque object as target
        assert result is target
        # Values should be updated
        assert list(result) == [1, 2]

    def test_tuple_not_modified_in_place(self):
        """Test that tuples are NOT modified in-place (new tuple is created)."""
        source = (1, 2, 3)
        target = (None, None)

        result, missing_paths = walk_checkpoint_structure(
            item_key="data",
            source=source,
            target=target,
        )

        # Result should NOT be the same tuple object (tuples are immutable)
        assert result is not target
        # Values should be from source
        assert result == (1, 2)

    def test_nested_dict_in_place_modification(self):
        """Test that nested dicts are modified in-place."""
        inner_target = {"x": None, "y": None}
        target = {"nested": inner_target}
        source = {"nested": {"x": 1, "y": 2}}

        result, missing_paths = walk_checkpoint_structure(
            item_key="data",
            source=source,
            target=target,
        )

        # Both outer and inner dicts should be the same objects
        assert result is target
        assert result["nested"] is inner_target
        assert result["nested"] == {"x": 1, "y": 2}

    # ========== DTensor tests ==========

    def test_dtensor_tensor_mismatch_raises_error(self):
        """Test that copying between DTensor and regular Tensor raises error."""
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor

        # Skip if distributed not available
        try:
            mesh = DeviceMesh("cpu", [0])
        except Exception:
            pytest.skip("DeviceMesh not available in this environment")

        src_tensor = torch.tensor([1.0, 2.0, 3.0])
        tgt_dtensor = DTensor.from_local(torch.zeros(3), mesh, [])

        source = {"weight": src_tensor}
        target = {"weight": tgt_dtensor}

        with pytest.raises(
            RuntimeError, match="Cannot copy between DTensor and regular Tensor"
        ):
            walk_checkpoint_structure(
                item_key="model",
                source=source,
                target=target,
            )

    # ========== Complex structure tests ==========

    def test_complex_nested_structure(self):
        """Test complex structure with dict, list, tuple, deque, tensors, strings."""
        tensor = torch.randn(3, 3)
        source = {
            "layers": [
                {"weights": tensor, "name": "layer1"},
                {"weights": torch.randn(2, 2), "name": "layer2"},
            ],
            "config": (1, 2, 3),
            "history": deque([10, 20, 30]),
            "metadata": {"version": "1.0", "tags": {"train", "v1"}},
        }
        # Request subset of data
        target = {
            "layers": [{"weights": None, "name": None}],  # Only first layer
            "config": (None, None),  # First two elements
            "history": deque([None]),  # First history item
            "metadata": None,  # All metadata
        }

        result, missing_paths = walk_checkpoint_structure(
            item_key="ckpt", source=source, target=target
        )

        # Verify structure is correct
        assert len(result["layers"]) == 1
        assert result["layers"][0]["name"] == "layer1"
        assert torch.equal(result["layers"][0]["weights"], tensor)
        assert result["config"] == (1, 2)
        assert result["history"] == deque([10])
        assert result["metadata"]["version"] == "1.0"
        assert result["metadata"]["tags"] == {"train", "v1"}
        # Verify container types are preserved
        assert isinstance(result["config"], tuple)
        assert isinstance(result["history"], deque)
        assert isinstance(result["metadata"]["tags"], set)

    # ========== Missing keys tracking tests ==========

    def test_missing_keys_tracked(self):
        """Test missing keys are tracked and returned."""
        source = {"a": 1}
        target = {"a": None, "b": None, "c": None}

        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=source, target=target
        )
        assert result == {"a": 1, "b": None, "c": None}
        assert len(missing_paths) == 2
        assert CheckpointPath("item", ("b",)) in missing_paths
        assert CheckpointPath("item", ("c",)) in missing_paths

    def test_missing_keys_tracking_dict(self):
        """Test that missing keys in dicts are tracked correctly."""
        loaded = {"a": 1, "b": 2}
        requested = {"a": None, "c": None, "d": None}  # c and d don't exist in loaded

        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )

        assert result == {"a": 1, "c": None, "d": None}
        assert len(missing_paths) == 2
        assert CheckpointPath("item", ("c",)) in missing_paths
        assert CheckpointPath("item", ("d",)) in missing_paths

    def test_missing_keys_tracking_nested_dict(self):
        """Test that nested missing keys in dicts are tracked with full path."""
        loaded = {"model": {"block1": {"w": 1}}}
        requested = {"model": {"block1": {"w": None, "b": None}, "block2": {"w": None}}}

        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )

        assert result["model"]["block1"]["w"] == 1
        assert result["model"]["block1"]["b"] is None
        assert len(missing_paths) == 2
        assert CheckpointPath("item", ("model", "block1", "b")) in missing_paths
        assert CheckpointPath("item", ("model", "block2")) in missing_paths

    def test_missing_keys_tracking_sequence(self):
        """Test that missing indices in sequences are tracked correctly."""
        loaded = [1, 2, 3]
        requested = [None, None, None, None, None]  # indices 3 and 4 don't exist

        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )

        assert result == [1, 2, 3, None, None]
        assert len(missing_paths) == 2
        assert CheckpointPath("item", (3,)) in missing_paths
        assert CheckpointPath("item", (4,)) in missing_paths

    def test_missing_keys_tracking_mixed(self):
        """Test missing keys tracking with mixed dict and sequence nesting."""
        loaded = {"items": [{"x": 1}, {"x": 2}]}
        requested = {
            "items": [{"x": None, "y": None}, {"x": None}, {"z": None}],
            "config": None,
        }

        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )

        # Check result structure
        assert result["items"][0]["x"] == 1
        assert result["items"][0]["y"] is None
        assert result["items"][1]["x"] == 2
        assert result["items"][2]["z"] is None
        assert result["config"] is None

        # Check missing keys
        assert len(missing_paths) == 3
        assert CheckpointPath("item", ("items", 0, "y")) in missing_paths
        assert CheckpointPath("item", ("items", 2)) in missing_paths
        assert CheckpointPath("item", ("config",)) in missing_paths

    # ========== Type compatibility tests ==========

    def test_ordereddict_compatibility(self):
        """Test OrderedDict is handled as Mapping."""
        source = OrderedDictType([("a", 1), ("b", 2), ("c", 3)])
        target = {"a": None, "c": None}

        result, missing_paths = walk_checkpoint_structure(
            item_key="data", source=source, target=target
        )
        assert result == {"a": 1, "c": 3}

    @pytest.mark.parametrize(
        "src,tgt,expected_error",
        [
            pytest.param({"a": 1}, [1, 2], "Target value at", id="dict_vs_list"),
            pytest.param([1, 2], {"a": 1}, "Target value at", id="list_vs_dict"),
            pytest.param("hello", [1, 2], "Target value at", id="str_vs_list"),
            pytest.param(
                {"a": [1, 2]}, {"a": {"b": 1}}, "Target value at", id="nested_mismatch"
            ),
        ],
    )
    def test_type_mismatch_raises_error(self, src, tgt, expected_error):
        """Test type mismatches raise RuntimeError."""
        with pytest.raises(RuntimeError, match=expected_error):
            walk_checkpoint_structure(item_key="item", source=src, target=tgt)

    # ========== Filtering tests ==========

    def test_none_target_returns_all_data(self):
        """When target is None, return all loaded data with structure derived from source."""
        loaded = {"a": 1, "b": 2}
        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=None
        )
        assert result == loaded
        assert missing_paths == []

    def test_dict_filtering(self):
        """Test filtering dict keys at various nesting levels."""
        # Top-level filtering
        loaded = {"a": 1, "b": 2, "c": 3}
        result, _ = walk_checkpoint_structure(
            item_key="item", source=loaded, target={"a": None, "c": None}
        )
        assert result == {"a": 1, "c": 3}

        # Nested filtering
        loaded = {"model": {"block1": {"w": 1}, "block2": {"w": 2}}, "opt": {}}
        result, _ = walk_checkpoint_structure(
            item_key="item", source=loaded, target={"model": {"block1": None}}
        )
        assert result == {"model": {"block1": {"w": 1}}}

        # Deeply nested filtering
        loaded = {
            "level1": {
                "level2": {"level3": {"a": 1, "b": 2, "c": 3}, "other": {"x": 10}},
                "sibling": {"y": 20},
            }
        }
        requested = {"level1": {"level2": {"level3": {"a": None, "c": None}}}}
        result, _ = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )
        assert result == {"level1": {"level2": {"level3": {"a": 1, "c": 3}}}}

    @pytest.mark.parametrize(
        "loaded,requested,expected",
        [
            # List: filter down from 5 to 3 elements
            pytest.param(
                [1, 2, 3, 4, 5],
                [None, None, None],
                [1, 2, 3],
                id="list_filter_down",
            ),
            # Tuple: filter down from 5 to 3 elements
            pytest.param(
                (1, 2, 3, 4, 5),
                (None, None, None),
                (1, 2, 3),
                id="tuple_filter_down",
            ),
            # Deque: filter down from 5 to 3 elements
            pytest.param(
                deque([1, 2, 3, 4, 5]),
                deque([None, None, None]),
                deque([1, 2, 3]),
                id="deque_filter_down",
            ),
            # Nested deque in dict: filter down from 3 to 2 elements
            pytest.param(
                {
                    "prefetch_buffer": deque([{"item": 1}, {"item": 2}, {"item": 3}]),
                    "other": "value",
                },
                {
                    "prefetch_buffer": deque([None, None]),
                    "other": None,
                },
                {
                    "prefetch_buffer": deque([{"item": 1}, {"item": 2}]),
                    "other": "value",
                },
                id="nested_deque_filter_down",
            ),
        ],
    )
    def test_sequence_filtering(self, loaded, requested, expected):
        """Test filtering sequences (loaded >= requested length)."""
        result, _ = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )
        assert result == expected
        # Verify deque type is preserved
        if isinstance(loaded, deque):
            assert isinstance(result, deque)
        elif isinstance(loaded, dict):
            # Check nested deques
            for key in loaded:
                if isinstance(loaded[key], deque):
                    assert isinstance(result[key], deque)

    def test_edge_cases(self):
        """Test edge cases: missing keys and empty requests."""
        # Missing key in loaded is kept from requested (preserves user's state)
        loaded = {"a": 1}
        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target={"a": None, "b": None}
        )
        assert result == {"a": 1, "b": None}
        assert len(missing_paths) == 1

        # Empty dict request means no keys requested
        loaded = {"a": 1, "b": 2}
        result, _ = walk_checkpoint_structure(item_key="item", source=loaded, target={})
        assert result == {}

    @pytest.mark.parametrize(
        "loaded,requested,expected",
        [
            # Sets: nested in dict with None request
            pytest.param(
                {"keys": {1, 2, 3}, "value": 10},
                {"keys": None, "value": None},
                {"keys": {1, 2, 3}, "value": 10},
                id="set_nested_none_request",
            ),
            # Sets: both have set values - returns loaded as-is
            pytest.param(
                {"keys": {1, 2, 3}},
                {"keys": {1}},
                {"keys": {1, 2, 3}},
                id="set_both_have_values",
            ),
            # Sets: direct set to set - returns loaded as-is
            pytest.param(
                {1, 2, 3, 4, 5},
                {1, 2},
                {1, 2, 3, 4, 5},
                id="set_direct",
            ),
            # Strings: nested in dict with None request
            pytest.param(
                {"name": "hello_world", "value": 42},
                {"name": None, "value": None},
                {"name": "hello_world", "value": 42},
                id="string_nested_none_request",
            ),
            # Strings: both have string values - returns loaded as-is
            pytest.param(
                {"name": "hello_world"},
                {"name": "h"},
                {"name": "hello_world"},
                id="string_both_have_values",
            ),
            # Strings: direct string to string - returns loaded as-is
            pytest.param(
                "hello_world",
                "h",
                "hello_world",
                id="string_direct",
            ),
        ],
    )
    def test_leaves_not_filtered(self, loaded, requested, expected):
        """Sets and strings are treated as leaf values and returned as-is."""
        result, _ = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )
        assert result == expected

    # ========== Merge tests ==========

    def test_merge_loaded_with_user_state(self):
        """Test merging: None loads from checkpoint, non-None keeps user value."""
        loaded = {"lr": 0.01, "step": 100}
        user_state = {"lr": None, "step": 200, "new_param": 0.5}

        result, missing_paths = walk_checkpoint_structure(
            item_key="optimizer", source=loaded, target=user_state
        )

        assert result == {"lr": 0.01, "step": 100, "new_param": 0.5}

    def test_mixed_none_and_dict_in_request(self):
        """Test mixing None (load all) and dict (filter) in request."""
        loaded = {
            "model": {"block1": {"w": 1}, "block2": {"w": 2}},
            "optimizer": {"state": {"a": 10, "b": 20}, "param_groups": []},
        }
        requested = {
            "model": {"block1": None},  # Only block1, load all of it
            "optimizer": None,  # Load all of optimizer
        }
        result, _ = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )
        assert result == {
            "model": {"block1": {"w": 1}},
            "optimizer": {"state": {"a": 10, "b": 20}, "param_groups": []},
        }

    @pytest.mark.parametrize(
        "loaded,requested,expected",
        [
            # Dict: mix of None (load from ckpt) and non-None (keep user value)
            pytest.param(
                {"a": 1, "b": 2},
                {"a": None, "b": 99, "c": 100, "d": 200},
                {"a": 1, "b": 2, "c": 100, "d": 200},
                id="dict_mixed_none_and_values",
            ),
            # Nested dict: mix at multiple levels
            pytest.param(
                {"model": {"block1": {"w": 1}}},
                {"model": {"block1": {"w": None, "b": 0.5}, "block2": {"w": 2}}},
                {"model": {"block1": {"w": 1, "b": 0.5}, "block2": {"w": 2}}},
                id="nested_dict_mixed",
            ),
            # List: some indices loaded (None), some kept (non-None), some new
            pytest.param(
                [1, 2, 3],
                [None, 99, None, 100, 200],
                [1, 2, 3, 100, 200],
                id="list_mixed_none_and_values",
            ),
            # Tuple: mix of None and non-None
            pytest.param(
                (1, 2),
                (None, 99, 100, 200, 300),
                (1, 2, 100, 200, 300),
                id="tuple_mixed_none_and_values",
            ),
            # Deque: mix of None and non-None
            pytest.param(
                deque([1, 2]),
                deque([None, 99, 100]),
                deque([1, 2, 100]),
                id="deque_mixed_none_and_values",
            ),
            # Complex nested: realistic checkpoint recovery scenario
            pytest.param(
                {
                    "items": [{"x": 1}, {"x": 2}],
                    "config": {"lr": 0.01},
                },
                {
                    "items": [{"x": None, "y": 10}, {"x": None}, {"x": 99, "y": 99}],
                    "config": {"lr": None, "momentum": 0.9},
                    "new_key": "new_value",
                },
                {
                    "items": [{"x": 1, "y": 10}, {"x": 2}, {"x": 99, "y": 99}],
                    "config": {"lr": 0.01, "momentum": 0.9},
                    "new_key": "new_value",
                },
                id="complex_nested_mixed",
            ),
        ],
    )
    def test_merge_loaded_with_requested(self, loaded, requested, expected):
        """Test merging: None means load from checkpoint, non-None means keep user value."""
        result, _ = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )
        assert result == expected
        # Verify container type is preserved for sequences
        if isinstance(loaded, deque):
            assert isinstance(result, deque)

    def test_empty_sequence_returns_loaded_data(self):
        """Test that empty sequence in target returns source as-is.

        This handles checkpoint restore scenarios where the current state has empty
        bins but the saved checkpoint has populated bins that need to be restored.
        """
        # Empty list target, should return loaded data
        loaded = [[1, 2], [3, 4, 5]]
        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=[]
        )
        assert result == [[1, 2], [3, 4, 5]]
        assert missing_paths == []

        # Empty tuple target
        loaded = ((1, 2), (3,))
        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=()
        )
        assert result == ((1, 2), (3,))
        assert missing_paths == []

        # Empty deque target
        loaded = deque([[1], [2, 3]])
        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=deque()
        )
        assert result == deque([[1], [2, 3]])
        assert missing_paths == []

        # Nested in dict: empty lists inside a dict
        loaded = {"bins": [[1, 2], [3]], "counts": [100, 50]}
        requested = {"bins": [], "counts": []}
        result, missing_paths = walk_checkpoint_structure(
            item_key="item", source=loaded, target=requested
        )
        assert result == {"bins": [[1, 2], [3]], "counts": [100, 50]}
        assert missing_paths == []
