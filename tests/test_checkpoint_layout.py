# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    merge_parts,
    RawSerialization,
    SafetensorsSerialization,
    serialization_format_from_dict,
    SerializationFormat,
    split_into_parts,
    TorchSerialization,
)


class _ExternalSerializationFormat(SerializationFormat):
    def to_dict(self) -> dict[str, object]:
        return {"type": type(self).__name__}

    @classmethod
    def from_dict(cls, d: dict[str, object]) -> "_ExternalSerializationFormat":
        return cls()


def test_supported_serialization_formats_are_exactly_the_builtins() -> None:
    assert SerializationFormat.supported_types() == (
        TorchSerialization,
        JsonSerialization,
        RawSerialization,
        SafetensorsSerialization,
    )
    assert _ExternalSerializationFormat not in SerializationFormat.supported_types()


@pytest.mark.parametrize(
    ("serialized", "expected"),
    [
        ({"type": "TorchSerialization"}, TorchSerialization()),
        (
            {"type": "JsonSerialization", "cls": "builtins.dict"},
            JsonSerialization(dict),
        ),
        ({"type": "RawSerialization"}, RawSerialization()),
        (
            {"type": "SafetensorsSerialization", "metadata": {"source": "test"}},
            SafetensorsSerialization(metadata={"source": "test"}),
        ),
    ],
)
def test_serialization_format_from_dict_preserves_builtin_behavior(
    serialized: dict[str, object], expected: SerializationFormat
) -> None:
    assert serialization_format_from_dict(serialized) == expected


@pytest.mark.parametrize(
    ("serialized", "message"),
    [
        ({}, "Unknown SerializationFormat type: None"),
        ({"type": []}, "Unknown SerializationFormat type: []"),
        (
            {"type": "UnknownSerialization"},
            "Unknown SerializationFormat type: UnknownSerialization",
        ),
    ],
)
def test_serialization_format_from_dict_preserves_unknown_type_error(
    serialized: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError) as error:
        serialization_format_from_dict(serialized)
    assert str(error.value) == message


def test_layout_names_one_file_per_part_and_round_trips_through_dict() -> None:
    layout = LayoutInfo("rank_0/optimizer.pt", TorchSerialization(), num_parts=3)

    assert layout.part_file_paths() == [
        "rank_0/optimizer.part0.pt",
        "rank_0/optimizer.part1.pt",
        "rank_0/optimizer.part2.pt",
    ]
    assert LayoutInfo.from_dict(layout.to_dict()) == layout
    # An unsplit layout serializes as it did before parts existed.
    assert LayoutInfo("model_0.pt", TorchSerialization()).to_dict() == {
        "file_path": "model_0.pt",
        "serialization_format": {"type": "TorchSerialization"},
    }


@pytest.mark.parametrize(
    ("serialization_format", "num_parts", "message"),
    [
        (TorchSerialization(), 0, "num_parts must be at least 1"),
        (SafetensorsSerialization(), 2, "num_parts > 1 requires TorchSerialization"),
    ],
)
def test_layout_rejects_parts_it_cannot_write(
    serialization_format: SerializationFormat, num_parts: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LayoutInfo("x.pt", serialization_format, num_parts=num_parts)


def test_split_into_parts_balances_tensor_bytes_and_merge_restores_the_dict() -> None:
    value = {
        "state": {
            "wq": {"exp_avg": torch.zeros(8), "exp_avg_sq": torch.zeros(8)},
            "wk": {"exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
        },
        "param_groups": [{"lr": 0.1}],
        "empty": {},
    }

    parts = split_into_parts(value, num_parts=2)

    # Largest first, each on the lighter part: 32 + 8 bytes on each.
    assert parts[0]["state"] == {
        "wq": {"exp_avg": value["state"]["wq"]["exp_avg"]},
        "wk": {"exp_avg": value["state"]["wk"]["exp_avg"]},
    }
    assert parts[1]["state"] == {
        "wq": {"exp_avg_sq": value["state"]["wq"]["exp_avg_sq"]},
        "wk": {"exp_avg_sq": value["state"]["wk"]["exp_avg_sq"]},
    }
    # Leaves without tensor bytes go to the lightest part, the first on a tie.
    assert parts[0]["param_groups"] is value["param_groups"]
    assert parts[0]["empty"] == {}

    merged = merge_parts(parts)
    assert merged.keys() == value.keys()
    for name in ("wq", "wk"):
        for moment in ("exp_avg", "exp_avg_sq"):
            assert merged["state"][name][moment] is value["state"][name][moment]
    assert merged["param_groups"] is value["param_groups"]
    assert merged["empty"] == {}


def test_split_into_parts_keeps_leaves_that_share_a_storage_in_one_part() -> None:
    # torch.save writes a view's whole storage, so a split storage would be written
    # once per part.
    buffer = torch.zeros(16)
    weight = torch.zeros(2)
    value = {
        "views": {str(index): view for index, view in enumerate(buffer.chunk(4))},
        "weight": weight,
        "tied_weight": weight,
    }

    parts = split_into_parts(value, num_parts=4)

    assert parts[0] == {"views": value["views"]}
    assert parts[1] == {"weight": weight, "tied_weight": weight}
    assert parts[2] == parts[3] == {}


class _WrapperTensor(torch.Tensor):
    """A wrapper subclass: its own storage is an empty placeholder."""

    @staticmethod
    def __new__(cls, inner: torch.Tensor) -> "_WrapperTensor":
        return torch.Tensor._make_wrapper_subclass(cls, inner.shape, dtype=inner.dtype)

    def __init__(self, inner: torch.Tensor) -> None:
        self.inner = inner

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        raise NotImplementedError(f"{func} is not supported by this test tensor")


def test_split_into_parts_weighs_a_wrapper_subclass_by_its_size() -> None:
    value = {
        "a": _WrapperTensor(torch.zeros(1024)),
        "b": _WrapperTensor(torch.zeros(1024)),
        "step": torch.zeros(1),
    }

    parts = split_into_parts(value, num_parts=2)

    # 4 KiB each, so one per part; the 4-byte step joins the first.
    assert parts == [{"a": value["a"], "step": value["step"]}, {"b": value["b"]}]


def test_split_into_parts_rejects_a_value_that_is_not_a_dict() -> None:
    with pytest.raises(ValueError, match="Only a dict can be split"):
        split_into_parts(torch.zeros(4), num_parts=2)
