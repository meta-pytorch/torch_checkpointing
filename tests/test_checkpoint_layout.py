# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    RawSerialization,
    SafetensorsSerialization,
    serialization_format_from_dict,
    SerializationFormat,
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
