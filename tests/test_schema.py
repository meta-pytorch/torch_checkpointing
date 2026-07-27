# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import pytest
from torch_checkpointing.checkpoint_layout import LayoutInfo, TorchSerialization
from torch_checkpointing.resharding import Resharder
from torch_checkpointing.schema import _CheckpointSchema, ItemSpec


class _StubResharder(Resharder):
    """Minimal concrete Resharder so has_resharder() has something to detect."""

    def extract_sharding_metadata(
        self, item_key: str, item_value: Any
    ) -> dict[Any, Any]:
        return {}

    def load(
        self,
        source_path: Any,
        item_key: str,
        target_metadata: Any,
        source_metadata: Any,
        target: Any,
        storage: Any,
    ) -> list[Any]:
        return []


def test_reconcile_overlays_value_onto_spec() -> None:
    schema = _CheckpointSchema(items={"model": ItemSpec(requires_copy=True)})

    items = schema.build_items({"model": [1, 2, 3]}, rank=0)

    assert set(items) == {"model"}
    assert items["model"].value == [1, 2, 3]
    assert items["model"].requires_copy is True
    assert items["model"].resharder is None


def test_default_applies_to_unnamed_keys() -> None:
    schema = _CheckpointSchema(items={}, default=ItemSpec(requires_copy=False))

    items = schema.build_items({"step": 10}, rank=0)

    assert items["step"].value == 10
    assert items["step"].requires_copy is False


def test_strict_schema_rejects_unnamed_key() -> None:
    schema = _CheckpointSchema(items={"model": ItemSpec()}, default=None)

    with pytest.raises(KeyError, match="strict"):
        schema.build_items({"model": 1, "unexpected": 2}, rank=0)


def test_strict_schema_reports_all_unnamed_keys() -> None:
    schema = _CheckpointSchema(items={"model": ItemSpec()}, default=None)

    with pytest.raises(KeyError) as excinfo:
        schema.build_items({"model": 1, "beta": 2, "alpha": 3}, rank=0)

    message = str(excinfo.value)
    assert "alpha" in message and "beta" in message
    assert "strict" in message


def test_missing_required_item_raises() -> None:
    schema = _CheckpointSchema(items={"model": ItemSpec(required=True)})

    with pytest.raises(KeyError, match="required"):
        schema.build_items({"step": 10}, rank=0)


def test_optional_item_may_be_absent() -> None:
    schema = _CheckpointSchema(
        items={"model": ItemSpec(required=False)}, default=ItemSpec()
    )

    items = schema.build_items({"step": 10}, rank=0)

    assert set(items) == {"step"}


def test_rank_placeholder_is_materialized() -> None:
    schema = _CheckpointSchema(
        items={
            "model": ItemSpec(
                layout=LayoutInfo(
                    file_path="model_rank_{rank}.pt",
                    serialization_format=TorchSerialization(),
                )
            )
        }
    )

    items = schema.build_items({"model": 1}, rank=3)

    assert items["model"].layout is not None
    assert items["model"].layout.file_path == "model_rank_3.pt"


def test_layout_without_placeholder_is_unchanged() -> None:
    layout = LayoutInfo(file_path="model.pt", serialization_format=TorchSerialization())
    schema = _CheckpointSchema(items={"model": ItemSpec(layout=layout)})

    items = schema.build_items({"model": 1}, rank=3)

    assert items["model"].layout is layout


def test_none_layout_passes_through() -> None:
    schema = _CheckpointSchema(items={"model": ItemSpec(layout=None)})

    items = schema.build_items({"model": 1}, rank=0)

    assert items["model"].layout is None


def test_has_resharder() -> None:
    assert not _CheckpointSchema(items={"model": ItemSpec()}).has_resharder()
    assert _CheckpointSchema(
        items={"model": ItemSpec(resharder=_StubResharder())}
    ).has_resharder()
    assert _CheckpointSchema(
        default=ItemSpec(resharder=_StubResharder())
    ).has_resharder()
