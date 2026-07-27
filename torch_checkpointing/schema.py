# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Declarative checkpoint schema.

A `CheckpointManager` binds the per-item "how" -- ``requires_copy``, on-disk
``layout``, ``resharder``, and whether the item is ``required`` -- exactly once,
as a mapping of item key to `ItemSpec`. At save/load time the caller supplies a
plain payload ``Mapping[ItemKey, Any]``; `_CheckpointSchema.build_items` overlays
each value onto its spec to produce the ``dict[str, CheckpointItem]`` describing
what to persist.

Keeping the spec separate from the value keeps the schema rank-agnostic and
declared in one place, so resharding cannot be half-wired: declaring a resharder
on any item enables resharding support automatically on both save and load.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .checkpoint_base import CheckpointItem
from .checkpoint_layout import LayoutInfo
from .resharding import Resharder
from .types import ItemKey

# Placeholder in ``LayoutInfo.file_path`` that the manager materializes with the
# current global rank at save/load time, so a single rank-agnostic schema can
# still produce per-rank shard files.
RANK_PLACEHOLDER = "{rank}"


@dataclass(frozen=True)
class ItemSpec:
    """The per-item "how" for one checkpoint item, bound once in the schema.

    Mirrors the non-value fields of `CheckpointItem` and adds ``required``. The
    payload supplies the value at save/load time; the spec supplies everything
    else.

    Attributes:
        requires_copy (bool): Whether the value must be copied during async staging so
            the training loop can keep mutating it without a race. See
            `CheckpointItem`.
        layout (LayoutInfo | None): Where/how to store the item. ``None`` delegates the
            on-disk layout to the checkpoint engine's default per-rank policy. A
            ``"{rank}"`` placeholder in ``file_path`` is materialized per rank at
            save/load time.
        resharder (Resharder | None): Optional resharder for loading across a different sharding
            than was saved. When any item declares one, resharding is enabled
            transparently on both save and load. Resharding relies on metadata
            captured at save time, which is only written when an item declares a
            resharder -- declare it from the first save if the checkpoint may need
            to be resharded later, since a checkpoint saved without it cannot be
            resharded retroactively.
        required (bool): Whether the item must be present in the payload. Enforced
            only for keys explicitly declared in ``Config.items``; a ``default``
            spec's ``required`` has no effect, since there is no notion of a
            required un-named key. A missing required item raises when the payload
            is resolved against the schema.
    """

    requires_copy: bool = True
    layout: LayoutInfo | None = None
    resharder: Resharder | None = None
    required: bool = True


@dataclass(frozen=True)
class _CheckpointSchema:
    """Maps item keys to `ItemSpec`, with a ``default`` for un-named keys.

    Attributes:
        items (Mapping[ItemKey, ItemSpec]): Explicit per-key specs.
        default (ItemSpec | None): Spec applied to any payload key not in ``items``. ``None`` makes
            the schema strict: an un-named key raises instead of silently getting
            a permissive default.
    """

    items: Mapping[ItemKey, ItemSpec] = field(default_factory=dict)
    default: ItemSpec | None = field(default_factory=ItemSpec)

    def has_resharder(self) -> bool:
        """Report whether any spec declares a resharder.

        Returns:
            bool: True if any explicit spec or the default declares a resharder.
        """
        if any(spec.resharder is not None for spec in self.items.values()):
            return True
        return self.default is not None and self.default.resharder is not None

    def build_items(
        self, payload: Mapping[ItemKey, Any], *, rank: int
    ) -> dict[str, CheckpointItem]:
        """Overlay each payload value onto its spec, yielding engine CheckpointItems.

        Materializes ``{rank}`` in each item's ``layout.file_path`` and enforces
        that every ``required`` explicitly-named item is present in ``payload``.

        Args:
            payload (Mapping[ItemKey, Any]): key -> value to checkpoint; each value
                is overlaid onto the spec resolved for its key.
            rank (int): global rank; materializes the ``{rank}`` placeholder in each
                item's `layout.file_path`.

        Returns:
            dict[str, CheckpointItem]: the resolved checkpoint items, keyed by
            item key.

        Raises:
            KeyError: if any ``required`` explicitly-named item is absent from
                `payload`, or if the schema is strict (``default`` is None) and
                `payload` contains keys with no `ItemSpec`.
        """
        missing = [
            key
            for key, spec in self.items.items()
            if spec.required and key not in payload
        ]
        if missing:
            raise KeyError(f"Missing required checkpoint item(s): {sorted(missing)}")

        # Report every un-named key at once when the schema is strict, rather than
        # failing on whichever happens to be encountered first.
        if self.default is None:
            extra = sorted(set(payload) - set(self.items))
            if extra:
                raise KeyError(
                    f"No ItemSpec for checkpoint key(s) {extra} and the schema is "
                    "strict (default=None). Declare them in Config.items or set a "
                    "Config.default."
                )

        items: dict[str, CheckpointItem] = {}
        for key, value in payload.items():
            spec = self.items.get(key, self.default)
            # Guaranteed non-None: either the key is explicit, or the strict-schema
            # check above already rejected un-named keys when default is None.
            assert spec is not None
            items[key] = CheckpointItem(
                value=value,
                requires_copy=spec.requires_copy,
                layout=_materialize_layout(spec.layout, rank),
                resharder=spec.resharder,
            )
        return items


def _materialize_layout(layout: LayoutInfo | None, rank: int) -> LayoutInfo | None:
    """Substitute the ``{rank}`` placeholder in a layout's ``file_path``, if present."""
    if layout is None or RANK_PLACEHOLDER not in layout.file_path:
        return layout
    return dataclasses.replace(
        layout, file_path=layout.file_path.replace(RANK_PLACEHOLDER, str(rank))
    )
