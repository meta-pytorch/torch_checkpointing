# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pickle
from pathlib import Path

import pytest
import torch_checkpointing.metadata_serialization as metadata_serialization
from torch_checkpointing.checkpoint_layout import (
    default_torch_layout_info,
    LayoutInfo,
    TorchSerialization,
)
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    DistributedMetadata,
)
from torch_checkpointing.metadata_serialization import METADATA_FILE_NAME
from torch_checkpointing.storage.base_storage import Storage
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig


def _metadata() -> DistributedMetadata:
    return DistributedMetadata(
        metadata={
            "model": DistributedItemMetadata(
                nested_path_to_metadata={},
                rank_to_layout_info={0: default_torch_layout_info("model", 0)},
            )
        },
        world_size=1,
    )


def _storage() -> Storage:
    return LocalFileSystemStorageConfig(use_direct_io=False).create_storage()


def test_load_distributed_metadata_returns_native_metadata(tmp_path: Path) -> None:
    expected = _metadata()
    (tmp_path / METADATA_FILE_NAME).write_bytes(pickle.dumps(expected.to_dict()))

    assert (
        metadata_serialization.load_distributed_metadata(
            tmp_path,
            _storage(),
            formats=(metadata_serialization.TorchDistributedMetadataFormat,),
        )
        == expected
    )


def test_load_distributed_metadata_returns_none_when_no_format_matches(
    tmp_path: Path,
) -> None:
    assert (
        metadata_serialization.load_distributed_metadata(
            tmp_path,
            _storage(),
            formats=(),
        )
        is None
    )


def test_torch_distributed_metadata_format_does_not_hide_corrupt_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / METADATA_FILE_NAME).write_bytes(b"not a pickle")

    with pytest.raises(pickle.UnpicklingError):
        metadata_serialization.load_distributed_metadata(
            tmp_path,
            _storage(),
            formats=(metadata_serialization.TorchDistributedMetadataFormat,),
        )


def test_load_distributed_metadata_returns_first_match(tmp_path: Path) -> None:
    expected = _metadata()
    format_base = metadata_serialization.DistributedMetadataFormat

    class FirstDistributedMetadataFormat(format_base):
        @classmethod
        def maybe_load(
            cls,
            checkpoint_dir: Path,
            storage: Storage,
        ) -> DistributedMetadata | None:
            return expected

    class SecondDistributedMetadataFormat(format_base):
        @classmethod
        def maybe_load(
            cls,
            checkpoint_dir: Path,
            storage: Storage,
        ) -> DistributedMetadata | None:
            raise AssertionError("loader continued after the first match")

    assert (
        metadata_serialization.load_distributed_metadata(
            tmp_path,
            _storage(),
            formats=(
                FirstDistributedMetadataFormat,
                SecondDistributedMetadataFormat,
            ),
        )
        == expected
    )


def test_load_distributed_metadata_propagates_corrupt_format_error(
    tmp_path: Path,
) -> None:
    format_base = metadata_serialization.DistributedMetadataFormat

    class CorruptMetadataError(Exception):
        pass

    class CorruptDistributedMetadataFormat(format_base):
        @classmethod
        def maybe_load(
            cls,
            checkpoint_dir: Path,
            storage: Storage,
        ) -> DistributedMetadata | None:
            raise CorruptMetadataError("corrupt metadata")

    class MustNotRunDistributedMetadataFormat(format_base):
        @classmethod
        def maybe_load(
            cls,
            checkpoint_dir: Path,
            storage: Storage,
        ) -> DistributedMetadata | None:
            raise AssertionError("loader fell through after corruption")

    with pytest.raises(CorruptMetadataError, match="corrupt metadata"):
        metadata_serialization.load_distributed_metadata(
            tmp_path,
            _storage(),
            formats=(
                CorruptDistributedMetadataFormat,
                MustNotRunDistributedMetadataFormat,
            ),
        )


def test_native_format_is_rank_addressable_with_the_writer_default() -> None:
    format_type = metadata_serialization.TorchDistributedMetadataFormat
    assert issubclass(
        format_type, metadata_serialization.RankAddressableDistributedMetadataFormat
    )
    assert format_type.default_layout_info("model", 7) == LayoutInfo(
        "model_7.pt", TorchSerialization()
    )


def test_reader_formats_are_rank_addressable() -> None:
    assert all(
        issubclass(
            format_type, metadata_serialization.RankAddressableDistributedMetadataFormat
        )
        for format_type in CheckpointReader._METADATA_FORMATS
    )
