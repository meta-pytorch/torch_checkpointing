# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Describing a Hugging Face checkpoint directory.

A Hugging Face export says what it holds either with an index naming the shard
each tensor lives in, or by putting everything in one ``model.safetensors``.
Neither carries sharding: every tensor is whole, so each is replicated on a
one-element mesh, and each shard file becomes a synthetic source rank so that
the per-rank file resolution used for native checkpoints works unchanged.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path, PurePosixPath

from ..checkpoint_layout import LayoutInfo, SafetensorsSerialization
from ..distributed_metadata import (
    DistributedItemMetadata,
    DistributedMetadata,
    GlobalObjectMetadata,
)
from ..dtensor_metadata import (
    DTensorShardingMetadata,
    get_device_mesh_spec,
    ReplicateSpec,
)
from ..metadata_serialization import DistributedMetadataFormat
from ..safetensors_metadata import SafetensorsFileMetadata
from ..serialized_tensor_slice import contiguous_strides, SerializedTensorSlice
from ..storage.base_storage import Storage
from ..types import NestedPath

HF_SAFETENSORS_INDEX_FILE_TEMPLATE = "{item_key}.safetensors.index.json"
_HF_SINGLE_FILE = "model.safetensors"


class HuggingFaceSafetensorsDistributedMetadataFormat(DistributedMetadataFormat):
    """Metadata synthesized from a Hugging Face safetensors export."""

    # Not rank addressable: each shard file holds whole tensors, so the
    # synthetic ranks below index files rather than readers, and no single file
    # answers a rank's request.

    @classmethod
    def maybe_load(
        cls,
        checkpoint_dir: Path,
        storage: Storage,
    ) -> DistributedMetadata | None:
        """Describe a Hugging Face directory under the canonical ``model`` item.

        Returns None when the directory does not follow the Hugging Face convention.
        A recognized Hugging Face checkpoint contains one unnamespaced state dict,
        represented as the canonical checkpoint item ``model``.
        """
        shard_names = _shard_names(checkpoint_dir, storage)
        if shard_names is None:
            return None

        # Each shard file stands in for a source rank, numbered by its position
        # here, so the file for a rank resolves as it does for a native checkpoint.
        headers = _read_headers(storage, checkpoint_dir, shard_names)
        nested_path_to_metadata: dict[NestedPath, list[GlobalObjectMetadata]] = {}
        for rank, header in enumerate(headers):
            for fqn, tensor in header.tensors.items():
                if (fqn,) in nested_path_to_metadata:
                    raise ValueError(
                        f"{fqn!r} appears in more than one Hugging Face shard "
                        f"in {checkpoint_dir}"
                    )
                nested_path_to_metadata[(fqn,)] = [
                    _shard_file_tensor_metadata(rank, tensor)
                ]
        if not nested_path_to_metadata:
            raise ValueError(
                f"Hugging Face safetensors metadata in {checkpoint_dir} contains no tensors"
            )

        return DistributedMetadata(
            world_size=len(shard_names),
            metadata={
                # TODO: Support loading into an item other than ``model``, e.g. by
                # remapping the item key where CheckpointManager builds the
                # CheckpointBase.
                "model": DistributedItemMetadata(
                    nested_path_to_metadata=nested_path_to_metadata,
                    rank_to_layout_info={
                        rank: LayoutInfo(name, SafetensorsSerialization())
                        for rank, name in enumerate(shard_names)
                    },
                )
            },
        )


def _shard_names(checkpoint_dir: Path, storage: Storage) -> list[str] | None:
    """Return the export's shard files, or None when it is not a Hugging Face export."""
    index_path = checkpoint_dir / HF_SAFETENSORS_INDEX_FILE_TEMPLATE.format(
        item_key="model"
    )
    if storage.exists(index_path):
        weight_map = json.loads(storage.read(index_path))["weight_map"]
        root = Path(os.path.normpath(checkpoint_dir))
        shard_names = set()
        for name in weight_map.values():
            # A shard must be a file inside the checkpoint directory. Normalizing
            # the joined path collapses "." and ".." and lets an absolute name
            # replace the directory, so anything that escapes it fails here.
            shard_path = Path(os.path.normpath(root / name))
            if shard_path == root or not shard_path.is_relative_to(root):
                raise ValueError(
                    f"Shard {name!r} in {index_path} is not a file inside "
                    f"{checkpoint_dir}"
                )
            # Equivalent spellings of one file name the same shard.
            shard_names.add(shard_path.relative_to(root).as_posix())
        return sorted(shard_names)
    if storage.exists(checkpoint_dir / _HF_SINGLE_FILE):
        return [_HF_SINGLE_FILE]
    return None


def _read_headers(
    storage: Storage,
    checkpoint_dir: Path,
    shard_names: list[str],
) -> list[SafetensorsFileMetadata]:
    """Read every shard's header in parallel; the result is indexed by rank."""
    paths = [str(checkpoint_dir / PurePosixPath(name)) for name in shard_names]
    with ThreadPoolExecutor(max_workers=max(1, len(paths))) as executor:
        return list(
            executor.map(
                partial(SafetensorsFileMetadata.from_file, storage),
                paths,
                range(len(paths)),
            )
        )


def _shard_file_tensor_metadata(
    shard_rank: int, tensor: SerializedTensorSlice
) -> GlobalObjectMetadata:
    """Describe a tensor stored whole in the shard file for ``shard_rank``."""
    return GlobalObjectMetadata(
        sharding_metadata=DTensorShardingMetadata(
            global_shape=tensor.slice_shape,
            dtype=str(tensor.torch_dtype),
            stride=contiguous_strides(tensor.slice_shape),
            mesh_spec=get_device_mesh_spec(
                device_type="cpu",
                mesh_shape=(1,),
                mesh_data=(shard_rank,),
            ),
            placements=(ReplicateSpec(),),
        ),
        ranks=(shard_rank,),
    )
