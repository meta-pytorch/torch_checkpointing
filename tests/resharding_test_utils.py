from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor._api import DTensor
from torch_checkpointing.checkpoint_base import CheckpointInfo, CheckpointItem
from torch_checkpointing.checkpoint_layout import LayoutInfo
from torch_checkpointing.distributed_metadata import (
    DistributedItemMetadata,
    ShardingMetadata,
)
from torch_checkpointing.dtensor_metadata import (
    DeviceMeshSpec,
    DTensorShardingMetadata,
    ReplicateSpec,
)
from torch_checkpointing.resharding import Resharder
from torch_checkpointing.storage.base_storage import Storage
from torch_checkpointing.types import CheckpointPath, NestedPath
from torch_checkpointing.walk_utils import walk_checkpoint_structure


class SimpleDTensorResharder(Resharder):
    """Resharder for DTensors - extracts DTensorShardingMetadata from DTensors.

    Only returns metadata for DTensor objects, skips others.
    """

    def extract_sharding_metadata(
        self,
        item_key: str,
        item_value: Any,
    ) -> dict[NestedPath, ShardingMetadata]:
        result: dict[NestedPath, ShardingMetadata] = {}

        def _collect(path: CheckpointPath, obj: Any, _: Any) -> None:
            if isinstance(obj, DTensor):
                result[path.nested_path] = DTensorShardingMetadata.from_dtensor(obj)

        walk_checkpoint_structure(
            item_key=item_key,
            source=item_value,
            target=None,
            leaf_fn=_collect,
        )
        return result

    def load(
        self,
        source_path: Path,
        item_key: str,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
        target: Any,
        storage: Storage,
    ) -> list[NestedPath]:
        # Test resharder - detect missing paths in source
        # Collect all source paths from nested_path_to_metadata
        source_paths = source_metadata.get_nested_paths()

        # Return paths that are in target but not in source
        return [
            nested_path
            for nested_path in target_metadata.keys()
            if nested_path not in source_paths
        ]


class CustomObjectResharder(Resharder):
    """Resharder for objects implementing __pt_sharding_metadata__ protocol.

    Returns ShardingMetadata from __pt_sharding_metadata__ if available, skips others.
    """

    def extract_sharding_metadata(
        self,
        item_key: str,
        item_value: Any,
    ) -> dict[NestedPath, ShardingMetadata]:
        result: dict[NestedPath, ShardingMetadata] = {}

        def _collect(path: CheckpointPath, obj: Any, _: Any) -> None:
            if hasattr(obj, "__pt_sharding_metadata__"):
                sharding_metadata = obj.__pt_sharding_metadata__()
                if sharding_metadata is not None:
                    result[path.nested_path] = sharding_metadata

        walk_checkpoint_structure(
            item_key=item_key,
            source=item_value,
            target=None,
            leaf_fn=_collect,
        )
        return result

    def load(
        self,
        source_path: Path,
        item_key: str,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
        target: Any,
        storage: Storage,
    ) -> list[NestedPath]:
        # Test resharder - detect missing paths in source
        source_paths = source_metadata.get_nested_paths()
        return [
            nested_path
            for nested_path in target_metadata.keys()
            if nested_path not in source_paths
        ]


class SimpleResharder(Resharder):
    """Basic resharder for replicated tensors.

    For testing scenarios without sharding metadata.
    """

    def extract_sharding_metadata(
        self,
        item_key: str,
        item_value: Any,
    ) -> dict[NestedPath, ShardingMetadata]:
        result: dict[NestedPath, ShardingMetadata] = {}

        def _collect(path: CheckpointPath, obj: Any, _: Any) -> None:
            if isinstance(obj, torch.Tensor):
                # Use static DeviceMesh with Replicate() - same pattern as CustomTensorObject
                mesh_spec = DeviceMeshSpec(
                    device_type="cpu",
                    mesh_shape=(1,),
                    mesh_data=(0,),
                    mesh_dim_names=None,
                )
                result[path.nested_path] = DTensorShardingMetadata(
                    global_shape=tuple(obj.shape),
                    dtype=str(obj.dtype),
                    stride=tuple(obj.stride()),
                    mesh_spec=mesh_spec,
                    placements=(ReplicateSpec(),),
                )

        walk_checkpoint_structure(
            item_key=item_key,
            source=item_value,
            target=None,
            leaf_fn=_collect,
        )
        return result

    def load(
        self,
        source_path: Path,
        item_key: str,
        target_metadata: dict[NestedPath, ShardingMetadata],
        source_metadata: DistributedItemMetadata,
        target: Any,
        storage: Storage,
    ) -> list[NestedPath]:
        # Test resharder - detect missing paths in source
        source_paths = source_metadata.get_nested_paths()
        return [
            nested_path
            for nested_path in target_metadata.keys()
            if nested_path not in source_paths
        ]


@dataclass(frozen=True)
class CustomShardingMetadata(ShardingMetadata):
    """Custom sharding metadata for testing __pt_sharding_metadata__ protocol."""

    custom_type: str
    # Use tuple of key-value pairs instead of dict for hashability
    # Dict representation is converted in to_dict/from_dict
    custom_info: tuple[tuple[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "custom_type": self.custom_type,
            "custom_info": dict(self.custom_info),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CustomShardingMetadata":
        custom_info_dict = d["custom_info"]
        return CustomShardingMetadata(
            custom_type=d["custom_type"],
            custom_info=tuple(sorted(custom_info_dict.items())),
        )

    @property
    def equivalent_ranks(self) -> tuple[int, ...] | None:
        """Return None - no compaction for custom metadata."""
        return None


class CustomObject:
    """Custom object for testing __pt_sharding_metadata__ protocol with CustomShardingMetadata."""

    def __init__(self, data: Any, metadata_type: str = "test_object"):
        self.data = data
        self.metadata_type = metadata_type

    def __pt_sharding_metadata__(self) -> CustomShardingMetadata:
        custom_info = (
            ("data_type", type(self.data).__name__),
            ("value", str(self.data)),
        )
        return CustomShardingMetadata(
            custom_type=self.metadata_type,
            custom_info=custom_info,
        )


class CustomTensorObject:
    """Custom object that returns DTensorShardingMetadata via __pt_sharding_metadata__."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __pt_sharding_metadata__(self) -> DTensorShardingMetadata:
        mesh_spec = DeviceMeshSpec(
            device_type="cpu", mesh_shape=(1,), mesh_data=(0,), mesh_dim_names=None
        )
        return DTensorShardingMetadata(
            global_shape=tuple(self.tensor.shape),
            dtype=str(self.tensor.dtype),
            stride=tuple(self.tensor.stride()),
            mesh_spec=mesh_spec,
            placements=(ReplicateSpec(),),
        )


def make_checkpoint_item(
    value: Any,
    layout: LayoutInfo | None = None,
    resharder: Resharder | None = None,
) -> CheckpointItem:
    """Create CheckpointItem with automatic resharder selection based on value type.

    Args:
        value: The value to checkpoint.
        layout: Optional LayoutInfo for specifying file path and serialization format.
        resharder: Optional resharder. If not provided, automatically selects based on value type:
            - CustomObjectResharder for objects with __pt_sharding_metadata__
            - SimpleDTensorResharder for other objects (handles DTensors and unknowns)

    Returns:
        A CheckpointItem configured with the appropriate resharder.
    """
    if resharder is not None:
        return CheckpointItem(value=value, layout=layout, resharder=resharder)

    if hasattr(value, "__pt_sharding_metadata__"):
        selected_resharder = CustomObjectResharder()
    else:
        selected_resharder = SimpleDTensorResharder()
    return CheckpointItem(value=value, layout=layout, resharder=selected_resharder)


def make_checkpoint_info_from_dict(
    state_dict: dict[str, Any],
    layout_info_mappings: dict[str, LayoutInfo] | None = None,
) -> CheckpointInfo:
    """Create CheckpointInfo from a state_dict with appropriate resharders.

    Args:
        state_dict: Dictionary mapping keys to values to checkpoint.
        layout_info_mappings: Optional mapping of keys to LayoutInfo for custom layouts.

    Returns:
        A CheckpointInfo with appropriate resharders for each item.
    """
    items = {}
    for key, value in state_dict.items():
        layout = layout_info_mappings.get(key) if layout_info_mappings else None
        items[key] = make_checkpoint_item(value, layout)
    return CheckpointInfo(items)


def make_checkpoint_path(fqn: str) -> CheckpointPath:
    """Helper to create CheckpointPath from FQN string.

    Args:
        fqn: Fully qualified name string, e.g., "model.weight" or "step".

    Returns:
        CheckpointPath with item_key and nested_path parsed from the FQN.
    """
    parts = fqn.split(".")
    if len(parts) == 1:
        return CheckpointPath(item_key=parts[0], nested_path=())
    return CheckpointPath(item_key=parts[0], nested_path=tuple(parts[1:]))


def get_checkpoint_paths(checkpoint_info: CheckpointInfo) -> set[CheckpointPath]:
    """Extract all CheckpointPath objects from a CheckpointInfo.

    This traverses the checkpoint items and collects all paths,
    providing a reliable way to verify expected paths in tests.

    Args:
        checkpoint_info: The CheckpointInfo to traverse.

    Returns:
        Set of all CheckpointPath objects found in the checkpoint.
    """
    paths: set[CheckpointPath] = set()

    def collect_path(checkpoint_path: CheckpointPath, obj: Any, _target: Any) -> Any:
        paths.add(checkpoint_path)
        return obj

    for item_key, checkpoint_item in checkpoint_info.checkpoint_items.items():
        walk_checkpoint_structure(
            item_key=item_key,
            source=checkpoint_item.value,
            target=None,
            leaf_fn=collect_path,
        )

    return paths
