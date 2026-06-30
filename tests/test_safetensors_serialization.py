# Owner(s): ["oncall: distributed checkpointing"]

import os
import shutil
import tempfile
from typing import Generator

import pytest
import torch
from torch_checkpointing.checkpoint_base import (
    CheckpointItem,
    CheckpointReadInfo,
    CheckpointWriteInfo,
)
from torch_checkpointing.checkpoint_layout import (
    JsonSerialization,
    LayoutInfo,
    SafetensorsSerialization,
    serialization_format_from_dict,
)
from torch_checkpointing.checkpoint_reader import CheckpointReader
from torch_checkpointing.checkpoint_writer import (
    CheckpointWriter,
    CheckpointWriterArgs,
    CheckpointWriterConfig,
)
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.types import RankInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def rank_info() -> RankInfo:
    return RankInfo(
        global_rank=0,
        global_world_size=1,
        role_rank=0,
        role_world_size=1,
    )


@pytest.fixture
def storage_config() -> LocalFileSystemStorageConfig:
    return LocalFileSystemStorageConfig()


@pytest.fixture
def writer(
    rank_info: RankInfo, storage_config: LocalFileSystemStorageConfig
) -> CheckpointWriter:
    args = CheckpointWriterArgs(
        config=CheckpointWriterConfig(),
        rank_info=rank_info,
        storage_config=storage_config,
    )
    return CheckpointWriter(args=args)


@pytest.fixture
def reader(
    rank_info: RankInfo, storage_config: LocalFileSystemStorageConfig
) -> CheckpointReader:
    return CheckpointReader(rank_info=rank_info, storage_config=storage_config)


# ---------------------------------------------------------------------------
# SafetensorsSerialization class tests
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_roundtrip_no_metadata() -> None:
    fmt = SafetensorsSerialization()
    d = fmt.to_dict()
    assert d == {"type": "SafetensorsSerialization"}
    restored = SafetensorsSerialization.from_dict(d)
    assert restored == fmt
    assert restored.metadata is None


def test_to_dict_from_dict_roundtrip_with_metadata() -> None:
    fmt = SafetensorsSerialization(metadata={"format": "pt", "version": "1"})
    d = fmt.to_dict()
    assert d["type"] == "SafetensorsSerialization"
    assert d["metadata"] == {"format": "pt", "version": "1"}
    restored = SafetensorsSerialization.from_dict(d)
    assert restored == fmt


def test_factory_registration() -> None:
    d = {"type": "SafetensorsSerialization"}
    fmt = serialization_format_from_dict(d)
    assert isinstance(fmt, SafetensorsSerialization)
    assert fmt.metadata is None


def test_factory_registration_with_metadata() -> None:
    d = {"type": "SafetensorsSerialization", "metadata": {"key": "value"}}
    fmt = serialization_format_from_dict(d)
    assert isinstance(fmt, SafetensorsSerialization)
    assert fmt.metadata == {"key": "value"}


def test_frozen_and_hashable() -> None:
    fmt1 = SafetensorsSerialization()
    fmt2 = SafetensorsSerialization(metadata={"a": "b"})
    # Should be hashable (needed for use as dict keys / in sets)
    assert hash(fmt1) != hash(fmt2)
    s = {fmt1, fmt2}
    assert len(s) == 2


def test_hash_and_eq_are_order_independent() -> None:
    """dict equality is unordered, and our custom __hash__ matches that contract:
    two instances with the same items in different insertion order are equal AND
    hash to the same value (so they collapse to one entry in a set)."""
    a = SafetensorsSerialization(metadata={"x": "1", "y": "2"})
    b = SafetensorsSerialization(metadata={"y": "2", "x": "1"})
    assert a == b
    assert hash(a) == hash(b)
    assert len({a, b}) == 1


def test_eq_against_non_safetensors_type() -> None:
    """__eq__ should return NotImplemented (not raise, not falsely equal)
    when compared to a different type."""
    fmt = SafetensorsSerialization(metadata={"a": "1"})
    assert fmt != "SafetensorsSerialization"
    assert fmt != ({"a": "1"},)
    assert fmt != JsonSerialization(cls=dict)


def test_metadata_roundtrip_preserves_contents() -> None:
    """to_dict() -> from_dict() should preserve every (key, value) pair."""
    original = SafetensorsSerialization(metadata={"z_last": "1", "a_first": "2"})
    restored = SafetensorsSerialization.from_dict(original.to_dict())
    assert restored == original
    assert restored.metadata == {"z_last": "1", "a_first": "2"}


# ---------------------------------------------------------------------------
# SafetensorsSerialization.prepare_tensors_for_save helper tests
# ---------------------------------------------------------------------------


def test_flatten_simple_dict() -> None:
    data = {"weight": torch.randn(3, 4), "bias": torch.randn(3)}
    flat = SafetensorsSerialization.prepare_tensors_for_save(data)
    assert set(flat.keys()) == {"weight", "bias"}
    assert flat["weight"].shape == (3, 4)
    assert flat["bias"].shape == (3,)


def test_flatten_nested_dict() -> None:
    data = {
        "layer1": {"weight": torch.randn(2, 3), "bias": torch.randn(2)},
        "layer2": {"weight": torch.randn(4, 2)},
    }
    flat = SafetensorsSerialization.prepare_tensors_for_save(data)
    assert set(flat.keys()) == {"layer1.weight", "layer1.bias", "layer2.weight"}


def test_flatten_dtensor_unwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """DTensor input should be unwrapped to ``_local_tensor`` before flattening.

    Real DTensors need a process group to construct, so we monkeypatch
    ``torch.distributed.tensor.DTensor`` to a Tensor subclass we control. The
    function's runtime ``from torch.distributed.tensor import DTensor`` resolves
    against this patched binding, exercising the actual unwrap branch.
    """

    local_tensor = torch.randn(3, 4)

    class _FakeDTensor(torch.Tensor):
        # Must be a torch.Tensor subclass so the outer `isinstance(value, torch.Tensor)`
        # check fires before our `isinstance(value, DTensor)` unwrap check.
        @staticmethod
        def __new__(cls, local: torch.Tensor) -> "torch.Tensor":
            return torch.Tensor._make_subclass(cls, local.detach())

        def __init__(self, local: torch.Tensor) -> None:
            self._local_tensor = local

    monkeypatch.setattr("torch.distributed.tensor.DTensor", _FakeDTensor)

    fake_dt = _FakeDTensor(local_tensor)
    assert isinstance(fake_dt, torch.Tensor)  # sanity: required for the unwrap branch

    data = {"param": fake_dt}
    flat = SafetensorsSerialization.prepare_tensors_for_save(data)
    # The unwrap should return the underlying local tensor, NOT the DTensor wrapper.
    assert flat["param"] is local_tensor


def test_flatten_noncontiguous_tensor() -> None:
    # Create a non-contiguous tensor via transpose
    t = torch.randn(4, 3).t()
    assert not t.is_contiguous()
    data = {"param": t}
    flat = SafetensorsSerialization.prepare_tensors_for_save(data)
    assert flat["param"].is_contiguous()
    assert torch.equal(flat["param"], t)


def test_flatten_non_tensor_value_raises() -> None:
    data = {"param": torch.randn(2), "epoch": 5}
    with pytest.raises(
        ValueError, match="SafetensorsSerialization only supports tensor values"
    ):
        SafetensorsSerialization.prepare_tensors_for_save(data)


def test_flatten_non_tensor_string_raises() -> None:
    data = {"name": "model_v1"}
    with pytest.raises(
        ValueError, match="SafetensorsSerialization only supports tensor values"
    ):
        SafetensorsSerialization.prepare_tensors_for_save(data)


def test_flatten_empty_dict() -> None:
    flat = SafetensorsSerialization.prepare_tensors_for_save({})
    assert flat == {}


def test_flatten_list_of_tensors() -> None:
    """Lists of tensors should flatten using stringified indices joined with '.'."""
    a = torch.randn(2, 3)
    b = torch.randn(2, 3)
    data = {"params": [a, b]}
    flat = SafetensorsSerialization.prepare_tensors_for_save(data)
    assert set(flat.keys()) == {"params.0", "params.1"}
    assert torch.equal(flat["params.0"], a)
    assert torch.equal(flat["params.1"], b)


def test_flatten_tuple_of_tensors() -> None:
    """Tuples behave like lists for flattening."""
    a = torch.randn(4)
    b = torch.randn(4)
    flat = SafetensorsSerialization.prepare_tensors_for_save({"shards": (a, b)})
    assert set(flat.keys()) == {"shards.0", "shards.1"}


def test_flatten_dict_inside_list() -> None:
    """Mixed nesting: list elements that are dicts should keep flattening."""
    data = {"group": [{"weight": torch.randn(2)}, {"weight": torch.randn(3)}]}
    flat = SafetensorsSerialization.prepare_tensors_for_save(data)
    assert set(flat.keys()) == {"group.0.weight", "group.1.weight"}


# ---------------------------------------------------------------------------
# Write / Read integration tests
# ---------------------------------------------------------------------------


def test_write_read_roundtrip_flat_tensors(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    original = {"weight": torch.randn(5, 3), "bias": torch.randn(5)}
    checkpoint_path = os.path.join(temp_dir, "ckpt")

    sf_fmt = SafetensorsSerialization()
    write_items = {
        "model": CheckpointItem(
            value=original,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))

    read_items = {
        "model": CheckpointItem(
            value=None,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    loaded, missing = reader.read(
        checkpoint_path, CheckpointReadInfo(checkpoint_items=read_items)
    )
    assert missing == []
    assert set(loaded["model"].keys()) == {"weight", "bias"}
    assert torch.allclose(loaded["model"]["weight"], original["weight"])
    assert torch.allclose(loaded["model"]["bias"], original["bias"])


def test_write_read_roundtrip_nested_tensors_no_target_returns_flat(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    """When the reader is called with value=None (no target), there is no shape to
    re-nest into, so the safetensors-native flat dotted keys are returned as-is."""
    original = {
        "layer1": {"weight": torch.randn(4, 3), "bias": torch.randn(4)},
        "layer2": {"weight": torch.randn(2, 4)},
    }
    checkpoint_path = os.path.join(temp_dir, "ckpt_nested_no_target")

    sf_fmt = SafetensorsSerialization()
    write_items = {
        "model": CheckpointItem(
            value=original,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))

    read_items = {
        "model": CheckpointItem(
            value=None,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    loaded, missing = reader.read(
        checkpoint_path, CheckpointReadInfo(checkpoint_items=read_items)
    )
    assert missing == []
    loaded_model = loaded["model"]
    assert set(loaded_model.keys()) == {"layer1.weight", "layer1.bias", "layer2.weight"}
    assert torch.allclose(loaded_model["layer1.weight"], original["layer1"]["weight"])


def test_write_read_roundtrip_nested_tensors_with_target_renests(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    """When the reader is given a nested target, the flat safetensors output should be
    re-nested to match the target's shape — restoring what the user originally wrote."""
    original = {
        "layer1": {"weight": torch.randn(4, 3), "bias": torch.randn(4)},
        "layer2": {"weight": torch.randn(2, 4)},
    }
    checkpoint_path = os.path.join(temp_dir, "ckpt_nested_with_target")

    sf_fmt = SafetensorsSerialization()
    write_items = {
        "model": CheckpointItem(
            value=original,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))

    # Provide a target with the original nested shape. Use fresh tensors of the right
    # shape — walk_checkpoint_structure does in-place copy_() into these.
    target = {
        "layer1": {"weight": torch.empty(4, 3), "bias": torch.empty(4)},
        "layer2": {"weight": torch.empty(2, 4)},
    }
    read_items = {
        "model": CheckpointItem(
            value=target,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    loaded, missing = reader.read(
        checkpoint_path, CheckpointReadInfo(checkpoint_items=read_items)
    )
    assert missing == []
    loaded_model = loaded["model"]
    # Structure should match the nested target, NOT the flat safetensors-native shape.
    assert set(loaded_model.keys()) == {"layer1", "layer2"}
    assert set(loaded_model["layer1"].keys()) == {"weight", "bias"}
    assert set(loaded_model["layer2"].keys()) == {"weight"}
    assert torch.allclose(
        loaded_model["layer1"]["weight"], original["layer1"]["weight"]
    )
    assert torch.allclose(loaded_model["layer1"]["bias"], original["layer1"]["bias"])
    assert torch.allclose(
        loaded_model["layer2"]["weight"], original["layer2"]["weight"]
    )


def test_write_read_roundtrip_list_of_tensors(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    """Writing a dict containing a list of tensors should round-trip back into a
    dict-with-list shape when the target has the same shape."""
    original = {"params": [torch.randn(3, 2), torch.randn(3, 2)]}
    checkpoint_path = os.path.join(temp_dir, "ckpt_list")

    sf_fmt = SafetensorsSerialization()
    writer.write(
        checkpoint_path,
        CheckpointWriteInfo(
            checkpoint_items={
                "model": CheckpointItem(
                    value=original, layout=LayoutInfo("model.safetensors", sf_fmt)
                )
            }
        ),
    )

    # Read with a target that matches the original shape — should recover the list.
    target = {"params": [torch.empty(3, 2), torch.empty(3, 2)]}
    loaded, missing = reader.read(
        checkpoint_path,
        CheckpointReadInfo(
            checkpoint_items={
                "model": CheckpointItem(
                    value=target, layout=LayoutInfo("model.safetensors", sf_fmt)
                )
            }
        ),
    )
    assert missing == []
    assert isinstance(loaded["model"]["params"], list)
    assert len(loaded["model"]["params"]) == 2
    assert torch.allclose(loaded["model"]["params"][0], original["params"][0])
    assert torch.allclose(loaded["model"]["params"][1], original["params"][1])


def test_write_read_roundtrip_nested_partial_target_reports_missing(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    """A target requesting keys absent from the file should surface as missing_keys,
    not be silently swallowed by the re-nest step."""
    original = {"layer1": {"weight": torch.randn(2, 2)}}
    checkpoint_path = os.path.join(temp_dir, "ckpt_partial_target")

    sf_fmt = SafetensorsSerialization()
    writer.write(
        checkpoint_path,
        CheckpointWriteInfo(
            checkpoint_items={
                "model": CheckpointItem(
                    value=original, layout=LayoutInfo("model.safetensors", sf_fmt)
                )
            }
        ),
    )

    # Target asks for layer1.weight (present) AND layer2.weight (absent).
    target = {
        "layer1": {"weight": torch.empty(2, 2)},
        "layer2": {"weight": torch.empty(3, 3)},
    }
    loaded, missing = reader.read(
        checkpoint_path,
        CheckpointReadInfo(
            checkpoint_items={
                "model": CheckpointItem(
                    value=target, layout=LayoutInfo("model.safetensors", sf_fmt)
                )
            }
        ),
    )
    assert torch.allclose(
        loaded["model"]["layer1"]["weight"], original["layer1"]["weight"]
    )
    # The absent leaf should be reported, not silently dropped.
    assert any("layer2" in str(m) for m in missing), (
        f"expected layer2 to appear in missing_keys, got {missing}"
    )


def test_write_read_model_state_dict(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    model = torch.nn.Linear(10, 5)
    state_dict = model.state_dict()
    checkpoint_path = os.path.join(temp_dir, "ckpt_model")

    sf_fmt = SafetensorsSerialization()
    write_items = {
        "model": CheckpointItem(
            value=state_dict,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))

    read_items = {
        "model": CheckpointItem(
            value=None,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    loaded, missing = reader.read(
        checkpoint_path, CheckpointReadInfo(checkpoint_items=read_items)
    )
    assert missing == []
    assert torch.allclose(loaded["model"]["weight"], state_dict["weight"])
    assert torch.allclose(loaded["model"]["bias"], state_dict["bias"])


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16, torch.int32],
    ids=["float32", "float16", "bfloat16", "int32"],
)
def test_write_read_multiple_dtypes(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
    dtype: torch.dtype,
) -> None:
    original = {"param": torch.randn(4, 3).to(dtype)}
    checkpoint_path = os.path.join(temp_dir, f"ckpt_{dtype}")

    sf_fmt = SafetensorsSerialization()
    write_items = {
        "model": CheckpointItem(
            value=original,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))

    read_items = {
        "model": CheckpointItem(
            value=None,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    loaded, _ = reader.read(
        checkpoint_path, CheckpointReadInfo(checkpoint_items=read_items)
    )
    assert loaded["model"]["param"].dtype == dtype
    assert torch.equal(loaded["model"]["param"], original["param"])


def test_write_read_with_metadata(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    original = {"weight": torch.randn(3, 2)}
    checkpoint_path = os.path.join(temp_dir, "ckpt_meta")

    sf_fmt = SafetensorsSerialization(metadata={"format_version": "2"})
    write_items = {
        "model": CheckpointItem(
            value=original,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))

    # Verify file was written and can be loaded
    read_items = {
        "model": CheckpointItem(
            value=None,
            layout=LayoutInfo("model.safetensors", sf_fmt),
        ),
    }
    loaded, missing = reader.read(
        checkpoint_path, CheckpointReadInfo(checkpoint_items=read_items)
    )
    assert missing == []
    assert torch.allclose(loaded["model"]["weight"], original["weight"])

    # Verify metadata was written by loading with safetensors directly
    from safetensors import safe_open

    sf_path = os.path.join(checkpoint_path, "model.safetensors")
    with safe_open(sf_path, framework="pt") as f:
        assert f.metadata() is not None
        assert f.metadata()["format_version"] == "2"


def test_mixed_serialization_formats(
    temp_dir: str,
    writer: CheckpointWriter,
    reader: CheckpointReader,
) -> None:
    model_data = {"weight": torch.randn(4, 3), "bias": torch.randn(4)}
    epoch_data = 42
    checkpoint_path = os.path.join(temp_dir, "ckpt_mixed")

    write_items = {
        "model": CheckpointItem(
            value=model_data,
            layout=LayoutInfo("model.safetensors", SafetensorsSerialization()),
        ),
        "epoch": CheckpointItem(
            value=epoch_data,
            layout=LayoutInfo("epoch.json", JsonSerialization(int)),
        ),
    }
    writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))

    read_items = {
        "model": CheckpointItem(
            value=None,
            layout=LayoutInfo("model.safetensors", SafetensorsSerialization()),
        ),
        "epoch": CheckpointItem(
            value=None,
            layout=LayoutInfo("epoch.json", JsonSerialization(int)),
        ),
    }
    loaded, missing = reader.read(
        checkpoint_path, CheckpointReadInfo(checkpoint_items=read_items)
    )
    assert missing == []
    assert torch.allclose(loaded["model"]["weight"], model_data["weight"])
    assert loaded["epoch"] == epoch_data


# ---------------------------------------------------------------------------
# Error case tests
# ---------------------------------------------------------------------------


def test_write_non_tensor_raises(
    temp_dir: str,
    writer: CheckpointWriter,
) -> None:
    bad_data = {"weight": torch.randn(3), "epoch": 5}
    checkpoint_path = os.path.join(temp_dir, "ckpt_bad")

    write_items = {
        "model": CheckpointItem(
            value=bad_data,
            layout=LayoutInfo("model.safetensors", SafetensorsSerialization()),
        ),
    }
    with pytest.raises(
        ValueError, match="SafetensorsSerialization only supports tensor values"
    ):
        writer.write(checkpoint_path, CheckpointWriteInfo(checkpoint_items=write_items))
