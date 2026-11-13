# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterator

import pytest
import torch
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.testing._internal.two_tensor import TwoTensor
from torch_checkpointing._tensor_utils import has_storage_data


@pytest.fixture
def fake_process_group() -> Iterator[None]:
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _dtensor(device: str, local_offset: int) -> DTensor:
    base = torch.arange(12, dtype=torch.float32, device=device)
    local = base[local_offset : local_offset + 4]
    return DTensor.from_local(
        local, init_device_mesh(device, (1,)), [Shard(0)], run_check=False
    )


def _fake_tensor() -> torch.Tensor:
    with FakeTensorMode():
        return torch.empty(4)


_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[
            pytest.mark.gpus_needed_1,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
        ],
    ),
]


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize(
    "make, expected",
    [
        pytest.param(lambda d: torch.arange(12.0, device=d), True, id="tensor"),
        pytest.param(lambda d: torch.arange(12.0, device=d)[2:6], True, id="view"),
        # data_ptr() is 0 for an empty view, but its storage holds data.
        pytest.param(
            lambda d: torch.arange(12.0, device=d)[2:2], True, id="empty_view"
        ),
        pytest.param(
            lambda d: torch.nn.Parameter(torch.ones(4, device=d)), True, id="parameter"
        ),
        pytest.param(lambda d: torch.empty(0, device=d), False, id="no_allocation"),
        pytest.param(lambda d: torch.eye(3, device=d).to_sparse(), False, id="sparse"),
        # Wrapper subclasses have a placeholder storage. data_ptr() is nonzero
        # when the wrapper has a nonzero storage offset.
        pytest.param(
            lambda d: TwoTensor(torch.ones(4, device=d), torch.ones(4, device=d)),
            False,
            id="wrapper_subclass",
        ),
        pytest.param(
            lambda d: TwoTensor(
                torch.arange(12.0, device=d)[4:8], torch.arange(12.0, device=d)[4:8]
            ),
            False,
            id="wrapper_subclass_with_offset",
        ),
    ],
)
def test_has_storage_data(
    device: str, make: Callable[[str], torch.Tensor], expected: bool
) -> None:
    assert has_storage_data(make(device)) is expected


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("local_offset", [0, 2])
def test_has_storage_data_for_dtensor(
    fake_process_group: None, device: str, local_offset: int
) -> None:
    dtensor = _dtensor(device, local_offset)
    assert dtensor.storage_offset() == local_offset
    assert not has_storage_data(dtensor)
    assert has_storage_data(dtensor.to_local())


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(lambda: torch.empty(4, device="meta"), id="meta"),
        pytest.param(_fake_tensor, id="fake"),
    ],
)
def test_has_storage_data_without_real_memory(
    make: Callable[[], torch.Tensor],
) -> None:
    assert not has_storage_data(make())


@pytest.mark.skipif(
    not hasattr(torch._C, "_set_storage_data_ptr_access_error_msg"),
    reason="forbidding access to a storage needs torch >= 2.14",
)
@pytest.mark.parametrize("make", [torch.Tensor, torch.nn.Parameter])
def test_has_storage_data_raises_when_access_to_real_data_is_forbidden(
    make: Callable[[torch.Tensor], torch.Tensor],
) -> None:
    # PyTorch can forbid access to a storage that holds data, e.g. a CUDA graph
    # output overwritten by a later replay. That is not a placeholder storage.
    tensor = make(torch.arange(4.0))
    torch._C._set_storage_data_ptr_access_error_msg(
        tensor.untyped_storage()._cdata, "access forbidden for this test"
    )
    with pytest.raises(RuntimeError, match="access forbidden for this test"):
        has_storage_data(tensor)
