# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def pin_memory(data_ptr: int, size: int) -> None:
    cudart = torch.cuda.cudart()
    assert cudart is not None, "CUDA runtime not available"
    succ = int(
        cudart.cudaHostRegister(
            data_ptr,
            size,
            1,  # lines up with 'cudaHostRegisterPortable'
        )
    )

    if succ != 0:
        raise RuntimeError(
            f"Registering memory failed with cudaError: {succ}."
            " It's possible that this is an asynchronous error raised from a previous cuda operation."
            " Consider launching with CUDA_LAUNCH_BLOCKING=1 to debug."
        )


def unpin_memory(data_ptr: int) -> None:
    cudart = torch.cuda.cudart()
    assert cudart is not None, "CUDA runtime not available"
    succ = int(cudart.cudaHostUnregister(data_ptr))
    if succ != 0:
        raise RuntimeError(
            f"Unpinning shared memory failed with cudaError: {succ}."
            " It's possible that this is an asynchronous error raised from a previous cuda operation."
            " Consider launching with CUDA_LAUNCH_BLOCKING=1 to debug."
        )
