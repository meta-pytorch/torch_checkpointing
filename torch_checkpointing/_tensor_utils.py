# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

# The error PyTorch raises for the placeholder storage of a wrapper subclass. See
# Note [Invalid Python Storages] in PyTorch.
_PLACEHOLDER_STORAGE_ERROR = "invalid python storage"


# WORKAROUND: PyTorch has no public, non-raising way to ask whether a tensor's
# storage holds data: https://github.com/pytorch/pytorch/issues/203866. Replace
# this helper with the upstream API once PyTorch provides one.
def has_storage_data(tensor: torch.Tensor) -> bool:
    """
    Return whether ``tensor``'s storage holds data.

    This is False for wrapper tensor subclasses such as DTensor, whose storage is a
    placeholder with a null data pointer (their data lives in inner tensors held as
    attributes); for tensors without an accessible storage, such as sparse tensors;
    and for meta tensors and storages with no allocation, such as
    ``torch.empty(0)``'s.

    Do not use ``tensor.data_ptr() == 0`` for this. ``data_ptr()`` is the storage's
    pointer plus ``tensor.storage_offset() * tensor.element_size()``, so it is
    nonzero for a wrapper subclass with a nonzero storage offset (a DTensor whose
    local tensor is a view, for example), and zero for a tensor with
    ``numel() == 0`` that views a non-empty storage.

    ``UntypedStorage.data_ptr()`` gives the right answer, but raises for a
    non-empty placeholder storage, hence the ``try``. Note that it also
    materializes a copy-on-write storage.

    Raises:
        RuntimeError: If PyTorch forbids access to the storage's data for another
            reason, e.g. a CUDA graph output that a later replay has overwritten.
            Such a storage does hold data, so treating it as a placeholder would
            let callers silently skip it.
    """
    try:
        storage = tensor.untyped_storage()
    except NotImplementedError:
        # Sparse, MKLDNN and functorch batched tensors have no accessible storage.
        return False
    try:
        return storage.data_ptr() != 0
    except RuntimeError as e:
        if _PLACEHOLDER_STORAGE_ERROR in str(e):
            return False
        raise
