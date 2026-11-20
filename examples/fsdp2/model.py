# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A small decoder-only transformer.

Adapted from the PyTorch FSDP2 example:
https://github.com/pytorch/examples/tree/main/distributed/FSDP2
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    n_layers: int = 4
    vocab_size: int = 256
    max_seq_len: int = 64
    dim: int = 128
    n_heads: int = 4


class Attention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.n_heads = args.n_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape

        def split_heads(t: torch.Tensor) -> torch.Tensor:
            # (bsz, seq_len, dim) -> (bsz, n_heads, seq_len, head_dim)
            return t.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            split_heads(self.wq(x)),
            split_heads(self.wk(x)),
            split_heads(self.wv(x)),
            is_causal=True,
        )
        return self.wo(out.transpose(1, 2).reshape(bsz, seq_len, dim))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(args.dim, hidden_dim=4 * args.dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))


class Transformer(nn.Module):
    """Maps a batch of token ids, shape (bsz, seq_len), to next-token logits."""

    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.layers = nn.ModuleList(
            TransformerBlock(args) for _ in range(args.n_layers)
        )
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, seq_len = tokens.shape
        assert seq_len <= self.max_seq_len
        positions = torch.arange(seq_len, device=tokens.device)
        h = self.tok_embeddings(tokens) + self.pos_embeddings(positions)
        for layer in self.layers:
            h = layer(h)
        # Logits in float32, so that the loss is computed in float32 even when
        # the model computes in bfloat16.
        return self.output(self.norm(h)).float()

    def reset_parameters(self) -> None:
        """Initialize every parameter, e.g. after ``to_empty()``."""
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.LayerNorm, nn.Linear)):
                module.reset_parameters()
