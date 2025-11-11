# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test parallel attention"""

import argparse
import numpy as np

import mindspore as ms
from mindspore import nn, ops, mint, Tensor
from mindspore.common import dtype as mstype
from mindspore.communication import init

from mindscience.distributed.modules import ColumnParallelLinear, RowParallelLinear
from mindscience.distributed.manager import (
    initialize_parallel, get_tensor_parallel_world_size, get_context_parallel_group, get_context_parallel_rank
)
from mindscience.distributed.mappings import (
    AllToAllFromSequenceToHidden, AllToAllFromHiddenToSequence, GatherFromSequence
)

class Attention(nn.Cell):
    """standalone Attention"""
    def __init__(self, features_dim: int, num_heads: int):
        super().__init__()
        self.local_heads = num_heads

        self.w_qkv = mint.nn.Linear(features_dim, features_dim * 3)
        self.w_o = mint.nn.Linear(features_dim, features_dim)

        self.scale = (features_dim // num_heads) ** -0.5

    def construct(self, x: Tensor, bs: int):
        """Attention construct"""
        qkv = self.w_qkv(x)  # S * B, H
        num_nodes = qkv.shape[0] // bs

        qkv = qkv.reshape(num_nodes, bs, self.local_heads, 3, -1).permute(3, 1, 2, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, N, S, D
        q, k, v = q.astype(mstype.bfloat16), k.astype(mstype.bfloat16), v.astype(mstype.bfloat16)

        attn = ops.flash_attention_score(q, k, v, head_num=self.local_heads, scalar_value=self.scale,
                                         input_layout="BNSD", sparse_mode=0).astype(mstype.float32)
        attn = attn.permute(2, 0, 1, 3).reshape(num_nodes * bs, -1)

        out = self.w_o(attn)
        return out


class TensorParallelAttention(nn.Cell):
    """Attention with tensor parallel"""
    def __init__(self, features_dim: int, num_heads: int):
        super().__init__()
        self.local_heads = num_heads // get_tensor_parallel_world_size()

        self.w_qkv = ColumnParallelLinear(features_dim, features_dim * 3,
                                          gather_output=False, compute_dtype=mstype.float32)
        self.w_o = RowParallelLinear(features_dim, features_dim,
                                     input_is_parallel=True, compute_dtype=mstype.float32)

        self.scale = (features_dim // num_heads) ** -0.5

    def construct(self, x: Tensor, bs: int):
        """TensorParallelAttention construct"""
        qkv = self.w_qkv(x)  # S * B, H
        num_nodes = qkv.shape[0] // bs

        qkv = qkv.reshape(num_nodes, bs, self.local_heads, 3, -1).permute(3, 1, 2, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, N, S, D
        q, k, v = q.astype(mstype.bfloat16), k.astype(mstype.bfloat16), v.astype(mstype.bfloat16)

        attn = ops.flash_attention_score(q, k, v, head_num=self.local_heads, scalar_value=self.scale,
                                         input_layout="BNSD", sparse_mode=0).astype(mstype.float32)
        attn = attn.permute(2, 0, 1, 3).reshape(num_nodes * bs, -1)

        out = self.w_o(attn)
        return out


class UlyssesContextParallelAttention(nn.Cell):
    """Attention with ulysses context parallel"""
    def __init__(self, features_dim: int, num_heads: int):
        super().__init__()
        self.cp_group = get_context_parallel_group()
        self.local_heads = num_heads // self.cp_group.size

        self.w_qkv = mint.nn.Linear(features_dim, features_dim * 3)
        self.w_o = mint.nn.Linear(features_dim, features_dim)

        self.scale = (features_dim // num_heads) ** -0.5

        self.all_to_all_s2h = AllToAllFromSequenceToHidden.apply
        self.all_to_all_h2s = AllToAllFromHiddenToSequence.apply

    def construct(self, x: Tensor, bs: int):
        """UlyssesContextParallelAttention construct"""
        qkv = self.w_qkv(x)  # S * B, H
        qkv = self.all_to_all_s2h(qkv, self.cp_group)
        num_nodes = qkv.shape[0] // bs

        qkv = qkv.reshape(num_nodes, bs, self.local_heads, 3, -1).permute(3, 1, 2, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, N, S, D
        q, k, v = q.astype(mstype.bfloat16), k.astype(mstype.bfloat16), v.astype(mstype.bfloat16)

        attn = ops.flash_attention_score(q, k, v, head_num=self.local_heads, scalar_value=self.scale,
                                         input_layout="BNSD", sparse_mode=0).astype(mstype.float32)
        attn = attn.permute(2, 0, 1, 3).reshape(num_nodes * bs, -1)

        attn = self.all_to_all_h2s(attn, self.cp_group)
        out = self.w_o(attn)
        return out


class ColossalContextParallelAttention(nn.Cell):
    """Attention with colossal context parallel"""
    def __init__(self, features_dim: int, num_heads: int):
        super().__init__()
        self.cp_group = get_context_parallel_group()
        self.local_heads = num_heads

        self.w_qkv = mint.nn.Linear(features_dim, features_dim * 3)
        self.w_o = mint.nn.Linear(features_dim, features_dim)

        self.scale = (features_dim // num_heads) ** -0.5

        self.all_gather = GatherFromSequence.apply

    def construct(self, x: Tensor, bs: int):
        """ColossalContextParallelAttention construct"""
        qkv = self.w_qkv(x)  # S * B, H
        num_nodes = qkv.shape[0] // bs

        split_size = qkv.shape[-1] // (self.local_heads * 3)
        q, k, v = mint.split(qkv.reshape(qkv.shape[0], self.local_heads, -1), split_size, dim=-1)

        k, v = self.all_gather(k, self.cp_group), self.all_gather(v, self.cp_group)

        q = q.reshape(-1, bs, self.local_heads, split_size).permute(1, 2, 0, 3).astype(mstype.bfloat16)
        k = k.reshape(-1, bs, self.local_heads, split_size).permute(1, 2, 0, 3).astype(mstype.bfloat16)
        v = v.reshape(-1, bs, self.local_heads, split_size).permute(1, 2, 0, 3).astype(mstype.bfloat16)
        # B, N, S, D

        attn = ops.flash_attention_score(q, k, v, head_num=self.local_heads, scalar_value=self.scale,
                                         input_layout="BNSD", sparse_mode=0).astype(mstype.float32)
        attn = attn.permute(2, 0, 1, 3).reshape(num_nodes * bs, -1)

        out = self.w_o(attn)
        return out


def test_parallel_attention(args):
    """
    Feature: Parallel attention (TP / context-parallel variants).

    Description: Builds a reference Attention and a parallel variant (TP, ulysses-CP,
    or colossal-CP) according to args.parallel_type and compares outputs for equality
    on the corresponding slices or gathered outputs.

    Expectation: Parallelized attention outputs match the reference Attention within
    numeric tolerances for the selected parallel type and worker slice.
    """
    features_dim, num_heads = 256, 8

    ms.set_seed(2048)
    attention = Attention(features_dim=features_dim, num_heads=num_heads)

    ms.set_seed(2048)
    if args.parallel_type == "tp":
        initialize_parallel(tensor_parallel_size=args.num_workers)
        parallel_attention = TensorParallelAttention(features_dim=features_dim, num_heads=num_heads)
    elif args.parallel_type == "ulysses_cp":
        initialize_parallel(context_parallel_size=args.num_workers)
        parallel_attention = UlyssesContextParallelAttention(features_dim=features_dim, num_heads=num_heads)
    elif args.parallel_type == "colossal_cp":
        initialize_parallel(context_parallel_size=args.num_workers)
        parallel_attention = ColossalContextParallelAttention(features_dim=features_dim, num_heads=num_heads)

    bs, num_nodes = 16, 128
    x = Tensor(np.random.randn(bs * num_nodes, features_dim), mstype.float32)
    attention_out = attention(x, bs)

    if args.parallel_type == "tp":
        parallel_attention_out = parallel_attention(x, bs)
        assert mint.allclose(attention_out, parallel_attention_out, rtol=1e-3, atol=1e-3)
    else:
        cp_rank = get_context_parallel_rank()
        stride = bs * num_nodes // args.num_workers
        parallel_attention_out = parallel_attention(x[cp_rank*stride:(cp_rank+1)*stride], bs)
        assert mint.allclose(attention_out[cp_rank*stride:(cp_rank+1)*stride], parallel_attention_out,
                             rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_device("Ascend")
    init()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_workers", type=int, default=8, help="Number of worker")
    parser.add_argument("--parallel_type", type=str, required=True, choices=["tp", "ulysses_cp", "colossal_cp"],
                        help="Number of output features dim")
    args_ = parser.parse_args()

    test_parallel_attention(args_)
