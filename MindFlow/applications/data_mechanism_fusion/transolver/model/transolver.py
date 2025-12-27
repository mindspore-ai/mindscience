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
# ==============================================================================
"""
Transolver Model for Structured Mesh
"""
import mindspore.nn as nn
from .embedding import Embedding2D
from .physics_attention import PhysicsAttention


class TransolverBlock(nn.Cell):
    """
    Transolver Block
    """
    def __init__(self, dim, heads, dim_head, mlp_ratio, dropout, slice_num, H=None, W=None):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-5)
        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-5)
        self.attn = PhysicsAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout,
                                     slice_num=slice_num)
        self.ffn = nn.SequentialCell(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(p=dropout)
        )

    def construct(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Transolver(nn.Cell):
    """
    Transolver Model for 2D Structured Mesh
    """
    def __init__(self, space_dim=2, n_layers=5, n_hidden=256, n_head=8, slice_num=32,
                 fun_dim=1, out_dim=1, H=32, W=32, unified_pos=False, ref=8,
                 mlp_ratio=1, dropout=0.0):
        super().__init__()
        self.preprocess = Embedding2D(fun_dim, n_hidden, unified_pos, H, W, ref)

        layers = []
        for _ in range(n_layers):
            layers.append(TransolverBlock(
                dim=n_hidden,
                heads=n_head,
                dim_head=n_hidden // n_head,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                slice_num=slice_num,
                H=H, W=W  
            ))
        self.blocks = nn.SequentialCell(layers)
        self.out_project = nn.Linear(n_hidden, out_dim)

    def construct(self, pos, x):
        x = self.preprocess(pos, x)
        x = self.blocks(x)
        x = self.out_project(x)
        return x
