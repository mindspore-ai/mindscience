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
"""transolver irregular mesh"""
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import initializer, TruncatedNormal
import numpy as np

from src.models.physics_attention import PhysicsAttentionIrregularMesh
from src.models.embedding import timestep_embedding


ACTIVATION = {'gelu': nn.GELU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU(0.1),
              'ELU': nn.ELU, 'silu': nn.SiLU}


class MLP(nn.Cell):
    """MLP"""
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()

        if act in ACTIVATION:
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.SequentialCell(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.CellList([nn.SequentialCell(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])

    def construct(self, x):
        """construct"""
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class TransolverBlock(nn.Cell):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm([hidden_dim])
        self.attn = PhysicsAttentionIrregularMesh(hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
                                                     dropout=dropout, slice_num=slice_num)
        self.ln_2 = nn.LayerNorm([hidden_dim])
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        if self.last_layer:
            self.ln_3 = nn.LayerNorm([hidden_dim])
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def construct(self, fx):
        fx = self.attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            fx = self.mlp2(self.ln_3(fx))
        return fx


class TransolverIrregular(nn.Cell):
    """TransolverIrregular"""
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 n_head=8,
                 time_input=False,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=1,
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False
                 ):
        super().__init__()
        self.ref = ref
        self.unified_pos = unified_pos
        self.time_input = time_input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        if self.unified_pos:
            self.preprocess = MLP(fun_dim + self.ref * self.ref, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim, n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        if time_input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden))

        self.blocks = nn.CellList([TransolverBlock(num_heads=n_head, hidden_dim=n_hidden,
                                                      dropout=dropout,
                                                      act=act,
                                                      mlp_ratio=mlp_ratio,
                                                      out_dim=out_dim,
                                                      slice_num=slice_num,
                                                      last_layer=i == (n_layers - 1))
                                     for i in range(n_layers)])
        self.initialize_weights()
        self.placeholder = ms.Parameter((1 / (n_hidden)) * ops.rand(n_hidden, dtype=ms.float32))

    def initialize_weights(self):
        """init"""
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """init"""
        if isinstance(m, nn.Linear):
            m.weight.set_data(initializer(TruncatedNormal(sigma=0.02), m.weight.shape, m.weight.dtype))
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.set_data(initializer(0, m.bias.shape, m.bias.dtype))

    def get_grid(self, x, batchsize=1):
        """get grid"""
        # x: B N 2
        # grid_ref
        gridx = ms.Tensor(np.linspace(0, 1, self.ref), dtype=ms.float32)
        gridx = gridx.reshape(1, self.ref, 1, 1).repeat([batchsize, 1, self.ref, 1])
        gridy = ms.Tensor(np.linspace(0, 1, self.ref), dtype=ms.float32)
        gridy = gridy.reshape(1, 1, self.ref, 1).repeat([batchsize, self.ref, 1, 1])
        grid_ref = ops.cat((gridx, gridy), dim=-1).reshape(batchsize, self.ref * self.ref, 2)  # B H W 8 8 2

        pos = ops.sqrt(ops.sum((x[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)). \
            reshape(batchsize, x.shape[1], self.ref * self.ref).contiguous()
        return pos

    def construct(self, x, fx, time=None):
        """construct"""
        if self.unified_pos:
            x = self.get_grid(x, x.shape[0])
        if fx is not None:
            fx = ops.cat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]

        if time is not None:
            time_emb = timestep_embedding(time, self.n_hidden).repeat(1, x.shape[1], 1)
            time_emb = self.time_fc(time_emb)
            fx = fx + time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
