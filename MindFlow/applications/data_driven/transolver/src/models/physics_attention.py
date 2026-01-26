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
"physics attention"
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Orthogonal

class PhysicsAttentionIrregularMesh(nn.Cell):
    """for irregular meshes in 1D, 2D or 3D space."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = ms.Parameter(ops.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num, weight_init=Orthogonal())
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.SequentialCell(
            nn.Linear(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

    def construct(self, x):
        """construct"""
        # b_size num C
        b_size, num, _ = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(b_size, num, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # b_size H num C
        x_mid = self.in_project_x(x).reshape(b_size, num, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # b_size H num C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # b_size H num G
        slice_norm = slice_weights.sum(2)  # b_size H G
        # slice_token = mint.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_weights_t = ops.transpose(slice_weights, (0, 1, 3, 2))  # b h g n
        slice_token = ops.matmul(slice_weights_t, fx_mid)  # b h g n @ b h n c -> b h g c
        slice_token = slice_token / (slice_norm.expand_dims(-1) + 1e-5)

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = ops.matmul(q_slice_token, k_slice_token.transpose((0, 1, -1, -2))) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = ops.matmul(attn, v_slice_token)  # b_size H G D

        ### (3) Deslice
        out_x = ops.matmul(slice_weights, out_slice_token)  # bhng @ bhgc -> bhnc
        # out_x = mint.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = ops.permute(out_x, (0, 2, 1, 3))  # b n h c
        out_x = out_x.reshape(out_x.shape[0], out_x.shape[1], -1)
        return self.to_out(out_x)
