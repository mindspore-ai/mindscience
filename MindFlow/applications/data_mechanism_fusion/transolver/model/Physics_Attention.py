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
Physics Attention Module
Optimized for Ascend NPU (using matmul instead of einsum)
"""
import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.common.initializer import Orthogonal

class Physics_Attention_Structured_Mesh_2D(nn.Cell):
    """
    Physics Attention mechanism.
    Compatible with both Irregular and Structured meshes (flattened).
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(p=dropout)
        
        # Temperature parameter for Gumbel-Softmax approximation
        self.temperature = ms.Parameter(ops.ones((1, heads, 1, 1), ms.float32) * 0.5)

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
        """
        Args:
            x: Input tensor of shape (batch_size, num_points, channels)
        Returns:
            Output tensor of shape (batch_size, num_points, channels)
        """
        # b_size, num, C
        b_size, num, _ = x.shape

        ### (1) Slice Integration
        # Project and reshape
        # (B, N, H, D) -> (B, H, N, D)
        fx_mid = self.in_project_fx(x).reshape(b_size, num, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3)
        x_mid = self.in_project_x(x).reshape(b_size, num, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3)
            
        # Calculate slice weights
        # (B, H, N, D) -> (B, H, N, S) where S is slice_num
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)  # (B, H, S)

        # Calculate slice tokens using matmul instead of einsum for NPU compatibility
        # logical: einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        # implementation: (B, H, S, N) @ (B, H, N, D) -> (B, H, S, D)
        slice_weights_t = ops.transpose(slice_weights, (0, 1, 3, 2))
        slice_token = ops.matmul(slice_weights_t, fx_mid)
        
        # Normalize
        slice_token = slice_token / (slice_norm.expand_dims(-1) + 1e-5)

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        
        # Scaled Dot-Product Attention
        # (B, H, S, D) @ (B, H, D, S) -> (B, H, S, S)
        k_transposed = ops.transpose(k_slice_token, (0, 1, 3, 2))
        dots = ops.matmul(q_slice_token, k_transposed) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        
        # (B, H, S, S) @ (B, H, S, D) -> (B, H, S, D)
        out_slice_token = ops.matmul(attn, v_slice_token)

        ### (3) Deslice (Project back to original points)
        # logical: einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        # implementation: (B, H, N, S) @ (B, H, S, D) -> (B, H, N, D)
        out_x = ops.matmul(slice_weights, out_slice_token)
        
        # Reshape back to (B, N, C)
        # (B, H, N, D) -> (B, N, H, D) -> (B, N, H*D)
        out_x = ops.permute(out_x, (0, 2, 1, 3))
        out_x = out_x.reshape(b_size, num, -1)
        
        return self.to_out(out_x)
        