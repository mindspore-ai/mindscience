# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes
"""
Physics Attention Mechanism implementations.
"""
import mindspore
from mindspore import nn, ops, Parameter
from mindspore.common.initializer import Orthogonal

class Physics_Attention_Irregular_Mesh(nn.Cell):
    """Irregular Mesh Attention.

    Args:
        dim (int): Input feature dimension.
        heads (int, optional): Number of attention heads. Default: 8.
        dim_head (int, optional): Dimension per attention head. Default: 64.
        dropout (float, optional): Dropout rate. Default: 0.0.
        slice_num (int, optional): Number of slicing groups. Default: 64.
    """
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = Parameter(ops.ones((1, heads, 1, 1), mindspore.float32) * 0.5)

        self.in_project_x = nn.Dense(dim, inner_dim)
        self.in_project_fx = nn.Dense(dim, inner_dim)
        self.in_project_slice = nn.Dense(dim_head, slice_num, weight_init=Orthogonal())

        self.to_q = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_k = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_v = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

    def construct(self, x):
        """Forward pass."""
        B, N, _ = x.shape
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head)
        fx_mid = ops.transpose(fx_mid, (0, 2, 1, 3))

        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head)
        x_mid = ops.transpose(x_mid, (0, 2, 1, 3))

        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)
        slice_norm = slice_weights.sum(2)

        slice_weights_t = ops.transpose(slice_weights, (0, 1, 3, 2))
        slice_token = ops.matmul(slice_weights_t, fx_mid)
        slice_token = slice_token / (slice_norm.expand_dims(-1) + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = ops.matmul(attn, v)

        out_x = ops.matmul(slice_weights, out_slice_token)
        out_x = ops.transpose(out_x, (0, 2, 1, 3)).reshape(B, N, -1)
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Cell):
    """Structured Mesh 2D Attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64, H=101, W=31, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = Parameter(ops.ones((1, heads, 1, 1), mindspore.float32) * 0.5)
        self.H = H
        self.W = W

        pad_val = kernel // 2
        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, stride=1,
                                      pad_mode='pad', padding=pad_val, has_bias=True)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, stride=1,
                                       pad_mode='pad', padding=pad_val, has_bias=True)
        self.in_project_slice = nn.Dense(dim_head, slice_num, weight_init=Orthogonal())

        self.to_q = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_k = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_v = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

    def construct(self, x):
        """Forward pass."""
        B, N, C = x.shape
        x_img = x.reshape(B, self.H, self.W, C)
        x_img = ops.transpose(x_img, (0, 3, 1, 2))

        fx_mid = self.in_project_fx(x_img)
        fx_mid = ops.transpose(fx_mid, (0, 2, 3, 1)).reshape(B, N, self.heads, self.dim_head)
        fx_mid = ops.transpose(fx_mid, (0, 2, 1, 3))

        x_mid = self.in_project_x(x_img)
        x_mid = ops.transpose(x_mid, (0, 2, 3, 1)).reshape(B, N, self.heads, self.dim_head)
        x_mid = ops.transpose(x_mid, (0, 2, 1, 3))

        temp_clamped = ops.clamp(self.temperature, 0.1, 5.0)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / temp_clamped)

        slice_norm = slice_weights.sum(2)
        slice_weights_t = ops.transpose(slice_weights, (0, 1, 3, 2))
        slice_token = ops.matmul(slice_weights_t, fx_mid)
        slice_token = slice_token / (slice_norm.expand_dims(-1) + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = ops.matmul(attn, v)

        out_x = ops.matmul(slice_weights, out_slice_token)
        out_x = ops.transpose(out_x, (0, 2, 1, 3)).reshape(B, N, -1)
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Cell):
    """Structured Mesh 3D Attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=32, H=32, W=32, D=32, kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(axis=-1)
        self.dropout = nn.Dropout(p=dropout)
        self.temperature = Parameter(ops.ones((1, heads, 1, 1), mindspore.float32) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        pad_val = kernel // 2
        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, stride=1,
                                      pad_mode='pad', padding=pad_val, has_bias=True)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, stride=1,
                                       pad_mode='pad', padding=pad_val, has_bias=True)
        self.in_project_slice = nn.Dense(dim_head, slice_num, weight_init=Orthogonal())

        self.to_q = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_k = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_v = nn.Dense(dim_head, dim_head, has_bias=False)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, dim),
            nn.Dropout(p=dropout)
        )

    def construct(self, x):
        """Forward pass."""
        B, N, C = x.shape
        x_vol = x.reshape(B, self.H, self.W, self.D, C)
        x_vol = ops.transpose(x_vol, (0, 4, 1, 2, 3))

        fx_mid = self.in_project_fx(x_vol)
        fx_mid = ops.transpose(fx_mid, (0, 2, 3, 4, 1)).reshape(B, N, self.heads, self.dim_head)
        fx_mid = ops.transpose(fx_mid, (0, 2, 1, 3))

        x_mid = self.in_project_x(x_vol)
        x_mid = ops.transpose(x_mid, (0, 2, 3, 4, 1)).reshape(B, N, self.heads, self.dim_head)
        x_mid = ops.transpose(x_mid, (0, 2, 1, 3))

        temp_clamped = ops.clamp(self.temperature, 0.1, 5.0)
        slice_weights = self.softmax(self.in_project_slice(x_mid) / temp_clamped)

        slice_norm = slice_weights.sum(2)
        slice_weights_t = ops.transpose(slice_weights, (0, 1, 3, 2))
        slice_token = ops.matmul(slice_weights_t, fx_mid)
        slice_token = slice_token / (slice_norm.expand_dims(-1) + 1e-5)

        q = self.to_q(slice_token)
        k = self.to_k(slice_token)
        v = self.to_v(slice_token)

        dots = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.dropout(self.softmax(dots))
        out_slice_token = ops.matmul(attn, v)

        out_x = ops.matmul(slice_weights, out_slice_token)
        out_x = ops.transpose(out_x, (0, 2, 1, 3)).reshape(B, N, -1)
        return self.to_out(out_x)
