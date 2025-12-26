# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, arguments-differ
"""
Transolver 2D Model Definition.
"""
import mindspore
from mindspore import nn, ops
import mindspore.numpy as mnp
from mindspore.common.initializer import initializer, TruncatedNormal, Constant, One
from .Embedding import timestep_embedding
from .Physics_Attention import Physics_Attention_Structured_Mesh_2D

def get_activation(act):
    """Return activation layer."""
    if act == 'gelu':
        return nn.GELU()
    if act == 'tanh':
        return nn.Tanh()
    if act == 'relu':
        return nn.ReLU()
    if act == 'silu':
        return nn.SiLU()
    if act == 'leaky_relu':
        return nn.LeakyReLU(0.1)
    return nn.GELU()

class MLP(nn.Cell):
    """Multi-Layer Perceptron."""
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        act_layer = get_activation(act)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res

        self.linear_pre = nn.SequentialCell([nn.Dense(n_input, n_hidden), act_layer])
        self.linear_post = nn.Dense(n_hidden, n_output)

        self.linears = nn.CellList([
            nn.SequentialCell([nn.Dense(n_hidden, n_hidden), act_layer])
            for _ in range(n_layers)
        ])

    def construct(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x

class TransolverBlock(nn.Cell):
    """Transolver Encoder Block."""
    def __init__(self, num_heads, hidden_dim, dropout, act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32, H=32, W=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm([hidden_dim])
        self.Attn = Physics_Attention_Structured_Mesh_2D(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num, H=H, W=W
        )
        self.ln_2 = nn.LayerNorm([hidden_dim])
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

        if self.last_layer:
            self.ln_3 = nn.LayerNorm([hidden_dim])
            self.mlp2 = nn.Dense(hidden_dim, out_dim)

    def construct(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx

class Model(nn.Cell):
    """Main Transolver Model."""
    def __init__(self, space_dim=2, n_layers=5, n_hidden=256, dropout=0.0, n_head=8,
                 Time_Input=False, act='gelu', mlp_ratio=1, fun_dim=1, out_dim=1,
                 slice_num=32, ref=8, unified_pos=False, H=32, W=32):
        super().__init__()
        self.H = H
        self.W = W
        self.ref = ref
        self.unified_pos = unified_pos
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim

        if self.unified_pos:
            self.pos = self.get_grid()
            self.preprocess = MLP(fun_dim + self.ref * self.ref,
                                  n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)
        else:
            self.preprocess = MLP(fun_dim + space_dim,
                                  n_hidden * 2, n_hidden, n_layers=0, res=False, act=act)

        if Time_Input:
            self.time_fc = nn.SequentialCell([
                nn.Dense(n_hidden, n_hidden), nn.SiLU(), nn.Dense(n_hidden, n_hidden)
            ])

        layers = []
        for i in range(n_layers):
            layers.append(TransolverBlock(
                num_heads=n_head, hidden_dim=n_hidden, dropout=dropout, act=act,
                mlp_ratio=mlp_ratio, out_dim=out_dim, slice_num=slice_num, H=H, W=W,
                last_layer=(i == n_layers - 1)
            ))
        self.blocks = nn.CellList(layers)

        self.placeholder = mindspore.Parameter(
            ops.rand((n_hidden,), dtype=mindspore.float32) * (1 / n_hidden)
        )
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer(TruncatedNormal(0.02),
                                                 cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer(Constant(0),
                                                   cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(initializer(One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer(Constant(0), cell.beta.shape, cell.beta.dtype))

    def get_grid(self, _batchsize=1):
        """Generate grid coordinates."""
        size_x, size_y = self.H, self.W
        x = mnp.linspace(0, 1, size_x)
        y = mnp.linspace(0, 1, size_y)
        grid_x, grid_y = mnp.meshgrid(x, y, indexing='ij')
        grid = ops.stack([grid_x, grid_y], axis=-1)

        xref = mnp.linspace(0, 1, self.ref)
        yref = mnp.linspace(0, 1, self.ref)
        grid_xref, grid_yref = mnp.meshgrid(xref, yref, indexing='ij')
        grid_ref = ops.stack([grid_xref, grid_yref], axis=-1)

        diff = grid[:, :, None, None, :] - grid_ref[None, None, :, :, :]
        dist = ops.sqrt(ops.reduce_sum(diff ** 2, axis=-1))

        pos = dist.reshape(1, size_x, size_y, -1)
        return pos

    def construct(self, x, fx, T=None):
        """
        Construct function for the Transolver Structured Mesh 2D Model.

        Args:
            x (Tensor): Input coordinates. Shape (batch_size, H*W, space_dim) if not unified_pos,
                        otherwise it's used to generate pos.
            fx (Tensor): Input features. Shape (batch_size, H*W, fun_dim).
            T (Tensor, optional): Time embedding input. Defaults to None.

        Returns:
            Tensor: Output features after processing through Transolver blocks.
        """
        if self.unified_pos:
            x = ops.tile(self.pos, (x.shape[0], 1, 1, 1))
            x = x.reshape(x.shape[0], self.H * self.W, -1)

        if fx is not None:
            fx = ops.concat((x, fx), -1)
            fx = self.preprocess(fx)
        else:
            fx = self.preprocess(x)
            fx = fx + self.placeholder[None, None, :]

        if T is not None and self.Time_Input:
            Time_emb = timestep_embedding(T, self.n_hidden)
            Time_emb = ops.tile(Time_emb.expand_dims(1), (1, x.shape[1], 1))
            Time_emb = self.time_fc(Time_emb)
            fx = fx + Time_emb

        for block in self.blocks:
            fx = block(fx)

        return fx
        