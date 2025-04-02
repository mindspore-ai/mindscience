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
"""Diffusion transformer api"""

import math

import numpy as np
from mindspore import nn, ops, Tensor
from mindspore import dtype as mstype
from mindflow.cell import AttentionBlock


class Mlp(nn.Cell):
    """MLP"""
    def __init__(self, in_channels, out_channels, dropout=0., compute_dtype=mstype.float32):
        super().__init__()
        self.fc1 = nn.Dense(
            in_channels, 4*in_channels).to_float(compute_dtype)
        self.act = nn.GELU()
        self.fc2 = nn.Dense(
            4*in_channels, out_channels).to_float(compute_dtype)
        self.drop = nn.Dropout(p=dropout)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SinusoidalPosEmb(nn.Cell):
    """sinusoidal embedding model"""
    def __init__(self, dim, max_period=10000, compute_dtype=mstype.float32):
        super().__init__()
        half_dim = dim // 2
        self.concat_zero = (dim % 2 == 1)
        freqs = np.exp(-math.log(max_period) *
                       np.arange(start=0, stop=half_dim) / half_dim)
        self.freqs = Tensor(freqs, compute_dtype)

    def construct(self, x):
        emb = x[:, None] * self.freqs[None, :]
        emb = ops.concat((ops.cos(emb), ops.sin(emb)), axis=-1)
        if self.concat_zero:
            emb = ops.concat([emb, ops.zeros_like(emb[:, :1])], axis=-1)
        return emb


class Transformer(nn.Cell):
    """Transformer backbone model"""
    def __init__(self, hidden_channels, layers, heads, compute_dtype=mstype.float32):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.layers = layers
        self.blocks = nn.CellList([
            AttentionBlock(
                in_channels=hidden_channels,
                num_heads=heads,
                drop_mode="dropout",
                dropout_rate=0.0,
                compute_dtype=compute_dtype,
            )
            for _ in range(layers)])

    def construct(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class DiffusionTransformer(nn.Cell):
    r"""
    Diffusion model with Transformer backbone implementation.

    Args:
        in_channels (int): The number of input channel.
        out_channels (int): The number of output channel.
        hidden_channels (int): The number of hidden channel.
        layers (int): The number of transformer block layers.
        heads (int): The number of transformer heads.
        time_token_cond (bool): Whether to use timestep as condition token. Default: ``True``.
        compute_dtype (mindspore.dtype): The dtype of compute, it can be ``mstype.float32`` or ``mstype.float16``.
            Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - The input has a shape of :math:`(batch\_size, sequence\_len, in\_channels)`.
        - **timestep** (Tensor) - The timestep input has a shape of :math:`(batch\_size,)`.

    Outputs:
        - **output** (Tensor) - The output has a shape of :math:`(batch\_size, sequence\_len, out\_channels)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import DiffusionTransformer
        >>> in_channels, out_channels, hidden_channels, layers, heads, batch_size, seq_len = 16, 16, 256, 3, 4, 8, 256
        >>> model = DiffusionTransformer(in_channels=in_channels,
        ...                              out_channels=out_channels,
        ...                              hidden_channels=hidden_channels,
        ...                              layers=layers,
        ...                              heads=heads)
        >>> x = ops.rand((batch_size, seq_len, in_channels))
        >>> timestep = ops.randint(0, 1000, (batch_size,))
        >>> output = model(x, timestep)
        >>> print(output.shape)
        (8, 256, 16)
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels,
                 layers,
                 heads,
                 time_token_cond=True,
                 compute_dtype=mstype.float32):
        super().__init__()
        self.time_token_cond = time_token_cond
        self.in_channels = in_channels
        self.timestep_emb = SinusoidalPosEmb(
            hidden_channels, compute_dtype=compute_dtype)
        self.time_embed = Mlp(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            dropout=0.,
            compute_dtype=compute_dtype
        )

        self.ln_pre = nn.LayerNorm(
            (hidden_channels,), epsilon=1e-5).to_float(mstype.float32)
        self.backbone = Transformer(
            hidden_channels=hidden_channels,
            layers=layers,
            heads=heads,
            compute_dtype=compute_dtype
        )
        self.ln_post = nn.LayerNorm(
            (hidden_channels,), epsilon=1e-5).to_float(mstype.float32)
        self.input_proj = nn.Dense(
            in_channels, hidden_channels).to_float(compute_dtype)
        self.output_proj = nn.Dense(
            hidden_channels, out_channels, weight_init='zeros', bias_init='zeros').to_float(compute_dtype)

    def construct(self, x, timestep):
        """construct"""
        t_embed = self.time_embed(self.timestep_emb(timestep))
        return self._forward_with_cond(x, [(t_embed, self.time_token_cond)])

    def _forward_with_cond(self, x, cond_token_list):
        """forward network with condition input"""
        h = self.input_proj(x)
        extra_tokens = []
        for tokens, as_token in cond_token_list:
            if as_token:
                if len(tokens.shape) == 2:
                    extra_tokens.append(tokens.unsqueeze(1))
                else:
                    extra_tokens.append(tokens)
            else:
                h = h + tokens.unsqueeze(1)

        if extra_tokens:
            h = ops.concat(extra_tokens + [h], axis=1)

        h = self.ln_pre(h)
        h = self.backbone(h)
        h = self.ln_post(h)
        if extra_tokens:
            # keep sequence length unchanged
            h = h[:, sum(token.shape[1] for token in extra_tokens):]
        h = self.output_proj(h)
        return h


class ConditionDiffusionTransformer(DiffusionTransformer):
    r"""
    Conditioned Diffusion Transformer implementation.

    Args:
        in_channels (int): The number of input channel.
        out_channels (int): The number of output channel.
        hidden_channels (int): The number of hidden channel.
        cond_channels (int): The number of condition channel.
        layers (int): The number of transformer block layers.
        heads (int): The number of transformer heads.
        time_token_cond (bool): Whether to use timestep as condition token. Default: ``True``.
        cond_as_token (bool): Whether to use condition as token. Default: ``True``.
        compute_dtype (mindspore.dtype): the dtype of compute, it can be ``mstype.float32`` or ``mstype.float16``.
            Default: ``mstype.float32``, indicates ``mindspore.float32``.

    Inputs:
        - **x** (Tensor) - The input has a shape of :math:`(batch\_size, sequence\_len, in\_channels)`.
        - **timestep** (Tensor) - The timestep input has a shape of :math:`(batch\_size,)`.
        - **condition** (Tensor) - The condition input has a shape of :math:`(batch\_size, cond\_size)`.
          Default: ``None``.

    Outputs:
        - **output** (Tensor) - The output has a shape of :math:`(batch\_size, sequence\_len, out\_channels)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindspore import ops
        >>> from mindflow.cell import DiffusionTransformer
        >>> in_channels, out_channels, cond_channels, hidden_channels = 16, 16, 10, 256
        >>> layers, heads, batch_size, seq_len = 3, 4, 8, 256
        >>> model = DiffusionTransformer(in_channels=in_channels,
        ...                              out_channels=out_channels,
        ...                              hidden_channels=hidden_channels,
        ...                              layers=layers,
        ...                              heads=heads)
        >>> x = ops.rand((batch_size, seq_len, in_channels))
        >>> cond = ops.rand((batch_size, cond_channels))
        >>> timestep = ops.randint(0, 1000, (batch_size,))
        >>> output = model(x, timestep, cond)
        >>> print(output.shape)
        (8, 256, 16)
    """

    def __init__(self, in_channels,
                 out_channels,
                 cond_channels,
                 hidden_channels,
                 layers,
                 heads,
                 time_token_cond=True,
                 cond_as_token=True,
                 compute_dtype=mstype.float32):
        super().__init__(in_channels,
                         out_channels,
                         hidden_channels,
                         layers,
                         heads,
                         time_token_cond,
                         compute_dtype)
        self.cond_as_token = cond_as_token
        self.cond_embed = nn.Dense(
            cond_channels, hidden_channels).to_float(compute_dtype)

    # pylint: disable=W0221
    def construct(self, x, timestep, condition=None):
        """forward network with timestep and condition input """
        t_embed = self.time_embed(self.timestep_emb(timestep))
        full_cond = [(t_embed, self.time_token_cond)]
        if condition is not None:
            cond_emb = self.cond_embed(condition)
            full_cond.append((cond_emb, self.cond_as_token))
        return self._forward_with_cond(x, full_cond)
