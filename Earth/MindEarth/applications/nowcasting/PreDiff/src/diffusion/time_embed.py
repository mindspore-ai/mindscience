# Copyright 2023 Huawei Technologies Co., Ltd
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
"TimeEmbedLayer and TimeEmbedResBlock"
from mindspore import nn, ops

from src.utils import conv_nd, apply_initialization, avg_pool_nd


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=padding
            )

    def construct(self, x):
        '''upsample forward'''
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = ops.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = ops.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Cell):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class TimeEmbedLayer(nn.Cell):
    """
    A neural network layer that embeds time information into a higher-dimensional space.

    The layer consists of two linear layers separated by a SiLU activation function.
    It takes an input tensor with a specified number of base channels and transforms it
    into a tensor with a specified number of time embedding channels.
    Parameters:
    - base_channels (int): Number of channels in the input tensor.
    - time_embed_channels (int): Number of channels in the output embedded tensor.
    - linear_init_mode (str, optional): Initialization mode for the linear layers. Defaults to "0".
    """

    def __init__(self, base_channels, time_embed_channels, linear_init_mode="0"):
        super().__init__()
        self.layer = nn.SequentialCell(
            nn.Dense(base_channels, time_embed_channels),
            nn.SiLU(),
            nn.Dense(time_embed_channels, time_embed_channels),
        )
        self.linear_init_mode = linear_init_mode

    def construct(self, x):
        """Forward pass through the TimeEmbedLayer."""
        return self.layer(x)

    def reset_parameters(self):
        """Reset the parameters of the linear layers in the TimeEmbedLayer."""
        apply_initialization(self.layer[0], linear_mode=self.linear_init_mode)
        apply_initialization(self.layer[2], linear_mode=self.linear_init_mode)


class TimeEmbedResBlock(nn.Cell):
    r"""
    Modifications:
    1. Change GroupNorm32 to use arbitrary `num_groups`.
    2. Add method `self.reset_parameters()`.
    3. Use gradient ckpt from mindspore instead of the stable diffusion implementation
    4. If no input time embed, it degrades to res block.
    """

    def __init__(
            self,
            channels,
            dropout,
            emb_channels=None,
            out_channels=None,
            use_conv=False,
            use_embed=True,
            use_scale_shift_norm=False,
            dims=2,
            up=False,
            down=False,
            norm_groups=32,
    ):
        r"""
        Parameters
        ----------
        channels
        dropout
        emb_channels
        out_channels
        use_conv
        use_embed:  bool
            include `emb` as input in `self.forward()`
        use_scale_shift_norm:   bool
            take effect only when `use_embed == True`
        dims
        up
        down
        norm_groups
        """
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.use_embed = use_embed
        if use_embed:
            assert isinstance(emb_channels, int)
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.SequentialCell(
            nn.GroupNorm(
                num_groups=norm_groups if channels % norm_groups == 0 else channels,
                num_channels=channels,
            ),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        if use_embed:
            self.emb_layers = nn.SequentialCell(
                nn.SiLU(),
                nn.Dense(
                    in_channels=emb_channels,
                    out_channels=(
                        2 * self.out_channels
                        if use_scale_shift_norm
                        else self.out_channels
                    ),
                ),
            )
        self.out_layers = nn.SequentialCell(
            nn.GroupNorm(
                num_groups=(
                    norm_groups
                    if self.out_channels % norm_groups == 0
                    else self.out_channels
                ),
                num_channels=self.out_channels,
            ),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            # nn.Dropout(p=0),
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1),
        )
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        self.reset_parameters()

    def construct(self, x, emb=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        Parameters
        ----------
        x: an [N x C x ...] Tensor of features.
        emb: an [N x emb_channels] Tensor of timestep embeddings.

        Returns
        -------
        out: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        if self.use_embed:
            emb_out = self.emb_layers(emb).astype(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = ops.chunk(emb_out, 2, axis=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)
        else:
            h = self.out_layers(h)
        n = self.skip_connection(x) + h
        return n

    def reset_parameters(self):
        for _, cell in self.cells_and_names():
            apply_initialization(cell)
        for p in self.out_layers[-1].get_parameters():
            p.set_data(ops.zeros(p.shape, dtype=p.dtype))
