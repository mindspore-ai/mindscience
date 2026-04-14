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
"resnet model"
from functools import partial
import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, nn, ops


class AvgPool1d(nn.Cell):
    """
    1D average pooling layer implementation with customizable kernel size, stride, and padding.
    Performs spatial downsampling by computing average values over sliding windows.
    """
    def __init__(self, kernel_size, stride=1, padding=0):
        """
        Initialize 1D average pooling parameters with validation checks.

        Args:
            kernel_size (int): Length of the pooling window
            stride (int): Stride size for window movement (default=1)
            padding (int): Zero-padding added to both sides of input (default=0)

        Raises:
            ValueError: If kernel_size ≤ 0, stride ≤ 0, or padding < 0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.mean = ops.ReduceMean(keep_dims=False)
        if stride <= 0:
            raise ValueError("stride must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if padding < 0:
            raise ValueError("padding must be non-negative")

    def construct(self, x):
        """
        Apply 1D average pooling to input tensor.
        """
        input_shape = x.shape
        n, c, l_in = input_shape[0], input_shape[1], input_shape[2]
        pad_left = self.padding
        pad_right = self.padding
        x = ops.Pad(((0, 0), (0, 0), (pad_left, pad_right)))(x)
        l_in += pad_left + pad_right
        l_out = (l_in - self.kernel_size) // self.stride + 1
        output = Tensor(np.zeros((n, c, l_out)), dtype=ms.float32)
        for i in range(l_out):
            start = i * self.stride
            end = start + self.kernel_size
            if end <= l_in:
                window = x[:, :, start:end]
                output[:, :, i] = self.mean(window, -1)

        return output


class Upsample1D(nn.Cell):
    """
    An upsampling layer with an optional convolution.

    Parameters:
            channels: channels in the inputs and outputs.
            use_conv: a bool determining if a convolution is applied.
            use_conv_transpose:
            out_channels:
    """

    def __init__(
            self,
            channels,
            use_conv=False,
            use_conv_transpose=False,
            out_channels=None,
            name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        self.conv = None
        if use_conv_transpose:
            self.conv = nn.Conv1dTranspose(
                channels,
                self.out_channels,
                kernel_size=4,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=True,
            )
        elif use_conv:
            self.conv = nn.Conv1d(
                self.channels,
                self.out_channels,
                3,
                padding=1,
                pad_mode="pad",
                has_bias=True,
            )

    def construct(self, x):
        """forward"""
        assert x.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(x)

        x = ops.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.use_conv:
            x = self.conv(x)

        return x


class Downsample1D(nn.Cell):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(
            self, channels, use_conv=False, out_channels=None, padding=1, name="conv"
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            self.conv = nn.Conv1d(
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
                pad_mode="pad",
                has_bias=True,
            )
        else:
            assert self.channels == self.out_channels
            self.conv = AvgPool1d(kernel_size=stride, stride=stride)

    def construct(self, x):
        assert x.shape[1] == self.channels
        return self.conv(x)


class Upsample2D(nn.Cell):
    """
    An upsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        use_conv_transpose:
        out_channels:
    """

    def __init__(
            self,
            channels,
            use_conv=False,
            use_conv_transpose=False,
            out_channels=None,
            name="conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name

        conv = None
        if use_conv_transpose:
            conv = nn.Conv2dTranspose(
                channels,
                self.out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                pad_mode="pad",
                has_bias=True,
            )
        elif use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=3,
                padding=1,
                pad_mode="pad",
                has_bias=True,
            )
        if name == "conv":
            self.conv = conv
        else:
            self.conv2d_0 = conv

    def construct(self, hidden_states, output_size=None):
        """forward"""
        assert hidden_states.shape[1] == self.channels

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        dtype = hidden_states.dtype
        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(ms.float32)
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()
        if output_size is None:
            hidden_states = ops.interpolate(
                hidden_states,
                scale_factor=2.0,
                recompute_scale_factor=True,
                mode="nearest",
            )
        else:
            hidden_states = ops.interpolate(
                hidden_states, size=output_size, mode="nearest"
            )

        if dtype == ms.bfloat16:
            hidden_states = hidden_states.to(dtype)
        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.conv2d_0(hidden_states)

        return hidden_states


class Downsample2D(nn.Cell):
    """
    A downsampling layer with an optional convolution.

    Parameters:
        channels: channels in the inputs and outputs.
        use_conv: a bool determining if a convolution is applied.
        out_channels:
        padding:
    """

    def __init__(
            self, channels, use_conv=False, out_channels=None, padding=1, name="conv"
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = nn.Conv2d(
                self.channels,
                self.out_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                pad_mode="pad",
                has_bias=True,
            )
        else:
            assert self.channels == self.out_channels
            conv = mint.nn.AvgPool2d(kernel_size=stride, stride=stride)
        if name == "conv":
            self.conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def construct(self, hidden_states):
        """forward"""
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            pad = (0, 1, 0, 1)
            hidden_states = ops.pad(hidden_states, pad, mode="constant", value=None)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states


class FirUpsample2D(nn.Cell):
    """
    2D upsampling layer with optional FIR filtering and convolutional projection.
    Implements pixel-shuffle based upsampling with optional convolutional transformation.
    """
    def __init__(
            self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)
    ):
        """
        Initialize upsample layer parameters.

        Args:
            channels (int): Number of input channels
            out_channels (int): Number of output channels (defaults to input channels if not specified)
            use_conv (bool): Whether to apply 3x3 convolution after upsampling
            fir_kernel (tuple): FIR filter kernel coefficients for antialiasing

        Raises:
            ValueError: If invalid kernel parameters are provided
        """
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.conv2d_0 = nn.Conv2d(
                channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=True,
            )
        self.use_conv = use_conv
        self.fir_kernel = fir_kernel
        self.out_channels = out_channels

    def _upsample_2d(self, hidden_states, weight=None, kernel=None, factor=2, gain=1):
        """
        Core upsampling operation with optional convolution and FIR filtering.
        """

        assert isinstance(factor, int) and factor >= 1

        # Setup filter kernel.
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = Tensor(kernel, dtype=ms.float32)
        if kernel.ndim == 1:
            kernel = ops.outer(kernel, kernel)
        kernel /= ops.sum(kernel)

        kernel = kernel * (gain * (factor**2))

        if self.use_conv:
            convh = weight.shape[2]
            convw = weight.shape[3]
            in_c = weight.shape[1]

            pad_value = (kernel.shape[0] - factor) - (convw - 1)

            stride = (factor, factor)
            # Determine data dimensions.
            output_shape = (
                (hidden_states.shape[2] - 1) * factor + convh,
                (hidden_states.shape[3] - 1) * factor + convw,
            )
            output_padding = (
                output_shape[0] - (hidden_states.shape[2] - 1) * stride[0] - convh,
                output_shape[1] - (hidden_states.shape[3] - 1) * stride[1] - convw,
            )
            assert output_padding[0] >= 0 and output_padding[1] >= 0
            num_groups = hidden_states.shape[1] // in_c

            # Transpose weights.
            weight = ops.reshape(weight, (num_groups, -1, in_c, convh, convw))
            weight = ops.flip(weight, dims=[3, 4]).permute(0, 2, 1, 3, 4)
            weight = ops.reshape(weight, (num_groups * in_c, -1, convh, convw))
            conv_transpose2d = nn.Conv2dTranspose(
                weight[0],
                weight[1],
                (weight[2], weight[3]),
                stride=stride,
                output_padding=output_padding,
                padding=0,
                pad_mode="pad",
            )
            inverse_conv = conv_transpose2d(hidden_states)

            output = upfirdn2d_native(
                inverse_conv,
                ms.tensor(kernel),
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2 + 1),
            )
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                ms.tensor(
                    kernel,
                ),
                up=factor,
                pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
            )

        return output

    def construct(self, hidden_states):
        """
        Apply upsampling transformation with optional convolutional projection.
        """
        if self.use_conv:
            height = self._upsample_2d(
                hidden_states, self.conv2d_0.weight, kernel=self.fir_kernel
            )
            height = height + self.conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            height = self._upsample_2d(hidden_states, kernel=self.fir_kernel, factor=2)

        return height


class FirDownsample2D(nn.Cell):
    """
    2D downsampling layer with optional FIR filtering and convolutional projection.
    Implements anti-aliased downsampling with optional 3x3 convolution.
    """
    def __init__(
            self, channels=None, out_channels=None, use_conv=False, fir_kernel=(1, 3, 3, 1)
    ):
        """
        Initialize downsampling layer parameters.
        Args:
            channels (int): Number of input channels
            out_channels (int): Number of output channels (defaults to input channels if not specified)
            use_conv (bool): Whether to apply 3x3 convolution before downsampling
            fir_kernel (tuple): FIR filter kernel coefficients for antialiasing

        Raises:
            ValueError: If invalid kernel parameters are provided
        """
        super().__init__()
        out_channels = out_channels if out_channels else channels
        if use_conv:
            self.conv2d_0 = nn.Conv2d(
                channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                pad_mode="pad",
                has_bias=True,
            )
        self.fir_kernel = fir_kernel
        self.use_conv = use_conv
        self.out_channels = out_channels

    def _downsample_2d(self, hidden_states, weight=None, kernel=None, factor=2, gain=1):
        """
        Core downsampling operation with optional convolution and FIR filtering.
        """
        assert isinstance(factor, int) and factor >= 1
        if kernel is None:
            kernel = [1] * factor

        # setup kernel
        kernel = ms.tensor(kernel, dtype=ms.float32)
        if kernel.ndim == 1:
            kernel = ms.outer(kernel, kernel)
        kernel /= ms.sum(kernel)

        kernel = kernel * gain

        if self.use_conv:
            _, _, _, convw = weight.shape
            pad_value = (kernel.shape[0] - factor) + (convw - 1)
            stride_value = [factor, factor]
            upfirdn_input = upfirdn2d_native(
                hidden_states,
                ms.tensor(kernel),
                pad=((pad_value + 1) // 2, pad_value // 2),
            )
            output = ops.conv2d(upfirdn_input, weight, stride=stride_value, padding=0)
        else:
            pad_value = kernel.shape[0] - factor
            output = upfirdn2d_native(
                hidden_states,
                ms.tensor(kernel),
                down=factor,
                pad=((pad_value + 1) // 2, pad_value // 2),
            )

        return output

    def construct(self, hidden_states):
        """
        Apply downsampling transformation with optional convolutional projection.
        """
        if self.use_conv:
            downsample_input = self._downsample_2d(
                hidden_states, weight=self.conv2d_0.weight, kernel=self.fir_kernel
            )
            hidden_states = downsample_input + self.conv2d_0.bias.reshape(1, -1, 1, 1)
        else:
            hidden_states = self._downsample_2d(
                hidden_states, kernel=self.fir_kernel, factor=2
            )

        return hidden_states


class ResnetBlock2D(nn.Cell):
    """
    2D ResNet block with optional time embeddings and spatial transformations.
    Implements pre-activation residual connections with optional upsampling/downsampling.
    """
    def __init__(
            self,
            *,
            in_channels,
            out_channels=None,
            conv_shortcut=False,
            dropout=0.0,
            temb_channels=512,
            groups=32,
            groups_out=None,
            pre_norm=True,
            eps=1e-6,
            non_linearity="swish",
            time_embedding_norm="default",
            kernel=None,
            output_scale_factor=1.0,
            use_in_shortcut=None,
            up=False,
            down=False,
    ):
        """
        Initialize ResNet block with configurable normalization and spatial transformations.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels (defaults to in_channels)
            conv_shortcut (bool): Use 1x1 convolution for shortcut connection
            dropout (float): Dropout probability (default=0)
            temb_channels (int): Time embedding dimension (default=512)
            groups (int): Number of groups for group normalization
            groups_out (int): Groups for second normalization layer (defaults to groups)
            pre_norm (bool): Apply normalization before non-linearity
            eps (float): Epsilon for numerical stability in normalization
            non_linearity (str): Activation function type ("swish", "mish", "silu")
            time_embedding_norm (str): Time embedding normalization mode ("default" or "scale_shift")
            kernel (str): Upsample/downsample kernel type ("fir", "sde_vp")
            output_scale_factor (float): Output scaling factor (default=1.0)
            use_in_shortcut (bool): Force shortcut connection usage
            up (bool): Enable upsampling transformation
            down (bool): Enable downsampling transformation

        Raises:
            ValueError: If invalid non_linearity or time_embedding_norm values are provided
        """
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.groups = groups
        self.in_channels = in_channels
        self.eps = eps
        if groups_out is None:
            groups_out = groups

        self.norm1 = nn.GroupNorm(
            num_groups=groups, num_channels=in_channels, eps=eps, affine=True
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            pad_mode="pad",
            has_bias=True,
        )

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(
                    f"unknown time_embedding_norm : {self.time_embedding_norm} "
                )

            self.time_emb_proj = nn.Dense(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = nn.GroupNorm(
            num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = mint.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if non_linearity == "swish":
            self.nonlinearity = ops.silu()
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.upsample = self.downsample = None
        if self.up:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.upsample = lambda x: upsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.upsample = partial(
                    ops.interpolate, scale_factor=2.0, mode="nearest"
                )
            else:
                self.upsample = Upsample2D(in_channels, use_conv=False)
        elif self.down:
            if kernel == "fir":
                fir_kernel = (1, 3, 3, 1)
                self.downsample = lambda x: downsample_2d(x, kernel=fir_kernel)
            elif kernel == "sde_vp":
                self.downsample = partial(mint.nn.AvgPool2d, kernel_size=2, stride=2)
            else:
                self.downsample = Downsample2D(
                    in_channels, use_conv=False, padding=1, name="op"
                )

        self.use_in_shortcut = (
            self.in_channels != self.out_channels
            if use_in_shortcut is None
            else use_in_shortcut
        )

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                pad_mode="pad",
                has_bias=True,
            )

    def construct(self, input_tensor, temb):
        """
        Forward pass of the ResNet block.

        Args:
            input_tensor (Tensor): Input tensor of shape (batch, channels, height, width).
            temb (Tensor): Optional time embedding tensor.

        Returns:
            Tensor: Output tensor after applying residual block operations.
        """
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        if self.upsample is not None:
            input_tensor = self.upsample(input_tensor)
            hidden_states = self.upsample(hidden_states)

        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor)
            hidden_states = self.downsample(hidden_states)
        hidden_states = self.conv1(hidden_states)
        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb
        hidden_states = self.norm2(hidden_states)
        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = ops.chunk(temb, 2, axis=1)
            hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        return output_tensor


class Mish(nn.Cell):
    """Implements the Mish activation function: x * tanh(softplus(x))."""
    def __init__(self):
        super().__init__()
        self.tanh = ops.Tanh()
        self.softplus = ops.Softplus()

    def construct(self, hidden_states):
        """Compute Mish activation on input tensor."""
        return hidden_states * self.tanh(self.softplus(hidden_states))


def rearrange_dims(tensor):
    """
    Adjust tensor dimensions based on input shape:
    - 2D → add two singleton dimensions
    - 3D → add one singleton dimension
    - 4D → squeeze spatial dimensions

    Args:
        tensor (Tensor): Input tensor.

    Returns:
        Tensor: Dimension-adjusted tensor.

    Raises:
        ValueError: If input tensor has invalid dimensions.
    """
    if len(tensor.shape) == 2:
        return tensor[:, :, None]
    if len(tensor.shape) == 3:
        return tensor[:, :, None, :]
    if len(tensor.shape) == 4:
        return tensor[:, :, 0, :]
    raise ValueError(f"`len(tensor)`: {len(tensor)} has to be 2, 3 or 4.")


class Conv1dBlock(nn.Cell):
    """
    1D Convolution block with GroupNorm and Mish activation.

    Args:
        inp_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size.
        n_groups (int): Number of groups for GroupNorm. Defaults to 8.
    """
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.conv1d = nn.Conv1d(
            inp_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            has_bias=True,
            pad_mode="valid",
        )
        self.group_norm = nn.GroupNorm(n_groups, out_channels)
        self.mish = ops.mish()

    def construct(self, x):
        """Apply convolution, normalization, dimension rearrangement and activation."""
        x = self.conv1d(x)
        x = rearrange_dims(x)
        x = self.group_norm(x)
        x = rearrange_dims(x)
        x = self.mish(x)
        return x


class ResidualTemporalBlock1D(nn.Cell):
    """ResidualTemporalBlock1D"""
    def __init__(self, inp_channels, out_channels, embed_dim, kernel_size=5):
        super().__init__()
        self.conv_in = Conv1dBlock(inp_channels, out_channels, kernel_size)
        self.conv_out = Conv1dBlock(out_channels, out_channels, kernel_size)

        self.time_emb_act = nn.Mish()
        self.time_emb = nn.Linear(embed_dim, out_channels)

        self.residual_conv = (
            nn.Conv1d(inp_channels, out_channels, 1, has_bias=True, pad_mode="valid")
            if inp_channels != out_channels
            else nn.Identity()
        )

    def construct(self, x, t):
        """
        Args:
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]

        returns:
            out : [ batch_size x out_channels x horizon ]
        """
        t = self.time_emb_act(t)
        t = self.time_emb(t)
        out = self.conv_in(x) + rearrange_dims(t)
        out = self.conv_out(out)
        return out + self.residual_conv(x)


def upsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """Upsample2D a batch of 2D images with the given filter."""
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = ms.tensor(kernel, dtype=ms.float32)
    if kernel.ndim == 1:
        kernel = ms.outer(kernel, kernel)
    kernel /= ms.sum(kernel)

    kernel = kernel * (gain * (factor**2))
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        up=factor,
        pad=((pad_value + 1) // 2 + factor - 1, pad_value // 2),
    )
    return output


def downsample_2d(hidden_states, kernel=None, factor=2, gain=1):
    """Downsample2D a batch of 2D images with the given filter."""
    assert isinstance(factor, int) and factor >= 1
    if kernel is None:
        kernel = [1] * factor

    kernel = ms.tensor(kernel, dtype=ms.float32)
    if kernel.ndim == 1:
        kernel = ms.outer(kernel, kernel)
    kernel /= ms.sum(kernel)

    kernel = kernel * gain
    pad_value = kernel.shape[0] - factor
    output = upfirdn2d_native(
        hidden_states,
        down=factor,
        pad=((pad_value + 1) // 2, pad_value // 2),
    )
    return output


def upfirdn2d_native(tensor, kernel=None, up=1, down=1, pad=(0, 0)):
    """upfirdn2d native"""
    up_x = up_y = up
    down_x = down_y = down
    pad_x0 = pad_y0 = pad[0]
    pad_x1 = pad_y1 = pad[1]

    _, channel, in_h, in_w = tensor.shape
    tensor = tensor.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = tensor.shape
    kernel_h, kernel_w = kernel.shape

    out = tensor.view(-1, in_h, 1, in_w, 1, minor)
    out = ops.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = ops.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = ms.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = ops.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)
