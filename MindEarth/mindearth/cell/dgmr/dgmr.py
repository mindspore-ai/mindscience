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
"""basic"""
from __future__ import absolute_import

import numpy as np

import mindspore as ms
from mindspore import set_seed
import mindspore.nn.probability.distribution as msd
from mindspore import nn, ops, Tensor, Parameter

from mindearth.cell.utils import SpectralNorm, PixelUnshuffle, PixelShuffle


set_seed(0)
np.random.seed(0)


def get_conv_layer(conv_type="standard"):
    if conv_type == "standard":
        conv_layer = nn.Conv2d
    elif conv_type == "coord":
        conv_layer = CoordConv
    elif conv_type == "3d":
        conv_layer = nn.Conv3d
    else:
        raise ValueError(f"{conv_type} is not a recognized Conv method")
    return conv_layer


class ConvGRUCell(nn.Cell):
    """
    ConvGRU cell of Dgmr generator.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 sn_eps=0.,
                 use_spectral_norm=False):
        super().__init__()
        self._kernel_size = kernel_size
        self._sn_eps = sn_eps

        self.read_gate_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            pad_mode="pad",
            padding=1,
            has_bias=True
        )
        if use_spectral_norm:
            self.read_gate_conv = SpectralNorm(
                self.read_gate_conv,
                eps=sn_eps,
            )

        self.update_gate_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            pad_mode="pad",
            padding=1,
            has_bias=True)
        if use_spectral_norm:
            self.update_gate_conv = SpectralNorm(
                self.update_gate_conv,
                eps=sn_eps,
            )

        self.output_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size),
            pad_mode="pad",
            padding=1,
            has_bias=True
        )
        if use_spectral_norm:
            self.output_conv = SpectralNorm(
                self.output_conv,
                eps=sn_eps,
            )

    def construct(self, x, prev_state):
        """ConvGRUCell forward function"""
        # Concatenate the inputs and previous state along the channel axis.
        concat_op = ops.Concat(axis=1)
        xh = concat_op((x, prev_state))

        # Read gate of the GRU.
        read_gate = ops.sigmoid(self.read_gate_conv(xh))

        # Update gate of the GRU.
        update_gate = ops.sigmoid(self.update_gate_conv(xh))

        # Gate the inputs.
        gated_input = concat_op((x, read_gate * prev_state))

        # Gate the cell and state / outputs.
        relu = ops.ReLU()
        c = relu(self.output_conv(gated_input))
        out = update_gate * prev_state + (1.0 - update_gate) * c
        new_state = out
        return out, new_state


class ConvGRU(nn.Cell):
    """
    ConvGRU used in Dgmr generation steps.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 sn_eps=0.0001):
        super().__init__()
        self.cell = ConvGRUCell(in_channels, out_channels, kernel_size, sn_eps)

    def construct(self, x: Tensor, hidden_state=None) -> Tensor:
        """ConvGRU forward function"""
        outputs = []
        for step in range(len(x)):
            # Compute current timestep
            output, hidden_state = self.cell(x[step], hidden_state)
            outputs.append(output)
        # Stack outputs to return as tensor
        output = ops.stack(outputs, 0)
        return output


class AddCoords(nn.Cell):
    """
    AddCoords in CoordConv module.
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def construct(self, input_tensor):
        """AddCoords forward function"""
        batch_size, _, x_dim, y_dim = input_tensor.shape

        xx_channel = ms.numpy.tile(ms.numpy.arange(x_dim), (1, y_dim, 1))
        yy_channel = ms.numpy.tile(ms.numpy.arange(y_dim), (1, x_dim, 1)).transpose((0, 2, 1))

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = ms.numpy.tile(xx_channel, (batch_size, 1, 1, 1)).transpose((0, 1, 3, 2))
        yy_channel = ms.numpy.tile(yy_channel, (batch_size, 1, 1, 1)).transpose((0, 1, 3, 2))

        concat_op = ops.Concat(axis=1)
        ret = concat_op(
            (input_tensor, ops.cast(xx_channel, input_tensor.dtype), ops.cast(yy_channel, input_tensor.dtype))
        )
        if self.with_r:
            rr = ops.sqrt(
                ops.pow(ops.cast(xx_channel, input_tensor.dtype) - 0.5, 2)
                + ops.pow(ops.cast(yy_channel, input_tensor.dtype) - 0.5, 2)
            )
            ret = concat_op((ret, rr))

        return ret


class CoordConv(nn.Cell):
    """
    An alternative implementation for mindspore with auto-infering the x-y dimensions.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_r=False,
                 **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def construct(self, x):
        """CoordConv forward function"""
        ret = self.addcoords(x)
        output = self.conv(ret)
        return output


def attention_einsum(q, k, v):
    """
    Self attention function.
    """
    # Reshape 3D tensors to 2D tensor with first dimension L = h x w.
    # [h, w, c] -> [L, c]
    k = ops.reshape(k, (k.shape[0] * k.shape[1], k.shape[2]))
    # [h, w, c] -> [L, c]
    v = ops.reshape(v, (v.shape[0] * v.shape[1], v.shape[2]))

    q0 = q.shape[0]
    k0 = k.shape[0]
    k1 = k.shape[1]
    tmp = Tensor(np.ones((q0, k1, k0)))
    k = k.T
    beta = ops.Softmax(axis=-1)(ops.bmm(q, k.expand_as(tmp)))

    # Einstein summation corresponding to the attention * value operation.
    tmp = Tensor(np.ones((q0, k0, k1)))
    output = ops.bmm(beta, v.expand_as(tmp))
    return output


class AttentionLayer(nn.Cell):
    """
    Attention Module in LatentConditioningStack.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ratio_kq=8,
                 ratio_v=8):
        super().__init__()

        self.ratio_kq = ratio_kq
        self.ratio_v = ratio_v
        self.out_channels = out_channels
        self.in_channels = in_channels

        # Compute query, key and value using 1x1 convolutions.
        self.query = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.out_channels // self.ratio_kq,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               pad_mode="valid",
                               has_bias=True,
                               )
        self.key = nn.Conv2d(in_channels=in_channels,
                             out_channels=self.out_channels // self.ratio_kq,
                             kernel_size=(1, 1),
                             stride=(1, 1),
                             pad_mode="valid",
                             has_bias=True,
                             )
        self.value = nn.Conv2d(in_channels=in_channels,
                               out_channels=self.out_channels // self.ratio_v,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               pad_mode="valid",
                               has_bias=True,)
        self.last_conv = nn.Conv2d(in_channels=self.out_channels // 8,
                                   out_channels=self.out_channels,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   pad_mode="valid",
                                   has_bias=True,
                                   )

        # Learnable gain parameter
        self.gamma = Parameter(ops.Zeros()(1, ms.float32))

    def construct(self, x):
        """AttentionLayer forward function"""
        # Compute query, key and value using 1x1 convolutions.
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        out = []
        for b in range(x.shape[0]):
            out.append(attention_einsum(query[b], key[b], value[b]))
        out = ops.stack(out, 0)
        output = self.gamma * self.last_conv(out) + x
        # Residual connection.
        return output


class DBlock(nn.Cell):
    """
    Residual downsampling block in ContextConditioningStack.
    """
    def __init__(self,
                 in_channels=12,
                 out_channels=12,
                 conv_type="standard",
                 first_relu=True,
                 keep_same_output=False,
                 use_spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_relu = first_relu
        self.keep_same_output = keep_same_output
        self.conv_type = conv_type
        conv2d = get_conv_layer(conv_type)
        if conv_type == "3d":
            # 3D Average pooling
            self.pooling = ops.AvgPool3D(kernel_size=2, strides=2)
        else:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv_1x1 = conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            has_bias=True
        )
        if use_spectral_norm:
            self.conv_1x1 = SpectralNorm(
                self.conv_1x1
            )

        self.first_conv_3x3 = conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.first_conv_3x3 = SpectralNorm(
                self.first_conv_3x3
            )

        self.last_conv_3x3 = conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            stride=1,
            has_bias=True
        )

        self.relu = nn.ReLU()

    def construct(self, x):
        """DBlock forward function"""
        if self.in_channels != self.out_channels:
            x1 = self.conv_1x1(x)
            if not self.keep_same_output:
                x1 = self.pooling(x1)
        else:
            x1 = x

        if self.first_relu:
            x = self.relu(x)
        x = self.first_conv_3x3(x)
        x = self.relu(x)
        x = self.last_conv_3x3(x)

        if not self.keep_same_output:
            x = self.pooling(x)
        output = x1 + x
        return output


class ContextConditioningStack(nn.Cell):
    """
    Context condition stack in Dgmr Sampler.
    """
    def __init__(self,
                 in_channels=1,
                 out_channels=768,
                 num_context_steps=4,
                 conv_type="standard",
                 use_spectral_norm=True):
        super().__init__()

        conv2d = get_conv_layer(conv_type)
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        self.d1 = DBlock(
            in_channels=4 * in_channels,
            out_channels=((out_channels // 4) * in_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d2 = DBlock(
            in_channels=((out_channels // 4) * in_channels) // num_context_steps,
            out_channels=((out_channels // 2) * in_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d3 = DBlock(
            in_channels=((out_channels // 2) * in_channels) // num_context_steps,
            out_channels=(out_channels * in_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.d4 = DBlock(
            in_channels=(out_channels * in_channels) // num_context_steps,
            out_channels=(out_channels * 2 * in_channels) // num_context_steps,
            conv_type=conv_type,
        )
        self.conv1 = conv2d(
            in_channels=(out_channels // 4) * in_channels,
            out_channels=(out_channels // 8) * in_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.conv1 = SpectralNorm(
                self.conv1
            )

        self.conv2 = conv2d(
            in_channels=(out_channels // 2) * in_channels,
            out_channels=(out_channels // 4) * in_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.conv2 = SpectralNorm(
                self.conv2
            )

        self.conv3 = conv2d(
            in_channels=out_channels * in_channels,
            out_channels=(out_channels // 2) * in_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.conv3 = SpectralNorm(
                self.conv3
            )

        self.conv4 = conv2d(
            in_channels=out_channels * 2 * in_channels,
            out_channels=out_channels * in_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.conv4 = SpectralNorm(
                self.conv4
            )

        self.relu = nn.ReLU()

    def construct(self, x):
        """ContextConditioningStack forward function"""
        # Each timestep processed separately
        x = self.space2depth(x)
        steps = x.shape[1]  # Number of timesteps
        scale_1 = []
        scale_2 = []
        scale_3 = []
        scale_4 = []
        for i in range(steps):
            s1 = self.d1(x[:, i, :, :, :])
            s2 = self.d2(s1)
            s3 = self.d3(s2)
            s4 = self.d4(s3)
            scale_1.append(s1)
            scale_2.append(s2)
            scale_3.append(s3)
            scale_4.append(s4)
        scale_1 = ops.stack(scale_1, 1)
        scale_2 = ops.stack(scale_2, 1)
        scale_3 = ops.stack(scale_3, 1)
        scale_4 = ops.stack(scale_4, 1)
        # Mixing layer
        scale_1 = self._mixing_layer(scale_1, self.conv1)
        scale_2 = self._mixing_layer(scale_2, self.conv2)
        scale_3 = self._mixing_layer(scale_3, self.conv3)
        scale_4 = self._mixing_layer(scale_4, self.conv4)

        return scale_1, scale_2, scale_3, scale_4

    def _mixing_layer(self,
                      inputs,
                      conv_block):
        # Convert from [batch_size, time, h, w, c] -> [batch_size, h, w, c * time]
        # then perform convolution on the output while preserving number of c.
        inputs = ops.transpose(inputs, (0, 2, 1, 3, 4))
        stacked_inputs = ops.reshape(inputs, (
            inputs.shape[0], inputs.shape[2] * inputs.shape[1], inputs.shape[3],
            inputs.shape[4]))  # [h, w, c] -> [L, c]
        return ops.relu(conv_block(stacked_inputs))


class LBlock(nn.Cell):
    """
    Residual block for the Latent Stack.
    """
    def __init__(self,
                 in_channels=12,
                 out_channels=12,
                 kernel_size=3,
                 conv_type="standard"):
        super().__init__()
        # Output size should be channel_out - channel_in
        self.in_channels = in_channels
        self.out_channels = out_channels
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=in_channels,
            out_channels=out_channels - in_channels,
            kernel_size=1,
            has_bias=True
        )

        self.first_conv_3x3 = conv2d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
            pad_mode="pad",
            stride=1,
            has_bias=True
        )

        self.relu = nn.ReLU()
        self.last_conv_3x3 = conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=1,
            pad_mode="pad",
            stride=1,
            has_bias=True
        )

    def construct(self, x):
        """LBlock forward function"""
        if self.in_channels < self.out_channels:
            sc = self.conv_1x1(x)
            concat_op = ops.Concat(axis=1)
            sc = concat_op((x, sc))
        else:
            sc = x

        x2 = self.relu(x)
        x2 = self.first_conv_3x3(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        output = x2 + sc
        return output


class LatentConditioningStack(nn.Cell):
    """
    Latent conditioning stack in Dgmr Sampler.
    """
    def __init__(self,
                 shape=(8, 8, 8, 1),
                 out_channels=768,
                 use_attention=True,
                 use_spectral_norm=True):
        super().__init__()
        self.shape = shape
        self.use_attention = use_attention
        self.distribution = msd.Normal(0.0, 1.0, seed=42)
        self.conv_3x3 = nn.Conv2d(
            in_channels=shape[0],
            out_channels=shape[0],
            kernel_size=(3, 3),
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.conv_3x3 = SpectralNorm(
                self.conv_3x3
            )

        self.l_block1 = LBlock(in_channels=shape[0], out_channels=out_channels // 32)
        self.l_block2 = LBlock(
            in_channels=out_channels // 32, out_channels=out_channels // 16
        )
        self.l_block3 = LBlock(
            in_channels=out_channels // 16, out_channels=out_channels // 4
        )

        if self.use_attention:
            self.att_block = AttentionLayer(
                in_channels=out_channels // 4, out_channels=out_channels // 4
            )
        self.l_block4 = LBlock(in_channels=out_channels // 4, out_channels=out_channels)

    def construct(self, x):
        """LatentConditioningStack forward function"""
        z = self.distribution.sample(self.shape)
        # Batch is at end for some reason, reshape
        z = ops.transpose(z, (3, 0, 1, 2))
        z = ops.cast(z, x.dtype)
        # 3x3 Convolution
        z = self.conv_3x3(z)

        # 3 L Blocks to increase number of channels
        z = self.l_block1(z)
        z = self.l_block2(z)
        z = self.l_block3(z)
        # Spatial attention module
        z = self.att_block(z)

        # L block to increase number of channel to 768
        output = self.l_block4(z)
        return output


class GBlock(nn.Cell):
    """
    Residual generator block without upsampling.
    """
    def __init__(self,
                 in_channels=12,
                 out_channels=12,
                 conv_type="standard",
                 spectral_normalized_eps=0.0001,
                 use_spectral_norm=True):
        super().__init__()
        self.out_channels = out_channels
        self.bn1 = nn.BatchNorm2d(in_channels, use_batch_statistics=True)
        self.bn2 = nn.BatchNorm2d(in_channels, use_batch_statistics=True)
        self.relu = nn.ReLU()
        # Upsample in the 1x1
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            has_bias=True
        )
        if use_spectral_norm:
            self.conv_1x1 = SpectralNorm(
                self.conv_1x1,
                eps=spectral_normalized_eps,
            )

        # Upsample 2D conv
        self.first_conv_3x3 = conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.first_conv_3x3 = SpectralNorm(
                self.first_conv_3x3,
                eps=spectral_normalized_eps,
            )

        self.last_conv_3x3 = conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
            pad_mode="pad", has_bias=True
        )
        if use_spectral_norm:
            self.last_conv_3x3 = SpectralNorm(
                self.last_conv_3x3,
                eps=spectral_normalized_eps,
            )

    def construct(self, x):
        """GBlock forward function"""
        # Optionally spectrally normalized 1x1 convolution
        if x.shape[1] != self.out_channels:
            sc = self.conv_1x1(x)
        else:
            sc = x
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        x2 = self.first_conv_3x3(x2)  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine, residual connection
        output = x2 + sc
        return output


class UpsampleGBlock(nn.Cell):
    """
    Residual generator block with upsampling.
    """
    def __init__(self,
                 in_channels=12,
                 out_channels=12,
                 conv_type="standard",
                 spectral_normalized_eps=0.0001,
                 use_spectral_norm=True):
        super().__init__()
        self.out_channels = out_channels
        self.bn1 = nn.BatchNorm2d(in_channels, use_batch_statistics=True)
        self.bn2 = nn.BatchNorm2d(in_channels, use_batch_statistics=True)
        self.relu = nn.ReLU()
        # Upsample in the 1x1
        conv2d = get_conv_layer(conv_type)
        self.conv_1x1 = conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            has_bias=True
        )
        if use_spectral_norm:
            self.conv_1x1 = SpectralNorm(
                self.conv_1x1,
                eps=spectral_normalized_eps,
            )

        self.upsample = nn.ResizeBilinear()
        # Upsample 2D conv
        self.first_conv_3x3 = conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            padding=1,
            pad_mode="pad",
            has_bias=True
        )
        if use_spectral_norm:
            self.first_conv_3x3 = SpectralNorm(
                self.first_conv_3x3,
                eps=spectral_normalized_eps
            )
        self.last_conv_3x3 = conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
            pad_mode="pad", has_bias=True
        )
        if use_spectral_norm:
            self.last_conv_3x3 = SpectralNorm(
                self.last_conv_3x3,
                eps=spectral_normalized_eps,
            )

    def construct(self, x):
        """UpsampleGBlock forward function"""
        # Spectrally normalized 1x1 convolution
        sc = self.upsample(x, scale_factor=2, align_corners=True)
        sc = self.conv_1x1(sc)
        x2 = self.bn1(x)
        x2 = self.relu(x2)
        # Upsample
        x2 = self.upsample(x2, scale_factor=2, align_corners=True)
        x2 = self.first_conv_3x3(x2)  # Make sure size is doubled
        x2 = self.bn2(x2)
        x2 = self.relu(x2)
        x2 = self.last_conv_3x3(x2)
        # Sum combine, residual connection
        output = x2 + sc
        return output


class Sampler(nn.Cell):
    """
    The sampler is a recurrent network formed with convolutional gated recurrent units (GRUs)
    that uses the context and latent representations as inputs.
    """
    def __init__(self,
                 forecast_steps=18,
                 latent_channels=768,
                 context_channels=384,
                 out_channels=1,
                 use_spectral_norm=False):
        super().__init__()
        self.forecast_steps = forecast_steps

        self.conv_gru1 = ConvGRU(
            in_channels=latent_channels + context_channels,
            out_channels=context_channels,
            kernel_size=3,
        )
        self.gru_conv_1x1 = nn.Conv2d(
            in_channels=context_channels, out_channels=latent_channels, kernel_size=1, has_bias=True
            )
        if use_spectral_norm:
            self.gru_conv_1x1 = SpectralNorm(
                self.gru_conv_1x1
            )

        self.g1 = GBlock(in_channels=latent_channels, out_channels=latent_channels)
        self.up_g1 = UpsampleGBlock(
            in_channels=latent_channels, out_channels=latent_channels // 2
        )

        self.conv_gru2 = ConvGRU(
            in_channels=latent_channels // 2 + context_channels // 2,
            out_channels=context_channels // 2,
            kernel_size=3,
        )

        self.gru_conv_1x1_2 = nn.Conv2d(
            in_channels=context_channels // 2,
            out_channels=latent_channels // 2,
            kernel_size=1,
            has_bias=True
        )
        if use_spectral_norm:
            self.gru_conv_1x1_2 = SpectralNorm(
                self.gru_conv_1x1_2
            )

        self.g2 = GBlock(in_channels=latent_channels // 2, out_channels=latent_channels // 2)
        self.up_g2 = UpsampleGBlock(
            in_channels=latent_channels // 2, out_channels=latent_channels // 4
        )

        self.conv_gru3 = ConvGRU(
            in_channels=latent_channels // 4 + context_channels // 4,
            out_channels=context_channels // 4,
            kernel_size=3,
        )

        self.gru_conv_1x1_3 = nn.Conv2d(
            in_channels=context_channels // 4,
            out_channels=latent_channels // 4,
            kernel_size=(1, 1),
            has_bias=True
        )
        if use_spectral_norm:
            self.gru_conv_1x1_3 = SpectralNorm(
                self.gru_conv_1x1_3
            )

        self.g3 = GBlock(in_channels=latent_channels // 4, out_channels=latent_channels // 4)
        self.up_g3 = UpsampleGBlock(
            in_channels=latent_channels // 4, out_channels=latent_channels // 8
        )

        self.conv_gru4 = ConvGRU(
            in_channels=latent_channels // 8 + context_channels // 8,
            out_channels=context_channels // 8,
            kernel_size=3,
        )

        self.gru_conv_1x1_4 = nn.Conv2d(
            in_channels=context_channels // 8,
            out_channels=latent_channels // 8,
            kernel_size=(1, 1),
            has_bias=True
        )
        if use_spectral_norm:
            self.gru_conv_1x1_4 = SpectralNorm(
                self.gru_conv_1x1_4
            )

        self.g4 = GBlock(in_channels=latent_channels // 8, out_channels=latent_channels // 8)
        self.up_g4 = UpsampleGBlock(
            in_channels=latent_channels // 8, out_channels=latent_channels // 16
        )

        self.bn = nn.BatchNorm2d(latent_channels // 16, use_batch_statistics=True)
        self.relu = nn.ReLU()

        self.conv_1x1 = nn.Conv2d(
            in_channels=latent_channels // 16,
            out_channels=4 * out_channels,
            kernel_size=(1, 1),
            has_bias=True
        )
        if use_spectral_norm:
            self.conv_1x1 = SpectralNorm(
                self.conv_1x1
            )

        self.depth2space = PixelShuffle(upscale_factor=2)

    def construct(self,
                  conditioning_states,
                  latent_dim):
        """Sampler forward function"""
        # Iterate through each forecast step
        # Initialize with conditioning state for first one, output for second one

        # Expand latent dim to match batch size
        latent_dim = ms.numpy.tile(latent_dim, (conditioning_states[0].shape[0], 1, 1, 1))
        hidden_states = [latent_dim] * self.forecast_steps
        # Layer 4 (bottom most)
        hidden_states = self.conv_gru1(hidden_states, conditioning_states[3])
        hidden_states = [self.gru_conv_1x1(hidden_states[index]) for index in range(hidden_states.shape[0])]
        hidden_states = [self.g1(h) for h in hidden_states]
        hidden_states = [self.up_g1(h) for h in hidden_states]
        #
        # Layer 3.
        hidden_states = self.conv_gru2(hidden_states, conditioning_states[2])
        hidden_states = [self.gru_conv_1x1_2(hidden_states[index]) for index in range(hidden_states.shape[0])]
        hidden_states = [self.g2(h) for h in hidden_states]
        hidden_states = [self.up_g2(h) for h in hidden_states]
        #
        # # Layer 2.
        hidden_states = self.conv_gru3(hidden_states, conditioning_states[1])
        hidden_states = [self.gru_conv_1x1_3(hidden_states[index]) for index in range(hidden_states.shape[0])]
        hidden_states = [self.g3(h) for h in hidden_states]
        hidden_states = [self.up_g3(h) for h in hidden_states]

        # Layer 1 (top-most).
        hidden_states = self.conv_gru4(hidden_states, conditioning_states[0])
        hidden_states = [self.gru_conv_1x1_4(hidden_states[index]) for index in range(hidden_states.shape[0])]
        hidden_states = [self.g4(h) for h in hidden_states]
        hidden_states = [self.up_g4(h) for h in hidden_states]

        # Output layer.
        hidden_states = [ops.relu(self.bn(h)) for h in hidden_states]
        hidden_states = [self.conv_1x1(h) for h in hidden_states]
        hidden_states = [self.depth2space(h) for h in hidden_states]

        # Convert forecasts to a Tensor
        output = ops.stack(hidden_states, 1)
        return output


class TemporalDiscriminator(nn.Cell):
    """
    Temporal Discriminator, which is a three-dimensional (3D) convolutional neural network
    that aims to distinguish observed and generated radar sequences.
    """
    def __init__(self,
                 in_channels=12,
                 num_layers=3,
                 conv_type="standard",
                 use_spectral_norm=True):
        super().__init__()
        self.downsample = ops.AvgPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        hidden_channels = 48
        self.d1 = DBlock(
            in_channels=4 * in_channels,
            out_channels=hidden_channels * in_channels,
            conv_type="3d",
            first_relu=False,
        )
        self.d2 = DBlock(
            in_channels=hidden_channels * in_channels,
            out_channels=2 * hidden_channels * in_channels,
            conv_type="3d",
        )
        self.intermediate_dblocks = nn.CellList()
        for _ in range(num_layers):
            hidden_channels *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    in_channels=hidden_channels * in_channels,
                    out_channels=2 * hidden_channels * in_channels,
                    conv_type=conv_type,
                )
            )

        self.d_last = DBlock(
            in_channels=2 * hidden_channels * in_channels,
            out_channels=2 * hidden_channels * in_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        self.fc = nn.Dense(2 * hidden_channels * in_channels, 1)
        if use_spectral_norm:
            self.fc = SpectralNorm(self.fc)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(2 * hidden_channels * in_channels, use_batch_statistics=True)

    def construct(self, x):
        """TemporalDiscriminator forward function"""
        x = self.downsample(x)
        x = self.space2depth(x)
        # Have to move time and channels
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        # 2 residual 3D blocks to halve resolution if image, double number of channels and reduce
        # number of time steps
        x = self.d1(x)
        x = self.d2(x)
        # Convert back to T x C x H x W
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        # Per Timestep part now, same as spatial discriminator
        representations = []
        for idx in range(x.shape[1]):
            # Intermediate DBlocks
            # Three residual D Blocks to halve the resolution of the image and double
            # the number of channels.
            rep = x[:, idx, :, :, :]
            for d in self.intermediate_dblocks:
                rep = d(rep)
            # One more D Block without downsampling or increase number of channels
            rep = self.d_last(rep)

            op = ops.ReduceSum()
            rep = op(ops.relu(rep), [2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)

            representations.append(rep)
        x = ops.stack(representations, 1)
        # Should be [Batch, N, 1]
        op = ops.ReduceSum(keep_dims=True)
        output = op(x, 1)
        return output


class SpatialDiscriminator(nn.Cell):
    """
    Spatial discriminator, which is a convolutional neural network
    that aims to distinguish individual observed radar fields from generated fields.
    """
    def __init__(
            self,
            in_channels=12,
            num_timesteps=8,
            num_layers=4,
            conv_type="standard",
            use_spectral_norm=True):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.mean_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.downsample = ops.AvgPool3D(kernel_size=(1, 2, 2), strides=(1, 2, 2))
        self.space2depth = PixelUnshuffle(downscale_factor=2)
        hidden_channels = 24
        self.d1 = DBlock(
            in_channels=4 * in_channels,
            out_channels=2 * hidden_channels * in_channels,
            first_relu=False,
            conv_type=conv_type,
        )
        self.intermediate_dblocks = nn.CellList()
        for _ in range(num_layers):
            hidden_channels *= 2
            self.intermediate_dblocks.append(
                DBlock(
                    in_channels=hidden_channels * in_channels,
                    out_channels=2 * hidden_channels * in_channels,
                    conv_type=conv_type,
                )
            )
        self.d6 = DBlock(
            in_channels=2 * hidden_channels * in_channels,
            out_channels=2 * hidden_channels * in_channels,
            keep_same_output=True,
            conv_type=conv_type,
        )

        self.fc = nn.Dense(2 * hidden_channels * in_channels, 1)
        if use_spectral_norm:
            self.fc = SpectralNorm(self.fc)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(2 * hidden_channels * in_channels, use_batch_statistics=True)

    def construct(self, x):
        """SpatialDiscriminator forward function"""
        # x should be the chosen 8 or so 22, 256, 256
        randperm = ops.Randperm(max_length=18)
        idxs = randperm(Tensor([18], dtype=ms.int32)) + 4
        idxs = idxs[:self.num_timesteps]
        x = ops.Gather()(x, idxs, 1)
        x = self.downsample(x)
        x = self.space2depth(x)
        representations = []
        for idx in range(x.shape[1]):
            rep = self.d1(x[:, idx, :, :, :])  # 32x32
            # Intermediate DBlocks
            for d in self.intermediate_dblocks:
                rep = d(rep)
            rep = self.d6(rep)  # 2x2
            op = ops.ReduceSum()
            rep = op(ops.relu(rep), [2, 3])
            rep = self.bn(rep)
            rep = self.fc(rep)
            representations.append(rep)

        # The representations are summed together before the ReLU
        x = ops.stack(representations, 1)
        # Should be [Batch, N, 1]
        op = ops.ReduceSum(keep_dims=True)
        output = op(x, 1)
        return output
