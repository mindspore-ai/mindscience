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
"""NowcastNet Encoder and Decoder"""
from mindspore import nn, ops

from .evolution import SpectralNormal, DoubleConv, Down


class ReflectPad(nn.Cell):
    """
    Reflect padding

    Args:
        padding (int): the padding size

    Inputs:
        - **x** (Tensor) - The input tensor.

    Outputs:
        - **out** (Tensor) - The tensor after padding.
    """
    def __init__(self, padding):
        super(ReflectPad, self).__init__()
        self.padding = padding

    def construct(self, x):
        out = ops.concat([x[..., 1:1 + self.padding][..., ::-1].copy(),
                          x,
                          x[..., -1 - self.padding:-1][..., ::-1].copy()], axis=3)
        out = ops.concat([out[:, :, 1:1 + self.padding][:, :, ::-1].copy(),
                          out,
                          out[:, :, -1 - self.padding:-1][:, :, ::-1]], axis=2)
        return out


class GenBlock(nn.Cell):
    """GenBlock"""
    def __init__(self, in_channels, out_channels, data_params, dilation=1, double_conv=False):
        super(GenBlock, self).__init__()
        self.learned_shortcut = (in_channels != out_channels)
        self.double_conv = double_conv
        mid_channels = min(in_channels, out_channels)
        self.pad = ReflectPad(dilation)
        self.conv_0 = nn.Conv2d(in_channels,
                                mid_channels,
                                kernel_size=3,
                                pad_mode='valid',
                                has_bias=True,
                                dilation=dilation
                                )
        self.conv_0 = SpectralNormal(self.conv_0)
        self.norm_0 = SPADE(in_channels, data_params.get("t_out", 20))
        self.conv_1 = nn.Conv2d(mid_channels,
                                out_channels,
                                kernel_size=3,
                                pad_mode='valid',
                                has_bias=True,
                                dilation=dilation
                                )
        self.conv_1 = SpectralNormal(self.conv_1)
        self.norm_1 = SPADE(mid_channels, data_params.get("t_out", 20))
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(in_channels, out_channels, kernel_size=1, pad_mode='pad')
            self.conv_s = SpectralNormal(self.conv_s)
            self.norm_s = SPADE(in_channels, data_params.get("t_out", 20))
        self.leaky_relu = nn.LeakyReLU(2e-1)

    def construct(self, x, evo):
        x_s = self.shortcut(x, evo)
        dx = self.conv_0(self.pad(self.leaky_relu(self.norm_0(x, evo))))
        if self.double_conv:
            dx = self.conv_1(self.pad(self.leaky_relu(self.norm_1(dx, evo))))
        out = x_s + dx
        return out

    def shortcut(self, x, evo):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, evo))
        else:
            x_s = x
        return x_s


class SPADE(nn.Cell):
    """SPADE class"""
    def __init__(self, norm_channels, label_nc, hidden=64, kernel_size=3):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(norm_channels, affine=False)
        self.pad_head = ReflectPad(kernel_size // 2)
        self.mlp_shared = nn.SequentialCell(
            nn.Conv2d(label_nc, hidden, kernel_size=kernel_size, pad_mode='pad', has_bias=True),
            nn.ReLU()
        )
        self.pad = ReflectPad(kernel_size // 2)
        self.mlp_gamma = nn.Conv2d(hidden, norm_channels, kernel_size=kernel_size, pad_mode='pad', has_bias=True)
        self.mlp_beta = nn.Conv2d(hidden, norm_channels, kernel_size=kernel_size, pad_mode='pad', has_bias=True)

    def construct(self, x, evo):
        normalized = self.param_free_norm(x)
        evo = ops.adaptive_avg_pool2d(evo, output_size=x.shape[2:])
        evo = self.pad_head(evo)
        evo_out = self.mlp_shared(evo)
        gamma = self.mlp_gamma(self.pad(evo_out))
        beta = self.mlp_beta(self.pad(evo_out))
        out = normalized * (1 + gamma) + beta
        return out


class NoiseProjector(nn.Cell):
    """Noise Projector"""
    def __init__(self, t_in):
        super(NoiseProjector, self).__init__()
        self.conv_first = SpectralNormal(nn.Conv2d(t_in,
                                                   t_in * 2,
                                                   kernel_size=3,
                                                   pad_mode='pad',
                                                   padding=1,
                                                   has_bias=True
                                                   )
                                         )
        self.block1 = ProjBlock(t_in * 2, t_in * 4)
        self.block2 = ProjBlock(t_in * 4, t_in * 8)
        self.block3 = ProjBlock(t_in * 8, t_in * 16)
        self.block4 = ProjBlock(t_in * 16, t_in * 32)

    def construct(self, x):
        x = self.conv_first(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        out = self.block4(x)
        return out


class ProjBlock(nn.Cell):
    """Projector block"""
    def __init__(self, in_channels, out_channels):
        super(ProjBlock, self).__init__()
        self.one_conv = SpectralNormal(nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=1, has_bias=True))
        self.double_conv = nn.SequentialCell(
            SpectralNormal(nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     pad_mode='pad',
                                     padding=1,
                                     has_bias=True)),
            nn.ReLU(),
            SpectralNormal(nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     pad_mode='pad',
                                     padding=1,
                                     has_bias=True))
        )

    def construct(self, x):
        x1 = ops.concat([x, self.one_conv(x)], axis=1)
        x2 = self.double_conv(x)
        out = x1 + x2
        return out


class GenerativeEncoder(nn.Cell):
    """Encoder of Generative"""
    def __init__(self, in_channels, hidden=64):
        super(GenerativeEncoder, self).__init__()
        self.inc = DoubleConv(in_channels, hidden, kernel=3)
        self.down1 = Down(hidden, hidden * 2, 3)
        self.down2 = Down(hidden * 2, hidden * 4, 3)
        self.down3 = Down(hidden * 4, hidden * 8, 3)

    def construct(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        out = self.down3(x)
        return out


class GenerativeDecoder(nn.Cell):
    """Decoder of Generative"""
    def __init__(self, config):
        super(GenerativeDecoder, self).__init__()
        data_params = config.get("data")
        model_params = config.get("model")
        scale = data_params.get("noise_scale") // 8
        nf = model_params.get("ngf")
        in_channels = (8 + nf // (scale**2)) * nf
        out_channels = data_params.get("t_out", 20)
        self.fc = nn.Conv2d(in_channels, 8 * nf, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.head_0 = GenBlock(8 * nf, 8 * nf, data_params)
        self.gen_middle_0 = GenBlock(8 * nf, 4 * nf, data_params, double_conv=True)
        self.gen_middle_1 = GenBlock(4 * nf, 4 * nf, data_params, double_conv=True)
        self.up_0 = GenBlock(4 * nf, 2 * nf, data_params)
        self.up_1 = GenBlock(2 * nf, nf, data_params, double_conv=True)
        self.up_2 = GenBlock(nf, nf, data_params, double_conv=True)
        self.conv_img = nn.Conv2d(nf, out_channels, kernel_size=3, pad_mode='pad', padding=1, has_bias=True)
        self.leaky_relu = nn.LeakyReLU(2e-1)

    def construct(self, x, evo):
        """decoder construct"""
        x = self.fc(x)
        x = self.head_0(x, evo)
        h, w = x.shape[2], x.shape[3]
        x = ops.interpolate(x, size=(h * 2, w * 2))
        x = self.gen_middle_0(x, evo)
        x = self.gen_middle_1(x, evo)
        h, w = x.shape[2], x.shape[3]
        x = ops.interpolate(x, size=(h * 2, w * 2))
        x = self.up_0(x, evo)
        h, w = x.shape[2], x.shape[3]
        x = ops.interpolate(x, size=(h * 2, w * 2))
        x = self.up_1(x, evo)
        x = self.up_2(x, evo)
        out = self.conv_img(self.leaky_relu(x))
        return out


class GenerationNet(nn.Cell):
    """Generation network"""
    def __init__(self, config):
        super(GenerationNet, self).__init__()
        self.config = config
        data_params = config.get("data")
        model_params = config.get("model")
        self.gen_enc = GenerativeEncoder(data_params.get("t_in", 9) + data_params.get("t_out", 20),
                                         hidden=model_params.get("ngf", 32))
        self.gen_dec = GenerativeDecoder(self.config)
        self.proj = NoiseProjector(model_params.get("ngf"))

    def construct(self, input_frames, evo_result, noise):
        batch = input_frames.shape[0]
        height = input_frames.shape[2]
        width = input_frames.shape[3]
        evo_feature = self.gen_enc(ops.concat([input_frames, evo_result], axis=1))
        noise_feature = self.proj(noise).reshape(batch, -1, 4, 4, 8, 8)
        noise_feature = noise_feature.permute(0, 1, 4, 5, 2, 3).reshape(batch, -1, height // 8, width // 8)
        feature = ops.concat([evo_feature, noise_feature], axis=1)
        out = self.gen_dec(feature, evo_result)
        return out
