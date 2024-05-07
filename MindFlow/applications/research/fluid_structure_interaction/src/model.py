# ============================================================================
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
# ============================================================================
"""hybrid deep neural network structure"""
from mindspore import nn, ops

from .conv_lstm import ConvLSTM


class Encoder(nn.Cell):
    """The Convolutional layer (12 layers) of Hybrid Deep Neural Network

    Args:
        in_channels (int): The number of channels in the input space.
        num_layers (int): The number of Convolutional layer.
        kernel_size(int): The size of Convolutional kernel in Convolutional layer.
        has_bias(bool): Whether set bias for Convolutional layer.
        weight_init(str): The way to perform weight initialization operation.

    Inputs:
        A tensor of size [B, C, H, W] = [16, 3, 192, 128]

    Outputs:
        A tensor of size [B, C, H, W] = [16, 128, 3, 2]

    Examples:
        >>> encoder = Encoder(in_channels=3, num_layers=12, kernel_size=4, has_bias=True, weight_init='XavierUniform')
    """

    def __init__(self, in_channels, num_layers, kernel_size, has_bias=True, weight_init='XavierUniform',
                 activation=nn.LeakyReLU()):
        super(Encoder, self).__init__()

        layers = []
        for num in range(1, num_layers + 1):
            if num == 1:
                layers.extend([nn.Conv2d(in_channels, 2 ** (num + 1), kernel_size, stride=2, padding=0, pad_mode='same',
                                         has_bias=has_bias, weight_init=weight_init, data_format='NCHW'), activation])
            elif num % 2 == 0:
                layers.extend([nn.Conv2d(int(2 ** (num / 2 + 1)), int(2 ** (num / 2 + 1)), kernel_size - 1, stride=1,
                                         padding=0, pad_mode='same', has_bias=has_bias, weight_init=weight_init,
                                         data_format='NCHW'), activation])
            elif num % 2 == 1:
                layers.extend([nn.Conv2d(int(2 ** ((num + 1) / 2)), int(2 ** ((num + 3) / 2)), kernel_size, stride=2,
                                         padding=0, pad_mode='same', has_bias=has_bias, weight_init=weight_init,
                                         data_format='NCHW'), activation])
        self.convlayers = nn.SequentialCell(layers)

    def construct(self, x):
        x = self.convlayers(x)
        return x


class Decoder(nn.Cell):
    """The DeConvolutional layer (12 layers) of Hybrid Deep Neural Network

    Args:
        in_channels (int): The number of channels in the input space.
        num_layers (int): The number of DeConvolutional layer.
        kernel_size(int): The size of DeConvolutional kernel in DeConvolutional layer.
        weight_init(str): The way to perform weight initialization operation.

    Inputs:
        A tensor of size [B, C, H, W] = [16, 128, 3, 2]

    Outputs:
        A tensor of size [B, C, H, W] = [16, 3, 192, 128]

    Examples:
        >>> Decoder = Decoder(in_channels=128, num_layers=12, kernel_size=4, weight_init='XavierUniform')
    """

    def __init__(self, in_channels, num_layers, kernel_size, weight_init='XavierUniform', activation=nn.LeakyReLU()):
        super(Decoder, self).__init__()

        layers = []
        for num in range(1, num_layers + 1):
            if num == num_layers:
                layers.extend(
                    [nn.Conv2d(in_channels, in_channels, kernel_size + 1, weight_init=weight_init, stride=1,
                               pad_mode='same', padding=0), activation])
            elif num == num_layers - 1:
                layers.extend([nn.Conv2dTranspose(in_channels + 1, in_channels, kernel_size, stride=2, pad_mode='same',
                                                  padding=0), activation])
            elif num % 2 == 1:
                layers.extend([nn.Conv2dTranspose(int(2 ** ((15 - num) / 2)), int(2 ** ((13 - num) / 2)), kernel_size,
                                                  stride=2, padding=0, pad_mode='same', weight_init=weight_init),
                               activation])
            elif num % 2 == 0:
                layers.extend([nn.Conv2d(int(2 ** ((14 - num) / 2)), int(2 ** ((14 - num) / 2)), kernel_size - 1,
                                         stride=1, padding=0, pad_mode='same', weight_init=weight_init), activation])
        self.deconv_layers = nn.SequentialCell(layers)

    def construct(self, x):
        x = self.deconv_layers(x)
        return x


class AEnet(nn.Cell):
    r"""
    A Hybrid Deep Neural Network Composed of Convolutional Layer, ConvLSTM, and Deconvolutional Layer

    Args:
        in_channels (int): The number of channels in the input space.
        num_layers (int): The number of Convolutional and DeConvolutional layer.
        kernel_size(int): The size of convolutional kernel in Convolutional and DeConvolutional layer.
        num_convlstm_layers (int): The number of ConvLSTM Layer.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, in\_channels)`.
    """

    def __init__(self,
                 in_channels,
                 num_layers,
                 kernel_size,
                 num_convlstm_layers):
        super(AEnet, self).__init__()
        self.encoder = Encoder(in_channels=in_channels, num_layers=num_layers, kernel_size=kernel_size)
        self.convlstm = ConvLSTM(input_dim=128, hidden_dim=128, kernel_size=(3, 3), num_layers=num_convlstm_layers,
                                 batch_first=True, bias=True)
        self.decoder = Decoder(in_channels=in_channels, num_layers=num_layers, kernel_size=kernel_size)

    def construct(self, x, velocity, ur):
        """
        Unpacking the input data x in five dimensions, passing through the reshape, and inputting it into the
        convolutional layer; Then send the output reshape and velocity to ConvLSTM; Then input the output result into
        the deconvolution layer and output the final result
        """
        b, t, c, h, w = x.shape

        con_in = ops.reshape(x, (b * t, c, h, w))

        con_out = self.encoder(con_in)

        con_out = con_out.reshape(b, t, con_out.shape[1], con_out.shape[2], con_out.shape[3])

        lstm_out = self.convlstm(con_out, velocity, ur)

        out = self.decoder(lstm_out)

        return out
