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
"""enhanced hybrid deep neural network structure"""
import numpy as np

from mindspore import Tensor, nn, ops
import mindspore.common.dtype as mstype
import mindspore.ops.operations as P


class ConvolutionalLayer(nn.Cell):
    r"""
    Convolutional layer (7layers)

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        num_layers (int): The number of Convolutional layer.
        kernel_size (int): The size of Convolutional kernel.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype is
            same as input `input` . For the values of str, refer to the function `initializer`. Default:'XavierUniform'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input `input` . The values of str refer to the function `initializer`. Default: "zeros".
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.
        activation (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected
            layer. Default: nn.LeakyReLU().

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        import numpy as np
        from mindspore import Tensor, nn, ops
        input = Tensor(np.array((16, 3, 200, 200)), np.float32))
        net = ConvolutionalLayer(3, 128, 7, 3)
        output = net(input)
        print(output.shape)
        (16, 128, 4, 4)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers,
                 kernel_size,
                 weight_init='XavierUniform',
                 has_bias=True,
                 bias_init='zeros',
                 activation=nn.LeakyReLU()):
        super(ConvolutionalLayer, self).__init__()
        layers = []
        for num in range(1, num_layers + 1):
            if num == 1:
                layers.extend([nn.Conv2d(in_channels, 2 ** (num + 1), kernel_size - 1, stride=1, padding=0,
                                         pad_mode='same', has_bias=has_bias, weight_init=weight_init,
                                         bias_init=bias_init, data_format='NCHW'), activation])
            else:
                if num < num_layers:
                    layers.extend([nn.Conv2d(2 ** num, 2 ** (num + 1), kernel_size, stride=2, padding=0,
                                             pad_mode='same', has_bias=has_bias, weight_init=weight_init,
                                             bias_init=bias_init, data_format='NCHW'), activation])
                else:
                    layers.extend([nn.Conv2d(2 ** num, out_channels, kernel_size, stride=2, padding=0,
                                             pad_mode='same', has_bias=has_bias, weight_init=weight_init,
                                             bias_init=bias_init, data_format='NCHW'), activation])
        self.build_block_conv = nn.SequentialCell(layers)

    def construct(self, x):
        return self.build_block_conv(x)


class MemoryLayer(nn.Cell):
    r"""
    Memory layer

    Args:
        input_channels (int): The number of channels in the input space.Default:1.
        hidden_channels (int):The number of channels in the hidden state.Default:1.
        num_layers (int): The number of layers of the whole Memory layer.
        kernel_size (int): The size of convolutional kernel.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype is
            same as input `input` . For the values of str, refer to the function `initializer`. Default:'XavierUniform'.
        bias_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable bias_init parameter. The dtype is
            same as input `input` . The values of str refer to the function `initializer`. Default: "zeros".
        has_bias (bool): Specifies whether the layer uses a bias vector. Default: ``True``.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, in\_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        import numpy as np
        from mindspore import Tensor, nn, ops
        input = Tensor(np.array((16, 128, 4, 4)), np.float32))
        net = MemoryLayer(2, 2)
        output = net(input)
        print(output.shape)
        (1, 128, 4, 4)

    """

    def __init__(self,
                 num_layers,
                 kernel_size,
                 input_channels=1,
                 hidden_channels=1,
                 weight_init='XavierUniform',
                 has_bias=True,
                 bias_init='zeros'):
        super(MemoryLayer, self).__init__()
        layers = []
        self.input_channels = input_channels
        self.num_layers = num_layers
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        num = 1
        while num <= self.num_layers:
            layers.extend([nn.Conv2d(self.input_channels + hidden_channels, 4 * hidden_channels,
                                     kernel_size, stride=1, pad_mode='same', padding=0, has_bias=has_bias,
                                     weight_init=weight_init, bias_init=bias_init, data_format='NCHW')])
            num += 1
        self.build_block_convlstm = nn.CellList(layers)

    def cell(self, x, h, c, layer):
        """
        ConvLSTM_Cell
        """
        x_h_conv = self.build_block_convlstm[layer](ops.concat((x, h), 1))
        i_x_h = ops.expand_dims(x_h_conv[:, 0, :, :], 1)
        f_x_h = ops.expand_dims(x_h_conv[:, 1, :, :], 1)
        c_x_h = ops.expand_dims(x_h_conv[:, 2, :, :], 1)
        o_x_h = ops.expand_dims(x_h_conv[:, 3, :, :], 1)

        ct = self.sigmoid(f_x_h) * c + self.sigmoid(i_x_h) * (self.tanh(c_x_h))
        ht = self.sigmoid(o_x_h) * self.tanh(ct)
        return ht, ct

    def construct(self, x):
        """
        ConvLSTM
        """
        time_length = len(x)
        input_layer = ops.transpose(x, (1, 0, 2, 3))
        output_layer = ops.ones_like(input_layer)
        layer_state_list = []
        param_0 = Tensor(np.zeros((128, 1, 4, 4)), mstype.float32)
        for num in range(self.num_layers):
            h, c = param_0, param_0
            for t in range(time_length):
                input_t = ops.expand_dims(input_layer[:, t, :, :], 1)
                h, c = self.cell(input_t, h, c, num)
                if t == 0:
                    output_layer = h
                else:
                    output_layer = ops.concat((output_layer, h), 1)
            layer_state_list.append([h, c])
            input_layer = output_layer
        out_state = layer_state_list[-1]
        return ops.transpose(out_state[0], (1, 0, 2, 3))


class DeConvolutionalLayer(nn.Cell):
    r"""
    DeConvolutional layer (7layers)

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        num_layers (int): The number of DeConvolutional layer.
        kernel_size (int): The size of DeConvolutional kernel.
        weight_init (Union[Tensor, str, Initializer, numbers.Number]): The trainable weight_init parameter. The dtype is
            same as input `input` . For the values of str, refer to the function `initializer`. Default:'XavierUniform'.
        has_bias (bool): Specifies whether the layer uses a bias vector. Default:False.
        activation (Union[str, Cell, Primitive, None]): activate function applied to the output of the fully connected
            layer. Default: nn.LeakyReLU().

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, out\_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        import numpy as np
        from mindspore import Tensor, nn, ops
        input = Tensor(np.array((1, 128, 4, 4)), np.float32))
        net = DeConvolutionalLayer(128, 3, 7, 3)
        output = net(input)
        print(output.shape)
        (1, 3, 200, 200)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers,
                 kernel_size,
                 weight_init='XavierUniform',
                 has_bias=False,
                 activation=nn.LeakyReLU()):
        super(DeConvolutionalLayer, self).__init__()
        layers = []
        for num in range(1, num_layers):
            if num < 4:
                if num == 1:
                    layers.extend([nn.Conv2dTranspose(in_channels, in_channels, kernel_size, stride=2,
                                                      pad_mode='pad', padding=1, has_bias=has_bias,
                                                      weight_init=weight_init), activation])
                else:
                    layers.extend(
                        [nn.Conv2dTranspose(int(in_channels / (2 ** (num - 2))), int(in_channels / (2 ** (num - 1))),
                                            kernel_size, stride=2, pad_mode='pad', padding=1, has_bias=has_bias,
                                            weight_init=weight_init), activation])
            else:
                layers.extend(
                    [nn.Conv2dTranspose(int(in_channels / (2 ** (num - 2))), int(in_channels / (2 ** (num - 1))),
                                        kernel_size, stride=2, pad_mode='same', padding=0, has_bias=has_bias,
                                        weight_init=weight_init), activation])

        layers.extend([nn.Conv2dTranspose(int(in_channels / (2 ** (num_layers - 2))), out_channels, kernel_size - 1,
                                          stride=1, pad_mode='same', padding=0, has_bias=has_bias,
                                          weight_init=weight_init), activation])

        self.build_block_deconv = nn.SequentialCell(layers)

    def construct(self, x):
        return self.build_block_deconv(x)


class EhdnnNet(nn.Cell):
    r"""
    EhdnnNet

    Args:
        in_channels (int): The number of channels in the input space.
        out_channels (int): The number of channels in the output space.
        num_layers (int): The number of Convolutional and DeConvolutional layer.
        num_memory_layers (int): The number of Memory Layer.
        kernel_size_conv (int): The size of convolutional kernel in Convolutional and DeConvolutional layer.
        kernel_size_lstm (int): The size of convolutional kernel in Memory Layer.
        compute_dtype (dtype): The data type for ForwardNet. Default: mstype.float32.


    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.

    Outputs:
        Tensor of shape :math:`(*, in\_channels)`.

    Supported Platforms:
        ``Ascend`` ``GPU````CPU``

    Examples:
        import numpy as np
        from mindspore import Tensor, nn, ops
        input = Tensor(np.array((16, 3, 200, 200)), np.float32))
        net = ForwardNet(3, 128, 7, 2, 3, 2)
        output = net(input)
        print(output.shape)
        (1, 3, 200, 200)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers,
                 num_memory_layers,
                 kernel_size_conv,
                 kernel_size_lstm,
                 compute_dtype=mstype.float32):
        super(EhdnnNet, self).__init__()
        self.compute_dtype = compute_dtype
        layers = []
        layers.extend([ConvolutionalLayer(in_channels=in_channels, out_channels=out_channels, num_layers=num_layers,
                                          kernel_size=kernel_size_conv).to_float(self.compute_dtype)])
        layers.extend(
            [MemoryLayer(num_layers=num_memory_layers, kernel_size=kernel_size_lstm).to_float(self.compute_dtype)])
        layers.extend([DeConvolutionalLayer(in_channels=out_channels, out_channels=in_channels, num_layers=num_layers,
                                            kernel_size=kernel_size_conv).to_float(self.compute_dtype)])
        self.build_block = nn.SequentialCell(layers)

    def construct(self, x):
        output = self.build_block(x)
        return P.Cast()(output, mstype.float32)


class HybridLoss(nn.Cell):
    """
    Hybrid_loss Function
    HE = -log(MSSIM) + 0.5*RMSE
    """

    def __init__(self):
        super(HybridLoss, self).__init__()
        self.sub = ops.Sub()
        self.log = ops.Log()
        self.rmse = nn.RMSELoss()
        self.mssim = nn.SSIM()

    def construct(self, prediction, real):
        loss_1 = self.rmse(prediction, real)
        loss_2 = self.log(self.mssim(prediction, real))
        return self.sub(0.5 * loss_1, loss_2)
