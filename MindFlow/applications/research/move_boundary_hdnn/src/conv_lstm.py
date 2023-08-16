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
from mindspore import nn, ops, numpy, float32


class ConvLSTMCell(nn.Cell):
    """
    The cell of ConvLSTM, which sequentially processes input data through convolution, regularization,  LSTM operations
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              stride=1,
                              pad_mode="same",
                              padding=0,
                              has_bias=self.bias,
                              data_format="NCHW")
        self.norm = nn.BatchNorm2d(4 * self.hidden_dim)

    def construct(self, input_tensor, cur_state):
        """
        Transform the input_tensor and cur_state, perform convolution and regularization, then perform LSTM operations
        """
        h_cur, c_cur = cur_state

        combined = ops.concat(input_x=(input_tensor, h_cur), axis=1)
        combined_conv = self.conv(combined)
        combined_conv = self.norm(combined_conv)
        cc_i, cc_f, cc_o, cc_g = ops.split(input_x=combined_conv, axis=1, output_num=4)

        i = ops.sigmoid(cc_i)
        f = ops.sigmoid(cc_f)
        o = ops.sigmoid(cc_o)
        g = ops.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * ops.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, h_ini):
        """
        Initial state tensor initialization. State tensor 0 initialization for the first timestamp
        Parameters
        ----------
        batch_size: int
            Minimum batch size of trained samples
        image_size: tuple of size[H,W]
            Height and width of data images
        """
        height, width = image_size
        h_ini = numpy.reshape(h_ini, (batch_size, 1, 1, 1))
        h_ini = numpy.broadcast_to(h_ini, (batch_size, self.hidden_dim, height, width))

        init_h = h_ini * numpy.ones(shape=(batch_size, self.hidden_dim, height, width)).astype(float32)
        init_c = numpy.zeros(shape=(batch_size, self.hidden_dim, height, width)).astype(float32)

        return (init_h, init_c)


class ConvLSTM(nn.Cell):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
    Input:
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]
    Output:
        layer_output_list--size=[B,T,hidden_dim,H,W]
        last_state_list--h.size=c.size = [B,hidden_dim,H,W]
        A tuple of two lists of length num_layers .
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
        self.cell_list = nn.CellList(cell_list)

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """Detect the input kernel_ Does the size meet the requirements and require a kernel_size is list or tuple"""
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """Expanding to multi-layer LSTM scenarios"""
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def construct(self, input_tensor, h0):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b1, _, _, h1, w1 = input_tensor.shape
        hidden_state = self._init_hidden(batch_size=b1, image_size=(h1, w1), h_ini=h0)

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.shape[1]
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = ops.stack(output_inner, axis=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size, h_ini):
        """
        Initialize the input state 0 of the first timestamp of all LSTM layers
        Parameters
        ----------
        batch_size: int
            Minimum batch size of trained samples
        image_size: tuple of size[H,W]
            Height and width of data images
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size, h_ini))
        return init_states
