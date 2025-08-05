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
"""KNO1D"""
import mindspore.common.dtype as mstype
from mindspore import ops, nn, Tensor

from .dft import SpectralConv1dDft
from ...utils.check_func import check_param_type


class KNO1D(nn.Cell):
    r"""
    The 1-dimensional Koopman Neural Operator (KNO1D) contains a encoder layer and a decoder layer,
    multiple Koopman layers.
    The details can be found in `KoopmanLab: machine learning for solving complex physics equations
     <https://arxiv.org/pdf/2301.01104.pdf>`_.

    Args:
        in_channels (int): The number of channels in the input space. Default: ``1``.
        channels (int): The number of channels after dimension lifting of the input. Default: ``32``.
        modes (int): The number of low-frequency components to keep. Default: ``16``.
        resolution (int): The spatial resolution of the input. Default: ``1024``.
        depths (int): The number of KNO layers. Default: ``4``.
        compute_dtype (dtype.Number): The computation type of dense. Default: ``mstype.float16``.
            Should be ``mstype.float32`` or ``mstype.float16``. mstype.float32 is recommended for
            the GPU backend, mstype.float16 is recommended for the Ascend backend.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(batch\_size, resolution, in\_channels)`.

    Outputs:
        Tensor, the output of this KNO network.

        - **output** (Tensor) -Tensor of shape :math:`(batch\_size, resolution, in\_channels)`.

    Raises:
        TypeError: If `in_channels` is not an int.
        TypeError: If `channels` is not an int.
        TypeError: If `modes` is not an int.
        TypeError: If `depths` is not an int.
        TypeError: If `resolution` is not an int.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindflow.cell.neural_operators import KNO1D
        >>> input_ = Tensor(np.ones([32, 1024, 1]), mstype.float32)
        >>> net = KNO1D()
        >>> x, x_reconstruct = net(input_)
        >>> print(x.shape, x_reconstruct.shape)
        (32, 1024, 1) (32, 1024, 1)
    """
    def __init__(self,
                 in_channels=1,
                 channels=32,
                 modes=16,
                 depths=4,
                 resolution=1024,
                 compute_dtype=mstype.float32):
        super().__init__()
        check_param_type(in_channels, "in_channels",
                         data_type=int, exclude_type=bool)
        check_param_type(channels, "channels",
                         data_type=int, exclude_type=bool)
        check_param_type(modes, "modes",
                         data_type=int, exclude_type=bool)
        check_param_type(depths, "depths",
                         data_type=int, exclude_type=bool)
        check_param_type(resolution, "resolution",
                         data_type=int, exclude_type=bool)
        self.in_channels = in_channels
        self.channels = channels
        self.modes = modes
        self.depths = depths
        self.resolution = resolution
        self.enc = nn.Dense(in_channels, channels, has_bias=True)
        self.dec = nn.Dense(channels, in_channels, has_bias=True)
        self.koopman_layer = SpectralConv1dDft(channels, channels, modes, resolution, compute_dtype=compute_dtype)
        self.w0 = nn.Conv1d(channels, channels, 1, has_bias=True)

    def construct(self, x: Tensor):
        """KNO1D forward function.

        Args:
            x (Tensor): Input Tensor.
        """
        # reconstruct
        x_reconstruct = self.enc(x)
        x_reconstruct = ops.tanh(x_reconstruct)
        x_reconstruct = self.dec(x_reconstruct)

        # predict
        x = self.enc(x)
        x = ops.tanh(x)
        x = x.transpose(0, 2, 1)
        x_w = x
        for _ in range(self.depths):
            x1 = self.koopman_layer(x)
            x = ops.tanh(x + x1)
        x = ops.tanh(self.w0(x_w) + x)
        x = x.transpose(0, 2, 1)
        x = self.dec(x)
        return x, x_reconstruct
