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
"""CTEFNet base class"""
import mindspore.nn as nn
from mindspore import ops


class CTEFNet(nn.Cell):
    r"""
    This class is used to define CHEFNet, a deeplearning model use to predict nino3.4 index.

    Args:
        cov_hidden_channels (int): The channels of hidden convolution layer.Default: ``60``.
        cov_out_channels (int): The channels of output convolution layer.Default: ``15``.
        heads (int): The number of heads of transformer encoder.Default: ``3``.
        num_layer (int): The number of transformer encoder layer.Default: ``4``.
        feedforward_dims (int): The dimensions of transformer feedforward layer.Default: ``256``.
        dropout (float): The dropout rate of transformer encoder.Default: ``0.1``.
        obs_time (int): The length of time the data can be observed.Default: ``12``.
        pred_time (int): The length of data model predited.Default: ``24``.

    Inputs:
         - **x** (Tensor) - Tensor of shape :math:`(batch\_size, obs\_time, channels, height\_size, width\_size)`.

    Outputs:
        Tensor, the output of the DgmrDiscriminator.

        - **output** (Tensor) - Tensor of shape :math:`(batch\_size, obs\_time + pred\_time)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.common.initializer import initializer, Normal
        >>> from src import CTEFNet
        >>> B, T, C, H, W = 16, 12, 3, 24, 72
        >>> x = initializer(Normal(), [B, T, C, H, W])
        >>> net = CTEFNet(obs_time=T, pred_time=24)
        >>> output = net(x)
        >>> print(output.shape)
        (16, 32)
    """
    def __init__(self,
                 cov_hidden_channels=60,
                 cov_out_channels=15,
                 heads=3,
                 num_layer=4,
                 feedforward_dims=256,
                 dropout=0.1,
                 obs_time=12,
                 pred_time=24
                 ):
        super().__init__()
        self.cov_hidden_channels = cov_hidden_channels
        self.cov_out_channels = cov_out_channels
        self.heads = heads
        self.num_layer = num_layer
        self.feedforward_dims = feedforward_dims
        self.dropout = dropout

        self.obs_time = obs_time
        self.pred_time = pred_time

        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=self.cov_hidden_channels, kernel_size=(4, 8), pad_mode="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=self.cov_hidden_channels, out_channels=self.cov_hidden_channels, kernel_size=(2, 4),
                      pad_mode="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=self.cov_hidden_channels, out_channels=self.cov_out_channels, kernel_size=(2, 4),
                      pad_mode="same"),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.AdaptiveAvgPool2d((3, 6)),
            nn.Flatten()])

        encoderlayer = nn.TransformerEncoderLayer(3 * 6 * self.cov_out_channels, self.heads,
                                                  dim_feedforward=feedforward_dims, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoderlayer, num_layers=self.num_layer)
        self.res = nn.Dense(18 * self.cov_out_channels * self.obs_time, 18 * self.cov_out_channels * self.obs_time)
        self.head = nn.Dense(18 * self.cov_out_channels * self.obs_time, self.obs_time + self.pred_time)

    def construct(self, x):
        """CTEFNet forward function"""
        fea = ops.unsqueeze(self.conv(x[:, 0, :, :, :]), dim=1)
        for c in range(1, self.obs_time):
            fea = ops.cat([fea, ops.unsqueeze(self.conv(x[:, c, :, :, :]), dim=1)], axis=1)
        out = self.encoder(fea)

        fea = ops.flatten(fea)
        out = ops.flatten(out)
        fea = self.res(fea)
        out = fea + out
        out = self.head(out)
        return out
