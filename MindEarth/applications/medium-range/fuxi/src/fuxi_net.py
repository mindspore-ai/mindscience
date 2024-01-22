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
"""FuXi base class"""
import mindspore.numpy as mnp
from mindspore import nn, ops

from .fuxi import CubeEmbed, PatchRecover
from .fuxi import DownSample, BaseBlock, UpSample


class FuXiNet(nn.Cell):
    r"""
    The FuXi is a cascaded ML weather forecasting system based on Swin Transformer.
    The details can be found in `FuXi: a cascade machine learning forecasting system for 15-day global weather forecast
    <https://www.nature.com/articles/s41612-023-00512-1>`_.

    Parameters:
         depths (int): The number of Swin Transformer Blocks.
         in_channels (int): The number of channels in the input space.
         out_channels (int): The number of channels in the output space.
         h_size (int): The height of ERA5 Data.
         w_size (int): The width of ERA5 Data.
         level_feature_size (int): The size of level feature.
         pressure_level_num (int): The number of pressure level num.

    Inputs:
        - **input** (Tensor) - Tensor of shape :math:`(batch\_size, height\_size * width\_size, feature\_size)` .

    Outputs:
        - **output** (Tensor) - Tensor of shape :math:`(level\_size, height\_size, width\_size, feature\_size)` .
        - **output_surface** (Tensor) - Tensor of shape :math:`(surface\_size, height\_size, width\_size)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import context, Tensor
        >>> import mindspore.common.dtype as mstype
        >>> from src.fuxi_net import FuXiNet
        >>>
        >>> depths = 18
        >>> in_channels = 96
        >>> out_channels = 192
        >>> h_size = 720
        >>> w_size = 1440
        >>> level_feature_size = 5
        >>> pressure_level_num = 13
        >>> surface_feature_size = 4
        >>> batch_size = 1
        >>> feature_num = 69
        >>> kernel_size = (2, 4, 4)
        >>> x = Tensor(np.random.rand(batch_size, h_size * w_size, feature_num).astype(np.float32), ms.float32)
        >>> fuxi_model = FuXiNet(depths=depths,
        >>>                      in_channels=in_channels,
        >>>                      out_channels=out_channels,
        >>>                      h_size=h_size,
        >>>                      w_size=w_size,
        >>>                      level_feature_size=level_feature_size,
        >>>                      pressure_level_num=pressure_level_num,
        >>>                      surface_feature_size=surface_feature_size,
        >>>                      kernel_size=kernel_size)
        >>> black_list = (nn.GELU, nn.Softmax, nn.BatchNorm2d, nn.LayerNorm, nn.SiLU)
        >>> fuxi_model.to_float(mstype.float16)
        >>> for _, cell in fuxi_model.cells_and_names():
        >>>     if isinstance(cell, black_list):
        >>>         cell.to_float(mstype.float32)
        >>> output, output_surface = fuxi_model(x)
        >>> print(output.shape, output_surface.shape)
        (5, 13, 720, 1440) (4, 720, 1440)
    """
    def __init__(self,
                 depths=18,
                 in_channels=96,
                 out_channels=192,
                 h_size=720,
                 w_size=1440,
                 level_feature_size=5,
                 pressure_level_num=13,
                 surface_feature_size=4,
                 kernel_size=(2, 4, 4)):
        super().__init__()
        self.out_channels = out_channels
        self.input_shape = [int(mnp.ceil(pressure_level_num / 2) + 1), h_size // 8, w_size // 8]
        self.cube_embed = CubeEmbed(in_channels, h_size, w_size, level_feature_size,
                                    pressure_level_num, surface_feature_size)
        self.down_sample = DownSample(in_channels=in_channels, out_channels=out_channels)

        swin_list = []
        for _ in range(depths):
            swin_list.append(BaseBlock(in_channels=out_channels, input_shape=self.input_shape))
        self.swin_block = nn.SequentialCell(swin_list)

        self.up_sample = UpSample(in_channels=in_channels * 4, out_channels=out_channels)

        self.patch_recover = PatchRecover(out_channels, h_size, w_size, level_feature_size,
                                          pressure_level_num, surface_feature_size, kernel_size)

    def construct(self, inputs):
        """FuXi forward function.

        Args:
            inputs (Tensor): Input Tensor.
        """
        out = self.cube_embed(inputs)
        out_down_sample = self.down_sample(out)
        _, z_size, h_size, w_size, _ = out_down_sample.shape
        out_skip = out_down_sample.reshape(1, -1, self.out_channels)

        out_swin_block = self.swin_block(out_down_sample)
        out_swin_block = out_swin_block.reshape(1, -1, self.out_channels)
        out_swin_block = ops.concat((out_skip, out_swin_block), axis=2)
        out_swin_block = out_swin_block.reshape(1, z_size, h_size, w_size, self.out_channels * 2)

        out_up_sample = self.up_sample(out_swin_block)
        output, output_surface = self.patch_recover(out_up_sample)

        return output, output_surface
