# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
pocket
"""
from mindspore import nn
from mindspore import ops
from .. import util
from ..core import Registry as R


class ConvNorm(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, active):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=1, group=1, has_bias=True, pad_mode='pad')
        self.norm = nn.BatchNorm3d(num_features=out_channels, eps=1e-05, momentum=0.1,
                                   affine=True, use_batch_statistics=True)
        self.active = active

    def construct(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        out = self.conv(x)
        out = self.norm(out)
        if self.active:
            out = ops.relu(out)
        return out


class SiteConvolution(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block11 = ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=1)
        self.block12 = ConvNorm(out_channels, out_channels, kernel_size=3, stride=1, padding=1, active=1)
        self.block13 = ConvNorm(out_channels, out_channels, kernel_size=1, stride=1, padding=0, active=0)
        self.block21 = ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=0)

    def construct(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        h11 = self.block11(x)
        h12 = self.block12(h11)
        h13 = self.block13(h12)
        h21 = self.block21(x)
        o = ops.add(h13, h21)
        o = ops.relu(o)
        return o


class SiteUpConvolution(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels, out_dimensions, stride=1):
        super().__init__()
        self.block10 = nn.Upsample(size=(out_dimensions, out_dimensions, out_dimensions), mode='area')
        self.block11 = ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=1)
        self.block12 = ConvNorm(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, active=1)
        self.block13 = ConvNorm(out_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=0)
        self.block20 = nn.Upsample(size=(out_dimensions, out_dimensions, out_dimensions), mode='area')
        self.block21 = ConvNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, active=0)

    def construct(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        h10 = self.block10(x)
        h11 = self.block11(h10)
        h12 = self.block12(h11)
        h13 = self.block13(h12)
        h20 = self.block20(x)
        h21 = self.block21(h20)
        o = ops.add(h13, h21)
        o = ops.relu(o)
        return o


class SiteIdentity(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block11 = ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, padding=0, active=1)
        self.block12 = ConvNorm(in_channels, out_channels, kernel_size=3, stride=1, padding=1, active=1)
        self.block21 = ConvNorm(in_channels, out_channels, kernel_size=1, stride=1, padding=0, active=0)

    def construct(self, x):
        """_summary_

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        h11 = self.block11(x)
        h12 = self.block12(h11)
        h21 = self.block21(x)
        o = ops.add(h12, h21)
        o = ops.relu(o)
        return o


@R.register('scenario.Pocket')
class DrugSiteMind(nn.Cell):
    """
    https://github.com/jivankandel/PUResNet  #https://gitlab.com/cheminfIBB/kalasanty
    ops.concat  nn.init.  Sequential
    ReLU/LeakyReLU/ELU BatchNorm3d/InstanceNorm3d MaxPool3d Upsample/ConvTranspose3d Dropout Softmax
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        ci = 18
        c1 = ci * 1
        c2 = ci*2
        c3 = ci*4
        c4 = ci*8
        c5 = ci*16
        self.smooth = smooth
        self.down_c1 = SiteConvolution(18, c1, stride=1)
        self.iden_d1 = SiteIdentity(c1, c1)
        self.iden_x1 = SiteIdentity(c1, c1)

        self.down_c2 = SiteConvolution(c1, c2, stride=2)
        self.iden_d2 = SiteIdentity(c2, c2)
        self.iden_x2 = SiteIdentity(c2, c2)

        self.down_c3 = SiteConvolution(c2, c3, stride=2)
        self.iden_d3 = SiteIdentity(c3, c3)
        self.iden_x3 = SiteIdentity(c3, c3)

        self.down_c4 = SiteConvolution(c3, c4, stride=3)
        self.iden_d4 = SiteIdentity(c4, c4)
        self.iden_x4 = SiteIdentity(c4, c4)

        self.down_c5 = SiteConvolution(c4, c5, stride=3)
        self.iden_d5 = SiteIdentity(c5, c5)

        # UNET, down(batch:-1, channel:288(c5), dimenson:1,dimenson:1,dimenson:1) to up...
        self.up_c4 = SiteUpConvolution(c5, c5, out_dimensions=3)
        self.iden_u4 = SiteIdentity(c5, c5)

        self.up_c3 = SiteUpConvolution(c5+c4, c4, out_dimensions=9)
        self.iden_u3 = SiteIdentity(c4, c4)

        self.up_c2 = SiteUpConvolution(c4+c3, c3, out_dimensions=18)
        self.iden_u2 = SiteIdentity(c3, c3)

        self.up_c1 = SiteUpConvolution(c3+c2, c2, out_dimensions=36)
        self.iden_u1 = SiteIdentity(c2, c2)

        self.out_conv = nn.Conv3d(in_channels=c2+c1, out_channels=1, kernel_size=1, stride=1,
                                  padding=0, dilation=1, group=1, has_bias=True, pad_mode='pad')

    def construct(self, x):
        """permute(0,4,1,2,3):(-1,36,36,36,18)(N,D,H,W,C) -> (-1,18,36,36,36) (N,C,D,H,W)

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        hd11 = self.down_c1(x)
        hd12 = self.iden_d1(hd11)
        i1 = self.iden_x1(hd12)

        hd21 = self.down_c2(hd12)
        hd22 = self.iden_d2(hd21)
        i2 = self.iden_x2(hd22)

        hd31 = self.down_c3(hd22)
        hd32 = self.iden_d3(hd31)
        i3 = self.iden_x3(hd32)

        hd41 = self.down_c4(hd32)
        hd42 = self.iden_d4(hd41)
        i4 = self.iden_x4(hd42)

        hd51 = self.down_c5(hd42)
        hd52 = self.iden_d5(hd51)

        # UNET, down to up ...
        hu41 = self.up_c4(hd52)
        hu42 = self.iden_u4(hu41)
        hu43 = ops.concat([hu42, i4], axis=1)

        hu31 = self.up_c3(hu43)
        hu32 = self.iden_u3(hu31)
        hu33 = ops.concat([hu32, i3], axis=1)

        hu21 = self.up_c2(hu33)
        hu22 = self.iden_u2(hu21)
        hu23 = ops.concat([hu22, i2], axis=1)

        hu11 = self.up_c1(hu23)
        hu12 = self.iden_u1(hu11)
        hu13 = ops.concat([hu12, i1], axis=1)

        oc = self.out_conv(hu13)
        o = ops.sigmoid(oc)
        return o

    def loss_fn(self, x, target):
        """
        #2*(x*y) / ((x+y)+1)  #https://zhuanlan.zhihu.com/p/86704421
        """
        x = x.permute(0, 4, 1, 2, 3)
        target = target.permute(0, 4, 1, 2, 3)
        predict = self(x)
        dice = 0.0
        position = ops.arange(36)
        site_numbers = target.sum(axis=(2, 3, 4)) + 1
        center_target_x = (target.sum(axis=(3, 4)) *
                           position).sum(2).reshape(target.shape[0], target.shape[1], 1) / (36 * site_numbers)
        center_target_y = (target.sum(axis=(2, 4)) *
                           position).sum(2).reshape(target.shape[0], target.shape[1], 1) / (36 * site_numbers)
        center_target_z = (target.sum(axis=(2, 3)) *
                           position).sum(2).reshape(target.shape[0], target.shape[1], 1) / (36 * site_numbers)

        max_predict_x = predict.max(axis=4).max(axis=3).argmax(axis=2) / 36.0
        max_predict_y = predict.max(axis=4).max(axis=2).argmax(axis=2) / 36.0
        max_predict_z = predict.max(axis=3).max(axis=2).argmax(axis=2) / 36.0

        center_p_t = (max_predict_x-center_target_x).pow(2) + \
            (max_predict_y - center_target_y).pow(2) + (max_predict_z-center_target_z).pow(2)

        for i in range(predict.shape[1]):  # predict = predict.squeeze(axis=1)
            dice += 2 * (predict[:, i] * target[:, i]).sum(axis=1).sum(axis=1).sum(axis=1) / \
                (predict[:, i].pow(2).sum(axis=1).sum(axis=1).sum(axis=1) +
                 target[:, i].pow(2).sum(axis=1).sum(axis=1).sum(axis=1) + self.smooth)
        dice = dice / predict.shape[1]
        loss = util.clip((1 - dice).mean(), 0, 1) + 0.1 * center_p_t.mean()
        return loss, (predict, target)
