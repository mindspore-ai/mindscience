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
"""network"""

import math
from mindspore import nn
from mindspore import ops
from mindspore import amp
from mindspore import Tensor
from mindspore.common.initializer import initializer, XavierUniform, Uniform, _calculate_fan_in_and_fan_out

from sciai.utils import print_log


def prepare_network(args):
    """Define network"""
    network = InversionNet(
        dim1=args.dims[0],
        dim2=args.dims[1],
        dim3=args.dims[2],
        dim4=args.dims[3],
        dim5=args.dims[4],
        sample_spatial=args.sample_spatial)

    # Define loss function
    lambda_g1v = Tensor(args.lambda_g1v)
    lambda_g2v = Tensor(args.lambda_g2v)
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()

    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        loss = lambda_g1v * loss_g1v + lambda_g2v * loss_g2v
        return loss, loss_g1v, loss_g2v

    if args.amp_level is not None:
        network = amp.auto_mixed_precision(network, args.amp_level)
        print_log('The model is run on amp_level: {}.'.format(args.amp_level))

    return network, criterion


NORM_LAYERS = {'bn': nn.BatchNorm2d, 'in': nn.InstanceNorm2d, 'ln': nn.LayerNorm}

def init_cells(net):
    """initialize the net"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            cell.weight.set_data(initializer(XavierUniform(), cell.weight.shape))
            if cell.has_bias:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1 / math.sqrt(fan_in)
                cell.bias.set_data(initializer(Uniform(bound), [cell.out_channels]))

class ConvBlock(nn.Cell):
    """based block"""
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn', relu_slop=0.2, dropout=None):
        super(ConvBlock, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            pad_mode='pad', padding=padding, has_bias=True,)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(relu_slop))
        if dropout:
            layers.append(nn.Dropout2d(0.8))
        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.layers(x)

class DeconvBlock(nn.Cell):
    """based deconvolution block"""
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, norm='bn'):
        super(DeconvBlock, self).__init__()
        layers = [nn.Conv2dTranspose(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                                     pad_mode='pad', padding=padding, has_bias=True)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.LeakyReLU(0.2))
        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.layers(x)

class ConvBlockTanh(nn.Cell):
    """deconvolution block with tanh"""
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, norm='bn'):
        super(ConvBlockTanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride,
                            pad_mode='pad', padding=padding, has_bias=True)]
        if norm in NORM_LAYERS:
            layers.append(NORM_LAYERS[norm](out_fea))
        layers.append(nn.Tanh())
        self.layers = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.layers(x)

class InversionNet(nn.Cell):
    """InversionNet"""
    def __init__(self, dim1=32, dim2=64, dim3=128, dim4=256, dim5=512, sample_spatial=1.0):
        super(InversionNet, self).__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 3, 0, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 1, 0, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 1, 0, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 1, 0, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 1, 0, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 1, 0, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 1, 0, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, math.ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlockTanh(dim1, 1)
        init_cells(self)

    def construct(self, x):
        """Network forward pass"""
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock6_1(x)
        x = self.convblock6_2(x)
        x = self.convblock7_1(x)
        x = self.convblock7_2(x)
        x = self.convblock8(x)
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = ops.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        x = self.deconv6(x)
        return x
