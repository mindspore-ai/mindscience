# Copyright 2020 Huawei Technologies Co., Ltd
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
"""resnet
The sample can be run on Ascend 910 AI processor.
mindspore 1.6
"""
import numpy as np
from mindspore import Tensor, Parameter, ops, nn
import mindspore.common.initializer as initializer
from mindspore.ops import operations as P


def weight_variable_0(shape):
    """weight_variable_0"""
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def weight_variable_1(shape):
    """weight_variable_1"""
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def conv3x3(in_channels, out_channels, stride=1, padding=0):
    """3x3 convolution """
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=padding, weight_init='HeUniform', has_bias=False,
                     pad_mode="same")


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=padding, weight_init='HeUniform',
                     has_bias=False, pad_mode="same")


def conv7x7(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=7, stride=stride, padding=padding, weight_init='HeUniform',
                     has_bias=False, pad_mode="same")


def fc_with_initialize(input_channels, out_channels):
    """fc_with_initialize"""
    return nn.Dense(input_channels, out_channels, weight_init='HeUniform', bias_init='Uniform')


class Bottleneck(nn.Cell):
    """bottleneck"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        """init block with down"""
        super(Bottleneck, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.in1 = InstanceNorm2d(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.in2 = InstanceNorm2d(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.in3 = InstanceNorm2d(out_channels)

        self.relu = ops.ReLU()

        if stride != 1 or in_channels != out_channels:
            self.down_sample = True
        else:
            self.down_sample = False

        if self.down_sample:
            self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride, padding=0)
            self.in_down_sample = InstanceNorm2d(out_channels)

        self.add = ops.Add()

    def construct(self, x):
        """construct"""
        identity = x

        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.in3(out)

        if self.down_sample:
            identity = self.conv_down_sample(identity)
            identity = self.in_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


# building block, 2层conv
class BasicBlock(nn.Cell):
    """basic block"""

    def __init__(self, in_channels, out_channels, stride=1):
        """init block with down"""
        super(BasicBlock, self).__init__()

        out_chls = out_channels  # // self.expansion
        self.conv1 = conv3x3(in_channels, out_chls, stride=stride, padding=0)
        self.in1 = InstanceNorm2d(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.in2 = InstanceNorm2d(out_chls)

        self.relu = ops.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.down_sample = True
        else:
            self.down_sample = False

        if self.down_sample:
            self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride, padding=0)
            self.in_down_sample = InstanceNorm2d(out_channels)

        self.add = ops.Add()

    def construct(self, x):
        """construct"""
        identity = x
        out = self.conv1(x)
        out = self.in1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.in2(out)

        if self.down_sample:
            identity = self.conv_down_sample(identity)
            identity = self.in_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class BasicBlockLn(nn.Cell):
    """basic block"""

    def __init__(self, in_channels, out_channels, ln_shape, stride=1):
        """init block with down"""
        super(BasicBlockLn, self).__init__()

        out_chls = out_channels  # // self.expansion
        self.conv1 = conv3x3(in_channels, out_chls, stride=stride, padding=0)
        self.ln_shape1 = self.calc_shape(ln_shape[0], ln_shape[1], 3, stride=(stride, stride), padding=(1, 1))
        self.ln1 = nn.LayerNorm([out_chls, *self.ln_shape1], begin_norm_axis=1, begin_params_axis=1)

        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.ln_shape2 = self.calc_shape(self.ln_shape1[0], self.ln_shape1[1], 3, padding=(1, 1))
        self.ln2 = nn.LayerNorm([out_chls, *self.ln_shape2], begin_norm_axis=1, begin_params_axis=1)

        self.relu = ops.ReLU()
        if stride != 1 or in_channels != out_channels:
            self.down_sample = True
        else:
            self.down_sample = False

        if self.down_sample:
            self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride, padding=0)
            self.ln_shape3 = self.calc_shape(ln_shape[0], ln_shape[1], 1, stride=(stride, stride))
            self.ln_down_sample = nn.LayerNorm([out_channels, *self.ln_shape3], begin_norm_axis=1, begin_params_axis=1)

        self.add = ops.Add()

    def calc_shape(self, h_in, w_in, ksize, dilation=(1, 1), stride=(1, 1), padding=(0, 0)):
        h_out = (h_in + 2 * padding[0] - dilation[0] * (ksize - 1) - 1) / stride[0] + 1
        w_out = (w_in + 2 * padding[1] - dilation[1] * (ksize - 1) - 1) / stride[1] + 1
        return int(h_out), int(w_out)

    def construct(self, x):
        """construct"""
        identity = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)

        if self.down_sample:
            identity = self.conv_down_sample(identity)
            identity = self.ln_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResnetLn(nn.Cell):
    """ResNet"""

    # resnet50: layer_nums = [3, 4, 6, 3], in_c = [64, 256, 512, 1024], out_c = [256, 512, 1024, 2048], strides = [1,
    # 2, 2, 2]
    def __init__(self, block, layer_nums, img_channel, in_channels, out_channels, strides, num_classes=100,
                 batch_size=32, first_stride=2, logit=True):
        """init"""
        super(ResnetLn, self).__init__()

        self.batch_size = batch_size

        self.conv1 = conv7x7(img_channel, in_channels[0], stride=first_stride, padding=0)
        ln_shape = self.calc_shape(40, 40, 7, stride=(first_stride, first_stride), padding=(3, 3))
        self.ln1 = nn.LayerNorm([in_channels[0], *ln_shape], begin_norm_axis=1, begin_params_axis=1)

        self.relu = ops.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=first_stride, pad_mode="same")  # 似乎用不到maxpooling

        self.layer1, ln_shape = self.make_layer(block, ln_shape, layer_nums[0], in_channel=in_channels[0],
                                                out_channel=out_channels[0], stride=strides[0])
        self.layer2, ln_shape = self.make_layer(block, ln_shape, layer_nums[1], in_channel=in_channels[1],
                                                out_channel=out_channels[1], stride=strides[1])
        self.layer3, ln_shape = self.make_layer(block, ln_shape, layer_nums[2], in_channel=in_channels[2],
                                                out_channel=out_channels[2], stride=strides[2])
        self.layer4, ln_shape = self.make_layer(block, ln_shape, layer_nums[3], in_channel=in_channels[3],
                                                out_channel=out_channels[3], stride=strides[3])

        self.pool = ops.ReduceMean(keep_dims=True)
        self.squeeze = ops.Squeeze(axis=(2, 3))

        self.logit = logit
        if logit:
            self.num_classes = num_classes
            self.fc = fc_with_initialize(out_channels[3], num_classes)  # 512 * block.expansion
        else:
            self.flatten = nn.Flatten()
            self.to_image_size = nn.Conv2d(out_channels[3], 1, 3, 1, pad_mode='same', padding=0, has_bias=False)
            self.tanh = nn.Tanh()

    def make_layer(self, block, ln_shape, layer_num, in_channel, out_channel, stride):
        """
        make layer
        Args:
            block: block
            ln_shape: ln_shape
            layer_num: layer_num
            in_channel: in_channel
            out_channel: out_channel
            stride: stride

        Returns: layer sequential cell

        """
        layers = []
        resnet_block = block(in_channel, out_channel, ln_shape, stride=stride)  # 只是init，没有construct，不能输出形状
        ln_shape = resnet_block.ln_shape2
        layers.append(resnet_block)

        for _ in range(1, layer_num - 1):
            resnet_block = block(out_channel, out_channel, ln_shape, stride=1)
            ln_shape = resnet_block.ln_shape2
            layers.append(resnet_block)

        return nn.SequentialCell(layers), ln_shape

    def calc_shape(self, h_in, w_in, ksize, dilation=(1, 1), stride=(1, 1), padding=(0, 0)):
        h_out = (h_in + 2 * padding[0] - dilation[0] * (ksize - 1) - 1) / stride[0] + 1
        w_out = (w_in + 2 * padding[1] - dilation[1] * (ksize - 1) - 1) / stride[1] + 1
        return int(h_out), int(w_out)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.logit:
            x = self.pool(x, (2, 3))
            x = self.squeeze(x)
            x = self.fc(x)
        else:
            x = self.to_image_size(x)
            x = self.tanh(x)
            x = self.flatten(x)

        return x


class _ResNet(nn.Cell):
    """ResNet"""

    # resnet50: layer_nums = [3, 4, 6, 3], in_c = [64, 256, 512, 1024], out_c = [256, 512, 1024, 2048], strides = [1,
    # 2, 2, 2]
    def __init__(self, block, layer_nums, img_channel, in_channels, out_channels, strides, num_classes=100,
                 batch_size=32, first_stride=2, logit=True):
        """init"""
        super(_ResNet, self).__init__()

        self.batch_size = batch_size

        self.conv1 = conv7x7(img_channel, in_channels[0], stride=first_stride, padding=0)
        self.in1 = InstanceNorm2d(in_channels[0])
        self.relu = ops.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=first_stride, pad_mode="same")  # 似乎用不到maxpooling

        self.layer1 = self.make_layer(block, layer_nums[0], in_channel=in_channels[0], out_channel=out_channels[0],
                                      stride=strides[0])
        self.layer2 = self.make_layer(block, layer_nums[1], in_channel=in_channels[1], out_channel=out_channels[1],
                                      stride=strides[1])
        self.layer3 = self.make_layer(block, layer_nums[2], in_channel=in_channels[2], out_channel=out_channels[2],
                                      stride=strides[2])
        self.layer4 = self.make_layer(block, layer_nums[3], in_channel=in_channels[3], out_channel=out_channels[3],
                                      stride=strides[3])

        self.pool = ops.ReduceMean(keep_dims=True)
        self.squeeze = ops.Squeeze(axis=(2, 3))

        self.logit = logit
        if logit:
            self.num_classes = num_classes
            self.fc = fc_with_initialize(out_channels[3], num_classes)  # 512 * block.expansion
        else:
            self.flatten = nn.Flatten()
            self.to_image_size = nn.Conv2d(out_channels[3], 1, 3, 1, pad_mode='same', padding=0, has_bias=False)
            self.tanh = nn.Tanh()

    def make_layer(self, block, layer_num, in_channel, out_channel, stride):
        layers = []
        resnet_block = block(in_channel, out_channel, stride=stride)
        layers.append(resnet_block)

        for _ in range(1, layer_num - 1):
            resnet_block = block(out_channel, out_channel, stride=1)
            layers.append(resnet_block)

        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.logit:
            x = self.pool(x, (2, 3))
            x = self.squeeze(x)
            x = self.fc(x)
        else:
            x = self.to_image_size(x)
            x = self.tanh(x)
            x = self.flatten(x)

        return x


class InstanceNorm2d(nn.Cell):
    """InstanceNorm2d"""

    def __init__(self, channel):
        super(InstanceNorm2d, self).__init__()
        self.gamma = Parameter(initializer.initializer(init=Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32)),
                                                       shape=[1, channel, 1, 1]), name='gamma')
        self.beta = Parameter(initializer.initializer(init=initializer.Zero(), shape=[1, channel, 1, 1]), name='beta')
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sub = P.Sub()
        self.add = P.Add()
        self.rsqrt = P.Rsqrt()
        self.mul = P.Mul()
        self.tile = P.Tile()
        self.reshape = P.Reshape()
        self.eps = Tensor(np.ones(shape=[1, channel, 1, 1], dtype=np.float32) * 1e-5)
        self.cast2fp32 = P.Cast()

    def construct(self, x):
        mean = self.reduce_mean(x, (2, 3))
        variance = self.reduce_mean(self.square(self.sub(x, mean)), (2, 3))  # _stop_grad
        variance = variance + self.eps
        inv = self.rsqrt(variance)
        normalized = self.sub(x, mean) * inv
        x_in = self.add(self.mul(self.gamma, normalized), self.beta)
        return x_in


def _resnet50(block=Bottleneck,
              layer_nums=(3, 4, 6, 3),
              img_channel=3,
              in_channels=(64, 256, 512, 1024),
              out_channels=(256, 512, 1024, 2048),
              strides=(1, 2, 2, 2),
              num_classes=100
              ):
    """create resnet50"""
    return _ResNet(block, layer_nums, img_channel, in_channels, out_channels, strides, num_classes=num_classes,
                   batch_size=32)


def _resnet34(block=BasicBlock,
              layer_nums=(3, 4, 6, 3),
              img_channel=3,
              in_channels=(64, 128, 256, 128),
              out_channels=(128, 256, 128, 64),
              strides=(1, 1, 1, 1),
              num_classes=100,
              first_stride=2,
              logit=True
              ):
    """create resnet34"""
    return _ResNet(block, layer_nums, img_channel, in_channels, out_channels, strides, num_classes=num_classes,
                   batch_size=32, first_stride=first_stride, logit=logit)


def _resnet34_ln(block=BasicBlockLn,
                 layer_nums=(3, 4, 6, 3),
                 img_channel=3,
                 in_channels=(64, 128, 256, 128),
                 out_channels=(128, 256, 128, 64),
                 strides=(1, 1, 1, 1),
                 num_classes=100,
                 first_stride=2,
                 logit=True
                 ):
    """create resnet34"""
    return ResnetLn(block, layer_nums, img_channel, in_channels, out_channels, strides, num_classes=num_classes,
                    batch_size=32, first_stride=first_stride, logit=logit)
