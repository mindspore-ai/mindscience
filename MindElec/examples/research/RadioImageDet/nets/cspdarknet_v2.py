# Copyright 2024 Huawei Technologies Co., Ltd
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
# ===========================================================================
"""backbone for YOLOv2"""
import mindspore as ms
import mindspore.nn as nn


class SiLU(nn.Cell):
    @staticmethod
    def construct(x):
        return x * ms.ops.sigmoid(x)


def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Focus(nn.Cell):
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def construct(self, x):
        # 320, 320, 12 => 320, 320, 64
        return self.conv(
            # 640, 640, 3 => 320, 320, 12
            ms.ops.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2]
                ], 1
            )
        )


class Conv(nn.Cell):
    """Convolution layer"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, pad_mode='pad', padding=autopad(
                k, p), group=g, has_bias=False, weight_init='normal')
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Cell):
    """
    Standard bottleneck
    ch_in, ch_out, shortcut, groups, expansion
    """
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def construct(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Cell):
    """
    CSP Bottleneck with 3 convolutions
    ch_in, ch_out, number, shortcut, groups, expansion
    """
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.SequentialCell(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def construct(self, x):
        return self.cv3(ms.ops.cat(
            (
                self.m(self.cv1(x)),
                self.cv2(x)
            ), axis=1))


class SPP(nn.Cell):
    """Spatial pyramid pooling layer"""
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.CellList([nn.MaxPool2d(
            kernel_size=x, stride=1, pad_mode='pad', padding=x // 2) for x in k])

    def construct(self, x):
        x = self.cv1(x)
        return self.cv2(ms.ops.cat([x] + [m(x) for m in self.m], 1))


class CSPDarknet(nn.Cell):
    """CSPDarknet for YOLOv2"""
    def __init__(self, base_channels, base_depth):
        super().__init__()
        self.stem = Focus(3, base_channels, k=3)
        self.dark2 = nn.SequentialCell(
            Conv(base_channels, base_channels * 2, 3, 2),
            C3(base_channels * 2, base_channels * 2, base_depth),
        )
        self.dark3 = nn.SequentialCell(
            Conv(base_channels * 2, base_channels * 4, 3, 2),
            C3(base_channels * 4, base_channels * 4, base_depth * 3),
        )
        self.dark4 = nn.SequentialCell(
            Conv(base_channels * 4, base_channels * 8, 3, 2),
            C3(base_channels * 8, base_channels * 8, base_depth * 3),
        )
        self.dark5 = nn.SequentialCell(
            Conv(base_channels * 8, base_channels * 16, 3, 2),
            SPP(base_channels * 16, base_channels * 16),
            C3(base_channels * 16, base_channels *
               16, base_depth, shortcut=False),
        )

    def construct(self, x):
        """construct the network"""
        x = self.stem(x)
        x = self.dark2(x)
        x = self.dark3(x)
        x = self.dark4(x)
        x = self.dark5(x)
        return x
