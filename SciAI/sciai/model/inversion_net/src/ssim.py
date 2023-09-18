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
"""Structural Similarity Index"""

from math import exp
import mindspore
from mindspore import nn
from mindspore import ops

def gaussian(window_size, sigma):
    gauss = mindspore.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    window = ops.repeat_elements(window_2d, channel, axis=0)
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    """Calculation for ssim """
    mu1 = ops.conv2d(img1, window, pad_mode="pad", padding=window_size//2, groups=channel)
    mu2 = ops.conv2d(img2, window, pad_mode="pad", padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = ops.conv2d(img1 * img1, window, pad_mode="pad", padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = ops.conv2d(img2 * img2, window, pad_mode="pad", padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = ops.conv2d(img1 * img2, window, pad_mode="pad", padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma12 + c2))/((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean((1, 2, 3))

class SSIM(nn.Cell):
    """cell for Structural Similarity"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def construct(self, img1, img2):
        channel = img1.shape[1]
        return _ssim(img1, img2, self.window, self.window_size, channel, self.size_average)
