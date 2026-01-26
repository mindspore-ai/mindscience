# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""normalizer"""
import numpy as np
from mindspore import Tensor

class UnitTransformer():
    """unit transformer"""
    def __init__(self, x):
        self.mean = Tensor(np.mean(x, axis=(0, 1), keepdims=True))
        self.std = Tensor(np.std(x, axis=(0, 1), keepdims=True) + 1e-8)

    def encode(self, x):
        """encode"""
        x = (x - self.mean.asnumpy()) / (self.std.asnumpy())
        return x

    def decode(self, x):
        """decode"""
        return x * self.std + self.mean

    def transform(self, x, inverse=True, component='all'):
        """transform"""
        if component in ('all', 'all-reduce'):
            if inverse:
                orig_shape = x.shape
                out = (x * (self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                out = (x - self.mean) / self.std
        else:
            if inverse:
                orig_shape = x.shape
                out = (x * (self.std[:, component] - 1e-8) + \
                       self.mean[:, component]).view(orig_shape)
            else:
                out = (x - self.mean[:, component]) / self.std[:, component]
        return out
