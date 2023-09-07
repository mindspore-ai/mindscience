# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
ESM1LayerNorm代码迁移： pytorch -> mindspore
"""
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore.ops import function as F
import numpy as np

class ESM1LayerNorm(nn.Cell):
    """ESM1LayerNorm"""
    def __init__(self, hidden_size, eps: float = 1e-12, affine: bool = True):
        """Construct a layernorm layer in the TF style (eps inside the sqrt)."""
        super().__init__()
        self.hidden_size = (hidden_size,) if isinstance(hidden_size, int) else tuple(hidden_size)
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(Tensor(np.ones(hidden_size)).astype(mstype.float32))
            self.bias = Parameter(Tensor(np.zeros(hidden_size)).astype(mstype.float32))
        else:
            self.weight, self.bias = None, None

    def construct(self, x: Tensor):
        dims = tuple(-(i + 1) for i in range(len(self.hidden_size)))
        means = x.mean(dims, keep_dims=True)
        x_zeromean = x - means
        variances = x_zeromean.pow(2).mean(keep_dims=True)
        x = x_zeromean / F.sqrt(variances + self.eps)
        if self.affine:
            x = (self.weight * x) + self.bias
        return x

if __name__ == "__main__":
    esm1_layernorm = ESM1LayerNorm(12)
    X = np.arange(24).reshape(2, 12)
    X = Tensor(X).astype(mstype.float32)
    res = esm1_layernorm(X)
    print(res)
