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
# ==============================================================================
"""get activation function."""
from __future__ import absolute_import

import mindspore.ops as ops
import mindspore.nn.layer.activation as activation

_activation = {
    'softmax': activation.Softmax,
    'logsoftmax': activation.LogSoftmax,
    'relu': activation.ReLU,
    'silu': activation.SiLU,
    'relu6': activation.ReLU6,
    'tanh': activation.Tanh,
    'gelu': activation.GELU,
    'fast_gelu': activation.FastGelu,
    'elu': activation.ELU,
    'sigmoid': activation.Sigmoid,
    'prelu': activation.PReLU,
    'leakyrelu': activation.LeakyReLU,
    'hswish': activation.HSwish,
    'hsigmoid': activation.HSigmoid,
    'logsigmoid': activation.LogSigmoid,
    'sin': ops.Sin
}
