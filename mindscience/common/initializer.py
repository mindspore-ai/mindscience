# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""initializer"""

import numpy as np
from mindspore.common.initializer import TruncatedNormal

TRUNCATED_NORMAL_STDDEV_FACTOR = np.asarray(.87962566103423978, dtype=np.float32)


def lecun_init(fan_in, initializer_name='linear'):
    r"""lecun init
    
    Args:
        fan_in (int): The number of input features.
        initializer_name (str, optional): The name of the initializer. Default: 'linear'.

    Returns:
        Initializer: The initializer.
    """
    scale = 1.0
    if initializer_name == 'relu':
        scale *= 2
    weight_init = TruncatedNormal(sigma=np.sqrt(scale / fan_in) / TRUNCATED_NORMAL_STDDEV_FACTOR)
    return weight_init


def glorot_uniform(fan_in, fan_out, weight_shape):
    r"""glorot uniform
    
    Args:
        fan_in (int): The number of input features.
        fan_out (int): The number of output features.
        weight_shape (tuple): The shape of the weight.

    Returns:
        numpy.ndarray: The weight.
    """
    limit = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=weight_shape)
