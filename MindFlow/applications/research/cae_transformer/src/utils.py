# ============================================================================
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
"""tool functions"""
import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import constexpr


@constexpr
def generate_tensor(t_shape):
    return ms.Tensor(np.ones(t_shape), ms.float32)


def mask_fill(mask, data, num):
    select = ops.Select()
    replace_tensor = generate_tensor(data.shape)
    replace_tensor[:] = num
    return select(mask, replace_tensor, data.astype(ms.float32))
