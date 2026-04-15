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
# ============================================================================
"""glorot orthogonal initiallizers"""

from mindspore.common.initializer import initializer


def glorot_orthogonal(x, scale):
    """Glorot orthogonal initialization.

    Args:
        x (Parameter): Parameter need to be initialized. Any shape of Parameter.
        scale (float): Scale

    Returns:
        (Parameter) Return a initialized parameter. The same shape as the input Parameter.
    """

    x_value = initializer("orthogonal", x.shape).init_data()
    scale /= ((x.shape[-2] + x.shape[-1]) * x_value.var())
    x_value *= scale.sqrt()
    x.set_data(x_value)
    return x
