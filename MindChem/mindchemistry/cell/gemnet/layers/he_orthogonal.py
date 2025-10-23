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
"""initiallizers"""

import mindspore as ms
import mindspore.mint as mint
from mindspore.common.initializer import initializer


def _standardize(kernel):
    """
    Standardize the input tensor.
    Makes sure that N*Var(W) = 1 and E[W] = 0

    Args:
        kernel (Tensor): Input tensor. Any shape of Tensor.

    Returns:
        (Tensor) Standardized tensor. The same shape as the input tensor.
    """
    eps = 1e-6

    if len(kernel.shape) == 3:
        axis = (0, 1)
    else:
        axis = (1,)
    var = ms.ops.var(kernel, axis=axis, keepdims=True)
    mean = mint.mean(kernel, dim=axis, keepdim=True)
    kernel = (kernel - mean) / mint.sqrt(var + eps)
    return kernel


def he_orthogonal_init(input_tensor):
    """
    Generate a weight matrix with variance according to He (Kaiming) initialization.
    Based on a random (semi-)orthogonal matrix neural networks.
    Expected to learn better when features are decorrelated.

    Args:
        input_tensor (Parameter): Parameter need to be initialized. Any shape of Parameter.

    Returns:
        (Parameter) Return a initialized parameter. The same shape as the input Parameter.
    """

    input_tensor_value = initializer(
        "orthogonal", input_tensor.shape).init_data()

    if len(input_tensor_value.shape) == 3:
        fan_in = input_tensor_value.shape[0] * input_tensor_value.shape[1]
    else:
        fan_in = input_tensor_value.shape[1]

    input_tensor_value = _standardize(input_tensor_value)
    input_tensor_value *= (1 / fan_in) ** 0.5
    input_tensor.set_data(input_tensor_value)
    return input_tensor
