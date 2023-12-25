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
Quaternion
"""

from mindspore import numpy as msnp
from mindspore import ops


def hamiltonian_product(quaternion_1, tensor_2):
    """ Get the Hamiltonian-product of the given quaternion and tensor.

    Args:
        quaternion_1 (Tensor): A tensor to calculate.
        tensor_2 (Tensor): A tensor to calculate.

    Returns:
        The hamiltonian product result.
    """
    if quaternion_1.ndim == 1:
        quaternion_1 = quaternion_1[None, :]
    if tensor_2.ndim == 1:
        tensor_2 = tensor_2[None, :]

    inverse_quaternion = quaternion_inverse(quaternion_1)
    op1 = quaternion_multiply(tensor_2, inverse_quaternion)
    res = quaternion_multiply(quaternion_1, op1)
    return res


def quaternion_multiply(tensor_1, tensor_2):
    """ Get the quaternion multiplication of the given tensor.

    Args:
        tensor_1 (Tensor): A tensor to calculate.
        tensor_2 (Tensor): The other tensor to calculate.

    Returns:
        The multiplication result.
    """
    if tensor_1.ndim == 1:
        tensor_1 = tensor_1[None, :]
    if tensor_2.ndim == 1:
        tensor_2 = tensor_2[None, :]

    if tensor_1.shape[-1] == 1 and tensor_2.shape[-1] == 4:
        return _constant_multiply(tensor_2, tensor_1)
    if tensor_2.shape[-1] == 1 and tensor_1.shape[-1] == 4:
        return _constant_multiply(tensor_1, tensor_2)
    if tensor_1.shape[-1] == 3:
        tensor_1 = msnp.pad(tensor_1, ((0, 0), (1, 0)), mode='constant', constant_value=0)
        return quaternion_multiply(tensor_1, tensor_2)
    if tensor_2.shape[-1] == 3:
        tensor_2 = msnp.pad(tensor_2, ((0, 0), (1, 0)), mode='constant', constant_value=0)
        return quaternion_multiply(tensor_1, tensor_2)
    return _quaternion_multiply(tensor_1, tensor_2)


def quaternion_inverse(tensor_1):
    """ Get the quaternion conjugate of the given tensor.

    Args:
        tensor_1 (Tensor): A tensor to calculate.

    Returns:
        tensor_2(Tensor), The multiplication result with shape (B, 4).
    """
    if tensor_1.ndim == 1:
        tensor_1 = tensor_1[None, :]

    if tensor_1.shape[-1] == 1:
        return msnp.pad(tensor_1, ((0, 0), (0, 3)), mode='constant', constant_value=0)

    if tensor_1.shape[-1] == 3:
        return -msnp.pad(tensor_1, ((0, 0), (0, 3)), mode='constant', constant_value=0) / (msnp.norm(
            tensor_1, axis=-1
        )[:, None] ** 2)

    return msnp.hstack((tensor_1[:, 0][:, None], -tensor_1[:, 1:])) / (msnp.norm(
        tensor_1, axis=-1
    )[:, None] ** 2)


def _quaternion_multiply(tensor_1, tensor_2):
    """ Get the quaternion multiplication of the given tensor.

    Args:
        tensor_1 (Tensor): A tensor with shape (B, 4).
        tensor_2 (Tensor): A tensor with shape (B, 4).

    Returns:
        q(Tensor), A tensor with shape (B, 4).
    """
    if tensor_1.shape[-1] != 4 or tensor_2.shape[-1] != 4:
        raise ValueError('The input tensor shape for quaternion_multiply should be like (B, 4) or (4, ).')

    s_1 = tensor_1[:, 0]
    s_2 = tensor_2[:, 0]
    v_1 = tensor_1[:, 1:]
    v_2 = tensor_2[:, 1:]
    s = s_1 * s_2
    d = ops.batch_dot(v_1, v_2, axes=-1)
    s -= d
    v = msnp.zeros_like(v_1)
    v += s_1 * v_2
    v += v_1 * s_2
    v += msnp.cross(v_1, v_2, axisc=-1)
    q = msnp.hstack((s, v))

    return q

def _constant_multiply(tensor_1, constant):
    """ Get the quaternion multiplication of the given tensor and constant.

    Args:
        tensor_1 (Tensor): A tensor with shape (B, 4).
        constant (Tensor): A tensor with shape (B, 1).

    Returns:
        A tensor with shape (B, 4).
    """
    return tensor_1 * constant
