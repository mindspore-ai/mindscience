# Copyright 2022 Huawei Technologies Co., Ltd
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
import math
import random

import numpy as np

from mindspore import Tensor, float32, ops

from ..utils.func import broadcast_args, _to_tensor, norm_keep

seed = int(random.random() * 10000)
zeros = ops.Zeros()
cos = ops.Cos()
sin = ops.Sin()
rand = ops.UniformReal(seed=seed)


def identity_angles(*shape, dtype=float32):
    r"""
    Give the identity set of Euler angles.

    Args:
        shape (Tuple[int]): The shape of additional dimensions.

    Returns:
        alpha (Tensor) - The alpha Euler angles.
        beta (Tensor) - The beta Euler angles.
        gamma (Tensor) - The gamma Euler angles.

    Raises:
        TypeError: If dtype of 'shape' is not tuple. 
        TypeError: If dtype of the element of 'shape' is not int. 
    """
    if not isinstance(shape, tuple):
        raise TypeError
    if not all(map(lambda x: isinstance(x, int), shape)):
        raise TypeError
    abc = zeros((3,) + shape, dtype)
    return abc[0], abc[1], abc[2]


def rand_angles(*shape):
    r"""
    Give a random set of Euler angles.

    Args:
        shape (Tuple[int]): The shape of additional dimensions.

    Returns:
        alpha (Tensor) - The alpha Euler angles.
        beta (Tensor) - The beta Euler angles.
        gamma (Tensor) - The gamma Euler angles.

    Raises:
        TypeError: If dtype of 'shape' is not tuple. 
        TypeError: If dtype of the element of 'shape' is not int. 
    """
    if not isinstance(shape, tuple):
        raise TypeError
    if not all(map(lambda x: isinstance(x, int), shape)):
        raise TypeError
    alpha, gamma = 2 * math.pi * rand((2,) + shape)
    beta = ops.acos(2 * rand(shape) - 1)
    return alpha, beta, gamma


def compose_angles(a1, b1, c1, a2, b2, c2):
    r"""
    Computes the composed Euler angles of two sets of Euler angles.

    .. math::

        R(a, b, c) = R(a_1, b_1, c_1) \circ R(a_2, b_2, c_2)

    Note:
        - The second set of Euler angles 'a2, b2, c2' are applied first, while the first set of Euler angles 'a2, b2, c2' are applied Second.
        - The elements of Euler angles should be one of the following types: float, float32, np.float32.

    Args:
        a1 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The second applied alpha Euler angles. 
        b1 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The second applied beta Euler angles. 
        c1 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The second applied gamma Euler angles. 
        a2 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The first applied alpha Euler angles. 
        b2 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The first applied beta Euler angles. 
        c2 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The first applied gamma Euler angles. 

    Returns:
        alpha (Tensor) - The composed alpha Euler angles.
        beta (Tensor) - The composed beta Euler angles.
        gamma (Tensor) - The composed gamma Euler angles.

    """

    a1, b1, c1, a2, b2, c2 = broadcast_args(a1, b1, c1, a2, b2, c2)
    return matrix_to_angles(ops.matmul(angles_to_matrix(a1, b1, c1), angles_to_matrix(a2, b2, c2)))


def matrix_x(angle):
    r"""
    Give the rotation matrices around x axis for given angle.

    Args:
        angle (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The rotation angles around x axis.
            The shape of 'angle' is :math:'(...)'

    Returns:
        Tensor, the rotation matrices around x axis.
            The shape of output is :math:'(..., 3, 3)'

    """
    angle = _to_tensor(angle)
    o = ops.ones_like(angle)
    z = ops.zeros_like(angle)
    return ops.stack([
        ops.stack([o, z, z], axis=-1),
        ops.stack([z, cos(angle), -sin(angle)], axis=-1),
        ops.stack([z, sin(angle), cos(angle)], axis=-1),
    ], axis=-2)


def matrix_y(angle):
    r"""
    Give the rotation matrices around y axis for given angle.

    Args:
        angle (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The rotation angles around y axis.
            The shape of 'angle' is :math:'(...)'

    Returns:
        Tensor, the rotation matrices around y axis.
            The shape of output is :math:'(..., 3, 3)'

    """
    angle = _to_tensor(angle)
    o = ops.ones_like(angle)
    z = ops.zeros_like(angle)
    return ops.stack([
        ops.stack([cos(angle), z, sin(angle)], axis=-1),
        ops.stack([z, o, z], axis=-1),
        ops.stack([-sin(angle), z, cos(angle)], axis=-1),
    ], axis=-2)


def matrix_z(angle):
    r"""
    Give the rotation matrices around z axis for given angle.

    Args:
        angle (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The rotation angles around z axis.
            The shape of 'angle' is :math:'(...)'

    Returns:
        Tensor, the rotation matrices around z axis.
            The shape of output is :math:'(..., 3, 3)'

    """
    angle = _to_tensor(angle)
    o = ops.ones_like(angle)
    z = ops.zeros_like(angle)
    return ops.stack([
        ops.stack([cos(angle), -sin(angle), z], axis=-1),
        ops.stack([sin(angle), cos(angle), z], axis=-1),
        ops.stack([z, z, o], axis=-1),
    ], axis=-2)


def angles_to_matrix(alpha, beta, gamma):
    r"""
    Conversion from angles to matrix.

    Args:
        alpha (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The alpha Euler angles.
            tensor of shape :math:`(...)`
        beta (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The beta Euler angles.
            tensor of shape :math:`(...)`
        gamma (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The gamma Euler angles.
            tensor of shape :math:`(...)`

    Returns:
        Tensor, the rotation matrices.
            matrices of shape :math:`(..., 3, 3)`

    """
    alpha, beta, gamma = broadcast_args(alpha, beta, gamma)
    return ops.matmul(ops.matmul(matrix_y(alpha), matrix_x(beta)), matrix_y(gamma))


def matrix_to_angles(R):
    r"""
    Conversion from matrix to angles.

    Args:
        R (Tensor): The rotation matrices.
            matrices of shape :math:`(..., 3, 3)`

    Returns:
        alpha (Tensor) - The alpha Euler angles.
            tensor of shape :math:`(...)`
        beta (Tensor) - The beta Euler angles.
            tensor of shape :math:`(...)`
        gamma (Tensor) - The gamma Euler angles.
            tensor of shape :math:`(...)`

    Raise:
        ValueError: If the det(R) is not equal to 1.
    """
    if not np.allclose(np.linalg.det(R.asnumpy()), 1., 1e-3, 1e-5):
        raise ValueError

    x = ops.matmul(R, Tensor([0.0, 1.0, 0.0]))
    a, b = xyz_to_angles(x)
    tmp_R = angles_to_matrix(a, b, ops.zeros_like(a))
    perm = tuple(range(len(tmp_R.shape)))
    R = ops.matmul(tmp_R.transpose(perm[:-2] + (perm[-1],) + (perm[-2],)), R)
    c = ops.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


def angles_to_xyz(alpha, beta):
    r"""
    Convert :math:`(\alpha, \beta)` into a point :math:`(x, y, z)` on the sphere.

    Args:
        alpha (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The alpha Euler angles.
            tensor of shape :math:`(...)`
        beta (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]): The beta Euler angles.
            tensor of shape :math:`(...)`

    Returns:
        Tensor, the point :math:`(x, y, z)` on the sphere.
            tensor of shape :math:`(..., 3)`
    """
    alpha, beta = broadcast_args(alpha, beta)
    x = sin(beta) * sin(alpha)
    y = cos(beta)
    z = sin(beta) * cos(alpha)
    return ops.stack([x, y, z], axis=-1)


def xyz_to_angles(xyz):
    r"""
    Convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`.

    .. math::
        \vec r = R(\alpha, \beta, 0) \vec e_z

    Args:
        xyz (Tensor): The point :math:`(x, y, z)` on the sphere.
            tensor of shape :math:`(..., 3)`

    Returns:
        alpha (Tensor) - The alpha Euler angles.
            tensor of shape :math:`(...)`
        beta (Tensor) - The beta Euler angles.
            tensor of shape :math:`(...)`
    """
    xyz = xyz / norm_keep(xyz, axis=-1)
    xyz = ops.clip_by_value(xyz, -1, 1)

    beta = ops.acos(xyz[..., 1])
    alpha = ops.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta
