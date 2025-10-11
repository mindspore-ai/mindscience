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
"""rotation"""
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
        dtype (mindspore.dtype): The type of input tensor. Default: ``mindspore.float32`` .

    Returns:
        alpha (Tensor) - The alpha Euler angles.

        beta (Tensor) - The beta Euler angles.

        gamma (Tensor) - The gamma Euler angles.

    Raises:
        TypeError: If dtype of 'shape' is not tuple.
        TypeError: If dtype of the element of 'shape' is not int.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import identity_angles
        >>> m = identity_angles((1))
        >>> print(m)
        (Tensor(shape=[1], dtype=Float32, value= [ 0.00000000e+00]), Tensor(shape=[1], dtype=Float32,
        value= [ 0.00000000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 0.00000000e+00]))
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

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import rand_angles
        >>> m = rand_angles((1))
        >>> print(m)
        (Tensor(shape=[1], dtype=Float32, value= [ 4.00494671e+00]), Tensor(shape=[1], dtype=Float32,
        value= [ 1.29240000e+00]), Tensor(shape=[1], dtype=Float32, value= [ 5.71690750e+00]))
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
        The second set of Euler angles 'a2, b2, c2' are applied first, while the first set of Euler angles a2, b2, c2'
        are applied Second.
        The elements of Euler angles should be one of the following types: float, float32, np.float32.

    Args:
        a1 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The second applied alpha Euler angles.
        b1 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The second applied beta Euler angles.
        c1 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The second applied gamma Euler angles.
        a2 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The first applied alpha Euler angles.
        b2 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The first applied beta Euler angles.
        c2 (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The first applied gamma Euler angles.

    Returns:
        - alpha (Tensor), The composed alpha Euler angles.
        - beta (Tensor), The composed beta Euler angles.
        - gamma (Tensor), The composed gamma Euler angles.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import compose_angles
        >>> m = compose_angles(0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        >>> print(m)
        (Tensor(shape=[], dtype=Float32, value= 1.34227), Tensor(shape=[], dtype=Float32, value= 1.02462),
        Tensor(shape=[], dtype=Float32, value= 1.47115))
    """

    a1, b1, c1, a2, b2, c2 = broadcast_args(a1, b1, c1, a2, b2, c2)
    return matrix_to_angles(
        ops.matmul(angles_to_matrix(a1, b1, c1), angles_to_matrix(a2, b2, c2)))


def matrix_x(angle):
    r"""
    Give the rotation matrices around x axis for given angle.

    Args:
        angle (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The rotation angles around x axis.
            The shape of 'angle' is :math:`(...)`.

    Returns:
        Tensor, the rotation matrices around x axis. The shape of output is :math:`(..., 3, 3)`

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import matrix_x
        >>> m = matrix_x(0.4)
        >>> print(m)
        [[ 1.          0.          0.        ]
        [ 0.          0.92106086 -0.38941833]
        [ 0.          0.38941833  0.92106086]]
    """
    angle = _to_tensor(angle)
    o = ops.ones_like(angle)
    z = ops.zeros_like(angle)
    return ops.stack([
        ops.stack([o, z, z], axis=-1),
        ops.stack([z, cos(angle), -sin(angle)], axis=-1),
        ops.stack([z, sin(angle), cos(angle)], axis=-1),
    ],
                     axis=-2)


def matrix_y(angle):
    r"""
    Give the rotation matrices around y axis for given angle.

    Args:
        angle (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The rotation angles around y axis.

    Returns:
        Tensor, the rotation matrices around y axis. The shape of output is :math:`(..., 3, 3)`

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import matrix_y
        >>> m = matrix_y(0.5)
        >>> print(m)
        [[ 0.87758255  0.          0.47942555]
        [ 0.          1.          0.        ]
        [-0.47942555  0.          0.87758255]]
    """
    angle = _to_tensor(angle)
    o = ops.ones_like(angle)
    z = ops.zeros_like(angle)
    return ops.stack([
        ops.stack([cos(angle), z, sin(angle)], axis=-1),
        ops.stack([z, o, z], axis=-1),
        ops.stack([-sin(angle), z, cos(angle)], axis=-1),
    ],
                     axis=-2)


def matrix_z(angle):
    r"""
    Give the rotation matrices around z axis for given angle.

    Args:
        angle (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The rotation angles around z axis.
            The shape of 'angle' is :math:`(...)`.

    Returns:
        Tensor, the rotation matrices around z axis. The shape of output is :math:`(..., 3, 3)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import matrix_z
        >>> m = matrix_z(0.6)
        >>> print(m)
        [[ 0.8253357 -0.5646425  0.       ]
        [ 0.5646425  0.8253357  0.       ]
        [ 0.         0.         1.       ]]
    """
    angle = _to_tensor(angle)
    o = ops.ones_like(angle)
    z = ops.zeros_like(angle)
    return ops.stack([
        ops.stack([cos(angle), -sin(angle), z], axis=-1),
        ops.stack([sin(angle), cos(angle), z], axis=-1),
        ops.stack([z, z, o], axis=-1),
    ],
                     axis=-2)


def angles_to_matrix(alpha, beta, gamma):
    r"""
    Conversion from angles to matrix.

    Args:
        alpha (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The alpha Euler angles. The shape of Tensor is :math:`(...)`.
        beta (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The beta Euler angles. The shape of Tensor is :math:`(...)`.
        gamma (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The gamma Euler angles. The shape of Tensor is :math:`(...)`.

    Returns:
        Tensor, the rotation matrices. Matrices of shape :math:`(..., 3, 3)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindchemistry.e3.o3 import angles_to_matrix
        >>> m = angles_to_matrix(0.4, 0.5, 0.6)
        >>> print(m)
        [[ 0.5672197   0.1866971   0.8021259 ]
        [ 0.27070403  0.87758255 -0.395687  ]
        [-0.77780527  0.44158012  0.4472424 ]]
    """
    alpha, beta, gamma = broadcast_args(alpha, beta, gamma)
    return ops.matmul(ops.matmul(matrix_y(alpha), matrix_x(beta)),
                      matrix_y(gamma))


def matrix_to_angles(r_param):
    r"""
    Conversion from matrix to angles.

    Args:
        r_param (Tensor): The rotation matrices. Matrices of shape :math:`(..., 3, 3)`.

    Returns:
        - alpha (Tensor), The alpha Euler angles. The shape of Tensor is :math:`(...)`.
        - beta (Tensor), The beta Euler angles. The shape of Tensor is :math:`(...)`.
        - gamma (Tensor), The gamma Euler angles. The shape of Tensor is :math:`(...)`.

    Raise:
        ValueError: If the det(R) is not equal to 1.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindchemistry.e3.o3 import matrix_to_angles
        >>> input = ms.Tensor([[0.5672197, 0.1866971, 0.8021259], [0.27070403, 0.87758255, -0.395687],
        ...                    [-0.77780527, 0.44158012,0.4472424]])
        >>> m = matrix_to_angles(input)
        >>> print(m)
        (Tensor(shape=[], dtype=Float32, value= 0.4), Tensor(shape=[], dtype=Float32, value= 0.5),
        Tensor(shape=[], dtype=Float32, value= 0.6))
    """
    if not np.allclose(np.linalg.det(r_param.asnumpy()), 1., 1e-3, 1e-5):
        raise ValueError

    x = ops.matmul(r_param, Tensor([0.0, 1.0, 0.0]))
    a, b = xyz_to_angles(x)
    tmp_r_param = angles_to_matrix(a, b, ops.zeros_like(a))
    perm = tuple(range(len(tmp_r_param.shape)))
    r_param = ops.matmul(
        tmp_r_param.transpose(perm[:-2] + (perm[-1],) + (perm[-2],)),
        r_param)
    c = ops.atan2(r_param[..., 0, 2], r_param[..., 0, 0])
    return a, b, c


def angles_to_xyz(alpha, beta):
    r"""
    Convert :math:`(\alpha, \beta)` into a point :math:`(x, y, z)` on the sphere.

    Args:
        alpha (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The alpha Euler angles. The shape of Tensor is :math:`(...)`.
        beta (Union[Tensor[float32], List[float], Tuple[float], ndarray[np.float32], float]):
            The beta Euler angles. The shape of Tensor is :math:`(...)`.

    Returns:
        Tensor, the point :math:`(x, y, z)` on the sphere. The shape of Tensor is :math:`(..., 3)`

    Supported Platforms:
        ``Ascend``

    Examples
        >>> import mindspore as ms
        >>> from mindchemistry.e3.o3 import angles_to_xyz
        >>> print(angles_to_xyz(ms.Tensor(1.7), ms.Tensor(0.0)).abs())
        [0., 1., 0.]
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
        xyz (Tensor): The point :math:`(x, y, z)` on the sphere. The shape of Tensor is :math:`(..., 3)`.

    Returns:
        alpha (Tensor) - The alpha Euler angles. The shape of Tensor is :math:`(...)`.
        beta (Tensor) - The beta Euler angles. The shape of Tensor is :math:`(...)`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> from mindchemistry.e3.o3 import xyz_to_angles
        >>> input = ms.Tensor([3, 3, 3])
        >>> m = xyz_to_angles(input)
        >>> print(m)
        (Tensor(shape=[], dtype=Float32, value= 0.785398), Tensor(shape=[], dtype=Float32, value= 0.955318))
    """
    xyz = xyz / norm_keep(xyz, axis=-1)
    xyz = ops.nan_to_num(ops.clamp(xyz, -1, 1), 1.0)

    beta = ops.acos(xyz[..., 1])
    alpha = ops.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta
