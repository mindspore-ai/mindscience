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
Common functions
"""

from typing import Union, List, Tuple
import numpy as np
from numpy import ndarray
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import ops
from mindspore import jit
from mindspore import Tensor, Parameter
from mindspore.ops import functional as F

__all__ = [
    'PI',
    'inv',
    'keepdims_sum',
    'keepdims_mean',
    'keepdims_prod',
    'reduce_any',
    'reduce_all',
    'reduce_prod',
    'concat_last_dim',
    'concat_penulti',
    'identity',
    'periodic_variable',
    'periodic_difference',
    'gather_vector',
    'gather_value',
    'pbc_box_reshape',
    'pbc_image',
    'coordinate_in_pbc',
    'vector_in_pbc',
    'calc_vector_nopbc',
    'calc_vector_pbc',
    'calc_vector',
    'calc_distance_nopbc',
    'calc_distance_pbc',
    'calc_distance',
    'calc_angle_by_vectors',
    'calc_angle_nopbc',
    'calc_angle_pbc',
    'calc_angle',
    'calc_torsion_by_vectors',
    'calc_torsion_nopbc',
    'calc_torsion_pbc',
    'calc_torsion',
    'coulomb_interaction',
    'lennard_jones_potential',
    'lennard_jones_potential2',
    'get_integer',
    'get_ndarray',
    'get_tensor',
    'get_ms_array',
    'check_broadcast',
    'any_none',
    'all_none',
    'any_not_none',
    'all_not_none',
]

PI = 3.141592653589793238462643383279502884197169399375105820974944592307
r""":math:`\pi`"""

inv = ops.Inv()
keepdims_sum = ops.ReduceSum(True)
keepdims_mean = ops.ReduceMean(True)
keepdims_prod = ops.ReduceProd(True)
reduce_any = ops.ReduceAny()
reduce_all = ops.ReduceAll()
reduce_prod = ops.ReduceProd()
concat_last_dim = ops.Concat(-1)
concat_penulti = ops.Concat(-2)
identity = ops.Identity()


@jit
def periodic_variable(variable: Tensor,
                      upper: Tensor,
                      lower: Tensor = 0,
                      mask: Tensor = None,
                      ) -> Tensor:
    r"""get the value in the periodic range.

    Args:
        variable (Tensor):  Tensor of shape `(...)`. Data type is float.
                            Periodic variable
        upper (Tensor):     Tensor of shape `(...)`. Data type is float.
                            Upper boundary of perodicity.
        lower (Tensor):     Tensor of shape `(...)`. Data type is float.
                            Lower boundary of perodicity. Default: 0
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool_.
                            Mask for the periodic variable.

    Returns:
        period_value (Tensor):  Tensor of shape `(...)`. Data type is float.
                                Variable with value in the periodic range.
    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    period = upper - lower
    period_value = variable - period * F.floor((variable - lower) / period)
    if mask is None:
        return period_value
    if mask.shape != variable.shape:
        mask = msnp.broadcast_to(mask, variable.shape)
    return F.select(mask, period_value, variable)


@jit
def periodic_difference(difference: Tensor,
                        period: Tensor,
                        mask: Tensor = None,
                        offset: float = -0.5,
                        ) -> Tensor:
    r"""get value of difference between periodic variables.

    Args:
        variable (Tensor):  Tensor of shape `(...)`. Data type is float.
                            Periodic variable
        period (Tensor):    Tensor of shape `(...)`. Data type is float.
                            Upper boundary of perodicity.
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool_.
                            Mask for the periodic variable.
        offset (float):     Offset ratio :math:`c` with relative to the period :math:`\theta`.
                            Default: -0.5

    Returns:
        period_diff (Tensor):   Tensor of shape `(...)`. Data type is float.
                                Variable with value in the periodic range.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    period_diff = difference - period * F.floor(difference / period - offset)
    if mask is None:
        return period_diff
    if mask.shape != difference.shape:
        mask = msnp.broadcast_to(mask, difference.shape)
    return F.select(mask, period_diff, difference)


@jit
def gather_vector(tensor: Tensor, index: Tensor) -> Tensor:
    r"""Gather vector from the penultimate axis (`axis=-2`) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape `(B, X, D)`.
        index (Tensor):     Tensor of shape `(B, ...,)`. Data type is int.

    Returns:
        vector (Tensor):    Tensor of shape `(B, ..., D)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Spatial dimension of the simulation system. Usually is 3.
        X:  Arbitrary value.

    """

    if index.shape[0] == 1:
        return F.gather(tensor, index[0], -2)
    if tensor.shape[0] == 1:
        return F.gather(tensor[0], index, -2)

    # (B, N, M)
    shape0 = index.shape
    # (B, N * M, 1) <- (B, N, M)
    index = F.reshape(index, (shape0[0], -1, 1))
    # (B, N * M, D) <- (B, N, D)
    vectors = msnp.take_along_axis(tensor, index, axis=-2)
    # (B, N, M, D) <- (B, N, M) + (D,)
    output_shape = shape0 + tensor.shape[-1:]
    # (B, N, M, D)
    return F.reshape(vectors, output_shape)


@jit
def gather_value(tensor: Tensor, index: Tensor) -> Tensor:
    r"""Gather value from the last axis (`axis=-1`) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape `(B, X)`.
        index (Tensor):     Tensor of shape `(B, ...,)`. Data type is int.

    Returns:
        value (Tensor): Tensor of shape `(B, ...,)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation.
        X:  Arbitrary value.

    """

    if index.shape[0] == 1:
        return F.gather(tensor, index[0], -1)
    if tensor.shape[0] == 1:
        return F.gather(tensor[0], index, -1)

    # (B, N, M)
    origin_shape = index.shape
    # (B, N * M) <- (B, N, M)
    index = F.reshape(index, (origin_shape[0], -1))
    # (B, N * M)
    values = F.gather_d(tensor, -1, index)
    # (B, N, M)
    return F.reshape(values, origin_shape)


@jit
def pbc_box_reshape(pbc_box: Tensor, ndim: int) -> Tensor:
    r"""Reshape the pbc_box as the same ndim.

    Args:
        pbc_box (Tensor):   Tensor of shape `(B, D)`. Data type is float.
        ndim (int):         The rank (number of dimension) of the pbc_box

    Returns:
        pbc_box (Tensor):   Tensor of shape `(B, 1, .., 1, D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    if ndim <= 2:
        return pbc_box
    shape = pbc_box.shape[:1] + (1,) * (ndim - 2) + pbc_box.shape[-1:]
    return F.reshape(pbc_box, shape)


@jit
def pbc_image(position: Tensor, pbc_box: Tensor, offset: float = 0) -> Tensor:
    r"""calculate the periodic image of the PBC box

    Args:
        position (Tensor):  Tensor of shape `(B, ..., D)`. Data type is float.
        pbc_box (Tensor):   Tensor of shape `(B, D)`. Data type is float.
        offset (float):     Offset ratio :math:`c` relative to box size :math:`\vec{L}`.
                            Default: 0

    Returns:
        image (Tensor): Tensor of shape `(B, ..., D)`. Data type is int32.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    pbc_box = pbc_box_reshape(F.stop_gradient(pbc_box), position.ndim)
    image = -F.floor(position / pbc_box - offset)
    return F.cast(image, ms.int32)


@jit
def coordinate_in_pbc(position: Tensor, pbc_box: Tensor, offset: float = 0) -> Tensor:
    r"""get coordinate in main PBC box

    Args:
        position (Tensor):  Tensor of shape `(B, ..., D)`. Data type is float.
                            Position coordinate :math:`R`
        pbc_box (Tensor):   Tensor of shape `(B, D)`. Data type is float.
                            Size of PBC box :math:`\vec{L}`
        offset (float):     Offset ratio :math:`c` relative to box size :math:`\vec{L}`.
                            Default: 0

    Returns:
        coordinate (Tensor):    Tensor of shape `(B, ..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    pbc_box = pbc_box_reshape(F.stop_gradient(pbc_box), position.ndim)
    return position - pbc_box * F.floor(position / pbc_box - offset)


@jit
def vector_in_pbc(vector: Tensor, pbc_box: Tensor, offset: float = -0.5) -> Tensor:
    r"""Make the value of vector :math:`\vec{v}` at a single PBC box :math:`\vec{L}`.

    Args:
        vector (Tensor):    Tensor of shape `(B, ..., D)`. Data type is float.
                            Vector :math:`\vec{v}
        pbc_box (Tensor):   Tensor of shape `(B, D)`. Data type is float.
                            Size of PBC box :math:`\vec{L}`
        offset (float):     Offset ratio :math:`c` of the vector relative to box size :math:`\vec{L}`.
                            The value of vector will be between :math:`c \vec{L}` and :math:`(c+1) \vec{L}`
                            Default: -0.5

    Returns:
        pbc_vector (Tensor):    Tensor of shape `(B, ..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    pbc_box = pbc_box_reshape(pbc_box, vector.ndim)
    box_nograd = F.stop_gradient(pbc_box)
    inv_box = msnp.reciprocal(box_nograd)
    vector -= box_nograd * F.floor(vector * inv_box - offset)
    return  vector * inv_box * pbc_box


@jit
def calc_vector_nopbc(initial: Tensor, terminal: Tensor) -> Tensor:
    r"""Compute vector from initial point to terminal point without perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape `(..., D)`. Data type is float.
                            Position coordinate of initial point
        terminal (Tensor):  Tensor of shape `(..., D)`. Data type is float.
                            Position coordinate of terminal point

    Returns:
        vector (Tensor):    Tensor of shape `(..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    return terminal - initial


@jit
def calc_vector_pbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor) -> Tensor:
    r"""Compute vector from initial point to terminal point at perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape `(..., D)`. Data type is float.
                            Position coordinate of initial point
        terminal (Tensor):  Tensor of shape `(..., D)`. Data type is float.
                            Position coordinate of terminal point
        pbc_box (Tensor):   Tensor of shape `(D)` or `(B, D)`. Data type is float.
                            Size of PBC box.

    Returns:
        vector (Tensor):    Tensor of shape `(..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    return vector_in_pbc(terminal-initial, pbc_box)


@jit
def calc_vector(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""Compute vector from initial point to terminal point.

    Args:
        initial (Tensor):   Tensor of shape `(..., D)`. Data type is float.
                            Position coordinate of initial point.
        terminal (Tensor):  Tensor of shape `(..., D)`. Data type is float.
                            Position coordinate of terminal point.
        pbc_box (Tensor):   Tensor of shape `(D)` or `(B, D)`. Data type is float.
                            Default: ``None``.

    Returns:
        vector (Tensor):    Tensor of shape `(..., D)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    vector = terminal - initial
    if pbc_box is None:
        return vector
    return vector_in_pbc(vector, pbc_box)


@jit
def calc_distance_nopbc(position_a: Tensor,
                        position_b: Tensor,
                        keepdims: bool = False,
                        ) -> Tensor:
    r"""Compute distance between position A and B without perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape :math:`(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        keepdims (bool):        If this is set to ``True``, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        distance (Tensor):      Tensor of shape :math:`(...)` or :math:(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    vec = calc_vector_nopbc(position_a, position_b)
    return msnp.norm(vec, axis=-1, keepdims=keepdims)


@jit
def calc_distance_pbc(position_a: Tensor,
                      position_b: Tensor,
                      pbc_box: Tensor = None,
                      keepdims: bool = False
                      ) -> Tensor:
    r"""Compute distance between position :math:`A` and :math:`B` at perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        pbc_box (Tensor):       Tensor of shape `(D)` or `(B, D)`. Data type is float.
                                Size of PBC box :math:`\vec{L}`
        keepdims (bool):        If this is set to `True`, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        distance (Tensor):      Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    vec = calc_vector_pbc(position_a, position_b, pbc_box)
    return msnp.norm(vec, axis=-1, keepdims=keepdims)


@jit
def calc_distance(position_a: Tensor,
                  position_b: Tensor,
                  pbc_box: Tensor = None,
                  keepdims: bool = False,
                  ) -> Tensor:
    r"""Compute distance between position :math:`A` and :math:`B`.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        pbc_box (Tensor):       Tensor of shape `(D)` or `(B, D)`. Data type is float.
                                Size of PBC box :math:`\vec{L}`
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        distance (Tensor):      Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    vec = calc_vector_nopbc(position_a, position_b)
    if pbc_box is not None:
        vec = vector_in_pbc(vec, pbc_box)
    return msnp.norm(vec, axis=-1, keepdims=keepdims)


@jit
def calc_angle_by_vectors(vector1: Tensor,
                          vector2: Tensor,
                          keepdims: bool = False
                          ) -> Tensor:
    r"""Compute angle between two vectors.
        For vector :math:`\vec {v_1} = (x_1, x_2, x_3, ..., x_n)` and
        :math:`\vec {v_2} = (y_1, y_2, y_3, ..., y_n)` , the formula is

    .. math::

        \theta = \arccos {\frac{|x_1y_1 + x_2y_2 + \cdots + x_ny_n|}{\sqrt{x_1^2 + x_2^2 +
                 \cdots + x_n^2}\sqrt{y_1^2 + y_2^2 + \cdots + y_n^2}}}

    Args:
        vector1 (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                             Vector of :math:`\vec{v_1}`.
        vector2 (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                             Vector of :math:`\vec{v_2}`.
        keepdims (bool):     If this is set to True, the last axis will be left
                             in the result as dimensions with size one.
                             Default: ``False``.

    Returns:
        angle (Tensor):      Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    # (...) or (..., 1) <- (..., D)
    dis1 = msnp.norm(vector1, axis=-1, keepdims=keepdims)
    dis2 = msnp.norm(vector2, axis=-1, keepdims=keepdims)
    dot12 = msnp.sum(vector1*vector2, axis=-1, keepdims=keepdims)
    # (...) or (..., 1)
    cos_theta = dot12 / dis1 / dis2
    return F.acos(cos_theta)


@jit
def calc_angle_nopbc(position_a: Tensor,
                     position_b: Tensor,
                     position_c: Tensor,
                     keepdims: bool = False,
                     ) -> Tensor:
    r"""Compute angle :math:`\angle{ABC}` formed by the position coordinates of three positions
        :math:`A`, :math:`B` and :math:`C` without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        angle (Tensor):         Tensor of shape `(...)` or `(..., 1)`. Data type is float.
                                Value of angle :math:`\angle{ABC}`.


    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    # (...,D)
    vec_ba = calc_vector_nopbc(position_b, position_a)
    vec_bc = calc_vector_nopbc(position_b, position_c)
    return calc_angle_by_vectors(vec_ba, vec_bc, keepdims=keepdims)


@jit
def calc_angle_pbc(position_a: Tensor,
                   position_b: Tensor,
                   position_c: Tensor,
                   pbc_box: Tensor,
                   keepdims: bool = False,
                   ) -> Tensor:
    r"""Compute angle :math:`\angle{ABC}` formed by the position coordinates of three positions
        :math:`A`, :math:`B` and :math:`C` at periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        pbc_box (Tensor):       Tensor of shape `(D)` or `(B, D)`. Data type is float.
                                Size of PBC box :math:`\vec{L}`
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        angle (Tensor):         Tensor of shape (...) or (..., 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    # (B, ..., D)
    vec_ba = calc_vector_pbc(position_b, position_a, pbc_box)
    vec_bc = calc_vector_pbc(position_b, position_c, pbc_box)
    return calc_angle_by_vectors(vec_ba, vec_bc, keepdims=keepdims)


@jit
def calc_angle(position_a: Tensor,
               position_b: Tensor,
               position_c: Tensor,
               pbc_box: Tensor = None,
               keepdims: bool = False,
               ) -> Tensor:
    r"""Compute angle formed by three positions :math:`A`, :math:`B` and :math:`C`
        with or without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        pbc_box (Tensor):       Tensor of shape `(D)` or `(B, D)`. Data type is float.
                                Size of PBC box :math:`\vec{L}`. Default: ``None``.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        angle (Tensor):         Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    if pbc_box is None:
        return calc_angle_nopbc(position_a, position_b, position_c, keepdims=keepdims)
    return calc_angle_pbc(position_a, position_b, position_c, pbc_box=pbc_box, keepdims=keepdims)


@jit
def calc_torsion_by_vectors(vector1: Tensor,
                            vector2: Tensor,
                            axis_vector: Tensor = None,
                            keepdims: bool = False,
                            ) -> Tensor:
    r"""Compute torsion angle formed by two direction vectors :math:`\vec{v_1}` and :math:`\vec{v_2}`
        and an axis vector :math:`\vec{v_{axis}}`.

    Args:
        vector1 (Tensor):       Tensor of shape `(..., D)`. Data type is float.
                                Direction vector :math:`\vec{v_1}`
        vector2 (Tensor):       Tensor of shape `(..., D)`. Data type is float.
                                Direction vector :math:`\vec{v_2}`
        axis_vector (Tensor):   Tensor of shape `(..., D)`. Data type is float.
                                Axis vector :math:`\vec{v_{axis}}`.
                                Default: ``None``.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        torsion (Tensor):   Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    if axis_vector is None:
        return calc_angle_by_vectors(vector1, vector2, keepdims=keepdims)

    # (..., D)
    vec_a = msnp.cross(vector1, axis_vector)
    vec_b = msnp.cross(vector2, axis_vector)
    cross_ab = msnp.cross(vec_a, vec_b)

    # (..., 1) <- (..., D)
    axis_norm = msnp.norm(axis_vector, axis=-1, keepdims=True)
    # (..., D) = (..., D) / (..., 1)
    axis_vector *= msnp.reciprocal(axis_norm)

    # (...) or (..., 1)
    sin_phi = msnp.sum(axis_vector*cross_ab, axis=-1, keepdims=keepdims)
    cos_phi = msnp.sum(vec_a*vec_b, axis=-1, keepdims=keepdims)

    return F.atan2(sin_phi, cos_phi)


@jit
def calc_torsion_nopbc(position_a: Tensor,
                       position_b: Tensor,
                       position_c: Tensor,
                       position_d: Tensor,
                       keepdims: bool = False,
                       ) -> Tensor:
    r"""Compute torsion angle :math:`A-B-C-D` formed by four positions :math:`A`, :math:`B`, :math:`C` and :math:`D`
        without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        position_d (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`D`.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        torsion (Tensor):   Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    vec_ba = calc_vector_nopbc(position_b, position_a)
    vec_cd = calc_vector_nopbc(position_c, position_d)
    vec_bc = calc_vector_nopbc(position_b, position_c)
    return calc_torsion_by_vectors(vec_ba, vec_cd, axis_vector=vec_bc, keepdims=keepdims)


@jit
def calc_torsion_pbc(position_a: Tensor,
                     position_b: Tensor,
                     position_c: Tensor,
                     position_d: Tensor,
                     pbc_box: Tensor,
                     keepdims: bool = False,
                     ) -> Tensor:
    r"""Compute torsion angle :math:`A-B-C-D` formed by four positions :math:`A`, :math:`B`, :math:`C` and :math:`D`
        at periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        position_d (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`D`.
        pbc_box (Tensor):       Tensor of shape `(D)` or `(B, D)`. Data type is float.
                                Size of PBC box :math:`\vec{L}`.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        torsion (Tensor):   Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    vec_ba = calc_vector_pbc(position_b, position_a, pbc_box)
    vec_cd = calc_vector_pbc(position_c, position_d, pbc_box)
    vec_bc = calc_vector_pbc(position_b, position_c, pbc_box)
    return calc_torsion_by_vectors(vec_ba, vec_cd, axis_vector=vec_bc, keepdims=keepdims)


@jit
def calc_torsion(position_a: Tensor,
                 position_b: Tensor,
                 position_c: Tensor,
                 position_d: Tensor,
                 pbc_box: Tensor = None,
                 keepdims: bool = False,
                 ) -> Tensor:

    r"""Compute torsion angle :math:`A-B-C-D` formed by four positions :math:`A`, :math:`B`, :math:`C` and :math:`D`
        with or without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`A`.
        position_b (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`B`.
        position_c (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`C`.
        position_d (Tensor):    Tensor of shape `(..., D)`. Data type is float.
                                Position coordinate of point :math:`D`.
        pbc_box (Tensor):       Tensor of shape `(D)` or `(B, D)`. Data type is float.
                                Size of PBC box :math:`\vec{L}`. Default: ``None``.
        keepdims (bool):        If this is set to True, the last axis will be left
                                in the result as dimensions with size one.
                                Default: ``False``.

    Returns:
        torsion (Tensor):   Tensor of shape `(...)` or `(..., 1)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Note:
        B:  Batchsize, i.e. number of walkers in simulation
        D:  Spatial dimension of the simulation system. Usually is 3.

    """

    if pbc_box is None:
        return calc_torsion_nopbc(
            position_a, position_b, position_c, position_d, keepdims=keepdims)

    return calc_torsion_pbc(
        position_a, position_b, position_c, position_d, pbc_box=pbc_box, keepdims=keepdims)


@jit
def coulomb_interaction(q_i: Tensor,
                        q_j: Tensor,
                        r_ij: Tensor,
                        mask: Tensor = None,
                        coulomb_const: float = 1,
                        ):
    r"""Calculate Coulomb interaction.

    Math:

    .. math::

        E_{coulomb}(r_{ij}) = k \frac{q_i q_j}{r_{ij}}

    Args:
        q_i (Tensor):           Tensor of shape `(...)`. Data type is float.
                                Charge of the :math:`i`-th atom :math:`q_i`.
        q_j (Tensor):           Tensor of shape `(...)`. Data type is float.
                                Charge of the :math:`j`-th atom :math:`q_j`.
        r_ij (Tensor):          Tensor of shape `(...)`. Data type is float.
                                Distance :math:`r_{ij}` between atoms :math:`i` and :math:`i`.
        mask (Tensor):          Tensor of shape `(...)`. Data type is bool.
                                Mask for distance :math:`r_{ij}`. Default: ``None``.
        coulomb_const (float):  Coulomb constant :math:`k`. Default: 1

    Returns:
        E_coulomb (Tensor):     Tensor of shape `(...)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    energy = coulomb_const * q_i * q_j * msnp.reciprocal(r_ij)

    if mask is None:
        return energy

    return energy * mask


@jit
def lennard_jones_potential(epsilon: Tensor, sigma: Tensor, r_ij: Tensor, mask: Tensor = None) -> Tensor:
    r"""Calculate Lennard-Jones (LJ) potential with :math:`\epsilon` and :math:`\sigma`.

    Math:

    .. math::

        E_{lj}(r_{ij}) = 4 \epsilon \left [\left ( \frac{\sigma}{r_{ij}} \right ) ^{12} -
                                           \left ( \frac{\sigma}{r_{ij}} \right ) ^{6} \right]

    Args:
        epsilon (Tensor):   Tensor of shape `(...)`. Data type is float.
                            Well depth :math:`\epsilon`.
        sigma (Tensor):     Tensor of shape `(...)`. Data type is float.
                            Characteristic distance :math:`\sigma`.
        r_ij (Tensor):      Tensor of shape `(...)`. Data type is float.
                            Distance :math:`r_{ij}` between atoms :math:`i` and :math:`i`.
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool.
                            Mask for distances :math:`r_{ij}`. Default: ``None``.

    Returns:
        E_coulomb (Tensor):     Tensor of shape (...). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    # (\sigma / r_{ij}) ^ 6
    r0_6 = F.pows(sigma * msnp.reciprocal(r_ij), 6)
    # 4 * \epsilon * (\sigma / r_{ij}) ^ 6
    ene_bcoeff = 4 * epsilon * r0_6
    # 4 * \epsilon * (\sigma / r_{ij}) ^ 12
    ene_acoeff = ene_bcoeff * r0_6

    energy = ene_acoeff - ene_bcoeff

    if mask is None:
        return energy
    return energy * mask


@jit
def lennard_jones_potential2(epsilon: Tensor, r_0: Tensor, r_ij: Tensor, mask: Tensor = None) -> Tensor:
    r"""Calculate Lennard-Jones (LJ) potential with :math:`\epsilon` and :math:`r_0`.

    Math:

    .. math::

        E_{lj}(r_{ij}) = 4 \epsilon \left [\frac{1}{4} \left ( \frac{r_0}{r_{ij}} \right ) ^{12} -
                                           \frac{1}{2} \left ( \frac{r_0}{r_{ij}} \right ) ^{6} \right]

    Args:
        epsilon (Tensor):   Tensor of shape `(...)`. Data type is float.
                            Well depth :math:`\epsilon`.
        r_0 (Tensor):       Tensor of shape `(...)`. Data type is float.
                            Atomic radius :math:`r_0`.
        r_ij (Tensor):      Tensor of shape `(...)`. Data type is float.
                            Distance :math:`r_{ij}` between atoms :math:`i` and :math:`i`.
        mask (Tensor):      Tensor of shape `(...)`. Data type is bool.
                            Mask for distances :math:`r_{ij}`. Default: ``None``.
    Returns:
        E_coulomb (Tensor):     Tensor of shape `(...)`. Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    # (\r_0 / r_{ij}) ^ 6
    r0_6 = F.pows(r_0 * msnp.reciprocal(r_ij), 6)
    # 2 * \epsilon * (r_0 / r_{ij}) ^ 6
    ene_bcoeff = 2 * epsilon * r0_6
    # \epsilon * (r_0 / r_{ij}) ^ 12
    ene_acoeff = epsilon * r0_6 * r0_6

    energy = ene_acoeff - ene_bcoeff

    if mask is None:
        return energy
    return energy * mask


def get_integer(value: Union[int, Tensor, Parameter, ndarray]) -> int:
    r"""get integer type of the input value

    Args:
        value (Union[int, Tensor, Parameter, ndarray]): Input value

    Returns:
        integer (int)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.asnumpy()
    return int(value)


def get_ndarray(value: Union[Tensor, Parameter, ndarray, List[float], Tuple[float]],
                dtype: type = None) -> ndarray:
    r"""get ndarray type of the input value

    Args:
        value (Union[Tensor, Parameter, ndarray]):  Input value
        dtype (type):                               Data type. Default: ``None``.

    Returns:
        array (ndarray)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if value is None:
        return None
    if isinstance(value, (Tensor, Parameter)):
        value = value.asnumpy()
        if dtype is not None:
            value = value.astype(dtype)
    else:
        value = np.array(value, dtype)
    return value


def get_tensor(value: Union[float, int, Tensor, Parameter, ndarray, List[float], Tuple[float]],
               dtype: type = None) -> Tensor:
    r"""get mindspore.Tensor type of the input value

    Args:
        value (Union[float, int, Tensor, Parameter, ndarray, list, tuple]):
                        Input value
        dtype (type):   Data type. Default: ``None``.

    Returns:
        tensor (Tensor)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if value is None:
        return None

    if isinstance(value, (float, int, list, tuple, ndarray)):
        value = Tensor(value, dtype)
    else:
        if isinstance(value, Parameter):
            value = identity(value)
        elif not isinstance(value, Tensor):
            raise TypeError(f'The type of input value must be '
                            f'Tensor, Parameter, ndarray, list or tuple but got: {type(value)}')
        if dtype is not None:
            value = F.cast(value, dtype)

    return value


def get_ms_array(value: Union[float, int, Tensor, Parameter, ndarray, list, tuple],
                 dtype: type = None
                 ) -> Union[Tensor, Parameter]:
    r"""get mindspore.Tensor type of the input value

    Args:
        value (Union[float, int, Tensor, Parameter, ndarray, list, tuple]):
                        Input value
        dtype (type):   Data type. Default: ``None``.

    Returns:
        array (Tensor or Parameter)

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """

    if value is None:
        return None

    if isinstance(value, (Tensor, Parameter)):
        if dtype is not None and value.dtype != dtype:
            value = F.cast(value, dtype)
        return value

    return Tensor(value, dtype)


def check_broadcast(shape0: tuple, shape1: tuple) -> tuple:
    r"""Check whether the two shapes match the rule of broadcast.

    Args:
        shape0 (tuple): First shape
        shape1 (tuple): Second shape

    Returns:
        shape (tuple):  Shape after broadcast

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    if shape0 is None:
        return shape1
    if shape1 is None:
        return shape0

    if len(shape0) < len(shape1):
        shape0 = (1,) * (len(shape1) - len(shape0)) + shape0
    if len(shape0) > len(shape1):
        shape1 = (1,) * (len(shape0) - len(shape1)) + shape1

    shape = ()
    for s0, s1 in zip(shape0, shape1):
        if s0 == s1:
            s = s0
        else:
            if s0 == 1:
                s = s1
            elif s1 == 1:
                s = s0
            else:
                raise ValueError(f'{shape0} and {shape1} cannot be broadcast to each other!')
        shape += (s,)
    return shape


def any_none(iterable: Union[list, tuple]) -> bool:
    r"""Return True if ANY values x in the iterable is None.

    Args:
        iterable (Union[list, tuple]): Iterable variable

    Returns:
        any (bool):  If any values x in the iterable is None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return any([i is None for i in iterable])


def all_none(iterable: Union[list, tuple]) -> bool:
    r"""Return True if ALL values `x` in the `iterable` is None..

    Args:
        iterable (Union[list, tuple]): Iterable variable

    Returns:
        all (bool):  If all values `x` in the `iterable` is None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return all([i is None for i in iterable])


def any_not_none(iterable: Union[list, tuple]) -> bool:
    r"""Return True if ANY values `x` in the `iterable` is NOT None.

    Args:
        iterable (Union[list, tuple]): Iterable variable

    Returns:
        any (bool):  If any values `x` in the `iterable` is not None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return any([i is not None for i in iterable])


def all_not_none(iterable: Union[list, tuple]) -> bool:
    r"""Return True if ALL values `x` in the `iterable` is Not None..

    Args:
        iterable (Union[list, tuple]): Iterable variable

    Returns:
        all (bool):  If all values `x` in the `iterable` is Not None

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    """
    return all([i is not None for i in iterable])
