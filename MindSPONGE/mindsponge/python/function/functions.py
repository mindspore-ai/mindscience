# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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

from typing import Union
import numpy as np
from numpy import ndarray
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import ops
from mindspore import jit
from mindspore import Tensor, Parameter

__all__ = [
    'PI',
    'inv',
    'keepdim_sum',
    'keepdim_mean',
    'keepdim_prod',
    'keep_norm_last_dim',
    'norm_last_dim',
    'reduce_any',
    'reduce_all',
    'concat_last_dim',
    'concat_penulti',
    'identity',
    'pbc_box_reshape',
    'periodic_image',
    'displace_in_box',
    'vector_in_box',
    'get_vector_without_pbc',
    'get_vector_with_pbc',
    'get_vector',
    'gather_vectors',
    'gather_values',
    'calc_distance_without_pbc',
    'calc_distance_with_pbc',
    'calc_distance',
    'calc_angle_between_vectors',
    'calc_angle_without_pbc',
    'calc_angle_with_pbc',
    'calc_angle',
    'calc_torsion_for_vectors',
    'calc_torsion_without_pbc',
    'calc_torsion_with_pbc',
    'calc_torsion',
    'get_kinetic_energy',
    'get_integer',
    'get_ndarray',
    'get_tensor',
    'get_ms_array',
]

PI = 3.141592653589793238462643383279502884197169399375105820974944592307

inv = ops.Inv()
keepdim_sum = ops.ReduceSum(keep_dims=True)
keepdim_mean = ops.ReduceMean(keep_dims=True)
keepdim_prod = ops.ReduceProd(keep_dims=True)
reduce_any = ops.ReduceAny()
reduce_all = ops.ReduceAll()
concat_last_dim = ops.Concat(-1)
concat_penulti = ops.Concat(-2)
identity = ops.Identity()
gather = ops.Gather()


@jit
def norm_last_dim(vector: Tensor) -> Tensor:
    r"""Compute the norm of vector, delete the last dims

    Args:
        vector (Tensor):    Tensor of shape (..., D). Data type is float.

    Returns:
        Tensor of shape (...,). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    """
    return msnp.norm(vector, axis=-1)


@jit
def keep_norm_last_dim(vector: Tensor) -> Tensor:
    r"""Compute the norm of vector, keep the last dims

    Args:
        vector (Tensor):    Tensor of shape (..., D). Data type is float.

    Returns:
        Tensor of shape (..., 1). Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    """
    return msnp.norm(vector, axis=-1, keep_dims=True)

@jit
def pbc_box_reshape(pbc_box: Tensor, ndim: int) -> Tensor:
    r"""
    Reshape the pbc_box as the same ndim.

    Args:
        pbc_box (Tensor):   Tensor of shape (B,D). Data type is float.
        ndim (int):         The rank (ndim) of the pbc_box.

    Returns:
        pbc_box (Tensor), Tensor of shape (B,1,..,1,D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    if ndim <= 2:
        return pbc_box
    shape = pbc_box.shape[:1] + (1,) * (ndim - 2) + pbc_box.shape[-1:]
    return ops.reshape(pbc_box, shape)


@jit
def periodic_image(position: Tensor, pbc_box: Tensor, shift: float = 0) -> Tensor:
    r"""
    calculate the periodic image of the PBC box.

    Args:
        position (Tensor):  Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
        shift (float):      Shift of PBC box. Default: 0

    Returns:
        image (Tensor), Tensor of shape (B, ..., D). Data type is int32.

    Symbols:
        - B:  Batchsize, i.e. number of walkers in simulation.
        - D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    pbc_box = pbc_box_reshape(ops.stop_gradient(pbc_box), position.ndim)
    image = -ops.floor(position / pbc_box - shift)
    return ops.cast(image, ms.int32)


@jit
def displace_in_box(position: Tensor, pbc_box: Tensor, shift: float = 0) -> Tensor:
    r"""
    displace the positions of system in a PBC box.

    Args:
        position (Tensor):  Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
        shift (float):      Shift of PBC box. Default: 0

    Returns:
        position_in box (Tensor), Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    pbc_box = pbc_box_reshape(ops.stop_gradient(pbc_box), position.ndim)
    image = -ops.floor(position / pbc_box - shift)
    return position + pbc_box * image


@jit
def vector_in_box(vector: Tensor, pbc_box: Tensor) -> Tensor:
    r"""
    Make the vector at the range from -0.5 box to 0.5 box
    at perodic bundary condition. (-0.5box < difference < 0.5box)

    Args:
        vector (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.

    Returns:
        diff_in_box (Tensor), Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        - B:  Batchsize, i.e. number of walkers in simulation.
        - D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    pbc_box = pbc_box_reshape(pbc_box, vector.ndim)
    box_nograd = ops.stop_gradient(pbc_box)
    inv_box = msnp.reciprocal(box_nograd)
    vector -= box_nograd * ops.floor(vector * inv_box + 0.5)
    return  vector * inv_box * pbc_box

@jit
def get_vector_without_pbc(initial: Tensor, terminal: Tensor, _pbc_box=None) -> Tensor:
    r"""
    Compute vector from initial point to terminal point without perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of initial point.
        terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of terminal point.
        _pbc_box (None):    Dummy.

    Returns:
        vector (Tensor), Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    #pylint: disable=invalid-name

    return terminal - initial


@jit
def get_vector_with_pbc(initial: Tensor, terminal: Tensor, pbc_box: Tensor) -> Tensor:
    r"""
    Compute vector from initial point to terminal point at perodic bundary condition.

    Args:
        initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of initial point.
        terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of terminal point.
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.

    Returns:
        vector (Tensor), Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    return vector_in_box(terminal-initial, pbc_box)

@jit
def get_vector(initial: Tensor, terminal: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""
    Compute vector from initial point to terminal point.

    Args:
        initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of initial point.
        terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                            Coordinate of terminal point.
        pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
                            Default: None

    Returns:
        vector (Tensor), Tensor of shape (B, ..., D). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    vector = terminal - initial
    if pbc_box is None:
        return vector
    return vector_in_box(vector, pbc_box)


@jit
def gather_vectors(tensor: Tensor, index: Tensor) -> Tensor:
    r"""
    Gather vectors from the penultimate axis (axis=-2) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape (B, A, D).
        index (Tensor):     Tensor of shape (B, ...,). Data type is int.

    Returns:
        vector (Tensor), Tensor of shape (B, ..., D).

    Symbols:
        B:  Batch size.
        A:  Atom nums.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    if index.shape[0] == 1:
        index1 = ops.reshape(index, index.shape[1:])
        if tensor.shape[0] == 1:
            tensor1 = ops.reshape(tensor, tensor.shape[1:])
            res = gather(tensor1, index1, len(tensor1.shape) - 2)
            res = ops.reshape(res, (1,) + res.shape)
            return res
        return gather(tensor, index1, len(tensor.shape) - 2)
    if tensor.shape[0] == 1:
        tensor1 = ops.reshape(tensor, tensor.shape[1:])
        return gather(tensor1, index, len(tensor1.shape) - 2)

    # (B, N, M):
    shape0 = index.shape
    # (B, N*M, 1) <- (B, N, M):
    index = ops.reshape(index, (shape0[0], -1, 1))
    # (B, N*M, D) <- (B, N, D):
    neigh_atoms = msnp.take_along_axis(tensor, index, axis=-2)
    # (B, N, M, D) <- (B, N, M) + (D,):
    output_shape = shape0 + tensor.shape[-1:]

    # (B, N, M, D):
    return ops.reshape(neigh_atoms, output_shape)


@jit
def gather_values(tensor: Tensor, index: Tensor) -> Tensor:
    r"""
    Gather values from the last axis (axis=-1) of the tensor according to index.

    Args:
        tensor (Tensor):    Tensor of shape (B, X).
        index (Tensor):     Tensor of shape (B, ...,). Data type is int.

    Returns:
        value (Tensor), Tensor of shape (B, ...,).

    Symbols:
        B:  Batch size.
        X:  Any value.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    if index.shape[0] == 1:
        index1 = ops.reshape(index, index.shape[1:])
        if tensor.shape[0] == 1:
            tensor1 = ops.reshape(tensor, tensor.shape[1:])
            res = gather(tensor1, index1, len(tensor1.shape) - 1)
            res = ops.reshape(res, (1,) + res.shape)
            return res
        return gather(tensor, index1, len(tensor.shape) - 1)
    if tensor.shape[0] == 1:
        tensor1 = ops.reshape(tensor, tensor.shape[1:])
        return gather(tensor1, index, len(tensor1.shape) - 1)

    # (B, N, M):
    origin_shape = index.shape
    # (B, N*M) <- (B, N, M):
    index = ops.reshape(index, (origin_shape[0], -1))

    # (B, N*M):
    neigh_values = ops.gather_d(tensor, -1, index)

    # (B, N, M):
    return ops.reshape(neigh_values, origin_shape)


@jit
def calc_distance_without_pbc(position_a: Tensor, position_b: Tensor, _pbc_box=None) -> Tensor:
    r"""
    Compute distance between position A and B without perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (..., D). Data type is float.
        _pbc_box (None):        Dummy.

    Returns:
        distance (Tensor), Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindsponge.function import calc_distance_without_pbc
        >>> from mindspore.common.tensor import Tensor
        >>> A = Tensor([[1.,2.,3.]])
        >>> B = Tensor([[1.,1.,1.]])
        >>> print (calc_distance_with_pbc(A,B))
        tensor(shape=[1,1],dtype = Float32, value = [[2.236060801]])
    """
    #pylint: disable=invalid-name

    vec = get_vector_without_pbc(position_a, position_b)
    return msnp.norm(vec, axis=-1, keepdims=True)


@jit
def calc_distance_with_pbc(position_a: Tensor, position_b: Tensor, pbc_box: Tensor) -> Tensor:
    r"""
    Compute distance between position A and B at perodic bundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        distance (Tensor), Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindsponge.function import calc_distance_with_pbc
        >>> from mindspore.common.tensor import Tensor
        >>> A = Tensor([[1.,2.,3.]])
        >>> B = Tensor([[1.,1.,1.]])
        >>> pbc_box = Tensor([[0.7,0.7,0.7]])
        >>> print (calc_distance_with_pbc(A,B,pbc_box))
        tensor(shape=[1,1],dtype = Float32, value = [[3.16227734e-01]])
    """

    vec = get_vector_with_pbc(position_a, position_b, pbc_box)
    return msnp.norm(vec, axis=-1, keepdims=True)


@jit
def calc_distance(position_a: Tensor, position_b: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""
    Compute distance between position A and B.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        distance (Tensor), Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore
        >>> from mindsponge.function import calc_distance
        >>> from mindspore.common.tensor import Tensor
        >>> A = Tensor([[1.,2.,3.]])
        >>> B = Tensor([[1.,1.,1.]])
        >>> pbc_box = Tensor([[0.7,0.7,0.7]])
        >>> print (calc_distance(A,B,pbc_box))
        tensor(shape=[1,1],dtype = Float32, value = [[3.16227734e-01]])
        >>> print (calc_distance(A,B))
        tensor(shape=[1,1],dtype = Float32, value = [[2.236060801]])
    """

    vec = get_vector_without_pbc(position_a, position_b)
    if pbc_box is not None:
        vec = vector_in_box(vec, pbc_box)
    return msnp.norm(vec, axis=-1, keepdims=True)


@jit
def calc_angle_between_vectors(vector1: Tensor, vector2: Tensor) -> Tensor:
    r"""
    Compute angle between two vectors. For vector :math:`\vec {V_1} = (x_1, x_2, x_3, ..., x_n)`
    and :math:`\vec {V_2} = (y_1, y_2, y_3, ..., y_n)` , the formula is

    .. math::

        \theta = \arccos {\frac{|x_1y_1 + x_2y_2 + \cdots + x_ny_n|}{\sqrt{x_1^2 + x_2^2 +
                 \cdots + x_n^2}\sqrt{y_1^2 + y_2^2 + \cdots + y_n^2}}}

    Args:
        vector1 (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
        vector1 (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.

    Returns:
        angle (Tensor), Tensor of shape :math:`(..., 1)`. Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> a = Tensor([[1., 2., 3.], [1., 2., 3.]])
        >>> b = Tensor([[1., 1., 1.], [2., 2., 2.]])
        >>> print(mindsponge.function.calc_angle_between_vectors(a, b))
        Tensor(shape=[2, 1], dtype=Float64, value=
        [[3.87596687e-01],
         [3.87596687e-01]])
    """

    # [..., 1] <- [..., D]
    dis1 = msnp.norm(vector1, axis=-1, keepdims=True)
    dis2 = msnp.norm(vector2, axis=-1, keepdims=True)
    dot12 = keepdim_sum(vector1 * vector2, -1)
    # [..., 1]
    cos_theta = dot12 / dis1 / dis2
    return ops.acos(cos_theta)


@jit
def calc_angle_without_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor) -> Tensor:
    r"""
    Compute angle :math:`\angle ABC` formed by three positions A, B, C without periodic boundary condition.

    Calculate the coordinates of vectors :math:`\vec{BA}` and :math:`\vec{BC}` according to the coordinates of A, B, C
    without periodic boundary condition, then use the vectors to calculate the angle.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
        position_b (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.
        position_c (Tensor):    Tensor of shape :math:`(..., D)` . Data type is float.

    Returns:
        angle (Tensor), Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> A = Tensor([[1., 2., 3.]])
        >>> B = Tensor([[1., 1., 1.]])
        >>> C = Tensor([[4., 5., 6.]])
        >>> print(mindsponge.function.calc_angle_without_pbc(A, B, C))
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 4.83361423e-01]])
    """

    # (...,D)
    vec_ba = get_vector_without_pbc(position_b, position_a)
    vec_bc = get_vector_without_pbc(position_b, position_c)
    return calc_angle_between_vectors(vec_ba, vec_bc)


@jit
def calc_angle_with_pbc(position_a: Tensor, position_b: Tensor, position_c: Tensor, pbc_box: Tensor) -> Tensor:
    r"""
    Compute angle :math:`\angle ABC` formed by three positions A, B, C with periodic boundary condition.
    Put in the coordinates of A, B, C and pbc_box, and get the angle :math:`\angle ABC` .

    Calculate the coordinates of vectors :math:`\vec{BA}` and :math:`\vec{BC}` according to the coordinates of A, B, C
    with periodic boundary condition, then use the vectors to calculate the angle.

    Args:
        position_a (Tensor):    Tensor of shape :math:`(B, ..., D)` . Data type is float.
        position_b (Tensor):    Tensor of shape :math:`(B, ..., D)` . Data type is float.
        position_c (Tensor):    Tensor of shape :math:`(B, ..., D)` . Data type is float.
        pbc_box (Tensor):       Tensor of shape :math:`(B, D)` . Data type is float.

    Returns:
        angle (Tensor), Tensor of shape :math:`(B, ..., 1)` . Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> A = Tensor([[1., 2., 3.]])
        >>> B = Tensor([[1., 1., 1.]])
        >>> C = Tensor([[4., 5., 6.]])
        >>> pbc_box = Tensor([[0.7, 0.7, 0.7]])
        >>> print(mindsponge.function.calc_angle_with_pbc(A, B, C, pbc_box=pbc_box))
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 2.40069723e+00]])
    """

    # (B, ..., D)
    vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
    vec_bc = get_vector_with_pbc(position_b, position_c, pbc_box)
    return calc_angle_between_vectors(vec_ba, vec_bc)


@jit
def calc_angle(position_a, position_b: Tensor, position_c: Tensor, pbc_box: Tensor = None) -> Tensor:
    r"""
    Compute angle :math:`\angle ABC` formed by three positions A, B, C.

    If pbc_box is provided, calculate the angle according to the coordinates with periodic boundary condition.
    If pbc_box is None, calculate the angle according to the coordinates without periodic boundary condition.

    Finally return the angle between vector :math:`\vec{BA}` and vector :math:`\vec{BC}` .

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float. Default: None

    Returns:
        angle (Tensor), Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindsponge
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindsponge.function import calc_angle
        >>> A = Tensor([[1., 2., 3.]])
        >>> B = Tensor([[1., 1., 1.]])
        >>> C = Tensor([[4., 5., 6.]])
        >>> print(calc_angle(A, B, C, pbc_box=None))
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 4.83361423e-01]])
        >>> A = Tensor([[1., 2., 3.]])
        >>> B = Tensor([[1., 1., 1.]])
        >>> C = Tensor([[4., 5., 6.]])
        >>> pbc_box = Tensor([[0.7, 0.7, 0.7]])
        >>> print(calc_angle(A, B, C, pbc_box=pbc_box))
        Tensor(shape=[1, 1], dtype=Float32, value=
        [[ 2.40069723e+00]])
    """

    # (B, ..., D)
    if pbc_box is None:
        vec_ba = get_vector_without_pbc(position_b, position_a)
        vec_bc = get_vector_without_pbc(position_b, position_c)
    else:
        vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
        vec_bc = get_vector_with_pbc(position_b, position_c, pbc_box)
    return calc_angle_between_vectors(vec_ba, vec_bc)


@jit
def calc_torsion_for_vectors(vector1: Tensor, vector2: Tensor, vector3: Tensor) -> Tensor:
    r"""
    Compute torsion angle formed by three vectors.

    Args:
        vector1 (Tensor):   Tensor of shape (..., D). Data type is float.
        vector2 (Tensor):   Tensor of shape (..., D). Data type is float.
        vector3 (Tensor):   Tensor of shape (..., D). Data type is float.

    Returns:
        torsion (Tensor), Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    # (B, ..., D) <- (B,...,1)
    v2norm = msnp.norm(vector2, axis=-1, keepdims=True)
    # (B, ..., D) = (B, ..., D) / (...,1)
    norm_vec2 = vector2 / v2norm

    # (B, ..., D)
    vec_a = msnp.cross(norm_vec2, vector1)
    vec_b = msnp.cross(vector3, norm_vec2)
    cross_ab = msnp.cross(vec_a, vec_b)

    # (B,...,1)
    sin_phi = keepdim_sum(cross_ab*norm_vec2, -1)
    cos_phi = keepdim_sum(vec_a*vec_b, -1)

    return ops.atan2(-sin_phi, cos_phi)


@jit
def calc_torsion_without_pbc(position_a: Tensor,
                             position_b: Tensor,
                             position_c: Tensor,
                             position_d: Tensor
                             ) -> Tensor:
    r"""
    Compute torsion angle formed by four positions A-B-C-D without periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (..., D). Data type is float.
        position_d (Tensor):    Tensor of shape (..., D). Data type is float.

    Returns:
        torsion (Tensor), Tensor of shape (..., 1). Data type is float.

    Symbols:
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    vec_ba = get_vector_without_pbc(position_b, position_a)
    vec_cb = get_vector_without_pbc(position_c, position_b)
    vec_dc = get_vector_without_pbc(position_d, position_c)
    return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)


@jit
def calc_torsion_with_pbc(position_a: Tensor,
                          position_b: Tensor,
                          position_c: Tensor,
                          position_d: Tensor,
                          pbc_box: Tensor
                          ) -> Tensor:
    r"""
    Compute torsion angle formed by four positions A-B-C-D at periodic boundary condition.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_d (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        torsion (Tensor), Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
    vec_cb = get_vector_with_pbc(position_c, position_b, pbc_box)
    vec_dc = get_vector_with_pbc(position_d, position_c, pbc_box)
    return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)


@jit
def calc_torsion(position_a: Tensor,
                 position_b: Tensor,
                 position_c: Tensor,
                 position_d: Tensor,
                 pbc_box: Tensor = None
                 ) -> Tensor:
    r"""
    Compute torsion angle formed by four positions A-B-C-D.

    Args:
        position_a (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_b (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_c (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        position_d (Tensor):    Tensor of shape (B, ..., D). Data type is float.
        pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.

    Returns:
        torsion (Tensor), Tensor of shape (B, ..., 1). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    if pbc_box is None:
        vec_ba = get_vector_without_pbc(position_b, position_a)
        vec_cb = get_vector_without_pbc(position_c, position_b)
        vec_dc = get_vector_without_pbc(position_d, position_c)
    else:
        vec_ba = get_vector_with_pbc(position_b, position_a, pbc_box)
        vec_cb = get_vector_with_pbc(position_c, position_b, pbc_box)
        vec_dc = get_vector_with_pbc(position_d, position_c, pbc_box)

    return calc_torsion_for_vectors(vec_ba, vec_cb, vec_dc)


@jit
def get_kinetic_energy(mass: Tensor, velocity: Tensor) -> Tensor:
    r"""
    Compute kinectic energy of the simulation system.

    Args:
        mass (Tensor):      Tensor of shape (B, A). Data type is float.
                            Mass of the atoms in system.
        velocity (Tensor):  Tensor of shape (B, A, D). Data type is float.
                            Velocities of the atoms in system.

    Returns:
        kinectics (Tensor), Tensor of shape (B). Data type is float.

    Symbols:
        B:  Batchsize, i.e. number of walkers in simulation.
        A:  Number of atoms in the simulation system.
        D:  Dimension of the simulation system. Usually is 3.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """

    # (B, A) <- (B, A, D)
    v2 = ops.reduce_sum(velocity*velocity, -1)
    # (B, A) * (B, A)
    kinectics = 0.5 * mass * v2
    # (B) <- (B, A)
    return ops.reduce_sum(kinectics, -1)


def get_integer(value: Union[int, Tensor, Parameter, ndarray]) -> int:
    r"""
    get integer type of the input value.

    Args:
        value (Union[int, Tensor, Parameter, ndarray]): Input value.

    Returns:
        integer, the integer type of the input value.

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    if value is None:
        return None
    if isinstance(value, Tensor):
        value = value.asnumpy()
    return int(value)


def get_ndarray(value: Union[Tensor, Parameter, ndarray, list, tuple], dtype: type = None) -> ndarray:
    r"""
    get ndarray type of the input value.

    Args:
        value (Union[Tensor, Parameter, ndarray]):  Input value.
        dtype (type):                               Data type. Default: None

    Returns:
        array (ndarray).

    Supported Platforms:
        ``Ascend`` ``GPU``
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


def get_tensor(value: Union[Tensor, Parameter, ndarray, list, tuple], dtype: type = None) -> Tensor:
    r"""
    get mindspore.Tensor type of the input value.

    Args:
        value (Union[Tensor, Parameter, ndarray, list, tuple]):  Input value
        dtype (type):                                            Data type. Default: None

    Returns:
        tensor (Tensor)

    """
    if value is None:
        return None

    if isinstance(value, (list, tuple, ndarray)):
        value = Tensor(value, dtype)
    else:
        if isinstance(value, Parameter):
            value = identity(value)
        elif not isinstance(value, Tensor):
            raise TypeError('The type of input value must be Tensor, Parameter, '
                            'ndarray, list or tuple but got: ' + str(type(value)))
        if dtype is not None:
            value = ops.cast(value, dtype)

    return value


def get_ms_array(value: Union[Tensor, Parameter, ndarray, list, tuple], dtype: type = None) -> Union[Tensor, Parameter]:
    r"""
    get mindspore.Tensor type of the input value.

    Args:
        value (Union[Tensor, Parameter, ndarray, list, tuple]):  Input value

    Returns:
        array (Tensor or Parameter)

    """

    if value is None:
        return None

    if isinstance(value, (Tensor, Parameter)):
        if dtype is not None and value.dtype != dtype:
            value = ops.cast(value, dtype)
        return value

    if isinstance(value, (list, tuple, np.ndarray)):
        return Tensor(value, dtype)

    return None
