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
Collective variables that accept index
"""

from inspect import signature

import mindspore as ms
from mindspore import ops
from mindspore.ops import functional as F
from mindspore import nn
from mindspore.nn import Cell
from mindspore.common import Tensor
from mindspore import numpy as msnp

from ..function import functions as func
from ..function import GetVector, get_integer

__all__ = [
    'IndexColvar',
    'IndexDistances',
    'IndexVectors',
]


class IndexColvar(Cell):
    r"""Collective variables based on index

    Args:
        use_pbc (bool, optional):     Whether to calculate the CV at periodic boundary condition (PBC).
                            If ``None`` is given, it will be determined at runtime based on
                            whether the `pbc_box` is given or not. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self, use_pbc: bool = None):

        super().__init__()

        self.get_vector = GetVector(use_pbc)
        self.use_pbc = use_pbc

        self.identity = ops.Identity()

    def vector_in_pbc(self, vector: Tensor, pbc_box: Tensor) -> Tensor:
        r"""Make the difference of vecters at the range from -0.5 box to 0.5 box.

        Args:
            vector (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
                                Coordinate of system
            pbc_box (Tensor):   Tensor of shape :math:`(B, D)`. Data type is float.
                                Periodic boundary condition Box.

        """
        return func.vector_in_pbc(vector, pbc_box)

    def set_pbc(self, use_pbc: bool):
        r"""Set periodic boundary condition

        Args:
            use_pbc (bool):     Whether to calculate the CV at periodic boundary condition (PBC).
                                If ``None`` is given, it will be determined at runtime based on
                                whether the `pbc_box` is given or not.

        """
        self.use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        return self

    def construct(self, coordinate: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        #pylint: disable=arguments-differ
        raise NotImplementedError


class IndexDistances(IndexColvar):
    r"""Calculate distance between atoms by neighbour index

    Args:
        use_pbc (bool, optional):     Whether to use periodic boundary condition. Default: ``None``.

        large_dis (float, optional):  A large value that added to the distance equal to zero to
                            prevent them from becoming zero values after Norm operation,
                            which could lead to auto-differentiation errors. Default: ``100.0``.

        keepdims (bool, optional):    If this is ``True``, the last axis will be left in the result as
                            dimensions with size one. Default: ``False``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import sponge
        >>> from sponge.partition import IndexDistances
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> idx_distances = IndexDistances(use_pbc=False)
        >>> coordinate = Tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        >>> index = Tensor([[[1],[0]]])
        >>> print(idx_distances(coordinate, index))
        [[[1.] [1.]]]

    """

    def __init__(self,
                 use_pbc: bool = None,
                 large_dis: float = 100,
                 keepdims: bool = False,
                 ):

        super().__init__(use_pbc=use_pbc)

        self.keepdims = keepdims
        self.large_dis = Tensor(large_dis, ms.float32)

        self.norm_last_dim = None
        # MindSpore < 2.0.0-rc1
        if 'ord' not in signature(ops.norm).parameters.keys():
            self.norm_last_dim = nn.Norm(-1, self.keepdims)

    def construct(self, coordinate: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        # pylint: disable=missing-docstring
        # Compute distances between atoms according to index.

        # Args:
        #     coordinate (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
        #                             Coordinate of system
        #     index (Tensor):         Tensor of shape :math:`(B, A, N)`. Data type is int.
        #                             Neighbour index
        #     mask (Tensor):          Tensor of shape :math:`(B, A, N)`. Data type is bool.
        #                             Mask of neighbour index
        #     pbc_box (Tensor):       Tensor of shape :math:`(B, D)`. Data type is float.
        #                             Periodic boundary condition Box.
        #                             Default: ``None``.

        # Returns:
        #     distances (Tensor):     Tensor of shape :math:`(B, A, N)`. Data type is float.

        # Note:

        #     - B:  Batchsize, i.e. number of simulation walker.
        #     - A:  Number of atoms.
        #     - N:  Number of neighbour atoms.
        #     - D:  Dimension of position coordinates.

        # (B, A, 1, D) <- (B, A, D)
        atoms = F.expand_dims(coordinate, -2)
        # (B, A, N, D) <- (B, A, D)
        neighbours = func.gather_vector(coordinate, index)
        vectors = self.get_vector(atoms, neighbours, pbc_box)

        # Add a non-zero value to the vectors whose mask value is False
        # to prevent them from becoming zero values after Norm operation,
        # which could lead to auto-differentiation errors
        if mask is not None:
            # (B, A, N, D) = (B, A, N, D) + (B, A, N, 1)
            large_dis = msnp.broadcast_to(self.large_dis, mask.shape)
            vectors += F.expand_dims(F.select(mask, F.zeros_like(large_dis), large_dis), -1)

        # (B, A, N) <- (B, A, N, D)
        if self.norm_last_dim is None:
            return ops.norm(vectors, None, -1, self.keepdims)

        return self.norm_last_dim(vectors)


class Vector2Distance(Cell):
    r"""Calculate distance of vector

    Args:
        axis (int, optional): Axis of vector to be calculated. Default: -1

        large_dis (float, optional): A large value that added to the distance equal to zero to prevent them from
            becoming zero values after Norm operation, which could lead to auto-differentiation errors.

        keepdims (bool, optional): If this is ``True``, the last axis will be left in the result
                                   as dimensions with size one.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 axis: int = -1,
                 large_dis: float = 100,
                 keepdims: bool = False,
                 ):

        self.axis = get_integer(axis)
        self.keepdims = keepdims
        self.large_dis = Tensor(large_dis, ms.float32)

        self.norm_last_dim = None
        # MindSpore < 2.0.0-rc1
        if 'ord' not in signature(ops.norm).parameters.keys():
            self.norm_last_dim = nn.Norm(self.axis, self.keepdims)

    def construct(self, vector: Tensor, mask: Tensor = None):
        # pylint: disable=missing-docstring
        # Compute distances between atoms according to index.

        # Args:
        #     coordinate (Tensor):    Tensor of shape :math:`(B, ..., D)`. Data type is float.
        #                             Vector
        #     mask (Tensor):          Tensor of shape :math:`(B, ...)`. Data type is bool.
        #                             Mask for Vector

        # Returns:
        #     distances (Tensor):     Tensor of shape :math:`(B, A, N)`. Data type is float.

        # Note:

        #     - B:  Batchsize, i.e. number of simulation walker.
        #     - A:  Number of atoms.
        #     - N:  Number of neighbour atoms.
        #     - D:  Dimension of position coordinates.

        # Add a non-zero value to the vectors whose mask value is False
        # to prevent them from becoming zero values after Norm operation,
        # which could lead to auto-differentiation errors
        if mask is not None:
            # (B, ...)
            large_dis = msnp.broadcast_to(self.large_dis, mask.shape)
            vector_shift = F.select(mask, F.zeros_like(large_dis), large_dis)
            # (B, ..., 1) <- (B, ...)
            vector_shift = F.expand_dims(vector_shift, self.axis)
            # (B, ..., D) = (B, ..., D) + (B, .., 1)
            vector += vector_shift

        # (B, ...) <- (B, ..., D) OR (B, ..., 1) <- (B, ..., D)
        if self.norm_last_dim is None:
            distance = ops.norm(vector, None, self.axis, self.keepdims)
        else:
            distance = self.norm_last_dim(vector)

        if mask is not None:
            if self.keepdims:
                mask = F.expand_dims(mask, self.axis)
            # (B, ...) * (B, ...) OR (B, ..., 1) * (B, ..., 1)
            distance *= mask

        return distance


class IndexVectors(IndexColvar):
    r"""Get vectors by index

    Args:
        use_pbc (bool, optional):     Whether to use periodic boundary condition.
                                      Default value: ``None``, defaults to ``False``.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import sponge
        >>> from sponge.partition import IndexVectors
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> idx_vector = IndexVectors(use_pbc=False)
        >>> coordinate = Tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
        >>> index = Tensor([[[1],[0]]])
        >>> print(idx_vector(coordinate, index))  # B=1, A=2, N=1
        [[[[ 0.  0.  1.]]
          [[ 0.  0. -1.]]]]

    """

    def __init__(self, use_pbc: bool = None):

        super().__init__(use_pbc=use_pbc)


    def construct(self, coordinate: Tensor, index: Tensor, mask: Tensor = None, pbc_box: Tensor = None):
        # pylint: disable=missing-docstring
        # get vector by index.

        # Args:
        #     coordinate (Tensor):    Tensor of shape :math:`(B, A, D)`. Data type is float.
        #                             Coordinate of system
        #     index (Tensor):         Tensor of shape :math:`(B, A, N)`. Data type is int.
        #                             Neighbour index
        #     mask (Tensor):          Tensor of shape :math:`(B, A, N)`. Data type is bool.
        #                             Mask of neighbour index
        #     pbc_box (Tensor):       Tensor of shape :math:`(B, D)`. Data type is float.
        #                             Periodic boundary condition Box.
        #                             Default: ``None``.

        # Returns:
        #     vector (Tensor):        Tensor of shape :math:`(B, A, D)`. Data type is float.

        # Note:

        #     - B:  Batchsize, i.e. number of simulation walker.
        #     - A:  Number of atoms.
        #     - N:  Number of neighbour atoms.
        #     - D:  Dimension of position coordinates.

        # (B,A,1,D) <- (B,A,D)
        atoms = F.expand_dims(coordinate, -2)
        # (B,A,N,D) <- (B,A,D)
        neighbours = func.gather_vector(coordinate, index)

        return self.get_vector(atoms, neighbours, pbc_box)
