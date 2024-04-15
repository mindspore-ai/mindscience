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
"""Vector"""

from mindspore import ops
from mindspore.common import Tensor
from mindspore.ops import functional as F

from .atoms import AtomsBase
from .get import get_atoms
from ...function import get_integer, check_broadcast, all_none, any_not_none


class Vector(AtomsBase):
    r"""Vector between specific atoms or virtual atoms.

    Args:
        atoms (AtomsBase):  Atoms of shape `(..., 2, D)` to form a vector of shape `(..., D)` or `(..., 1, D)`.
                            Cannot be used with `atoms0` or `atoms1`.
                            Default: ``None``. `D` means Spatial dimension of the simulation system. Usually is 3.

        atoms0 (AtomsBase): The initial point of atoms of shape `(..., D)` to form a vector of shape `(..., D)`.
                            Must be used with `atoms1`, and cannot be used with `atoms`.
                            Default: ``None``.

        atoms1 (AtomsBase): The terminal point of atoms of shape `(..., D)` to form a vector of shape `(..., D)`.
                            Must be used with `atoms0`, and cannot be used with `atoms`.
                            Default: ``None``.

        batched (bool):     Whether the first dimension of index is the batch size.
                            Default: ``False``.

        use_pbc (bool):     Whether to calculate distance under periodic boundary condition.
                            Default: ``None``.

        keepdims (bool):    If this is set to True, the axis which is take from the `atoms` will be left,
                            and the shape of the vector will be `(..., 1, D)`
                            If this is set to False, the shape of the vector will be `(..., D)`
                            if None, its value will be determined according to the rank (number of dimension) of
                            the input atoms: False if the rank is greater than 2, otherwise True.
                            It only works when initialized with `atoms`.
                            Default: ``None``.

        axis (int):         Axis along which the coordinate of atoms are take, of which the dimension must be 2.
                            It only works when initialized with `atoms`.
                            Default: -2.

        name (str):         Name of the Colvar. Default: 'vector'.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.colvar import Vector
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> crd
        Tensor(shape=[4, 3], dtype=Float32, value=
        [[ 2.47492954e-01,  9.78153408e-01,  1.44034222e-01],
         [ 2.36211464e-01,  3.35842371e-01,  8.39536846e-01],
         [ 8.82235169e-01,  5.98322928e-01,  6.68052316e-01],
         [ 7.17712820e-01,  4.72498119e-01,  1.69098437e-01]])
        >>> vc02 = Vector(atoms0=[0], atoms1=[2])
        >>> vc02(crd)
        Tensor(shape=[1, 3], dtype=Float32, value=
        [[ 6.34742200e-01, -3.79830480e-01,  5.24018109e-01]])
    """
    def __init__(self,
                 atoms: AtomsBase = None,
                 atoms0: AtomsBase = None,
                 atoms1: AtomsBase = None,
                 batched: bool = False,
                 use_pbc: bool = None,
                 keepdims: bool = None,
                 axis: int = -2,
                 name: str = 'vector',
                 ):

        super().__init__(
            keep_in_box=False,
            name=name,
        )

        if all_none([atoms, atoms0, atoms1]):
            raise ValueError('No input atoms!')

        self.atoms = None
        self.atoms0 = None
        self.atoms1 = None

        self.split2 = None
        self.squeeze = None
        if atoms is None:
            if atoms0 is None:
                raise ValueError('atoms0 cannot be None when atoms1 is given!')
            if atoms1 is None:
                raise ValueError('atoms1 cannot be None when atoms0 is given!')

            # (..., D)
            self.atoms0 = get_atoms(atoms0, batched, False)
            self.atoms1 = get_atoms(atoms1, batched, False)

            if self.atoms0.ndim > self.atoms1.ndim:
                new_shape = (1,) * (self.atoms0.ndim - self.atoms1.ndim)
                self.atoms1.reshape(new_shape)
            if self.atoms0.ndim < self.atoms1.ndim:
                new_shape = (1,) * (self.atoms1.ndim - self.atoms0.ndim)
                self.atoms0.reshape(new_shape)

            # (..., D)
            self._set_shape(check_broadcast(self.atoms0.shape, self.atoms1.shape))
        else:
            if any_not_none([atoms0, atoms1]):
                raise ValueError('When atoms is given, atoms0 and atoms1 must be None!')

            # (..., 2, D)
            self.atoms = get_atoms(atoms, batched, False)

            axis = get_integer(axis)
            # (1, ..., 2, D)
            shape = (1,) + self.atoms.shape
            if shape[axis] != 2:
                raise ValueError(f'The dimension at axis must be 2 but got: {shape[axis]}')

            self.split2 = ops.Split(axis, 2)

            if keepdims is None:
                if self.atoms.ndim > 2:
                    keepdims = False
                else:
                    keepdims = True

            if keepdims:
                # (1, ..., 1, D) <- (1, ..., 2, D)
                shape = shape[:axis] + (1,) + shape[axis+1:]
            else:
                # (1, ..., D) <- (1, ..., 2, D)
                shape = shape[:axis] + shape[axis+1:]
                self.squeeze = ops.Squeeze(axis)

            # (..., D) <- (1, ..., D)
            self._set_shape(shape[1:])

        self.set_pbc(use_pbc)

    @property
    def ndim(self) -> int:
        """rank (number of dimensions) of the vector"""
        return self._ndim

    @property
    def shape(self) -> tuple:
        """shape of the vector"""
        return self._shape

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get vector between specific atoms or virtual atoms.

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means number of atoms in system.
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Default: ``None``.

        Returns:
            vector (Tensor):        Tensor of shape `(B, ..., D)`. Data type is float.

        """

        if self.atoms is None:
            # (..., D)
            atoms0 = self.atoms0(coordinate, pbc_box)
            atoms1 = self.atoms1(coordinate, pbc_box)
        else:
            # (B, ..., 2, D)
            atoms = self.atoms(coordinate, pbc_box)
            # (B, ..., 1, D) <- (B, ..., 2, D)
            atoms0, atoms1 = self.split2(atoms)
            if self.squeeze is not None:
                # (B, ..., D) <- (B, ..., 1, D)
                atoms0 = self.squeeze(atoms0)
                atoms1 = self.squeeze(atoms1)

        # (B, ..., D) or (B, ..., 1, D)
        vector = self.get_vector(atoms0, atoms1, pbc_box)

        if self.do_reshape:
            new_shape = coordinate.shape[0] + self._shape
            vector = F.reshape(vector, new_shape)

        # (B, ..., D) or (B, ..., 1, D)
        return vector
