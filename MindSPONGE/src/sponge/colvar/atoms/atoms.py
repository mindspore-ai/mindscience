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
Atoms
"""

from typing import Union, List
from numpy import ndarray

import mindspore as ms
import mindspore.numpy as msnp
from mindspore.ops import functional as F
from mindspore.common import Tensor

from ..colvar import Colvar
from ...function import functions as func
from ...function import get_ms_array, get_integer, Units

__all__ = [
    'AtomsBase',
    'Atoms',
    'BatchedAtoms',
]


class AtomsBase(Colvar):
    r"""
    Base class for specific atoms group, used as the "atoms group module" in MindSPONGE.

    The `AtomsBase` Cell is a special subclass of `Colvar`. It has the shape `(a_1, a_2, ... , a_n, D)`,
    where `D` is the dimension of the atomic coordinates (usually 3). As with the Colvar Cell, when it takes
    as input coordinates of shape `(B, A, D)`, it returns the shape of the Tensor with an extra dimension `B`,
    i.e. `(B, a_1, a_2, ... , a_n, D)`. B means Batchsize, i.e. number of walkers in simulation.
    {a_i} means Dimensions of the Atoms Cell.

    Args:
        keep_in_box (bool): Whether to keep the coordinate in PBC box.
                            Default: ``False``.

        dimension (int):    Spatial dimension of the simulation system. Default: 3

        name (str):         Name of the Colvar. Default: 'atoms'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 keep_in_box: bool = False,
                 dimension: int = 3,
                 name: str = 'atoms'
                 ):

        super().__init__(
            shape=(1, dimension),
            periodic=False,
            name=name,
        )

        self.dimension = get_integer(dimension)
        self.keep_in_box = keep_in_box

    @property
    def ndim(self) -> int:
        """rank (number of dimensions) of the atoms"""
        return self._ndim

    @property
    def shape(self) -> tuple:
        """shape of the atoms"""
        return self._shape

    def set_dimension(self, dimension: int = 3):
        """set the spatial dimension of the simulation system"""
        self.dimension = get_integer(dimension)
        self._shape = self.shape[:-1] + (self.dimension,)
        return self

    def get_unit(self, units: Units = None) -> str:
        """return unit of the collective variables"""
        return units.length_unit_name

    def reshape(self, input_shape: tuple):
        """rearranges the shape of atoms"""
        if input_shape != self._shape:
            self.do_reshape = True
            self._shape = input_shape
            self._ndim = len(self._shape)
            self._periodic = msnp.full(self._shape, False)
        return self

    def coordinate_in_pbc(self, coordinate: Tensor, pbc_box: Tensor = None) -> Tensor:
        """displace the coordinate in PBC box."""
        if pbc_box is None:
            return coordinate
        return func.coordinate_in_pbc(coordinate, pbc_box)

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get position coordinate(s) of specific atom(s)

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            position (Tensor):  Tensor of shape (B, ..., D). Data type is float.
        """

        # (B, a_1, a_2, ..., a_n, D)
        raise NotImplementedError

    def _set_shape(self, shape: tuple):
        """set shape of Atoms"""
        super()._set_shape(shape)
        self.dimension = self._shape[-1]
        return self


class Atoms(AtomsBase):
    r"""
    Specific atoms group initialized using an array of atomic indices. It is a subclass of `AtomsBase`.

    When initializing, the Atoms Cell accepts as input an array of atomic indices, which can either be common
    to all walkers or have a separate index for each walker.

    To set a common atomic index, set `batched` to `False`, where the shape of the `index` is the same as
    the shape of the `Atoms` Cell, which is `(a_1, a_2, ... , a_n)`, while the shape of the returned Tensor is
    `(B, a_1, a_2, ... , a_n, D)`. `B` means Batchsize, i.e. number of walkers in simulation.
    `{a_i}` means Dimensions of the Atoms Cell. `D` means Dimension of the simulation system. Usually is 3.

    To set a separate atomic index for each walker, set `Batched` to `True`. In this case, the shape of `index`
    should be `(B, a_1, a_2, ... , a_n)`, while the shape of the `Atoms` Cell would be `(a_1, a_2, ... , a_n)`.
    The batch size `B` of the atomic indices should be the same as the batch size of the simulation system.
    The shape of the returned Tensor of the `Atoms` Cell is `(B, a_1, a_2, ... , a_n, D)`.

    Args:
        index (Union[Tensor, ndarray, List[int]]):
                            Array of the indices of specific atoms.
                            The shape of tensor is (a_1, a_2, ..., a_n) or (B, a_1, a_2, ..., a_n), and the
                            data type is int.

        batched (bool):     Whether the first dimension of index is the batch size.
                            Default: ``False``.

        keep_in_box (bool): Whether to displace the coordinate in PBC box.
                            Default: ``False``.

        dimension (int):    Spatial dimension of the simulation system. Default: 3

        name (str):         Name of the Colvar. Default: 'atoms'

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from sponge.colvar import Atoms
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> crd
        Tensor(shape=[4, 3], dtype=Float32, value=
        [[ 2.47492954e-01,  9.78153408e-01,  1.44034222e-01],
         [ 2.36211464e-01,  3.35842371e-01,  8.39536846e-01],
         [ 8.82235169e-01,  5.98322928e-01,  6.68052316e-01],
         [ 7.17712820e-01,  4.72498119e-01,  1.69098437e-01]])
        >>> atom_1 = Atoms(0)
        >>> atom_1(crd)
        Tensor(shape=[1, 3], dtype=Float32, value=
        [[ 2.47492954e-01,  9.78153408e-01,  1.44034222e-01]])
    """
    def __init__(self,
                 index: Union[Tensor, ndarray, List[int]],
                 batched: bool = False,
                 keep_in_box: bool = False,
                 dimension: int = 3,
                 name: str = 'atoms',
                 ):

        super().__init__(
            keep_in_box=keep_in_box,
            dimension=dimension,
            name=name,
        )

        self.index = get_ms_array(index, ms.int32)
        if self.index.ndim == 0:
            self.index = self.index.reshape((1,))
        if batched:
            # shape of self.index: (B, a_1, a_2, ..., a_n)
            if self.index.ndim == 1:
                raise ValueError('The rank of index must be larger than 1 '
                                 'when using batched!')
        else:
            # (1, a_1, a_2, ..., a_n) <- (a_1, a_2, ..., a_n)
            self.index = F.expand_dims(self.index, 0)

        # (a_1, a_2, ..., a_n, -1)
        self._set_shape(self.index.shape[1:] + (self.dimension,))

        self.keep_in_box = keep_in_box

    def reshape(self, input_shape: tuple):
        """rearranges the shape of atoms"""
        shape = self._shape[0] + input_shape[:-1]
        self.index = F.reshape(self.index, shape)
        self._shape = self.index.shape[1:] + (self.dimension,)
        self._ndim = self.index.ndim
        self._periodic = msnp.full(self._shape, False)
        return self

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get position coordinate(s) of specific atom(s)

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
                                    Position coordinate of atoms in system. `A` means Number of atoms in system.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            position (Tensor):  Tensor of shape (B, a_1, a_2, ..., a_{n}, D). Data type is float.

        """
        # (B, a_1, a_2, ..., a_{n}, D) <- (B, A, D)
        atoms = func.gather_vector(coordinate, self.index)
        if self.keep_in_box:
            atoms = self.coordinate_in_pbc(atoms, pbc_box)
        return atoms


class BatchedAtoms(Atoms):
    r"""
    A batched version of Atoms Cell. It is a subclass of `Atoms`.

    the shape of `index` should be `(B, a_1, a_2, ... , a_n)`, while the shape of the `Atoms` Cell would be
    `(a_1, a_2, ... , a_n)`. The batch size `B` of the atomic indices should be the same as the batch size
    of the simulation system. The shape of the returned Tensor of the `Atoms` Cell is `(B, a_1, a_2, ... , a_n, D)`.
    `{a_i}` means Dimensions of the Atoms Cell. `D` means Dimension of the simulation system. Usually is 3.

    Args:
        index (Union[Tensor, ndarray, List[int]]):
                            Array of the indices of specific atoms.
                            The shape of tensor is (B, a_1, a_2, ..., a_{n}), and the data type is int

        keep_in_box (bool): Whether to displace the coordinate in PBC box.
                            Default: ``False``.

        dimension (int):    Spatial dimension of the simulation system. Default: 3

        name (str):         Name of the Colvar. Default: 'atoms'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 index: Union[Tensor, ndarray, List[int]],
                 keep_in_box: bool = False,
                 dimension: int = 3,
                 name: str = 'atoms',
                 ):

        super().__init__(
            index=index,
            batched=True,
            keep_in_box=keep_in_box,
            dimension=dimension,
            name=name,
        )
