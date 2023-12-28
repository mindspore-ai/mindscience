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
Collective variables
"""

from typing import Union, Tuple, List

import mindspore as ms
from mindspore import ops
from mindspore.common import Tensor
from mindspore.nn import Cell
from mindspore.ops import functional as F

from ..function import functions as func
from ..function.operations import GetVector
from ..function import check_broadcast, get_ms_array, Units


class Colvar(Cell):
    r"""
    Base class for generalized collective variables (CVs) :math:`s(R)`.

    In mathematics, CVs :math:`s(R)` are defined as a low dimensional function of
    the atomistic coordinate :math:`R` of the simulation system, which should refer to
    the variable describing the slow motion in the process of interest.

    In MindSPONGE, Colvar Cell is the base class for ``"generalized"`` CVs. A narrow CV is
    generally a vector, i.e., its rank (ndim) is 1. For example, a CV of shape `(S)`.
    Whereas a Colvar Cell can be of higher rank (ndim), for example, a Colvar of
    shape `(S_1, S_2, ..., S_n)`

    For a Colvar, multiple values can be calculated using multiple sets of coordinates.
    Therefore, for a Colvar Cell of shape `(S_1, S_2, ... , S_n)`, a calculation using
    the `B` set of atomic coordinates represented by a tensor with shape `(B, A, D)`
    yields a Tensor with shape `(B, S_1, S_2, ... , S_n)`.
    `B` means Batchsize, i.e. number of walkers in simulation.
    `A` means Number of colvar in system.
    `D` means Dimension of the simulation system. Usually is 3.
    `{S_i}` means Dimensions of the collective variables.

    Reference:
        Yang, Y. I.; Shao, Q.; Zhang, J.; Yang, L.; Gao, Y. Q.
        Enhanced Sampling in Molecular Dynamics [J].
        The Journal of Chemical Physics, 2019, 151(7): 070902.

    Args:
        shape (Tuple):      Shape of collective variables. Default: ()

        periodic (bool):    Whether the collective variables is periodic. Default: ``False``.

        use_pbc (bool):     Whether to use periodic boundary condition.
                            If `None` is given, it will determine whether to use periodic boundary
                            conditions based on whether the `pbc_box` is provided.
                            Default: ``None``.

        name (str):         Name of the collective variables. Default: 'colvar'

        unit (str):         Unit of the collective variables.
                            NOTE: This is not the `Units` Cell that wraps length and energy.
                            Default: ``None``.

        dtype (type):       Data type of the collective variables. Default: float32

    Supported Platforms:
        ``Ascend`` ``GPU``

    """

    def __init__(self,
                 shape: Tuple[int] = (),
                 periodic: Union[bool, List[bool]] = False,
                 use_pbc: bool = None,
                 name: str = 'colvar',
                 unit: str = None,
                 dtype: type = ms.float32,
                 ):

        super().__init__()

        self._name = name

        self._periodic = get_ms_array(periodic, ms.bool_)

        # (s_1, s_2, ..., s_n)
        self._shape = None
        # rank: n
        self._ndim = None
        self._set_shape(shape)

        self._dtype = dtype

        self.get_vector = GetVector(use_pbc)
        self._use_pbc = use_pbc

        self._unit = unit

        self.identity = ops.Identity()

        self.do_reshape = False

    @property
    def use_pbc(self) -> bool:
        """whether to use periodic boundary condition

        Returns:
            bool, whether to use periodic boundary condition.

        """
        return self._use_pbc

    @use_pbc.setter
    def use_pbc(self, use_pbc_: bool):
        """set whether to use periodic boundary condition"""
        self.set_pbc(use_pbc_)

    @property
    def shape(self) -> tuple:
        """shape of the collective variables (S_1, S_2, ..., S_n)

        Returns:
            shape (tuple),  Shape of the Colvar

        """
        return self._shape

    @shape.setter
    def shape(self, shape_: tuple):
        """set shape of colvar"""
        self._set_shape(shape_)

    @property
    def name(self) -> str:
        r"""name of the collective variables

        Returns:
            str, name of the CV

        """
        return self._name

    @property
    def ndim(self) -> int:
        r"""rank (number of dimensions) of the collective variables

        Returns:
            int, rank of the CV

        """
        return self._ndim

    @property
    def dtype(self) -> type:
        """data type of the collective variables.

        Returns:
            type, data type of the Colvar

        """
        return self._dtype

    @property
    def periodic(self) -> Tensor:
        """return a Tensor of data type `bool` to indicate whether the CV is periodic or not"""
        return self._periodic

    @property
    def any_periodic(self) -> bool:
        """whether any dimension is periodic"""
        return self._periodic.any()

    @property
    def all_periodic(self) -> bool:
        """whether all dimensions are periodic"""
        return self._periodic.all()

    @classmethod
    def vector_in_pbc(cls, vector: Tensor, pbc_box: Tensor) -> Tensor:
        """Make the difference of vectors at the range from -0.5 box to 0.5 box"""
        return func.vector_in_pbc(vector, pbc_box)

    def set_name(self, name: str):
        """set the name of the collective variables"""
        if not isinstance(name, str):
            raise ValueError(f'The type of name must be `str` but got: {type(name)}')
        self._name = name
        return self

    def get_unit(self, units: Units = None) -> str:
        """return unit of the collective variables"""
        #pylint: disable=unused-argument
        return self._unit

    def reshape(self, input_shape: tuple):
        """rearranges the shape"""
        if input_shape != self._shape:
            self.do_reshape = True
            self._shape = input_shape
            self._ndim = len(self._shape)
            self._periodic = F.reshape(self._periodic, self._shape)
        return self

    def set_pbc(self, use_pbc: bool):
        """set whether to use periodic boundary condition"""
        self._use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        return self

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get the value of a collective variables :math:`s(R)` with shape `(B, S_1, S_2, ..., S_n)`
            at system coordinate :math:`R`

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
                                    Position coordinate of atoms in system
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            colvar (Tensor):        Tensor of shape `(B, S_1, S_2, ..., S_n)`.

        """

        # (B, S_1, S_2, ..., S_n)
        raise NotImplementedError

    def _set_shape(self, shape: tuple):
        """set shape of colvar"""
        self._shape = shape
        self._ndim = len(self.shape)
        if self._periodic.shape != self._shape:
            if not check_broadcast(self._periodic.shape, self._shape):
                raise ValueError(f'The shape of periodic {self._periodic.shape} can not be broadcast to '
                                 f'the shape of CVs: {self._shape}')
            self._periodic = F.broadcast_to(self._periodic, self._shape)
        return self
