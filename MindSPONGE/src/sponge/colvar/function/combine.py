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
Combine Colvar
"""

from typing import Union, List, Tuple
from numpy import ndarray
import mindspore as ms
import mindspore.numpy as msnp
from mindspore.ops import functional as F
from mindspore import Tensor
from mindspore.nn import CellList

from ..colvar import Colvar
from ...function import get_ms_array, any_none, any_not_none, check_broadcast


class CombineCV(Colvar):
    r"""
    Polynomial combination of a set of Colvar :math:`{s_i}` with shape (S_1, S_2, ..., S_n).
    `{S_i}` means dimensions of collective variables.

    .. math::

        S = \sum_i^n{w_i (s_i - o_i)^{p_i}}

    Args:
        colvar (Union[List[Colvar], Tuple[Colvar]]): Array of `Colvar` to be combined :math:`{s_i}`.

        weights (Union[List[float], Tuple[Float], float, Tensor]): Weights :math:`{w_i}` for each Colvar.
            If a list or tuple is given, the number of the elements should be equal to the number of CVs.
            If a float or Tensor is given, the value will be used for all Colvar. Default: 1

        offsets (Union[List[float], Tuple[Float], float, Tensor]): Offsets :math:`{o_i}` for each Colvar.
            If a list or tuple is given, the number of the elements should be equal to the number of CVs.
            If a float or Tensor is given, the value will be used for all Colvar. Default: 0

        exponents (Union[List[float], Tuple[Float], float, Tensor]): Exponents :math:`{p_i}` for each Colvar.
            If a list or tuple is given, the number of the elements should be equal to the number of CVs.
            If a float or Tensor is given, the value will be used for all Colvar. Default: 1

        normal (bool): Whether to normalize all weights to 1. Default: ``False``.

        periodic_min (Union[float, ndarray, Tensor]): The periodic minimum of the output of the combination of the CVs.
            If the output is not periodic, it should be None. Default: ``None``.

        periodic_max (Union[float, ndarray, Tensor]): The periodic maximum of the output of the combination of the CVs.
            If the output is not periodic, it should be None. Default: ``None``.

        periodic_mask (Union[Tensor, ndarray]): Mask for the periodicity of the outputs.
            The shape of the tensor should be as the same as the outputs, i.e. `(S_1, S_2, ..., S_n)`.
            Default: ``None``.

        use_pbc (bool): Whether to use periodic boundary condition. If `None` is given, it will determine whether
            to use periodic boundary conditions based on whether the `pbc_box` is provided. Default: ``None``.

        name (str): Name of the collective variables. Default: 'combine'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Union[List[Colvar], Tuple[Colvar]],
                 weights: Union[float, List[float], Tuple[float], Tensor] = 1,
                 offsets: Union[float, List[float], Tuple[float], Tensor] = 0,
                 exponents: Union[float, List[float], Tuple[float], Tensor] = 1,
                 normal: bool = False,
                 periodic_min: Union[float, ndarray, Tensor] = None,
                 periodic_max: Union[float, ndarray, Tensor] = None,
                 periodic_mask: Union[Tensor, ndarray] = None,
                 use_pbc: bool = None,
                 name: str = 'combine',
                 ):

        super().__init__(
            periodic=(periodic_min is not None),
            name=name
        )

        if any_none([periodic_min, periodic_max]) and any_not_none([periodic_min, periodic_max]):
            raise ValueError('The "periodic_min" and "periodic_max" must both be None, or neither.')

        if isinstance(colvar, Colvar):
            colvar = [colvar]
        elif not isinstance(colvar, (list, tuple)):
            raise TypeError(f'The type of "colvar" must be list of Colvar but got: {type(colvar)}')

        self.num_colvar = len(colvar)

        def _check_parameters(parameters: Union[list, tuple, float, Tensor], name: str):
            """check parameters for combination"""
            if isinstance(parameters, (list, tuple)):
                num_ = len(parameters)
                if num_ == self.num_colvar:
                    return [get_ms_array(p, ms.float32) for p in parameters]

                if num_ != 1:
                    raise ValueError(f'The number of {name} ({num_}) does not match '
                                     f'the number of colvar {self.num_colvar}')
                return [get_ms_array(parameters[0], ms.float32)] * self.num_colvar

            return [get_ms_array(parameters, ms.float32)] * self.num_colvar

        self.weights = _check_parameters(weights, 'weights')
        self.offsets = _check_parameters(offsets, 'offsets')
        self.exponents = [(None if (e == 1).all() else e) for e in _check_parameters(exponents, 'exponents')]

        if normal:
            norm_factor = 0
            for w in self.weights:
                norm_factor += w
            self.weights = [w / norm_factor for w in self.weights]

        shape = None
        colvar_ = []
        for i, cv in enumerate(colvar):
            try:
                shape = check_broadcast(shape, cv.shape)
            except ValueError:
                raise ValueError(f'The shape of the {i}-th colvar {cv.shape} cannot be '
                                 f'broadcast to the shape of the output: {shape}')

            wshape = self.weights[i].shape
            try:
                check_broadcast(wshape, cv.shape)
            except ValueError:
                raise ValueError(f'The shape of the {i}-th weight {wshape} cannot be broadcast to '
                                 f'the shape of the corresponding colvar: {cv.shape}')

            oshape = self.offsets[i].shape
            try:
                check_broadcast(oshape, cv.shape)
            except ValueError:
                raise ValueError(f'The shape of the {i}-th offset {oshape} cannot be broadcast to '
                                 f'the shape of the corresponding colvar: {cv.shape}')

            if self.exponents[i] is not None:
                eshape = self.exponents[i].shape
                try:
                    check_broadcast(eshape, cv.shape)
                except ValueError:
                    raise ValueError(f'The shape of the {i}-th exponent {eshape} cannot be broadcast to '
                                     f'the shape of the corresponding colvar: {cv.shape}')

            if use_pbc is not None:
                cv.set_pbc(use_pbc)

            colvar_.append(cv)

        self.colvar: List[Colvar] = CellList(colvar_)

        self._shape = shape
        self._ndim = len(self._shape)

        self.periodic_min = None
        self.periodic_max = None
        self.periodic_range = None
        self.periodic_mask = None
        if self._periodic:
            self.periodic_min = msnp.broadcast_to(get_ms_array(periodic_min, ms.float32), self._shape)
            self.periodic_max = msnp.broadcast_to(get_ms_array(periodic_max, ms.float32), self._shape)
            self.periodic_range = self.periodic_max - self.periodic_min
            if (self.periodic_range <= 0).any():
                raise ValueError(f'periodic_max {self.periodic_max} must be greater than'
                                 f'periodic_min {self.periodic_min}!')
            if periodic_mask is not None:
                self.periodic_mask = msnp.broadcast_to(get_ms_array(periodic_mask, ms.bool_), self._shape)

        self._periodic = msnp.broadcast_to(get_ms_array(self._periodic, ms.bool_), self._shape)

    def set_pbc(self, use_pbc: bool):
        """set whether to use periodic boundary condition"""
        self._use_pbc = use_pbc
        self.get_vector.set_pbc(use_pbc)
        for i in range(self.num_colvar):
            self.colvar[i].set_pbc(use_pbc)
        return self

    def construct(self, coordinate: Tensor, pbc_box: Tensor = None):
        r"""get position coordinates of colvar group

        Args:
            coordinate (Tensor):    Tensor of shape `(B, A, D)`. Data type is float.
                                    Position coordinate of colvar in system.
                                    `B` means batchsize, i.e. number of walkers in simulation.
                                    `A` means number of colvar in system.
                                    `D` means dimension of the simulation system. Usually is 3
            pbc_box (Tensor):       Tensor of shape `(B, D)`. Data type is float.
                                    Tensor of PBC box. Default: ``None``.

        Returns:
            combine (Tensor):       Tensor of shape `(B, S_1, S_2, ..., S_n)`. Data type is float.

        """
        colvar = 0
        for i in range(self.num_colvar):
            colvar_ = self.colvar[i](coordinate, pbc_box) - self.offsets[i]
            if self.exponents[i] is not None:
                colvar_ = msnp.power(colvar_, self.exponents[i])
            colvar += colvar_ * self.weights[i]

        if self.periodic_range is None:
            return colvar

        period = F.floor((colvar - self.periodic_min) / self.periodic_range)
        period_colvar = colvar - self.periodic_range * period

        if self.periodic_mask is None:
            return period_colvar

        return F.select(self.periodic_mask, period_colvar, colvar)


class ColvarCombine(CombineCV):
    r"""
    See `CombineCV`. NOTE: This module will be removed in a future release, please use `CombineCV` instead.

    .. math::

        S = \sum_i^n{w_i (s_i - o_i)^{p_i}}

    Args:
        colvar (list or tuple): Array of `Colvar` to be combined :math:`{s_i}`.

        weights (list, tuple, float, Tensor):
                        Weights :math:`{w_i}` for each Colvar.
                        If a list or tuple is given, the number of the elements should be equal to the number of CVs.
                        If a float or Tensor is given, the value will be used for all Colvar.
                        Default: 1

        offsets (list, tuple, float, Tensor):
                        Offsets :math:`{o_i}` for each Colvar.
                        If a list or tuple is given, the number of the elements should be equal to the number of CVs.
                        If a float or Tensor is given, the value will be used for all Colvar.
                        Default: 0

        exponents (list, tuple, float, Tensor):
                        Exponents :math:`{p_i}` for each Colvar.
                        If a list or tuple is given, the number of the elements should be equal to the number of CVs.
                        If a float or Tensor is given, the value will be used for all Colvar.
                        Default: 1

        normal (bool):  Whether to normalize all weights to 1. Default: ``False``.

        periodic_min (float, ndarray, Tensor):
                        The periodic minimum of the output of the combination of the CVs.
                        If the output is not periodic, it should be None.
                        Default: ``None``.

        periodic_max (float, ndarray, Tensor):
                        The periodic maximum of the output of the combination of the CVs.
                        If the output is not periodic, it should be None.
                        Default: ``None``.

        periodic_mask (Tensor, ndarray):
                        Mask for the periodicity of the outputs.
                        The shape of the tensor should be as the same as the outputs, i.e. `(S_1, S_2, ..., S_n)`.
                        Default: ``None``.

        use_pbc (bool): Whether to use periodic boundary condition.
                        If `None` is given, it will determine whether to use periodic boundary
                        conditions based on whether the `pbc_box` is provided.
                        Default: ``None``.

        name (str):     Name of the collective variables. Default: 'colvar_combination'

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self,
                 colvar: Union[List[Colvar], Tuple[Colvar]],
                 weights: Union[float, List[float], Tuple[float], Tensor] = 1,
                 offsets: Union[float, List[float], Tuple[float], Tensor] = 0,
                 exponents: Union[float, List[float], Tuple[float], Tensor] = 1,
                 normal: bool = False,
                 periodic_min: Union[float, ndarray, Tensor] = None,
                 periodic_max: Union[float, ndarray, Tensor] = None,
                 periodic_mask: Union[Tensor, ndarray] = None,
                 use_pbc: bool = None,
                 name: str = 'colvar_combination',
                 ):

        super().__init__(
            colvar=colvar,
            weights=weights,
            offsets=offsets,
            exponents=exponents,
            normal=normal,
            periodic_min=periodic_min,
            periodic_max=periodic_max,
            periodic_mask=periodic_mask,
            use_pbc=use_pbc,
            name=name
        )
        print('[WARNING] This module will be removed in a future release, please use `CombineCV` instead.')
