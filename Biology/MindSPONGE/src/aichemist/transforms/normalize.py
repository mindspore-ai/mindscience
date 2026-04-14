# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
Modules for normalization
"""

from typing import Union
from numpy import ndarray

import mindspore as ms
from mindspore.numpy import count_nonzero
from mindspore import Tensor, Parameter
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F


class ScaleShift(Cell):
    r"""A network to scale and shift the label of dataset or prediction.

    Args:
        scale (float): Scale value. Default: 1

        shift (float): Shift value. Default: 0

        type_ref (Union[Tensor, ndarray]): Tensor of shape (T, E). Data type is float
            Reference values of label for each atom type. Default: ``None``.

        by_atoms (bool): Whether to do atomwise scale and shift. Default: ``None``.

        axis (int): Axis to summation the reference value of molecule. Default: -2

    Note:

        B:  Batch size

        A:  Number of atoms

        T:  Number of total atom types

        Y:  Number of labels

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 scale: Union[float, Tensor, ndarray] = 1,
                 shift: Union[float, Tensor, ndarray] = 0,
                 type_ref: Union[Tensor, ndarray] = None,
                 shift_by_atoms: bool = True,
                 ):

        super().__init__()

        scale = ms.Tensor(scale, ms.float32)
        self._scale = Parameter(scale, name='scale', requires_grad=False)
        shift = ms.Tensor(shift, ms.float32)
        self._shift = Parameter(shift, name='shift', requires_grad=False)

        if type_ref is None:
            self._type_ref = Parameter(Tensor(0, ms.float32), name='type_ref', requires_grad=False)
        else:
            type_ref = ms.Tensor(type_ref, ms.float32)
            self._type_ref = Parameter(type_ref, name='type_ref', requires_grad=False)

        self.shift_by_atoms = shift_by_atoms

        self.identity = ops.Identity()

    @property
    def scale(self) -> Tensor:
        return self.identity(self._scale)

    @scale.setter
    def scale(self, scale_: Union[float, Tensor, ndarray]):
        self._scale.set_data(ms.Tensor(scale_, ms.float32), True)

    @property
    def shift(self) -> Tensor:
        return self.identity(self._shift)

    @shift.setter
    def shift(self, shift_: Union[float, Tensor, ndarray]):
        self._shift.set_data(ms.Tensor(shift_, ms.float32), True)

    @property
    def type_ref(self) -> Tensor:
        return self.identity(self._type_ref)

    @type_ref.setter
    def type_ref(self, type_ref_: Union[float, Tensor, ndarray]):
        if type_ref_ is None:
            type_ref_ = 0
        self._type_ref.set_data(ms.Tensor(type_ref_, ms.float32), True)

    def set_scaleshift(self,
                       scale: Union[float, Tensor, ndarray],
                       shift: Union[float, Tensor, ndarray],
                       type_ref: Union[Tensor, ndarray] = None):
        self._scale.set_data(ms.Tensor(scale, ms.float32), True)
        self._shift.set_data(ms.Tensor(shift, ms.float32), True)
        if type_ref is not None:
            self._type_ref.set_data(ms.Tensor(type_ref, ms.float32), True)
        return self

    def convert_energy_from(self, unit) -> float:
        """returns a scale factor that converts the energy from a specified unit."""
        return self.units.convert_energy_from(unit)

    def convert_energy_to(self, unit) -> float:
        """returns a scale factor that converts the energy to a specified unit."""
        return self.units.convert_energy_to(unit)

    def print_info(self, num_retraction: int = 0, num_gap: int = 3, char: str = '-'):
        """print the information of readout"""
        ret = char * num_retraction
        gap = char * num_gap
        print(ret+gap+f" Scale: {self.scale.asnumpy()}")
        print(ret+gap+f" Shift: {self.shift.asnumpy()}")
        if self.type_ref.ndim > 1:
            print(ret+gap+" Reference value for atom types:")
            for i, ref in enumerate(self.type_ref):
                out = f'{i}: {ref}'
                print(ret+gap+gap+f' No.{out:<5}')
        else:
            print(ret+gap+f" Reference value for atom types: {self.type_ref.asnumpy()}")
        print(ret+gap+f" Scale the shift by the number of atoms: {self.shift_by_atoms}")
        print('-'*80)
        return self

    def scale_force(self, force: Tensor) -> Tensor:
        return force * self._scale

    def normalize_force(self, force: Tensor) -> Tensor:
        return force / self._scale

    def normalize(self, label: Tensor, atom_type: Tensor, num_atoms: Tensor = None) -> Tensor:
        """Normalize outputs.

        Args:
            label (Tensor):       Tensor with shape (B, ...). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_type (Tensor):    Tensor with shape (B, A). Data type is float.
                                    Default: ``None``.

        Returns:
            outputs (Tensor):       Tensor with shape (B, ...). Data type is float.

        """
        ref = 0
        if self._type_ref.ndim > 0:
            # The shape chanages (B, A, ...) <- (T, ...)
            ref = F.gather(self._type_ref, atom_type, 0)
            # The shape chanages (B, ...) <- (B, A, ...)
            ref = F.reduce_sum(ref, 1)

        label -= ref

        shift = self._shift
        if self.shift_by_atoms:
            if num_atoms is None:
                num_atoms = count_nonzero(F.cast(atom_type > 0, ms.int16), axis=-1, keepdims=True)
            if label.ndim > 2:
                num_atoms = F.reshape(num_atoms, (num_atoms.shape[0],) + (1,) * (label.ndim - 1))
            # shape is (B, ...) = (...) * (B, ...)
            shift *= num_atoms

        return (label - shift) / self._scale

    def construct(self, output: Tensor, atom_type: Tensor, num_atoms: Tensor = None) -> Tensor:
        """Scale and shift output.

        Args:
            outputs (Tensor):       Tensor with shape (B, ...). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_type (Tensor):     Tensor with shape (B, A). Data type is float.
                                    Default: ``None``.

        Returns:
            outputs (Tensor):       Tensor with shape (B, ...). Data type is float.

        """
        ref = 0
        if self._type_ref.ndim > 0:
            ref = F.gather(self._type_ref, atom_type, 0)
            ref = F.reduce_sum(ref, 1)

        output = output * self._scale + ref

        shift = self._shift
        if self.shift_by_atoms:
            if num_atoms is None:
                num_atoms = count_nonzero(F.cast(atom_type > 0, ms.int16), axis=-1, keepdims=True)
            if output.ndim > 2:
                num_atoms = F.reshape(num_atoms, (num_atoms.shape[0],) + (1,) * (output.ndim - 1))
            # (B, ...) = (...) * (B, ...)
            shift *= num_atoms

        return output + shift
