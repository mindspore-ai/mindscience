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
Common operations
"""

from inspect import signature
import numpy as np
import mindspore as ms
from mindspore import numpy as msnp
from mindspore.ops import functional as F
from mindspore import ops, nn
from mindspore import Tensor
from mindspore.nn import Cell

from . import functions as func
from .functions import get_integer
from .units import Units, GLOBAL_UNITS

__all__ = [
    'GetVector',
    'GetDistance',
    'VelocityGenerator',
    'GetDistanceShift',
    'GetShiftGrad',
]


class GetVector(Cell):
    r"""The class to get vector with or without PBC box

    Args:
        use_pbc (bool): Whether to calculate vector under periodic boundary condition.
                        If ``None`` is given, it will determine whether to use periodic boundary
                        conditions based on whether the ``pbc_box`` is provided.
                        Default: ``None`` .

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from sponge.function import GetVector
        >>> from mindspore import Tensor
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> pbc_box = Tensor([[0.5, 0.5, 0.5]], ms.float32)
        >>> gd = GetVector(use_pbc=True)
        >>> gd(crd[0], crd[1], pbc_box)
        Tensor(shape=[1, 3], dtype=Float32, value=
        [[ 5.14909625e-02, -4.62748706e-02, -1.20763242e-01]])
        >>> gd = GetVector(use_pbc=False)
        >>> gd(crd[0], crd[1])
        Tensor(shape=[3], dtype=Float32, value= [ 5.14909625e-02,  4.53725129e-01, -1.20763242e-01])
    """

    def __init__(self, use_pbc: bool = None):
        super().__init__()

        self._use_pbc = use_pbc
        if use_pbc is None:
            self.calc_vector = self.calc_vector_default
        else:
            if use_pbc:
                self.calc_vector = self.calc_vector_pbc
            else:
                self.calc_vector = self.calc_vector_nopbc

    @property
    def use_pbc(self) -> bool:
        """whether to use periodic boundary condition

        Returns:
            bool, whether to use periodic boundary condition

        """
        return self._use_pbc

    @use_pbc.setter
    def use_pbc(self, use_pbc_: bool):
        """set whether to use periodic boundary condition"""
        self.set_pbc(use_pbc_)

    def set_pbc(self, use_pbc: bool):
        """
        set whether to use periodic boundary condition.

        Args:
            use_pbc (bool): Whether to calculate vector under periodic boundary condition.
                Default: ``None``.

        """
        self._use_pbc = use_pbc
        if use_pbc is None:
            self.calc_vector = self.calc_vector_default
        else:
            if not isinstance(use_pbc, bool):
                raise TypeError(f'The type of use_pbc must be bool or None but got: {type(use_pbc)}')
            if use_pbc:
                self.calc_vector = self.calc_vector_pbc
            else:
                self.calc_vector = self.calc_vector_nopbc
        return self

    def calc_vector_default(self, initial: Tensor, terminal: Tensor, pbc_box: Tensor = None) -> Tensor:
        """
        get vector.

        Args:
            initial (Tensor):   Tensor of shape :math:`(B, ..., D)` .
                                B means batchsize, i.e. number of walkers in simulation.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Data type is float.
                                Coordinate of initial point
            terminal (Tensor):  Tensor of shape :math:`(B, ..., D)` . Data type is float.
                                Coordinate of terminal point
            pbc_box (Tensor):   Tensor of shape :math:`(B, D)` . Data type is float.
                                Default: ``None``.

        """
        return func.calc_vector(initial, terminal, pbc_box)

    def calc_vector_pbc(self, initial: Tensor, terminal: Tensor, pbc_box: Tensor = None) -> Tensor:
        """
        get vector with perodic bundary condition.

        Args:
            initial (Tensor):   Tensor of shape :math:`(B, ..., D)` .
                                B means batchsize, i.e. number of walkers in simulation.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Data type is float.
                                Coordinate of initial point
            terminal (Tensor):  Tensor of shape :math:`(B, ..., D)` . Data type is float.
                                Coordinate of terminal point
            pbc_box (Tensor):   Tensor of shape :math:`(B, D)` . Data type is float.
                                Default: ``None``.

        """
        return func.calc_vector_pbc(initial, terminal, pbc_box)

    def calc_vector_nopbc(self, initial: Tensor, terminal: Tensor, pbc_box: Tensor = None) -> Tensor:
        """
        get vector without perodic bundary condition.

        Args:
            initial (Tensor):   Tensor of shape :math:`(B, ..., D)` .
                                B means batchsize, i.e. number of walkers in simulation.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Data type is float.
                                Coordinate of initial point
            terminal (Tensor):  Tensor of shape :math:`(B, ..., D)` . Data type is float.
                                Coordinate of terminal point
            pbc_box (Tensor):   Tensor of shape :math:`(B, D)` . Data type is float.
                                Default: ``None``.

        """
        #pylint: disable=unused-argument
        return terminal - initial

    def construct(self, initial: Tensor, terminal: Tensor, pbc_box: Tensor = None):
        r"""Compute vector from initial point to terminal point.

        Args:
            initial (Tensor):   Tensor of shape :math:`(B, ..., D)` .
                                B means batchsize, i.e. number of walkers in simulation.
                                D means spatial dimension of the simulation system. Usually is 3.
                                Data type is float.
                                Coordinate of initial point
            terminal (Tensor):  Tensor of shape :math:`(B, ..., D)` . Data type is float.
                                Coordinate of terminal point
            pbc_box (Tensor):   Tensor of shape :math:`(B, D)` . Data type is float.
                                Default: ``None``.

        Returns:
            vector (Tensor):    Tensor of shape :math:`(B, ..., D)` . Data type is float.

        """
        return self.calc_vector(initial, terminal, pbc_box)


class GetDistance(GetVector):
    r"""The class to calculate distance with or without PBC box

    Args:
        use_pbc (bool):     Whether to calculate distance under periodic boundary condition.
                            If this is "None", it will determine whether to calculate the distance under
                            periodic boundary condition based on whether the pbc_box is given.
                            Default: ``None``.

        keepdims (bool):    Whether to keep the last dimension of the output Tensor of distance after norm.
                            If this is "True", the last dimension of the output Tensor will be 1.
                            Default: ``False``.

        axis (int):         The axis of the space dimension of the coordinate. Default: ``-1`` .

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from sponge.function import GetDistance
        >>> from mindspore import Tensor
        >>> crd = Tensor(np.random.random((4, 3)), ms.float32)
        >>> pbc_box = Tensor([[0.5, 0.5, 0.5]], ms.float32)
        >>> gd = GetDistance(use_pbc=True, keepdims=False)
        >>> gd(crd[0], crd[1], pbc_box)
        Tensor(shape=[1], dtype=Float32, value= [ 1.39199302e-01])
        >>> gd = GetDistance(use_pbc=False, keepdims=False)
        >>> gd(crd[0], crd[1])
        Tensor(shape=[], dtype=Float32, value= 0.472336)
    """

    def __init__(self,
                 use_pbc: bool = None,
                 keepdims: bool = False,
                 axis: int = -1,
                 ):

        super().__init__(use_pbc=use_pbc)

        self.axis = get_integer(axis)
        self.keepdims = keepdims

        self.norm = None
        # MindSpore < 2.0.0-rc1
        if 'ord' not in signature(ops.norm).parameters.keys():
            self.norm = nn.Norm(self.axis, self.keepdims)

    def construct(self, initial: Tensor, terminal: Tensor, pbc_box: Tensor = None):
        r"""Compute the distance from initial point to terminal point.

        Args:
            initial (Tensor):   Tensor of shape (B, ..., D). Data type is float.
                                Coordinate of initial point
                                B means batchsize, i.e. number of walkers in simulation.
                                D means spatial dimension of the simulation system. Usually is 3.
            terminal (Tensor):  Tensor of shape (B, ..., D). Data type is float.
                                Coordinate of terminal point
            pbc_box (Tensor):   Tensor of shape (B, D). Data type is float.
                                Default: ``None``.

        Returns:
            distance (Tensor):  Tensor of shape (B, ...). Data type is float.

        """
        if self._use_pbc:
            vector = self.calc_vector(initial, terminal, pbc_box)
        else:
            vector = self.calc_vector(initial, terminal)

        if self.norm is None:
            return ops.norm(vector, None, self.axis, self.keepdims)

        return self.norm(vector)


class VelocityGenerator(Cell):
    r"""A class to generate velocities for atoms in system according to temperature

    Args:
        temperature (float):        Temperature

        remove_translation (bool):  Whether to calculate distance under periodic boundary condition.
                                    Default: ``True``

        seed (int):                 Random seed for standard normal. Default: ``0``

        seed2 (int):                Random seed2 for standard normal. Default: ``0``

        length_unit (str):          Length unit. Default: ``None``

        energy_unit (str):          energy unit. Default: ``None``
    Examples:
        >>> from sponge import UpdaterMD
        >>> from sponge.function import VelocityGenerator
        >>> vgen = VelocityGenerator(300)
        >>> velocity = vgen(system.shape, system.atom_mass)
        >>> opt = UpdaterMD(system=system,
        ...                 time_step=1e-3,
        ...                 velocity=velocity,
        ...                 integrator='velocity_verlet',
        ...                 temperature=300,
        ...                 thermostat='langevin')

    """
    #pylint: disable=invalid-name

    def __init__(self,
                 temperature: float = 300,
                 remove_translation: bool = True,
                 seed: int = 0,
                 seed2: int = 0,
                 length_unit: str = None,
                 energy_unit: str = None,
                 ):

        super().__init__()

        if length_unit is None and energy_unit is None:
            self.units = GLOBAL_UNITS
        else:
            self.units = Units(length_unit, energy_unit)

        self.temperature = Tensor(temperature, ms.float32).reshape(-1, 1, 1)

        self.standard_normal = ops.StandardNormal(seed, seed2)

        self.kb = Tensor(self.units.boltzmann, ms.float32)
        self.kbT = self.kb * self.temperature
        self.sigma = F.sqrt(self.kbT)

        self.kinectics_unit_scale = self.units.kinetic_ref
        self.remove_translation = remove_translation
        self.identity = ops.Identity()

        self.multi_temp = False

    def set_temperature(self, temperature: float):
        """
        set temperature.

        Args:
            temperature (float): temperature
        """
        self.temperature = Tensor(temperature, ms.float32).reshape(-1, 1, 1)
        self.multi_temp = False
        if self.temperature is not None and self.temperature.size > 1:
            self.multi_temp = False
        return self

    def construct(self, shape: tuple, atom_mass: Tensor, mask: Tensor = None):
        r"""Randomly generate velocities for atoms in system.

        Args:
            shape (tuple):      Shape of velocity
            atom_mass (Tensor): Tensor of shape (B, A). Data type is float.
                                Atom mass in system.
                                B means batchsize, i.e. number of walkers in simulation.
                                A means number of atoms
            mask (Tensor):      Tensor of shape (B, A). Data type is bool.
                                Mask for atoms. Default: ``None``.

        Returns:
            velocity (Tensor):  Tensor of shape (B, A, D). Data type is float.
                                D means spatial dimension of the simulation system. Usually is 3.

        """
        # (B,A,1)
        atom_mass = F.expand_dims(self.identity(atom_mass), -1)
        inv_mass = msnp.reciprocal(atom_mass)
        velocity_scale = self.sigma * \
            msnp.sqrt(inv_mass / self.kinectics_unit_scale)
        if mask is not None:
            velocity_scale = msnp.where(
                F.expand_dims(mask, -1), velocity_scale, 0)

        velocity = self.standard_normal(shape) * velocity_scale

        if self.remove_translation:
            # (B,A,D) * (1,A,1)
            momentum = atom_mass * velocity
            # (1,1,1) or (B,1,1) <- (1,A,1) or (B,A,1)

            dp = func.keepdims_mean(momentum, -2)
            if mask is not None:
                sp = func.keepdims_sum(momentum, -2)
                n = func.keepdims_sum(F.cast(mask, ms.int32), -2)
                dp = sp / n
            # (B,A,D) - (B,1,D) = (B,A,D)
            momentum -= dp
            velocity = momentum * inv_mass

        return velocity


class GetDistanceShift(Cell):
    r"""Module for calculating B matrix whose dimensions are: C.

    Args:
        bonds (Tensor):     Tensor of shape (C, 2). Data type is int.
                            Bonds need to be constraint.

        num_atoms (int):    Number of atoms in system.

        num_walkers (int):  Number of multiple walkers.

        use_pbc (bool):     Whether to use periodic boundary condition.

    """

    def __init__(self,
                 bonds: Tensor,
                 num_atoms: int,
                 num_walkers: int = 1,
                 use_pbc: bool = None
                 ):

        super().__init__(auto_prefix=False)

        # (C,2)
        self.bonds = bonds

        # (B,C,A)
        shape = (num_walkers, bonds.shape[-2], num_atoms)

        # (1,C,1)
        bond0 = self.bonds[..., 0].reshape(1, -1, 1).asnumpy()
        # (B,C,A) <- (B,A,1)
        mask0 = np.zeros(shape)
        np.put_along_axis(mask0, bond0, 1, axis=-1)
        # (B,C,A,1)
        self.mask0 = F.expand_dims(Tensor(mask0, ms.int32), -1)

        # (1,C,1)
        bond1 = self.bonds[..., 1].reshape(1, -1, 1).asnumpy()
        # (B,C,A) <- (B,A,1)
        mask1 = np.zeros(shape)
        np.put_along_axis(mask1, bond1, 1, axis=-1)
        # (B,C,A,1)
        self.mask1 = F.expand_dims(Tensor(mask1, ms.int32), -1)

        self.get_distance = GetDistance(use_pbc)

    def construct(self, coordinate_new: Tensor, coordinate_old: Tensor, pbc_box: Tensor = None):
        """Module for calculating B matrix whose dimensions are: C.

        Args:
            coordinate_new (Tensor):    Tensor of shape (B,A,D). Data type is float.
                                        The new coordinates of the system.
            coordinate_old (Tensor):    Tensor of shape (B,A,D). Data type is float.
                                        The old coordinates of the system.
            pbc_box (Tensor):           Tensor of shape (B,D). Data type is float.
                                        Tensor of PBC box
        Returns:
            shift (Tensor): Tensor of shape (B,A,D). Data type is float.

        """
        # (B,C,A,D) = (B,C,A,1) * (B,1,A,D)
        pos_new_0 = F.reduce_sum(self.mask0 * coordinate_new, -2)
        pos_new_1 = F.reduce_sum(self.mask1 * coordinate_new, -2)
        # (B,C,A)
        dis_new = self.get_distance(pos_new_0, pos_new_1, pbc_box)

        # (B,C,A,D) = (B,C,A,1) * (B,1,A,D)
        pos_old_0 = F.reduce_sum(self.mask0 * coordinate_old, -2)
        pos_old_1 = F.reduce_sum(self.mask1 * coordinate_old, -2)
        dis_old = self.get_distance(pos_old_0, pos_old_1, pbc_box)

        # (B,C,A)
        return dis_new - dis_old


class GetShiftGrad(Cell):
    """Module for calculating the differentiation of B matrix whose dimensions are: K*N*D.

    Args:
        num_atoms (int):    Number of atoms in system.

        bonds (Tensor):     Tensor of shape :math:`(C, 2)` . Data type is int.
                            Bonds need to be constraint.

        num_walkers (int):  Number of multiple walkers.

        dimension (int):    Dimension.

        use_pbc (bool):     Whether to use periodic boundary condition.

    """

    def __init__(self,
                 num_atoms: int,
                 bonds: Tensor,
                 num_walkers: int = 1,
                 dimension: int = 3,
                 use_pbc: bool = None
                 ):

        super().__init__(auto_prefix=False)

        # (B,C,A,D)
        shape = (num_walkers, bonds.shape[-2], num_atoms, dimension)
        self.broadcast = ops.BroadcastTo(shape)
        self.net = GetDistanceShift(
            bonds=bonds,
            num_atoms=num_atoms,
            num_walkers=num_walkers,
            use_pbc=use_pbc
        )

        self.grad = ops.GradOperation()
        self.zero_shift = ops.Zeros()((num_walkers, num_atoms - 1, num_atoms, dimension), ms.float32)

    def construct(self, coordinate_new: Tensor, coordinate_old: Tensor, pbc_box: Tensor = None):
        """Module for calculating the differentiation of B matrix whose dimensions are: K*N*D.

        Args:
            coordinate_new (Tensor):    Tensor of shape :math:`(B,A,D)` . Data type is float.
                                        The new coordinates of the system.
            coordinate_old (Tensor):    Tensor of shape :math:`(B,A,D)` . Data type is float.
                                        The old coordinates of the system.
            pbc_box (Tensor):           Tensor of shape :math:`(B,D)` . Data type is float.
                                        Tensor of PBC box
        Returns:
            shift (Tensor): Tensor of shape :math:`(B,A,D)` . Data type is float.

        """
        # (B,C,A,D)
        coordinate_new = self.broadcast(coordinate_new[:, None, :, :])
        coordinate_old = self.broadcast(coordinate_old[:, None, :, :])
        shift_grad = self.grad(self.net)(coordinate_new, coordinate_old, pbc_box)
        if msnp.isnan(shift_grad.sum()):
            shift_grad = self.zero_shift
        return shift_grad
