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
Controller
"""

from typing import Union, Tuple, List
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.numpy.utils_const import _raise_value_error

from ..system import Molecule
from ..function import functions as func
from ..function.functions import get_integer, get_ms_array, get_arguments


class Controller(Cell):
    r"""
    Base class for the controller module in MindSPONGE.
    The :class:`sponge.control.Controller` is used in
    :class:`sponge.control.Updater` to control the values of
    seven variables during the simulation process, including
    coordinate, velocity, force, energy, kinetics, virial and pbc_box.

    Args:
        system(:class:`sponge.system.Molecule`): Simulation system.
        control_step(int, optional):  Step interval for controller execution. Default: ``1``.
        kwargs(dict): Other parameters for extension.

    Inputs:
        - **coordinate** (Tensor) - Coordinate. Tensor of shape :math:`(B, A, D)`.
          Data type is float.
          Here :math:`B` is the number of walkers in simulation,
          :math:`A` is the number of atoms and
          :math:`D` is the spatial dimension of the simulation system, which is usually 3.
        - **velocity** (Tensor) - Velocity. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **force** (Tensor) - Force. Tensor of shape :math:`(B, A, D)`. Data type is float.
        - **energy** (Tensor) - Energy. Tensor of shape :math:`(B, 1)`. Data type is float.
        - **kinetics** (Tensor) - Kinetics. Tensor of shape :math:`(B, D)`. Data type is float.
        - **virial** (Tensor) - Virial. Tensor of shape :math:`(B, D)`. Data type is float.
        - **pbc_box** (Tensor) - Pressure boundary condition box. Tensor of shape :math:`(B, D)`.
          Data type is float.
        - **step** (int) - Simulation step. Default: ``0``.

    Outputs:
        - coordinate, Tensor of shape :math:`(B, A, D)`. Coordinate. Data type is float.
        - velocity, Tensor of shape :math:`(B, A, D)`. Velocity. Data type is float.
        - force, Tensor of shape :math:`(B, A, D)`. Force. Data type is float.
        - energy, Tensor of shape :math:`(B, 1)`. Energy. Data type is float.
        - kinetics, Tensor of shape :math:`(B, D)`. Kinetics. Data type is float.
        - virial, Tensor of shape :math:`(B, D)`. Virial. Data type is float.
        - pbc_box, Tensor of shape :math:`(B, D)`. Periodic boundary condition box.
          Data type is float.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from sponge import Sponge, Molecule, ForceField, UpdaterMD, WithEnergyCell
        >>> from sponge.control import Controller
        >>> from sponge.callback import RunInfo
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> potential = ForceField(system, parameters='SPCE')
        >>> withenergy = WithEnergyCell(system, potential)
        >>> class MyController(Controller):
        ...     def construct(self,
        ...             coordinate: Tensor,
        ...             velocity: Tensor,
        ...             force: Tensor,
        ...             energy: Tensor,
        ...             kinetics: Tensor,
        ...             virial: Tensor = None,
        ...             pbc_box: Tensor = None,
        ...             step: int = 0,
        ...             **kwargs):
        ...         return super().construct(coordinate, velocity/100,
        ...                                 force, energy, kinetics,
        ...                                 virial, pbc_box, step,
        ...                                 **kwargs)
        >>> velocity = Tensor([[0.1008,0.,0.],[-0.8,0.,0.],[-0.8,0.,0.]])
        >>> updater = UpdaterMD(
        ...     system=system,
        ...     time_step=1e-3,
        ...     velocity=velocity,
        ...     integrator='velocity_verlet',
        ...     temperature=300,
        ...     controller=MyController(system)
        ... )
        >>> mini = Sponge(withenergy, optimizer=updater)
        >>> run_info = RunInfo(1)
        >>> mini.run(5, callbacks=[run_info])
    """
    def __init__(self,
                 system: Molecule,
                 control_step: int = 1,
                 **kwargs,
                 ):

        super().__init__(auto_prefix=False)
        self._kwargs = get_arguments(locals(), kwargs)

        self.system = system
        self.num_walker = self.system.num_walker
        self.num_atoms = system.num_atoms
        self.dimension = system.dimension

        self.sys_dofs = system.degrees_of_freedom
        self.degrees_of_freedom = system.degrees_of_freedom

        self.time_step = Tensor(1e-3, ms.float32)

        self._coordinate = self.system.coordinate
        self._pbc_box = self.system.pbc_box

        self.units = self.system.units
        self.kinetic_unit_scale = Tensor(self.units.kinetic_ref, ms.float32)
        self.press_unit_scale = Tensor(self.units.pressure_ref, ms.float32)

        # (B, A)
        self.atom_mass = self.system.atom_mass
        self.inv_mass = self.system.inv_mass
        # (B, A, 1)
        self._atom_mass = F.expand_dims(self.atom_mass, -1)
        self._inv_mass = F.expand_dims(self.inv_mass, -1)

        # (B, 1)
        self.system_mass = self.system.system_mass
        self.system_natom = self.system.system_natom

        self.control_step = get_integer(control_step)
        if self.control_step <= 0:
            raise ValueError('The "control_step" must be larger than 0!')

        self.num_constraints = 0

        self.identity = ops.Identity()
        self.keepdims_sum = ops.ReduceSum(True)

    @property
    def boltzmann(self) -> float:
        """
        Boltzmann constant in current unit.

        Returns:
            float, Boltzmann constant in current unit.
        """
        return self.units.boltzmann

    def set_time_step(self, dt: float):
        r"""
        Set simulation time step.

        Args:
            dt (float):  Time step.
        """
        self.time_step = get_ms_array(dt, ms.float32)
        return self

    def set_degrees_of_freedom(self, dofs: int):
        """
        Set degrees of freedom (DOFs).

        Args:
            dofs (int):  Degrees of freedom.
        """
        self.degrees_of_freedom = get_integer(dofs)
        return self

    def update_coordinate(self, coordinate: Tensor) -> Tensor:
        r"""
        Update the coordinate of the simulation system.

        Args:
            coordinate (Tensor): Tensor of atomic coordinates. Tensor shape is :math:`(B, A, D)`.
              Data type is float.

        Returns:
            Tensor, has the same data type and shape as original `coordinate`.
        """
        return F.assign(self._coordinate, coordinate)

    def update_pbc_box(self, pbc_box: Tensor) -> Tensor:
        r"""
        Update the parameter of PBC box.

        Args:
            pbc_box (Tensor): Tensor of PBC box. Tensor shape is :math:`(B, D)`.
              Data type is float.

        Returns:
            Tensor, has the same data type and shape as original `pbc_box`.
        """
        if self._pbc_box is None:
            return pbc_box
        return F.assign(self._pbc_box, pbc_box)

    def get_kinetics(self, velocity: Tensor) -> Tensor:
        r"""
        Calculate kinetics according to velocity.

        Args:
            velocity (Tensor):  Tensor of atomic velocities. Tensor shape is :math:`(B, A, D)`.
              Data type is float.

        Returns:
            Tensor, kinetics. Tensor shape is :math:`(B, D)`. Data type is float.
        """
        if velocity is None:
            return None
        # (B, A, D) * (B, A, 1)
        k = 0.5 * self._atom_mass * velocity**2
        # (B, D) <- (B, A, D)
        kinetics = F.reduce_sum(k, -2)
        return kinetics * self.kinetic_unit_scale

    def get_temperature(self, kinetics: Tensor = None) -> Tensor:
        r"""
        Calculate temperature according to velocity.

        Args:
            kinetics (Tensor): Kinetics. Tensor shape is :math:`(B, D)`.
                Data type is float. Default: ``None``.

        Returns:
            Tensor, temperature. Tensor shape is :math:`(B)`. Data type is float.
        """
        if kinetics is None:
            return None
        # (B) <- (B, D)
        kinetics = F.reduce_sum(kinetics, -1)
        return 2 * kinetics / self.degrees_of_freedom / self.boltzmann

    def get_volume(self, pbc_box: Tensor) -> Tensor:
        r"""
        Calculate volume according to PBC box.

        Args:
            pbc_box (Tensor): Tensor of PBC box. Tensor shape is :math:`(B, D)`. Data type is float.

        Returns:
            Tensor, Tensor of volume. Shape is :math:`(B)`. The data type is float.
        """
        if self._pbc_box is None:
            return None
        # (B, 1) <- (B, D)
        return func.keepdims_prod(pbc_box, -1)

    def get_pressure(self, kinetics: Tensor, virial: Tensor, pbc_box: Tensor) -> Tensor:
        r"""
        Calculate pressure according to kinetics, viral and PBC box.

        Args:
            kinetics (Tensor): Kinetics. Tensor shape is :math:`(B, D)`. Data type is float.
            virial (Tensor): Virial. Tensor shape is :math:`(B, D)`. Data type is float.
            pbc_box (Tensor): PBC box. Tensor shape is :math:`(B, D)`. Data type is float.

        Returns:
            Tensor, pressure. Tensor shape is :math:`(B, D)`. Data type is float.
        """
        if self._pbc_box is None:
            return None
        volume = func.keepdims_prod(pbc_box, -1)
        # (B, D) = ((B, D) - (B, D)) / (B, 1)
        pressure = 2 * (kinetics - virial) / volume
        return pressure * self.press_unit_scale

    def get_com(self, coordinate: Tensor, keepdims: bool = True) -> Tensor:
        r"""
        Get coordinate of center of mass.

        Args:
            coordinate (Tensor): Atomic coordinates. Tensor shape is :math:`(B, A, D)`.
              Data type is float.
            keepdims (bool, optional): If this is set to ``True``, the second axis will be left
              in the result as dimensions with size one.
              Default: ``True``.

        Returns:
            Tensor, the coordinate of the center of mass.
              The shape is :math:`(B, A, D)` or :math:`(B, D)`.
              Data type is float.
        """

        # (B, A, D) = (B, A, D) * (B, A, 1)
        weight_coord = coordinate * self._atom_mass
        if keepdims:
            # (B, 1, D) <- (B, A, D)
            tot_coord = self.keepdims_sum(weight_coord, -2)
            # (B, 1, 1) <- (B, 1)
            tot_mass = F.expand_dims(self.system_mass, -1)
        else:
            # (B, D) <- (B, A, D)
            tot_coord = F.reduce_sum(weight_coord, -2)
            # (B, 1)
            tot_mass = self.system_mass

        # (B, 1, D) = (B, 1, D) / (B, 1, 1)
        # OR
        # (B, D) = (B, D) / (B, 1)
        com = tot_coord / tot_mass
        return com

    def get_com_velocity(self, velocity: Tensor, keepdims: bool = True) -> Tensor:
        r"""
        Calculate velocity of center of mass.

        Args:
            velocity (Tensor): Velocity. The shape is :math:`(B, A, D)`. Data type is float.
            keepdims(bool): If this is set to ``True``, the second axis will be left
              in the result as dimensions with size one. Default: ``True``.

        Returns:
            Tensor, Tensor of the velocity of the center of mass.
              The shape is :math:`(B, A, D)` or :math:`(B, D)`. Data type is float.
        """

        # (B, A, D) = (B, A, D) * (B, A, 1)
        weight_vel = velocity * self._atom_mass
        if keepdims:
            # (B, 1, D) <- (B, A, D)
            tot_vel = self.keepdims_sum(weight_vel, -2)
            # (B, 1, 1) <- (B, 1)
            tot_mass = F.expand_dims(self.system_mass, -1)
        else:
            # (B, D) <- (B, A, D)
            tot_vel = F.reduce_sum(weight_vel, -2)
            # (B, 1)
            tot_mass = self.system_mass

        # (B, 1, D) = (B, 1, D) / (B, 1, 1)
        # OR
        # (B, D) = (B, D) / (B, 1)
        com_vel = tot_vel / tot_mass
        return com_vel

    def construct(self,
                  coordinate: Tensor,
                  velocity: Tensor,
                  force: Tensor,
                  energy: Tensor,
                  kinetics: Tensor,
                  virial: Tensor = None,
                  pbc_box: Tensor = None,
                  step: int = 0,
                  ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        r"""
        Control the parameters during the simulation

        Args:
            coordinate (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            velocity (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            force (Tensor): Tensor of shape :math:`(B, A, D)`. Data type is float.
            energy (Tensor): Tensor of shape :math:`(B, 1)`. Data type is float.
            kinetics (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            virial (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            pbc_box (Tensor): Tensor of shape :math:`(B, D)`. Data type is float.
            step (int): Simulation step. Default: ``0``.

        Returns:
            - **coordinate** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **velocity** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **force** (Tensor) - Tensor of shape :math:`(B, A, D)`. Data type is float.
            - **energy** (Tensor) - Tensor of shape :math:`(B, 1)`. Data type is float.
            - **kinetics** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **virial** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.
            - **pbc_box** (Tensor) - Tensor of shape :math:`(B, D)`. Data type is float.

        Note:
            :math:`B` is the number of walkers in simulation.
            :math:`A` is the number of atoms.
            :math:`D` is the spatial dimension of the simulation system. Usually is 3.
        """
        #pylint: disable=unused-argument

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box

    def _get_mw_tensor(self,
                       value: Union[float, ndarray, Tensor, List[float]],
                       name: str) -> Tensor:
        """get tensor for multiple walkers"""
        value = func.get_tensor(value, ms.float32)
        if value.size == 1:
            # ()
            return F.reshape(value, ())

        if value.size != self.num_walker:
            error_info = f'The size of {name} must be equal to 1 or ' \
                         f'the number of multiple walker ({self.num_walker}) but got '
            _raise_value_error(error_info, value.size)
        # (B, 1)
        return F.reshape(value, (self.num_walker, 1))
