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
Controller
"""

import mindspore as ms
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore import ops
from mindspore.ops import functional as F

from ..system import Molecule
from ..function import functions as func
from ..function.functions import get_integer


class Controller(Cell):
    r"""
    The controller for control the parameters in the simulation process,
    including integrator, thermostat, barostat, constraint, etc.

    Args:
        system (Molecule):  Simulation system.
        control_step (int): Step interval for controller execution. Default: 1

    Supported Platforms:
        ``Ascend`` ``GPU``
    """
    def __init__(self,
                 system: Molecule,
                 control_step: int = 1,
                 ):

        super().__init__(auto_prefix=False)

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
        self.boltzmann = self.units.boltzmann
        self.kinetic_unit_scale = self.units.kinetic_ref
        self.press_unit_scale = self.units.pressure_ref

        # (B,A)
        self.atom_mass = self.system.atom_mass
        self.inv_mass = self.system.inv_mass
        # (B,A,1)
        self._atom_mass = F.expand_dims(self.atom_mass, -1)
        self._inv_mass = F.expand_dims(self.inv_mass, -1)

        # (B,1)
        self.system_mass = self.system.system_mass
        self.system_natom = self.system.system_natom

        self.control_step = get_integer(control_step)
        if self.control_step <= 0:
            raise ValueError('The "control_step" must be larger than 0!')

        self.num_constraints = 0

        self.identity = ops.Identity()
        self.keepdim_sum = ops.ReduceSum(keep_dims=True)

    def set_time_step(self, dt: float):
        """
        set simulation time step.

        Args:
            dt (float): Time of a time step.
        """
        self.time_step = Tensor(dt, ms.float32)
        return self

    def set_degrees_of_freedom(self, dofs: int):
        """
        set degrees of freedom (DOFs).

        Args:
            dofs (int): degrees of freedom.
        """
        self.degrees_of_freedom = get_integer(dofs)
        return self

    def update_coordinate(self, coordinate: Tensor, success: bool = True) -> bool:
        """
        update the parameter of coordinate.

        Args:
            coordinate (Tensor):    A tensor of parameters of coordinate.
            success (bool):         Whether update the parameters successfully.

        Returns:
            bool.
        """
        success = F.depend(success, F.assign(self._coordinate, coordinate))
        return success

    def update_pbc_box(self, pbc_box: Tensor, success: bool = True) -> bool:
        """
        update the parameter of PBC box.

        Args:
            pbc_box (Tensor):   A tensor of parameters of PBC box.
            success (bool):     Whether update the parameters successfully.

        Returns:
            bool.
        """
        if self._pbc_box is None:
            return success
        return F.depend(success, F.assign(self._pbc_box, pbc_box))

    def get_kinetics(self, velocity: Tensor) -> Tensor:
        """
        calculate kinetics according to velocity.

        Args:
            velocity (Tensor):  A tensor of velocity.

        Returns:
            Tensor, kinetics according to velocity.
        """
        if velocity is None:
            return None
        # (B,A,D) * (B,A,1)
        k = 0.5 * self._atom_mass * velocity**2
        # (B,D) <- (B,A,D)
        kinetics = F.reduce_sum(k, -2)
        return kinetics * self.kinetic_unit_scale

    def get_temperature(self, kinetics: Tensor = None) -> Tensor:
        """
        calculate temperature according to velocity.

        Args:
            kinetics (Tensor):  A tensor of kinetics.

        Returns:
            Tensor, temperature according to velocity.
        """
        if kinetics is None:
            return None
        # (B) <- (B,D)
        kinetics = F.reduce_sum(kinetics, -1)
        return 2 * kinetics / self.degrees_of_freedom / self.boltzmann

    def get_volume(self, pbc_box: Tensor) -> Tensor:
        """
        calculate volume according to PBC box.

        Args:
            pbc_box (Tensor):   A PBC box tensor used to calculate volume.

        Returns:
            Tensor, volume according to PBC box.
        """
        if self._pbc_box is None:
            return None
        # (B,1) <- (B,D)
        return func.keepdim_prod(pbc_box, -1)

    def get_virial(self, pbc_grad, pbc_box):
        """
        calculate virial according to the PBC box and its gradients.

        Args:
            pbc_grad (Tensor):  Tensor of PBC box's gradients.
            pbc_box (Tensor):   Tensor of PBC box

        Returns:
            Tensor, virial.
        """
        # (B,D)
        return 0.5 * pbc_grad * pbc_box

    def get_pressure(self, kinetics: Tensor, virial: Tensor, pbc_box: Tensor) -> Tensor:
        """
        calculate pressure according to kinetics, virial and PBC box.

        Args:
            kinetics (Tensor):  Tensor of kinetics.
            virials (Tensor):   Tensor of virials.
            pbc_box (Tensor):   Tensor of PBC box.

        Returns:
            Tensor, pressure according to kinetics, viral and PBC box.
        """
        if self._pbc_box is None:
            return None
        volume = func.keepdim_prod(pbc_box, -1)
        # (B,D) = ((B,D) - (B, D)) / (B,1)
        pressure = 2 * (kinetics - virial) / volume
        return pressure * self.press_unit_scale

    def get_com(self, coordinate: Tensor) -> Tensor:
        """
        get coordinate of center of mass.

        Args:
            coordinate (Tensor):    Tensor of coordinate.

        Returns:
            Tensor, coordinate of center of mass.
        """
        return self.keepdim_sum(coordinate * self._atom_mass, -2) / F.expand_dims(self.system_mass, -1)

    def get_com_velocity(self, velocity: Tensor) -> Tensor:
        """
        calculate velocity of center of mass.

        Args:
            velocity (Tensor):  Tensor of velocity.

        Returns:
            Tensor, velocity of center of mass.
        """
        # (B,A,D) * (B,A,1) -> (B,1,D)
        # (B,1,D) / (B,1,1)
        return self.keepdim_sum(velocity * self._atom_mass, -2) / F.expand_dims(self.system_mass, -1)

    def construct(self,
                  coordinate: Tensor,
                  velocity: Tensor,
                  force: Tensor,
                  energy: Tensor,
                  kinetics: Tensor,
                  virial: Tensor = None,
                  pbc_box: Tensor = None,
                  step: int = 0,
                  ):

        r"""
        Control the parameters during the simulation.

        Args:
            coordinate (Tensor):    Tensor of shape (B, A, D). Data type is float.
            velocity (Tensor):      Tensor of shape (B, A, D). Data type is float.
            force (Tensor):         Tensor of shape (B, A, D). Data type is float.
            energy (Tensor):        Tensor of shape (B, 1). Data type is float.
            kinetics (Tensor):      Tensor of shape (B, D). Data type is float.
            virial (Tensor):        Tensor of shape (B, D). Data type is float.
            pbc_box (Tensor):       Tensor of shape (B, D). Data type is float.
            step (int):             Simulation step. Default: 0

        Returns:
            - coordinate (Tensor), Tensor of shape (B, A, D). Data type is float.
            - velocity (Tensor), Tensor of shape (B, A, D). Data type is float.
            - force (Tensor), Tensor of shape (B, A, D). Data type is float.
            - energy (Tensor), Tensor of shape (B, 1). Data type is float.
            - kinetics (Tensor), Tensor of shape (B, D). Data type is float.
            - virial (Tensor), Tensor of shape (B, D). Data type is float.
            - pbc_box (Tensor), Tensor of shape (B, D). Data type is float.

        Symbols:
            B:  Number of walkers in simulation.
            A:  Number of atoms.
            D:  Dimension of the simulation system. Usually is 3.
        """
        #pylint: disable=unused-argument

        return coordinate, velocity, force, energy, kinetics, virial, pbc_box
