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
"""Space updater"""

from typing import Union, List
from numpy import ndarray

from mindspore import Tensor
from mindspore.nn.optim.optimizer import opt_init_args_register

from . import Updater
from ..system import Molecule
from ..control.controller import Controller
from ..control.integrator import Integrator, get_integrator
from ..control.thermostat import Thermostat, get_thermostat
from ..control.barostat import Barostat, get_barostat
from ..control.constraint import Constraint, get_constraint
from ..function import get_arguments


class UpdaterMD(Updater):
    r"""
    A updater for molecular dynamics (MD) simulation, which is the subclass of `Updater`.

    UpdaterMD uses four different `Controllers` to control the different
    variables in the simulation process. The `integrator` is used for update
    the atomic coordinates and velocities, the `thermostat` is used for
    temperature coupling, the `barostat` is used for pressure coupling, and
    the `constraint` is used for bond constraint.

    Args:
        system (:class:`sponge.system.Molecule`): Simulation system.
        time_step (float, optional): Time step. Default: 1e-3.
        velocity (Union[Tensor, ndarray, List[float]], optional): Array of atomic velocity.
          The shape of array is :math:`(A, D)` or :math:`(B, A, D)`.
          Here :math:`B` is the number of walkers in simulation,
          :math:`A` is the number of atoms,
          :math:`D` is the spatial dimension of the simulation system, which is usually 3.
          Data type is float. Default: ``None``.
        temperature (float, optional): Reference temperature for coupling.
          Only valid if `thermostat` is set to type `str`. Default: ``None``.
        pressure (float, optional): Reference pressure for temperature coupling.
          Only valid if `barostat` is set to type `str`. Default: ``None``.
        integrator (Union[`sponge.control.Integrator`, str], optional): Integrator
          for MD simulation.
          It can be an object of `Integrator` or the `str` of an integrator name.
          Default: 'leap_frog'
        thermostat (Union[`sponge.control.Thermostat`, str], optional): Thermostat
          for temperature coupling.
          It can be an object of `sponge.control.Thermostat`
          or the `str` of a thermostat name.
          If a `str` is given, then it will only valid if the `temperature` is not ``None``.
          Default: 'berendsen'
        barostat (Union[`sponge.control.Barostat`, str], optional): Barostat
          for pressure coupling.
          It can be an object of `Barostat` or the `str` of a barostat name.
          If a `str` is given, then it will only valid if the `pressure` is not `None`.
          Default: ``'berendsen'``.
        constraint (Union[`sponge.control.Constraint`, List[`sponge.control.Constraint`]], optional):
          Constraint controller(s) for bond constraint. Default: ``None``.
        controller (Union[:class:`sponge.control.Controller`, List[:class:`sponge.control.Controller`]], optional):
          Other controller(s). It will work after the four specific
          controllers (integrator, thermostat, barostat and constraint).
          Default: ``None``.
        weight_decay (float, optional): An value for the weight decay. Default: ``0.0``.
        loss_scale (float, optional): A value for the loss scale. Default: ``1.0``.

    Inputs:
        - **energy** (Tensor) - Total potential energy of the simulation system.
          Tensor of shape :math:`(B, 1)`.
          Here :math:`B` is the batch size, i.e. the number of walkers in simulation.
          Data type is float.
        - **force** (Tensor) - Force on each atoms of the simulation system.
          Tensor of shape :math:`(B, A, D)`.
          Here :math:`A` is the number of atoms,
          :math:`D` is the spatial dimension of the simulation system, which is usually 3.
        - **virial** (Tensor) - Virial tensor of the simulation system.
          Tensor of shape :math:`(B, D, D)`.
          Data type is float. Default: ``None``.

    Outputs:
        - **success** (bool) - whether successfully finish the current optimization step and move to next step.


    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> from sponge import UpdaterMD, Molecule
        >>> from mindspore import Tensor
        >>> system = Molecule(template='water.tip3p.yaml')
        >>> velocity = Tensor([[0.1008,0.,0.],[-0.8,0.,0.],[-0.8,0.,0.]])
        >>> opt = UpdaterMD(system=system,
        ... time_step=1e-3,
        ... velocity=velocity,
        ... integrator='leap_frog',
        ... temperature=None,
        ... thermostat=None)
        >>> # In actual usage, the energy and force are calculated by the
        >>> # potential cell or the neural network model
        >>> print(opt.coordinate.value())
        >>> # [[[ 0. 0. 0. ]
        >>> # [ 0.07907964 0.06120793 0. ]
        >>> # [-0.07907964 0.06120793 0. ]]]
        >>> force = Tensor([[0.,0.,0.],[0.,0.,0.],[0.,0.,0.]])
        >>> opt(energy=Tensor([0.0]), force=force)
        >>> print(opt.coordinate.value())
        >>> # [[[ 0.0001008 0. 0. ]
        >>> # [ 0.07827964 0.06120793 0. ]
        >>> # [-0.07987964 0.06120793 0. ]]]
    """
    @opt_init_args_register
    def __init__(self,
                 system: Molecule,
                 time_step: float = 1e-3,
                 velocity: Union[Tensor, ndarray, List[float]] = None,
                 temperature: float = None,
                 pressure: float = None,
                 integrator: Union[Integrator, str] = 'leap_frog',
                 thermostat: Union[Thermostat, str] = 'berendsen',
                 barostat: Union[Barostat, str] = 'berendsen',
                 constraint: Union[Constraint, List[Constraint], str] = None,
                 controller: Union[Controller, List[Controller]] = None,
                 weight_decay: float = 0.0,
                 loss_scale: float = 1.0,
                 **kwargs,
                 ):

        super().__init__(
            system=system,
            controller=controller,
            time_step=time_step,
            velocity=velocity,
            weight_decay=weight_decay,
            loss_scale=loss_scale,
        )
        self._kwargs = get_arguments(locals(), kwargs)
        self._kwargs.pop('velocity')

        self.integrator: Integrator = get_integrator(integrator, self.system)
        self.integrator.set_time_step(self.time_step)

        self.constraint: Constraint = None
        self.set_constraint(constraint)

        self.thermostat: Thermostat = None
        self.set_thermostat(thermostat, temperature)

        self.barostat: Barostat = None
        self.set_barostat(barostat, pressure)

    @property
    def ref_temp(self):
        r"""
        Reference temperature for thermostat.

        Returns:
            float, reference temperature for thermostat.
        """
        if self.thermostat is None:
            return None
        return self.thermostat.temperature

    @property
    def ref_press(self):
        r"""
        Reference pressure for barostat.

        Returns:
            float, reference pressure for barostat.
        """
        if self.barostat is None:
            return None
        return self.barostat.pressure

    def set_temperature(self, temperature: float):
        r"""
        Set reference temperature for thermostat

        Args:
            temperature (float): Reference temperature for thermostat.
        """
        if self.thermostat is not None:
            return self.thermostat.set_temperature(temperature)
        return None

    def set_pressure(self, pressure: float):
        r"""
        Set reference pressure for barostat

        Args:
            pressure (float): Reference pressure for barostat.
        """
        if self.barostat is not None:
            self.barostat.set_pressure(pressure)
        return self

    def set_thermostat(self, thermostat: Thermostat, temperature: float = None):
        r"""
        Set thermostat

        Args:
            thermostat (Thermostat): Thermostat for temperature coupling.
            temperature (float): Reference temperature for thermostat.
        """
        if temperature is None:
            temperature = self.ref_temp
        thermostat = get_thermostat(thermostat, self.system, temperature)
        if temperature is not None and thermostat is None:
            raise ValueError('The `thermostat` cannot be None when setting the `temperature`')

        if thermostat is None:
            self.thermostat = None
        else:
            self.integrator.set_thermostat(thermostat)
            self.thermostat = self.integrator.thermostat
        return self

    def set_barostat(self, barostat: Barostat, pressure: float = None):
        r"""
        Set barostat

        Args:
            barostat (Barostat): Barostat for pressure coupling.
            pressure (float): Reference pressure for barostat.
        """
        if pressure is None:
            pressure = self.ref_press
        barostat = get_barostat(barostat, self.system, pressure)
        if pressure is not None and barostat is None:
            raise ValueError('The `barostat` cannot be None when setting the `pressure`')
        if barostat is None:
            self.barostat = None
        else:
            if self.pbc_box is None:
                raise ValueError('Barostat cannot be used for the system without periodic boundary condition.')
            self.integrator.set_barostat(barostat)
            self.barostat = self.integrator.barostat
        return self

    def set_constraint(self, constraint: Union[Constraint, List[Constraint]]):
        r"""
        Set constraint.

        Args:
            constraint (Union[`sponge.control.Constraint`, List[`sponge.control.Constraint`]]):
              Constraint controller(s) for bond constraint.
        """
        constraint = get_constraint(constraint, self.system)
        if constraint is None:
            self.constraint = None
        else:
            self.integrator.set_constraint(constraint, self.num_constraints)
            self.constraint = self.integrator.constraint
            self.set_degrees_of_freedom(self.integrator.degrees_of_freedom)
        return self

    def construct(self, energy: Tensor, force: Tensor, virial: Tensor = None):
        r"""
        Update the parameters of system

        Args:
            energy (Tensor): Total potential energy of the simulation system.
              Tensor of shape :math:`(B, 1)`.
              Here :math:`B` is the batch size.
              Data type is float.
            force (Tensor): Force on each atoms of the simulation system.
              Tensor of shape :math:`(B, A, D)`.
              Here :math:`A` is the number of atoms,
              :math:`D` is the spatial dimension of the simulation system,
              which is usually 3.
              Data type is float.
            virial (Tensor): Virial tensor of the simulation system.
              Tensor of shape :math:`(B, D, D)`.
              Data type is float.
              Default: ``None``.

        Returns:
            bool, whether successfully finish the current optimization step and move to next step.
        """

        force, virial = self.decay_and_scale_grad(force, virial)

        coordinate = self.coordinate
        velocity = self.velocity
        kinetics = self.kinetics
        pbc_box = self.pbc_box

        step = self.identity(self.step)
        coordinate, velocity, force, energy, kinetics, virial, pbc_box = \
            self.integrator(coordinate, velocity, force, energy, kinetics, virial, pbc_box, step)

        if self.controller is not None:
            for i in range(self.num_controller):
                coordinate, velocity, force, energy, kinetics, virial, pbc_box = \
                    self.controller[i](coordinate, velocity, force, energy, kinetics, virial, pbc_box, step)

        temperature = self.get_temperature(kinetics)
        pressure = self.get_pressure(kinetics, virial, pbc_box)

        success = True
        success = self.update_coordinate(coordinate, success)
        success = self.update_velocity(velocity, success)
        success = self.update_pbc_box(pbc_box, success)
        success = self.update_kinetics(kinetics, success)
        success = self.update_temperature(temperature, success)
        success = self.update_virial(virial, success)
        success = self.update_pressure(pressure, success)

        return self.next_step(success)
