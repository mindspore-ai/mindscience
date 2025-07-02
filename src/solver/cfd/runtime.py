# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""simulation run time controller."""
from mindspore import numpy as mnp
from mindspore import Tensor, jit_class


@jit_class
class RunTime:
    r"""
    Simulation run time controller.

    Args:
        config (dict): The dict of parameters.
        mesh_info (MeshInfo): The information of the compute mesh.
        material (Material): The fluid material model.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> from mindflow import cfd
        >>> config = {'mesh': {'dim': 1, 'nx': 100, 'gamma': 1.4, 'x_range': [0, 1], 'pad_size': 3},
        ...           'material': {'type': 'IdealGas', 'heat_ratio': 1.4, 'specific_heat_ratio': 1.4,
        ...           'specific_gas_constant': 1.0}, 'runtime': {'CFL': 0.9, 'current_time': 0.0, 'end_time': 0.2},
        ...           'integrator': {'type': 'RungeKutta3'}, 'space_solver': {'is_convective_flux': True,
        ...           'convective_flux': {'reconstructor': 'WENO5', 'riemann_computer': 'Rusanov'},
        ...           'is_viscous_flux': False}, 'boundary_conditions': {'x_min': {'type': 'Neumann'},
        ...           'x_max': {'type': 'Neumann'}}}
        >>> s = cfd.Simulator(config)
        >>> r = cfd.RunTime(c, s.mesh_info, s.material)
    """

    def __init__(self, config, mesh_info, material):
        self.current_time = Tensor(config.get('current_time', 0.0))
        self.end_time = config.get('end_time', 0.0)
        self.fixed_timestep = config.get('fixed_timestep', False)
        if self.fixed_timestep:
            self.timestep = Tensor(config['timestep'])
        else:
            self.timestep = None
        self.cfl = config.get('CFL', 0.0)
        self.mesh_info = mesh_info
        self.material = material
        self.eps = 1e-8

    def compute_timestep(self, pri_var):
        """
        Computes the physical time step size.

        Args:
            pri_var (Tensor): The primitive variables.
        """
        if not self.fixed_timestep:
            tmp = []
            for axis in self.mesh_info.active_axis:
                tmp.append(self.mesh_info.cell_sizes[axis])
            min_cell_size = min(tmp)

            sound_speed = self.material.sound_speed(pri_var)

            abs_velocity = 0.0
            for i in self.mesh_info.active_axis:
                abs_velocity += (mnp.abs(pri_var[i + 1, :, :, :]) + sound_speed)
            dt = min_cell_size / (mnp.max(abs_velocity) + self.eps)
            self.timestep = self.cfl * dt
        print("current time = {:.6f}, time step = {:.6f}".format(self.current_time.asnumpy(), self.timestep.asnumpy()))

    def advance(self):
        """
        Simulation advance according to the timestep.

        Raises:
            NotImplementedError: If `timestep` is invalid.
        """
        if not self.timestep:
            raise NotImplementedError()
        self.current_time += self.timestep

    def time_loop(self, pri_var):
        """
        Weather to continue the simulation. When current time reaches end time or ``NAN`` value detected,
        return False.

        Args:
            pri_var (Tensor): The primitive variables.

        Returns:
            Bool. Weather to continue the simulation.

        Raises:
            ValueError: If `pri_var` has ``NAN`` values.
        """
        if mnp.isnan(pri_var).sum() > 0:
            raise ValueError('Nan value detected!')
        if self.timestep and self.timestep < self.eps:
            return False
        return self.current_time < self.end_time
