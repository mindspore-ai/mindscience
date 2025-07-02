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
"""CFD Simulator. It is the top-level class in MindFlow CFD."""
from mindspore import jit, jit_class

from .mesh_info import MeshInfo
from .material import define_material
from .integrator import define_integrator
from .space_solver import SpaceSolver
from .boundary_conditions import BoundaryManager
from .utils import cal_con_var, cal_pri_var


@jit_class
class Simulator:
    r"""
    CFD Simulator. It is the top-level class in MindFlow CFD.

    Args:
        config (dict): The dict of parameters.
        net_dict (dict): The dict of netwoks. Default: ``None``.

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
    """

    def __init__(self, config, net_dict=None):
        self.mesh_info = MeshInfo(config['mesh'])
        self.material = define_material(config['material'])
        self.integrator = define_integrator(config['integrator'])
        self.space_solver = SpaceSolver(config['space_solver'], self.mesh_info, self.material, net_dict)
        self.boundary = BoundaryManager(config['boundary_conditions'], mesh_info=self.mesh_info)

    @jit
    def integration_step(self, con_var, timestep):
        """Do integration in a timestep.

        Args:
            con_var (Tensor): Conservative variables.
            timestep (float): Timestep of integration.

        Returns:
            Tensor. Conservative variables at the next timestep.
        """
        rhs = None
        init_con_var = con_var.copy()
        for stage in range(self.integrator.number_of_stages):
            pri_var = cal_pri_var(con_var, self.material)
            filled_pri_var = self.boundary.fill_boundarys(pri_var)
            filled_con_var = cal_con_var(filled_pri_var, self.material)
            rhs = self.space_solver.compute_rhs(filled_con_var)
            con_var = self.integrator.integrate(con_var, init_con_var, rhs, timestep, stage)
        return con_var
