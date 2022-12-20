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
"""godunov flux for conservative variables."""
from mindspore import jit_class

from .riemann_computer import define_riemann_computer
from .reconstructor import define_reconstructor


@jit_class
class GodunovFlux:
    """Godunov flux computer for conservative variables."""

    def __init__(self, material, mesh_info, config, net_dict=None):
        self.material = material
        self.reconstructor = define_reconstructor(config['reconstructor'])(mesh_info)
        self.riemann_solver = define_riemann_computer(config['riemann_computer'])(material, net_dict)

    def compute_flux(self, con_var, axis):
        """
        Compute conservative variables flux.

        Args:
            con_var : Tensor. conservative variables.
            axis : int. The dimension to compute flux on.
        Returns:
            Tensor. conservative variables flux.
        """
        con_var_left = self.reconstructor.reconstruct_from_left(con_var, axis)
        con_var_right = self.reconstructor.reconstruct_from_right(con_var, axis)

        flux = self.riemann_solver.compute_riemann_flux(con_var_left, con_var_right, axis)

        return flux
