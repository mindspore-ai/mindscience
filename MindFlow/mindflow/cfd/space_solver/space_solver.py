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
"""space solver of NS equations."""
from mindspore import jit_class

from .godunov_flux import GodunovFlux
from .viscous_flux import ViscousFlux


@jit_class
class SpaceSolver:
    """Space solver of NS equations."""

    def __init__(self, config, mesh_info, material, net_dict=None):
        convective_config = config.get("convective_flux", None)
        if convective_config:
            self.convective_flux_computer = GodunovFlux(material, mesh_info, convective_config, net_dict)
        else:
            self.convective_flux_computer = None

        viscous_config = config.get("viscous_flux", None)
        if viscous_config:
            self.viscous_flux_computer = ViscousFlux(material, mesh_info, viscous_config)
        else:
            self.viscous_flux_computer = None

        self.active_axis = mesh_info.active_axis
        self.cell_sizes = mesh_info.cell_sizes
        self.number_of_cells = mesh_info.number_of_cells

    @staticmethod
    def _get_flux_var(var, axis):
        """get flux variables."""
        flux_1 = None
        flux_2 = None
        if axis == 0:
            flux_1 = var[:, :-1, :, :]
            flux_2 = var[:, 1:, :, :]
        if axis == 1:
            flux_1 = var[:, :, :-1, :]
            flux_2 = var[:, :, 1:, :]
        if axis == 2:
            flux_1 = var[:, :, :, :-1]
            flux_2 = var[:, :, :, 1:]
        return flux_1, flux_2

    def compute_rhs(self, con_var):
        """
        Compute the Right Hand Side(RHS) of NS equation in conservative form.

        Args:
            con_var : Tensor. Conservative variables of NS equations.
        Returns:
            Tensor : Tensor.
        """
        rhs = 0.0

        for axis in self.active_axis:
            axis_flux = 0.0

            if self.convective_flux_computer is not None:
                axis_flux += self.convective_flux_computer.compute_flux(con_var, axis)

            if self.viscous_flux_computer is not None:
                axis_flux -= self.viscous_flux_computer.compute_flux(con_var, axis)

            if self.convective_flux_computer is not None or self.viscous_flux_computer is not None:
                axis_flux_1, axis_flux_2 = self._get_flux_var(axis_flux, axis)
                rhs += (axis_flux_1 - axis_flux_2) / self.cell_sizes[axis]

        return rhs
