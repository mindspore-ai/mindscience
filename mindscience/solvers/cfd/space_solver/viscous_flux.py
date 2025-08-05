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
"""viscous flux computer"""
from mindspore import numpy as mnp
from mindspore import jit_class

from .derivative_computer import define_derivative_computer
from .interpolator import define_interpolator
from ..utils import cal_pri_var


@jit_class
class ViscousFlux:
    """Viscous flux computer"""

    def __init__(self, material, mesh_info, config):
        self.material = material
        self.mesh_info = mesh_info
        self.active_axis = mesh_info.active_axis

        self.interpolator = define_interpolator(config['interpolator'])(mesh_info)
        self.face_derivative_computer = define_derivative_computer(config['face_derivative_computer'])(mesh_info)
        self.central_derivative_computer = define_derivative_computer(config['central_derivative_computer'])(mesh_info)
        self.cell_sizes = mesh_info.cell_sizes

    def compute_flux(self, con_var, axis):
        """
        Compute viscous flux in an axis.

        Args:
            con_var : Tensor. Input conservative variables.
            axis : int. Axis to compute flux on.
        Returns:
            flux : Tensor. Viscous flux.
        """
        output_size = [5,] + self.mesh_info.number_of_cells
        output_size[axis + 1] += 1

        # compute temperature
        pri_var = cal_pri_var(con_var, self.material)
        face_pri_var = self._slice(self.interpolator.interpolate(pri_var, axis), output_size)
        face_dynamic_viscosity = self.material.dynamic_viscosity(face_pri_var)
        face_bulk_viscosity = self.material.bulk_viscosity

        # compute velocity and stress
        face_stress = self._compute_stress(axis, output_size, pri_var, face_dynamic_viscosity, face_bulk_viscosity)

        # interpolate velocity at face
        face_velocity = self._slice(self.interpolator.interpolate(pri_var, axis=axis), output_size)[1:4]

        energy_flux = 0.0
        for k in self.active_axis:
            energy_flux += face_stress[k, ...] * face_velocity[k, ...]

        rho_flux = mnp.zeros_like(energy_flux)

        viscous_flux = mnp.stack([rho_flux, face_stress[0, ...], face_stress[1, ...], face_stress[2, ...], energy_flux])

        return viscous_flux

    def _compute_stress(self, axis, output_size, pri_var, dynamic_viscosity, bulk_viscosity):
        """
        Compute viscous stress vector.

        Args:
            axis : int. Axis to compute flux on.
            output_size : List. Output size of viscous flux.
            pri_var : Tensor. Input primitive variables.
            dynamic_viscosity : Tensor. Dynamic viscosity of material.
            bulk_viscosity : Tensor. Bulk viscosity of material.
        Returns:
            Tensor. Viscous flux.
        """
        face_velocity_grad = self._compute_velocity_grad(axis, output_size, pri_var)
        mu_1 = dynamic_viscosity
        mu_2 = bulk_viscosity - 2.0 / 3.0 * dynamic_viscosity

        stress_list = []
        hydrostatic_pressure = 0.0
        for i in range(3):
            if i in self.active_axis:
                stress_list.append(mu_1 * (face_velocity_grad[axis, i, ...] + face_velocity_grad[i, axis, ...]))
                hydrostatic_pressure += mu_2 * face_velocity_grad[i, i, ...]
            else:
                stress_list.append(mnp.zeros_like(stress_list[-1]))
        stress_list[axis] += hydrostatic_pressure

        return mnp.stack(stress_list, axis=0)

    def _compute_velocity_grad(self, axis, output_size, pri_var):
        r"""
        Compute velocity grad tensor.

        We use 2 different ways to compute this tensor `\frac{\partial u}{\partial x_{i}}`. When i is same as axis, the
        derivative is computed directly. When i is different from axis, we compute derivative at mesh center first. Then
        it is interpolated on face.

        Args:
            axis : int. Axis to compute flux on.
            output_size : List. Output size of viscous flux.
            pri_var : Tensor. Input pri_var.

        Returns:
            Tensor. Velocity grad tensor.
        """
        velocity_grad = []
        for i in range(3):
            tmp = 0.0
            if i not in self.active_axis:
                tmp = mnp.zeros_like(velocity_grad[-1])
            elif i == axis:
                tmp = self.face_derivative_computer.derivative(pri_var, dxi=self.cell_sizes[axis], axis=axis)
                tmp = self._slice(tmp, output_size)[1:4, ...]
            else:
                tmp = self.central_derivative_computer.derivative(pri_var, dxi=self.cell_sizes[axis], axis=i)
                tmp = self.interpolator.interpolate(tmp, axis)
                tmp = self._slice(tmp, output_size)[1:4, ...]
            velocity_grad.append(tmp)
        return mnp.stack(velocity_grad, axis=0)

    def _slice(self, inputs, output_size):
        """
        Take slice of the input tensor according to output_size.

        Args:
            inputs: Tensor. Input tensor.
            output_size : List. Output size of viscous flux.
        Returns:
            Tensor. Output tensor with shape of output_size.
        """
        starts = []
        ends = []

        for i in range(3):
            if inputs.shape[i + 1] == output_size[i + 1]:
                starts.append(0)
                ends.append(inputs.shape[i + 1])
            else:
                starts.append(self.mesh_info.pad)
                ends.append(-self.mesh_info.pad)

        return inputs[:, starts[0]: ends[0], starts[1]: ends[1], starts[2]: ends[2]]
