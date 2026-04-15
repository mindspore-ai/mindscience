# Copyright 2021 Huawei Technologies Co., Ltd
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
#pylint: disable=W0613
"""
Class for 3D Antenna.
"""
from mindspore import nn
from .utils import ones, tensor, sum_fields, vstack


class Antenna(nn.Cell):
    """
    Antenna class with lumped elements.

    Args:
        grid_helper (GridHelper): Helper class for FDTD modeling.
        background_epsr (float): Relative permittivity of the background. Default: 1.
    """

    def __init__(self, grid_helper, background_epsr=1):
        super().__init__(auto_prefix=False)
        self.grid_helper = grid_helper
        self.cell_numbers = grid_helper.cell_numbers
        self.cell_lengths = grid_helper.cell_lengths
        self.background_epsr = tensor(background_epsr)
        self.background_sige = tensor(0.)
        self.grid = ones(self.cell_numbers)

    def construct(self, *args, **kwargs):
        """Generate material tensors.

        Returns:
            epsr (Tensor, shape=(self.cell_nunbers)): Relative permittivity in the whole domain.
            sige (Tensor, shape=(self.cell_nunbers)): Conductivity in the whole domain.
        """
        return self.generate_object()

    def generate_object(self):
        """Generate Antenna.

        Args:
            *args: Dummy parameters

        Returns:
            epsr (Tensor, shape=(self.cell_nunbers)): Relative permittivity in the whole domain.
            sige (Tensor, shape=(self.cell_nunbers)): Conductivity in the whole domain.
        """
        # generate background material tensors
        epsr = self.background_epsr * self.grid
        sige = self.background_sige * self.grid

        for obj in self.grid_helper.objects_in_cells:
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = obj.indices
            epsr[i_s:i_e, j_s:j_e, k_s:k_e] = obj.epsr
            sige[i_s:i_e, j_s:j_e, k_s:k_e] = obj.sigma

        return epsr, sige

    def modify_object(self, args):
        """
        Generate special objects, such as PEC or lumped elements.

        Args:
            args (tuple): collections of material tensors.

        Returns:
            material_tensors (tuple)
        """
        epsrx, epsry, epsrz, sigex, sigey, sigez = args

        # objects defined at the edges
        for obj in self.grid_helper.objects_on_edges:
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = obj.indices
            if obj.direction == 'x':
                epsrx[..., i_s:i_e, j_s:j_e, k_s:k_e] += obj.epsr
                sigex[..., i_s:i_e, j_s:j_e, k_s:k_e] += obj.sigma

            elif obj.direction == 'y':
                epsry[..., i_s:i_e, j_s:j_e, k_s:k_e] += obj.epsr
                sigey[..., i_s:i_e, j_s:j_e, k_s:k_e] += obj.sigma

            elif obj.direction == 'z':
                epsrz[..., i_s:i_e, j_s:j_e, k_s:k_e] += obj.epsr
                sigez[..., i_s:i_e, j_s:j_e, k_s:k_e] += obj.sigma

            else:
                raise ValueError(f'Cannot match direction {obj.direction}')

        # objects defined on the interfaces
        for obj in self.grid_helper.objects_on_faces:
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = obj.indices
            if obj.direction == 'x':
                epsry[..., i_s:i_s+1, j_s:j_e, k_s:k_e+1] = obj.epsr
                epsrz[..., i_s:i_s+1, j_s:j_e+1, k_s:k_e] = obj.epsr
                sigey[..., i_s:i_s+1, j_s:j_e, k_s:k_e+1] = obj.sigma
                sigez[..., i_s:i_s+1, j_s:j_e+1, k_s:k_e] = obj.sigma

            elif obj.direction == 'y':
                epsrx[..., i_s:i_e, j_s:j_s+1, k_s:k_e+1] = obj.epsr
                epsrz[..., i_s:i_e+1, j_s:j_s+1, k_s:k_e] = obj.epsr
                sigex[..., i_s:i_e, j_s:j_s+1, k_s:k_e+1] = obj.sigma
                sigez[..., i_s:i_e+1, j_s:j_s+1, k_s:k_e] = obj.sigma

            elif obj.direction == 'z':
                epsrx[..., i_s:i_e, j_s:j_e+1, k_s:k_s+1] = obj.epsr
                epsry[..., i_s:i_e+1, j_s:j_e, k_s:k_s+1] = obj.epsr
                sigex[..., i_s:i_e, j_s:j_e+1, k_s:k_s+1] = obj.sigma
                sigey[..., i_s:i_e+1, j_s:j_e, k_s:k_s+1] = obj.sigma

            else:
                raise ValueError(f'Cannot match direction {obj.direction}')

        material_tensors = (epsrx, epsry, epsrz, sigex, sigey, sigez)
        return material_tensors

    def update_sources(self, sources, hidden_states, waveform, dt):
        """
        Set locations of sources.

        Args:
            sources = (jx, jy, jz), where
                jx (Tensor, shape=(ns, 1, nx, ny+1, nz+1)): Jx tensor.
                jy (Tensor, shape=(ns, 1, nx+1, ny, nz+1)): Jy tensor.
                jz (Tensor, shape=(ns, 1, nx+1, ny+1, nz)): Jz tensor.

            hidden_states = (ex, ey, ez), where
                ex (Tensor, shape=(ns, 1, nx, ny+1, nz+1)): Ex tensor.
                ey (Tensor, shape=(ns, 1, nx+1, ny, nz+1)): Ey tensor.
                ez (Tensor, shape=(ns, 1, nx+1, ny+1, nz)): Ez tensor.

            waveform (Tensor, shape=()): waveform at time
            dt (float): time interval

        Returns:
            sources (tuple): Collection of source location tensors.
        """
        jx, jy, jz = sources
        ex, ey, ez = hidden_states

        for obj in self.grid_helper.dynamic_sources_on_edges:
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = obj.indices
            if obj.direction == 'x':
                jx[..., i_s:i_e, j_s:j_e, k_s:k_e] = jx[..., i_s:i_e, j_s:j_e, k_s:k_e]\
                    + dt * obj.coef * ex[..., i_s:i_e, j_s:j_e, k_s:k_e]

            elif obj.direction == 'y':
                jy[..., i_s:i_e, j_s:j_e, k_s:k_e] = jy[..., i_s:i_e, j_s:j_e, k_s:k_e]\
                    + dt * obj.coef * ey[..., i_s:i_e, j_s:j_e, k_s:k_e]

            elif obj.direction == 'z':
                jz[..., i_s:i_e, j_s:j_e, k_s:k_e] = jz[..., i_s:i_e, j_s:j_e, k_s:k_e]\
                    + dt * obj.coef * ez[..., i_s:i_e, j_s:j_e, k_s:k_e]

            else:
                raise ValueError(f'Cannot match direction {obj.direction}')

        for source_id, obj in enumerate(self.grid_helper.sources_on_edges):
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = obj.indices
            if obj.direction == 'x':
                jx[source_id, 0, i_s:i_e, j_s:j_e, k_s:k_e] = obj.j * waveform

            elif obj.direction == 'y':
                jy[source_id, 0, i_s:i_e, j_s:j_e, k_s:k_e] = obj.j * waveform

            elif obj.direction == 'z':
                jz[source_id, 0, i_s:i_e, j_s:j_e, k_s:k_e] = obj.j * waveform

            else:
                raise ValueError(f'Cannot match direction {obj.direction}')

        return (jx, jy, jz)

    def get_outputs_at_each_step(self, hidden_states):
        """Compute output each step.

        Args:
            hidden_states (tuple): Collection of field tensors.

        Returns:
            rx (Tensor, shape=(ns, nr, 2)): Voltages and currents at monitors.
        """
        ex, ey, ez, hx, hy, hz = hidden_states

        voltages = []

        for monitor in self.grid_helper.voltage_monitors:
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = monitor.indices
            if monitor.direction == 'x':
                voltages.append(monitor.coef *
                                sum_fields(ex, i_s, i_e, j_s, j_e, k_s, k_e))

            elif monitor.direction == 'y':
                voltages.append(monitor.coef *
                                sum_fields(ey, i_s, i_e, j_s, j_e, k_s, k_e))

            elif monitor.direction == 'z':
                voltages.append(monitor.coef *
                                sum_fields(ez, i_s, i_e, j_s, j_e, k_s, k_e))

            else:
                raise ValueError(
                    f'Cannot match direction {monitor.direction}')

        voltages = vstack(voltages)

        currents = []
        dx, dy, dz = self.grid_helper.cell_lengths

        for monitor in self.grid_helper.current_monitors:
            (i_s, i_e), (j_s, j_e), (k_s, k_e) = monitor.indices
            if monitor.direction == 'x':
                currents.append(monitor.coef * (
                    dy * sum_fields(hy, i_s, i_s+1, j_s, j_e, k_s-1, k_s) -
                    dy * sum_fields(hy, i_s, i_s+1, j_s, j_e, k_e-1, k_e) +
                    dz * sum_fields(hz, i_s, i_s+1, j_e-1, j_e, k_s, k_e) -
                    dz * sum_fields(hz, i_s, i_s+1, j_s-1, j_s, k_s, k_e)
                ))

            elif monitor.direction == 'y':
                currents.append(monitor.coef * (
                    dx * sum_fields(hx, i_s, i_e, j_s, j_s+1, k_e-1, k_e) -
                    dx * sum_fields(hx, i_s, i_e, j_s, j_s+1, k_s-1, k_s) +
                    dz * sum_fields(hz, i_s-1, i_s, j_s, j_s+1, k_s, k_e) -
                    dz * sum_fields(hz, i_e-1, i_e, j_s, j_s+1, k_s, k_e)
                ))

            elif monitor.direction == 'z':
                currents.append(monitor.coef * (
                    dx * sum_fields(hx, i_s, i_e, j_s-1, j_s, k_s, k_s+1) -
                    dx * sum_fields(hx, i_s, i_e, j_e-1, j_e, k_s, k_s+1) +
                    dy * sum_fields(hy, i_e-1, i_e, j_s, j_e, k_s, k_s+1) -
                    dy * sum_fields(hy, i_s-1, i_s, j_s, j_e, k_s, k_s+1)
                ))

            else:
                raise ValueError(
                    f'Cannot match direction {monitor.direction}')

        currents = vstack(currents)

        return vstack([voltages, currents])
