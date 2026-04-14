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
2D TE-Mode differentiable FDTD.
"""
import mindspore as ms
from mindspore import nn, ops
from .constants import epsilon0, mu0
from .utils import tensor, zeros_like, ones, hstack
from .utils import create_zero_tensor, fcmpt


class FDTDLayer(nn.Cell):
    """
    One-step 2D TE-Mode FDTD.

    Args:
        cell_lengths (tuple): Lengths of Yee cells.
        cpmlx_e (Tensor): Updating coefficients for electric fields in the x-direction CPML.
        cpmlx_m (Tensor): Updating coefficients for magnetic fields in the x-direction CPML.
        cpmly_e (Tensor): Updating coefficients for electric fields in the y-direction CPML.
        cpmly_m (Tensor): Updating coefficients for magnetic fields in the y-direction CPML.
    """

    def __init__(self,
                 cell_lengths,
                 cpmlx_e, cpmlx_m,
                 cpmly_e, cpmly_m,
                 ):
        super(FDTDLayer, self).__init__()

        dx = cell_lengths[0]
        dy = cell_lengths[1]

        self.cpmlx_e = cpmlx_e
        self.cpmlx_m = cpmlx_m
        self.cpmly_e = cpmly_e
        self.cpmly_m = cpmly_m

        # operators
        self.dx_oper = ops.Conv2D(out_channel=1, kernel_size=(2, 1))
        self.dy_oper = ops.Conv2D(out_channel=1, kernel_size=(1, 2))
        self.dx_wghts = tensor([-1., 1.]).reshape((1, 1, 2, 1)) / dx
        self.dy_wghts = tensor([-1., 1.]).reshape((1, 1, 1, 2)) / dy
        self.pad_x = ops.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0)))
        self.pad_y = ops.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1)))

    def construct(self, mz_t, hz, ex, ey, phzx, phzy, pexy, peyx,
                  chzh, chze, cexe, cexh, ceye, ceyh):
        """One-step forward propagation

        Args:
            mz_t (Tensor): Source at time t.
            hz, ex, ey (Tensor): Hz, Ex, Ey fields.
            phzx, phzy, pexy, peyx (Tensor): CPML auxiliary fields.
            chzh, chze (Tensor): Updating coefficients for Hz fields.
            cexe, cexh (Tensor): Updating coefficients for Ex fields.
            ceye, ceyh (Tensor): Updating coefficients for Ey fields.

        Returns:
            hidden_states (tuple)
        """
        # -------------------------------------------------
        # Step 1: Update H's at n+1/2 step
        # -------------------------------------------------
        # compute curl E
        deydx = self.dx_oper(ey, self.dx_wghts) / self.cpmlx_m[2]
        dexdy = self.dy_oper(ex, self.dy_wghts) / self.cpmly_m[2]

        # update auxiliary fields
        phzx = self.cpmlx_m[0] * phzx + self.cpmlx_m[1] * deydx
        phzy = self.cpmly_m[0] * phzy + self.cpmly_m[1] * dexdy

        # update H
        hz = chzh * hz + chze * ((dexdy + phzy) - (deydx + phzx) - mz_t)

        # -------------------------------------------------
        # Step 2: Update E's at n+1 step
        # -------------------------------------------------
        # compute curl H
        dhzdx = self.pad_x(self.dx_oper(hz, self.dx_wghts)) / self.cpmlx_e[2]
        dhzdy = self.pad_y(self.dy_oper(hz, self.dy_wghts)) / self.cpmly_e[2]

        # update auxiliary fields
        peyx = self.cpmlx_e[0] * peyx + self.cpmlx_e[1] * dhzdx
        pexy = self.cpmly_e[0] * pexy + self.cpmly_e[1] * dhzdy

        # update E
        ex = cexe * ex + cexh * (dhzdy + pexy)
        ey = ceye * ey - ceyh * (dhzdx + peyx)

        hidden_states = (hz, ex, ey, phzx, phzy, pexy, peyx)

        return hidden_states


class ADFDTD(nn.Cell):
    """2D TE-Mode Differentiable FDTD Network.

    Args:
        cell_numbers (tuple): Number of Yee cells in (x, y) directions.
        cell_lengths (tuple): Lengths of Yee cells.
        nt (int): Number of time steps.
        dt (float): Time interval.
        ns (int): Number of sources.
        designer (BaseTopologyDesigner): Customized Topology designer.
        cfs_pml (CFSParameters): CFS parameter class.
        init_weights (Tensor): Initial weights.

    Returns:
        outputs (Tensor): Customized outputs.
    """

    def __init__(self,
                 cell_numbers,
                 cell_lengths,
                 nt, dt,
                 ns,
                 designer,
                 cfs_pml,
                 init_weights,
                 ):
        super(ADFDTD, self).__init__()

        self.nx = cell_numbers[0]
        self.ny = cell_numbers[1]
        self.dx = cell_lengths[0]
        self.dy = cell_lengths[1]
        self.nt = nt
        self.ns = ns
        self.dt = tensor(dt)

        self.designer = designer
        self.cfs_pml = cfs_pml
        self.rho = ms.Parameter(init_weights) if init_weights is not None else None

        self.mur = tensor(1.)
        self.sigm = tensor(0.)

        if self.cfs_pml is not None:
            # CFS-PML Coefficients
            cpmlx_e, cpmlx_m = self.cfs_pml.get_update_coefficients(
                self.nx, self.dx, self.dt, self.designer.background_epsr.asnumpy())
            cpmly_e, cpmly_m = self.cfs_pml.get_update_coefficients(
                self.ny, self.dy, self.dt, self.designer.background_epsr.asnumpy())

            cpmlx_e = tensor(cpmlx_e.reshape((3, 1, 1, -1, 1)))
            cpmlx_m = tensor(cpmlx_m.reshape((3, 1, 1, -1, 1)))
            cpmly_e = tensor(cpmly_e.reshape((3, 1, 1, 1, -1)))
            cpmly_m = tensor(cpmly_m.reshape((3, 1, 1, 1, -1)))

        else:
            # PEC boundary
            cpmlx_e = cpmlx_m = tensor([0., 0., 1.]).reshape((3, 1))
            cpmly_e = cpmly_m = tensor([0., 0., 1.]).reshape((3, 1))

        # FDTD layer
        self.fdtd_layer = FDTDLayer(
            cell_lengths, cpmlx_e, cpmlx_m, cpmly_e, cpmly_m)

        # auxiliary variables
        self.dte = tensor(dt / epsilon0)
        self.dtm = tensor(dt / mu0)

        # material parameters smoother
        self.smooth_kx = 0.5 * ones((1, 1, 2, 1))
        self.smooth_ky = 0.5 * ones((1, 1, 1, 2))
        self.smooth_x = ops.Conv2D(
            out_channel=1, kernel_size=(2, 1), pad_mode='pad', pad=(1, 1, 0, 0))
        self.smooth_y = ops.Conv2D(
            out_channel=1, kernel_size=(1, 2), pad_mode='pad', pad=(0, 0, 1, 1))

    def construct(self, waveform_t):
        """
        ADFDTD-based forward propagation.

        Args:
            waveform_t (Tensor, shape=(nt,)): Time-domain waveforms.

        Returns:
            outputs (Tensor): Customized outputs.
        """
        # ----------------------------------------
        # Initialization
        # ----------------------------------------
        # constants
        ns, nt, nx, ny = self.ns, self.nt, self.nx, self.ny
        dt = self.dt

        # material grid
        epsr, sige = self.designer.generate_object(self.rho)

        # delectric smoothing
        epsrx = self.smooth_y(epsr[None, None], self.smooth_ky)
        sigex = self.smooth_y(sige[None, None], self.smooth_ky)
        epsry = self.smooth_x(epsr[None, None], self.smooth_kx)
        sigey = self.smooth_x(sige[None, None], self.smooth_kx)

        # set materials on the interfaces
        (epsrx, epsry, sigex, sigey) = self.designer.modify_object(
            epsrx, epsry, sigex, sigey)

        # non-magnetic & magnetically lossless material
        murz = self.mur
        sigmz = self.sigm

        # updating coefficients
        cexe, cexh = fcmpt(self.dte, epsrx, sigex)
        ceye, ceyh = fcmpt(self.dte, epsry, sigey)
        chzh, chze = fcmpt(self.dtm, murz, sigmz)

        # hidden states
        hz = create_zero_tensor((ns, 1, nx, ny))
        ex = create_zero_tensor((ns, 1, nx, ny + 1))
        ey = create_zero_tensor((ns, 1, nx + 1, ny))

        # CFS-PML auxiliary fields
        phzx = zeros_like(hz)
        phzy = zeros_like(hz)
        pexy = zeros_like(ex)
        peyx = zeros_like(ey)

        # set source location
        mz_t = zeros_like(hz)

        # ----------------------------------------
        # Update
        # ----------------------------------------
        outputs = []

        t = 0

        while t < nt:

            mz_t = self.designer.update_sources(
                (mz_t,), (hz,), waveform_t[t], dt)

            # RNN-Style Update
            (hz, ex, ey, phzx, phzy, pexy, peyx) = self.fdtd_layer(
                mz_t, hz, ex, ey, phzx, phzy, pexy, peyx,
                chzh, chze, cexe, cexh, ceye, ceyh)

            # Compute outputs
            outputs.append(self.get_outputs_at_each_step((hz, ex, ey)))

            t = t + 1

        outputs = hstack(outputs)
        return outputs
