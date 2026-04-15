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
# pylint: disable=W0613
"""
3D Differentiable FDTD
"""
from mindspore import nn, ops

from .constants import epsilon0, mu0
from .utils import create_zero_tensor, fcmpt
from .utils import tensor, zeros_like, ones, hstack


class FDTDLayer(nn.Cell):
    """
    One-step 3D FDTD.

    Args:
        cell_lengths (tuple): Lengths of Yee cells.
        cpmlx_e (Tensor): Updating coefficients for electric fields in the x-direction CPML.
        cpmlx_m (Tensor): Updating coefficients for magnetic fields in the x-direction CPML.
        cpmly_e (Tensor): Updating coefficients for electric fields in the y-direction CPML.
        cpmly_m (Tensor): Updating coefficients for magnetic fields in the y-direction CPML.
        cpmlz_e (Tensor): Updating coefficients for electric fields in the z-direction CPML.
        cpmlz_m (Tensor): Updating coefficients for magnetic fields in the z-direction CPML.
    """

    def __init__(self,
                 cell_lengths,
                 cpmlx_e, cpmlx_m,
                 cpmly_e, cpmly_m,
                 cpmlz_e, cpmlz_m,
                 ):
        super(FDTDLayer, self).__init__()

        dx = cell_lengths[0]
        dy = cell_lengths[1]
        dz = cell_lengths[2]

        self.cpmlx_e = cpmlx_e
        self.cpmlx_m = cpmlx_m
        self.cpmly_e = cpmly_e
        self.cpmly_m = cpmly_m
        self.cpmlz_e = cpmlz_e
        self.cpmlz_m = cpmlz_m

        # operators
        self.dx_wghts = tensor([-1., 1.]).reshape((1, 1, 2, 1, 1)) / dx
        self.dy_wghts = tensor([-1., 1.]).reshape((1, 1, 1, 2, 1)) / dy
        self.dz_wghts = tensor([-1., 1.]).reshape((1, 1, 1, 1, 2)) / dz

        self.dx_oper = ops.Conv3D(out_channel=1, kernel_size=(2, 1, 1))
        self.dy_oper = ops.Conv3D(out_channel=1, kernel_size=(1, 2, 1))
        self.dz_oper = ops.Conv3D(out_channel=1, kernel_size=(1, 1, 2))

        self.pad_x = ops.Pad(paddings=((0, 0), (0, 0), (1, 1), (0, 0), (0, 0)))
        self.pad_y = ops.Pad(paddings=((0, 0), (0, 0), (0, 0), (1, 1), (0, 0)))
        self.pad_z = ops.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0), (1, 1)))

    def construct(self, jx_t, jy_t, jz_t,
                  ex, ey, ez, hx, hy, hz,
                  pexy, pexz, peyx, peyz, pezx, pezy,
                  phxy, phxz, phyx, phyz, phzx, phzy,
                  cexe, cexh, ceye, ceyh, ceze, cezh,
                  chxh, chxe, chyh, chye, chzh, chze):
        """One-step forward propagation

        Args:
            jx_t (Tensor): Source at time t + 0.5 * dt.
            jy_t (Tensor): Source at time t + 0.5 * dt.
            jz_t (Tensor): Source at time t + 0.5 * dt.
            ex, ey, ez, hx, hy, hz (Tensor): E and H fields.
            pexy, pexz, peyx, peyz, pezx, pezy (Tensor): CPML auxiliary fields.
            phxy, phxz, phyx, phyz, phzx, phzy (Tensor): CPML auxiliary fields.
            cexe, cexh, ceye, ceyh, ceze, cezh (Tensor): Updating coefficients.
            chxh, chxe, chyh, chye, chzh, chze (Tensor): Updating coefficients.

        Returns:
            hidden_states (tuple)
        """

        # -------------------------------------------------
        # Step 1: Update H's at n+1/2 step
        # -------------------------------------------------
        # compute curl E
        deydx = self.dx_oper(ey, self.dx_wghts) / self.cpmlx_m[2]
        dezdx = self.dx_oper(ez, self.dx_wghts) / self.cpmlx_m[2]
        dexdy = self.dy_oper(ex, self.dy_wghts) / self.cpmly_m[2]
        dezdy = self.dy_oper(ez, self.dy_wghts) / self.cpmly_m[2]
        dexdz = self.dz_oper(ex, self.dz_wghts) / self.cpmlz_m[2]
        deydz = self.dz_oper(ey, self.dz_wghts) / self.cpmlz_m[2]

        # update auxiliary fields in CFS-PML
        phyx = self.cpmlx_m[0] * phyx + self.cpmlx_m[1] * dezdx
        phzx = self.cpmlx_m[0] * phzx + self.cpmlx_m[1] * deydx
        phxy = self.cpmly_m[0] * phxy + self.cpmly_m[1] * dezdy
        phzy = self.cpmly_m[0] * phzy + self.cpmly_m[1] * dexdy
        phxz = self.cpmlz_m[0] * phxz + self.cpmlz_m[1] * deydz
        phyz = self.cpmlz_m[0] * phyz + self.cpmlz_m[1] * dexdz

        # update H
        hx = chxh * hx + chxe * ((deydz + phxz) - (dezdy + phxy))
        hy = chyh * hy + chye * ((dezdx + phyx) - (dexdz + phyz))
        hz = chzh * hz + chze * ((dexdy + phzy) - (deydx + phzx))

        # -------------------------------------------------
        # Step 2: Update E's at n+1 step
        # -------------------------------------------------
        # compute curl H
        dhydx = self.pad_x(self.dx_oper(hy, self.dx_wghts)) / self.cpmlx_e[2]
        dhzdx = self.pad_x(self.dx_oper(hz, self.dx_wghts)) / self.cpmlx_e[2]
        dhxdy = self.pad_y(self.dy_oper(hx, self.dy_wghts)) / self.cpmly_e[2]
        dhzdy = self.pad_y(self.dy_oper(hz, self.dy_wghts)) / self.cpmly_e[2]
        dhxdz = self.pad_z(self.dz_oper(hx, self.dz_wghts)) / self.cpmlz_e[2]
        dhydz = self.pad_z(self.dz_oper(hy, self.dz_wghts)) / self.cpmlz_e[2]

        # update auxiliary fields in CFS-PML
        peyx = self.cpmlx_e[0] * peyx + self.cpmlx_e[1] * dhzdx
        pezx = self.cpmlx_e[0] * pezx + self.cpmlx_e[1] * dhydx
        pexy = self.cpmly_e[0] * pexy + self.cpmly_e[1] * dhzdy
        pezy = self.cpmly_e[0] * pezy + self.cpmly_e[1] * dhxdy
        pexz = self.cpmlz_e[0] * pexz + self.cpmlz_e[1] * dhydz
        peyz = self.cpmlz_e[0] * peyz + self.cpmlz_e[1] * dhxdz

        # update E
        ex = cexe * ex + cexh * ((dhzdy + pexy) - (dhydz + pexz) - jx_t)
        ey = ceye * ey + ceyh * ((dhxdz + peyz) - (dhzdx + peyx) - jy_t)
        ez = ceze * ez + cezh * ((dhydx + pezx) - (dhxdy + pezy) - jz_t)

        hidden_states = (ex, ey, ez, hx, hy, hz, pexy, pexz, peyx,
                         peyz, pezx, pezy, phxy, phxz, phyx, phyz, phzx, phzy)

        return hidden_states


class ADFDTD:
    """3D Differentiable FDTD Network.

    Args:
        cell_numbers (tuple): Number of Yee cells in (x, y, z) directions.
        cell_lengths (tuple): Lengths of Yee cells.
        nt (int): Number of time steps.
        dt (float): Time interval.
        ns (int): Number of sources.
        designer (BaseTopologyDesigner): Customized Topology designer.
        cfs_pml (CFSParameters): CFS parameter class.
        init_weights (Tensor): Initial weights, Default: ``None``.

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
                 ):

        self.nx = cell_numbers[0]
        self.ny = cell_numbers[1]
        self.nz = cell_numbers[2]
        self.dx = cell_lengths[0]
        self.dy = cell_lengths[1]
        self.dz = cell_lengths[2]
        self.nt = nt
        self.ns = ns
        self.dt = tensor(dt)

        self.designer = designer
        self.cfs_pml = cfs_pml

        self.mur = tensor(1.)
        self.sigm = tensor(0.)

        if self.cfs_pml is not None:
            # CFS-PML Coefficients
            cpmlx_e, cpmlx_m = self.cfs_pml.get_update_coefficients(
                self.nx, self.dx, self.dt, self.designer.background_epsr.asnumpy())
            cpmly_e, cpmly_m = self.cfs_pml.get_update_coefficients(
                self.ny, self.dy, self.dt, self.designer.background_epsr.asnumpy())
            cpmlz_e, cpmlz_m = self.cfs_pml.get_update_coefficients(
                self.nz, self.dz, self.dt, self.designer.background_epsr.asnumpy())

            cpmlx_e = tensor(cpmlx_e.reshape((3, 1, 1, -1, 1, 1)))
            cpmlx_m = tensor(cpmlx_m.reshape((3, 1, 1, -1, 1, 1)))
            cpmly_e = tensor(cpmly_e.reshape((3, 1, 1, 1, -1, 1)))
            cpmly_m = tensor(cpmly_m.reshape((3, 1, 1, 1, -1, 1)))
            cpmlz_e = tensor(cpmlz_e.reshape((3, 1, 1, 1, 1, -1)))
            cpmlz_m = tensor(cpmlz_m.reshape((3, 1, 1, 1, 1, -1)))

        else:
            # PEC boundary
            cpmlx_e = cpmlx_m = tensor([0., 0., 1.]).reshape((3, 1))
            cpmly_e = cpmly_m = tensor([0., 0., 1.]).reshape((3, 1))
            cpmlz_e = cpmlz_m = tensor([0., 0., 1.]).reshape((3, 1))

        # FDTD layer
        self.fdtd_layer = FDTDLayer(
            cell_lengths, cpmlx_e, cpmlx_m, cpmly_e, cpmly_m,
            cpmlz_e, cpmlz_m)

        # auxiliary variables
        self.dte = tensor(dt / epsilon0)
        self.dtm = tensor(dt / mu0)

        # define material parameters smoother
        self.smooth_yz = 0.25 * ones((1, 1, 1, 2, 2))
        self.smooth_xz = 0.25 * ones((1, 1, 2, 1, 2))
        self.smooth_xy = 0.25 * ones((1, 1, 2, 2, 1))

        self.smooth_yz_oper = ops.Conv3D(
            out_channel=1, kernel_size=(1, 2, 2), pad_mode='pad', pad=(0, 0, 1, 1, 1, 1))
        self.smooth_xz_oper = ops.Conv3D(
            out_channel=1, kernel_size=(2, 1, 2), pad_mode='pad', pad=(1, 1, 0, 0, 1, 1))
        self.smooth_xy_oper = ops.Conv3D(
            out_channel=1, kernel_size=(2, 2, 1), pad_mode='pad', pad=(1, 1, 1, 1, 0, 0))

    def __call__(self, waveform_t):
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
        nx, ny, nz, ns, nt = self.nx, self.ny, self.nz, self.ns, self.nt
        dt = self.dt

        epsr, sige = self.designer.generate_object()

        # delectric smoothing
        epsrx = self.smooth_yz_oper(epsr[None, None], self.smooth_yz)
        sigex = self.smooth_yz_oper(sige[None, None], self.smooth_yz)
        epsry = self.smooth_xz_oper(epsr[None, None], self.smooth_xz)
        sigey = self.smooth_xz_oper(sige[None, None], self.smooth_xz)
        epsrz = self.smooth_xy_oper(epsr[None, None], self.smooth_xy)
        sigez = self.smooth_xy_oper(sige[None, None], self.smooth_xy)

        (epsrx, epsry, epsrz, sigex, sigey, sigez) = self.designer.modify_object(
            (epsrx, epsry, epsrz, sigex, sigey, sigez)
        )

        # non-magnetic & magnetically lossless material
        murx = mury = murz = self.mur
        sigmx = sigmy = sigmz = self.sigm

        # updating coefficients
        cexe, cexh = fcmpt(self.dte, epsrx, sigex)
        ceye, ceyh = fcmpt(self.dte, epsry, sigey)
        ceze, cezh = fcmpt(self.dte, epsrz, sigez)
        chxh, chxe = fcmpt(self.dtm, murx, sigmx)
        chyh, chye = fcmpt(self.dtm, mury, sigmy)
        chzh, chze = fcmpt(self.dtm, murz, sigmz)

        # hidden states
        ex = create_zero_tensor((ns, 1, nx, ny + 1, nz + 1))
        ey = create_zero_tensor((ns, 1, nx + 1, ny, nz + 1))
        ez = create_zero_tensor((ns, 1, nx + 1, ny + 1, nz))
        hx = create_zero_tensor((ns, 1, nx + 1, ny, nz))
        hy = create_zero_tensor((ns, 1, nx, ny + 1, nz))
        hz = create_zero_tensor((ns, 1, nx, ny, nz + 1))

        # CFS-PML auxiliary fields
        pexy = zeros_like(ex)
        pexz = zeros_like(ex)
        peyx = zeros_like(ey)
        peyz = zeros_like(ey)
        pezx = zeros_like(ez)
        pezy = zeros_like(ez)
        phxy = zeros_like(hx)
        phxz = zeros_like(hx)
        phyx = zeros_like(hy)
        phyz = zeros_like(hy)
        phzx = zeros_like(hz)
        phzy = zeros_like(hz)

        # set source location
        jx_t = zeros_like(ex)
        jy_t = zeros_like(ey)
        jz_t = zeros_like(ez)

        # ----------------------------------------
        # Update
        # ----------------------------------------
        outputs = []

        for t in range(nt):
            jx_t, jy_t, jz_t = self.designer.update_sources(
                (jx_t, jy_t, jz_t), (ex, ey, ez),
                waveform_t[t], dt)

            # RNN-Style Update
            ex, ey, ez, hx, hy, hz, \
            pexy, pexz, peyx, peyz, pezx, pezy, \
            phxy, phxz, phyx, phyz, phzx, phzy = \
                self.fdtd_layer(jx_t, jy_t, jz_t,
                                ex, ey, ez, hx, hy, hz,
                                pexy, pexz, peyx, peyz, pezx, pezy,
                                phxy, phxz, phyx, phyz, phzx, phzy,
                                cexe, cexh, ceye, ceyh, ceze, cezh,
                                chxh, chxe, chyh, chye, chzh, chze)

            # Compute outputs
            outputs.append(self.designer.get_outputs_at_each_step(
                (ex, ey, ez, hx, hy, hz)))
            print(f"iter: {t} / {nt}")

        outputs = hstack(outputs)
        return outputs
