#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2023 Huawei Technologies Co., Ltd
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
r"""Generate dataset of the PDE with polynomial (and optionally sinusoidal) terms."""
import argparse
from typing import Optional, Callable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray
import dedalus.public as d3

import common

# Parameters
X_L = -1
X_R = 1
N_X_GRID = 256
DEALIAS = 2  # polynomial term of maximum order 3

# Substitutions
d3_xcoord = d3.Coordinate('x')
dist = d3.Distributor(d3_xcoord, dtype=np.float64)


def d3_dx(operand):
    r""" Obtain the spatial derivative of a Dedalus operator. """
    return d3.Differentiate(operand, d3_xcoord)


class SinusoidalTermFi(common.PDERandomCoefBase):
    r"""
    Generate coefficients for $f_0(u)$ or $f_1(u)$, where
    $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
             + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$.
    """
    coef: NDArray[float]

    def __init__(self,
                 num_sinusoid: int = 0,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(coef_distribution, coef_magnitude)
        self.max_sinusoid = num_sinusoid

    @property
    def is_linear(self) -> bool:
        r"""Check whether the coefficient has the form [[0, *, 0., 0.]]."""
        return (self.coef.shape[0] == 1 and self.coef[0, 2] == 0
                and self.coef[0, 3] == 0)

    @classmethod
    def add_cli_args_(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--num_sinusoid", "-J", type=int, default=0,
                            help="total number of sinusoidal terms")
        super().add_cli_args_(parser)

    def reset(self,  # pylint: disable=W0221
              rng: np.random.Generator,
              num_sinusoid: Optional[int] = None) -> None:
        if num_sinusoid is None:
            num_sinusoid = self.max_sinusoid
        coef = self._random_value(rng, size=(num_sinusoid + 1, 4))

        # polynomial part, coef kept with prob 0.5
        mask = rng.choice(2, size=4).astype(bool)
        coef[0, mask] = 0
        coef[0, 0] = 0  # do not keep the constant term

        # sinusoidal part
        for j in range(1, num_sinusoid + 1):
            op_j_type = rng.choice(3)
            if op_j_type == 1:  # remove term u in sinusoidal terms
                coef[j, 1] = 0
            elif op_j_type == 2:  # remove term u^2 in sinusoidal terms
                coef[j, 2] = 0

        self.coef = coef

    def reset_debug(self) -> None:
        self.coef = np.zeros((1, 4))

    def gen_dedalus_ops(self, u_op) -> Tuple:  # pylint: disable=W0221
        coef = self.coef

        # polynomial part
        lin_op = coef[0, 1] * u_op
        u2_op = u_op**2
        nonlin_op = coef[0, 2] * u2_op + coef[0, 3] * u_op**3

        # sinusoidal part
        for j in range(1, coef.shape[0]):
            if coef[j, 0] == 0:
                continue
            op_j = coef[j, 1] * u_op + coef[j, 2] * u2_op
            if coef[j, 3] > 0:
                gj_op = np.sin(op_j)
            else:
                gj_op = np.cos(op_j)
            nonlin_op = nonlin_op + coef[j, 0] * gj_op

        return lin_op, nonlin_op

    def get_data_dict(self, prefix: str) -> Dict[str, NDArray[float]]:
        # pad empty sinusoidal part
        pad_j = 1 + self.max_sinusoid - self.coef.shape[0]
        # [1 + num_sinusoid, 4] -> [1 + max_sinusoid, 4]
        coef = np.pad(self.coef, ((0, pad_j), (0, 0)))
        return {prefix: coef}


class SinusoidalPDE(common.PDEDataGenBase):
    r"""
    Generate dataset of 1D time-dependent PDE solutions with Dedalus-v3.
    ======== PDE with sinusoidal terms ========
    The PDE takes the form $u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x)u_x)_x=0$,
        $(t,x)\in[0,1]\times[-1,1]$,
    where $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                   + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$.
    Here, the sinusoidal function $h_{ij}\in\{\sin,\cos\}$ is selected
        with equal probability, $J_0+J_1=J$ with $J_0\in\{0,1,\dots,J\}$
        selected with equal probability.
    Unless periodic boundary is specified, the boundary condition at each
    endpoint is randomly selected from Dirichlet, Neumann and Robin types.
    """
    info_dict = {"version": 4.23, "preprocess_dag": True, "pde_type_id": 1}

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.num_sinusoid = args.num_sinusoid
        self.periodic = args.periodic

        # Dedalus Bases
        basis_cls = d3.RealFourier if self.periodic else d3.Chebyshev
        self.xbasis = basis_cls(d3_xcoord, size=N_X_GRID,
                                bounds=(X_L, X_R), dealias=DEALIAS)
        self.x_coord = dist.local_grid(self.xbasis)
        if not np.all(np.diff(self.x_coord) > 0):
            raise ValueError("'x_coord' should be strictly increasing.")

        # PDE terms
        self.u_ic = common.InitialCondition(self.x_coord, self.periodic)
        self.f0_term = SinusoidalTermFi(
            num_sinusoid=args.num_sinusoid,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.f1_term = SinusoidalTermFi(
            num_sinusoid=args.num_sinusoid,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.s_term = common.RandomFieldCoef(
            self.x_coord, self.periodic,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.kappa_term = common.NonNegativeCoefField(
            self.x_coord, self.periodic, min_val=args.kappa_min,
            max_val=args.kappa_max)
        if not self.periodic:
            self.bc_l = common.RandomBoundaryCondition(
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
            self.bc_r = common.RandomBoundaryCondition(
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)

    @property
    def _term_obj_dict(self) -> Dict[str, common.PDETermBase]:
        coef_obj_dict = {"f0": self.f0_term, "f1": self.f1_term,
                         "s": self.s_term, "kappa": self.kappa_term}
        if not self.periodic:
            coef_obj_dict.update({"bc_l": self.bc_l, "bc_r": self.bc_r})
        return coef_obj_dict

    @staticmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        bc_type = "circ" if args.periodic else "robin"
        return (f"sinus{args.num_sinusoid}_{bc_type}"
                f"_c{args.coef_distribution}{args.coef_magnitude:g}"
                f"_k{args.kappa_min:.0e}_{args.kappa_max:g}")

    @classmethod
    def _get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--periodic", action="store_true",
                            help="use periodic (circular) boundary conditions")
        SinusoidalTermFi.add_cli_args_(parser)
        common.NonNegativeCoefField.add_cli_args_(parser)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        self.u_sol = None

        # coefficients of f_0(u), f_1(u)
        f0_num_sinusoid = rng.integers(self.num_sinusoid + 1)
        f1_num_sinusoid = self.num_sinusoid - f0_num_sinusoid
        self.f0_term.reset(rng, f0_num_sinusoid)
        self.f1_term.reset(rng, f1_num_sinusoid)

        # random fields
        self.u_ic.reset(rng)
        self.s_term.reset(rng)
        allow_zero_kappa = self.periodic and self.f1_term.is_linear
        self.kappa_term.reset(rng, zero_prob=float(allow_zero_kappa))

        if not self.periodic:
            (ic_l, ic_r, dx_ic_l, dx_ic_r) = self.u_ic.boundary_values()
            self.bc_l.reset(rng, ic_l, dx_ic_l)
            self.bc_r.reset(rng, ic_r, dx_ic_r)

    def _get_dedalus_problem(self) -> Tuple:
        # Fields
        u_op = dist.Field(name="u", bases=self.xbasis)
        s_field = dist.Field(name="s", bases=self.xbasis)
        kappa_field = dist.Field(name="kappa", bases=self.xbasis)
        u_list = [u_op]

        # Initial condition
        u_op = self.u_ic.gen_dedalus_ops(u_op)

        # PDE Terms
        dx_u = d3_dx(u_op)
        f0_lin, f0_nonlin = self.f0_term.gen_dedalus_ops(u_op)
        f1_lin, f1_nonlin = self.f1_term.gen_dedalus_ops(u_op)
        s_op = self.s_term.gen_dedalus_ops(s_field)
        kappa_op = self.kappa_term.gen_dedalus_ops(kappa_field)
        lin = f0_lin + d3_dx(f1_lin - kappa_op * dx_u)
        nonlin = f0_nonlin + d3_dx(f1_nonlin) + s_op
        if self.periodic and not self.kappa_term.is_const:
            pass
            # In this case, we found the solution has a relatively lower
            # accuracy, especially when kappa contains jump discontinuities.
            # This could be remedied if we treat the diffusion term (i.e.
            # d3_dx(kappa_op * dx_u) ) as a nonlinear term rather than a linear
            # term, but a much smaller time stepsize (hence a much longer
            # solving time, >40x) is required.

        # Periodic case
        if self.periodic:
            problem = d3.IVP([u_op])
            problem.add_equation([d3.dt(u_op) + lin, -nonlin])
            return u_list, problem

        # Non-periodic case
        # Tau polynomials, following
        # https://dedalus-project.readthedocs.io/en/latest/notebooks/dedalus_tutorial_3.html

        tau1 = dist.Field(name='tau1')
        tau2 = dist.Field(name='tau2')
        tau_basis = self.xbasis.derivative_basis(2)
        p1_op = dist.Field(name='p_tau1', bases=tau_basis)
        p2_op = dist.Field(name='p_tau2', bases=tau_basis)
        p1_op['c'][-1] = 1
        p2_op['c'][-2] = 2
        lin_with_tau = lin + tau1 * p1_op + tau2 * p2_op

        problem = d3.IVP([u_op, tau1, tau2])
        problem.add_equation([d3.dt(u_op) + lin_with_tau, -nonlin])

        # boundary conditions
        op_l, bc_val_l = self.bc_l.gen_dedalus_ops(u_op, dx_u)
        problem.add_equation([op_l(x="left"), bc_val_l])
        op_r, bc_val_r = self.bc_r.gen_dedalus_ops(u_op, dx_u)
        problem.add_equation([op_r(x="right"), bc_val_r])

        return u_list, problem

    def _accept_u_sol(self, print_fn: Optional[Callable] = None) -> bool:
        if not super()._accept_u_sol(print_fn):
            return False

        u_sol = self.u_sol
        thres = 0.05 * (u_sol.max() - u_sol.min())

        def local_extremum_ratio(array):
            delta_arr = array[1:] - array[:-1]  # [n, *] -> [n - 1, *]
            sgn_flip1 = np.logical_and(delta_arr[1:] > thres,
                                       delta_arr[:-1] < -thres)
            sgn_flip2 = np.logical_and(delta_arr[:-1] > thres,
                                       delta_arr[1:] < -thres)
            sgn_flip = np.logical_or(sgn_flip1, sgn_flip2)  # [n - 2, *]
            return sgn_flip.mean()

        # detect checkerboard artifact along x-axis
        # [nt, nx, 1] -> [nx, nt] -> float
        x_extremum_ratio = local_extremum_ratio(u_sol[:, :, 0].T)
        if x_extremum_ratio > 0.01:
            if callable(print_fn):
                print_fn(f"rejected: x_extremum_ratio {x_extremum_ratio:.4f}")
            return False

        if self.periodic:
            return True

        # Non-periodic BCs; early time steps (first 10) will be ignored.
        # detect boundary checkerboard artifact along t-axis
        t_extremum_ratio = local_extremum_ratio(u_sol[10:, [1, -2], 0])
        if t_extremum_ratio > 0.1:
            if callable(print_fn):
                print_fn("rejected: boundary t_extremum_ratio "
                         f"{t_extremum_ratio:.4f}")
            return False

        # examine finite-difference BC violation
        u_arr = u_sol[10:, :, 0]  # [nt - 10, nx]
        x_c = self.x_coord  # [nx]
        dx_u_l = (u_arr[:, 1] - u_arr[:, 0]) / (x_c[1] - x_c[0])  # [nt - 10]
        # Abuse use of gen_dedalus_ops.
        op_l, bc_val_l = self.bc_l.gen_dedalus_ops(u_arr[:, 0], dx_u_l)
        max_err_l = np.max(np.abs(op_l - bc_val_l))
        if max_err_l > 1:
            if callable(print_fn):
                print_fn(f"rejected: left BC FD error {max_err_l:.2e}")
            return False
        dx_u_r = (u_arr[:, -1] - u_arr[:, -2]) / (x_c[-1] - x_c[-2])
        op_r, bc_val_r = self.bc_r.gen_dedalus_ops(u_arr[:, -1], dx_u_r)
        max_err_r = np.max(np.abs(op_r - bc_val_r))
        if max_err_r > 1:
            if callable(print_fn):
                print_fn(f"rejected: right BC FD error {max_err_r:.2e}")
            return False

        return True


if __name__ == "__main__":
    my_args = SinusoidalPDE.get_cli_args()
    pde_data_gen = SinusoidalPDE(my_args)
    common.gen_data(my_args, pde_data_gen)
