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
r"""Generate inverse problem dataset of the wave equation."""
from typing import List, Optional, Callable
import numpy as np
from numpy.typing import NDArray
import common
from custom_wave import WaveEquation


class InverseWaveEquation(WaveEquation, common.PDEInverseDataGenBase):
    r"""
    Generating inverse problem dataset for the wave equation.
    The PDE takes the form $u_{tt}+\mu u_t+Lu+bu_x+f(u)+s_T(t)s_X(x)=0$,
        $(t,x)\in[0,1]\times[-1,1]$.
    Here, the wave term is randomly selected from the non-divergence form
    $Lu=-c(x)^2u_{xx}$, the factored form $Lu=-c(x)(c(x)u_x)_x$, and the
    divergence form $Lu=-(c(x)^2u_x)_x$ with equal probability, where
    $c(x)^2=\kappa(x)$.
    We take $f(u) = \sum_{k=1}^3c_{0k}u^k
                   + \sum_{j=1}^{J}c_{j0}h_{j}(c_{j1}u+c_{j2}u^2)$.
    Unless periodic boundary is specified, the boundary condition is taken
    to be homogeneous, with Mur type on the left and Neumann type on the right.
    For each PDE, solution samples with different initial conditions and
    spatial-temporal source terms are generated.
    """
    info_dict = {"version": 4.2, "preprocess_dag": False, "pde_type_id": 5}
    field_pde_all: List[NDArray[float]]

    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        self.field_pde_all = []
        self.wave_term.reset(rng, coef_type=self.wave_term.FIELD_COEF)
        if not self.periodic:
            # For the case bc_val_type==ZERO_VAL, the initial values of u, u_x
            # and u_t at the endpoints will not be used (they may also differ
            # across different data samples), and hence we take zeros as the
            # corresponding input to reset the boundary conditions.
            (c_l, c_r) = self.wave_term.boundary_values()
            self.bc_l.reset(rng, c_l, 0, 0, 0, bc_type=self.bc_l.MUR,
                            bc_val_type=self.bc_l.ZERO_VAL)
            self.bc_r.reset(rng, c_r, 0, 0, 0, bc_type=self.bc_r.NEUMANN,
                            bc_val_type=self.bc_r.ZERO_VAL)

    def reset_sample(self, rng: np.random.Generator) -> None:
        super().reset_sample(rng)
        self.u_ic.reset(rng)
        self.ut_ic.reset(rng)
        self.s_term.reset(rng, coef_type=self.s_term.VARYING_FIELD_COEF)

    def _accept_u_sol(self, print_fn: Optional[Callable] = None) -> bool:
        accepted = super()._accept_u_sol(print_fn)
        if accepted:
            field_sample = np.array([self.ut_ic.field,
                                     self.s_term.temporal_coef,
                                     self.s_term.field])
            self.field_pde_all.append(field_sample)  # [3, n_x_grid]
        return accepted

    def _record_solution(self, u_sol: NDArray[float]) -> None:
        super()._record_solution(u_sol)
        key = "persample_field"
        if key not in self.coef_all_dict:
            self.coef_all_dict[key] = []
        # Shape is [num_sample_per_pde, 3, n_x_grid].
        field_pde_all = np.array(self.field_pde_all)
        self.coef_all_dict[key].append(field_pde_all)


if __name__ == "__main__":
    my_args = InverseWaveEquation.get_cli_args()
    pde_data_gen = InverseWaveEquation(my_args)
    common.gen_data(my_args, pde_data_gen)
