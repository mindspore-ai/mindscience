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
r"""Generate dataset of the PDE with multiple components."""
import argparse
from typing import Optional, List, Dict, Union, Tuple
import numpy as np
from numpy.typing import NDArray
import dedalus.public as d3

import common

# Parameters
X_L = -1
X_R = 1
N_X_GRID = 256
DEALIAS = 2  # >=1.5 is enough, since polynomial terms has maximum order 2

# Substitutions
d3_xcoord = d3.Coordinate('x')
dist = d3.Distributor(d3_xcoord, dtype=np.float64)
xbasis = d3.RealFourier(d3_xcoord, size=N_X_GRID,
                        bounds=(X_L, X_R), dealias=DEALIAS)
x_coord = dist.local_grid(xbasis)


def d3_dx(operand):
    r""" Obtain the spatial derivative of a Dedalus operator. """
    return d3.Differentiate(operand, d3_xcoord)


class SparseCOOCoef(common.PDERandomCoefBase):
    r"""
    Generate sparse coefficient matrices/tensors in the COOrdinate format.
    """
    coo_len: int
    coo_vals: NDArray[float]
    coo_i: NDArray[int]
    coo_j: NDArray[int]
    coo_k: NDArray[int]

    def __init__(self,
                 n_vars: int = 2,
                 max_len: Optional[int] = None,
                 is_3d: bool = False,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(coef_distribution, coef_magnitude)
        self.n_vars = n_vars
        self.is_3d = is_3d
        if max_len is None:
            max_len = 2 * self.n_vars
        self.max_len = max_len

    def __str__(self) -> str:
        return str(self.todense())

    def reset(self, rng: np.random.Generator) -> None:
        self.coo_len = rng.integers(self.max_len + 1)
        self.coo_vals = self._random_value(rng, size=self.coo_len)
        if self.is_3d:
            # only generate entries with j <= k
            coo_j_all, coo_k_all = np.triu_indices(self.n_vars)
            num_jk_all = coo_j_all.shape[0]
            coo_ijk = rng.choice(self.n_vars * num_jk_all,
                                 self.coo_len, replace=False)
            self.coo_i, coo_jk_idx = divmod(coo_ijk, num_jk_all)
            self.coo_j = coo_j_all[coo_jk_idx]
            self.coo_k = coo_k_all[coo_jk_idx]
        else:
            coo_ij = rng.choice(self.n_vars**2, self.coo_len, replace=False)
            self.coo_i, self.coo_j = divmod(coo_ij, self.n_vars)

    def reset_debug(self) -> None:
        self.coo_len = 0
        self.coo_vals = np.zeros(0)
        self.coo_i = np.zeros(0, dtype=int)
        self.coo_j = np.zeros(0, dtype=int)
        if self.is_3d:
            self.coo_k = np.zeros(0, dtype=int)

    def is_linear_wrt(self, i: int) -> bool:
        r""" check whether this term is linear with respect to $u_i$. """
        return not (self.is_3d and i in self.coo_i)

    def todense(self) -> NDArray[float]:
        r""" Convert the sparse 2D/3D array to dense format. """
        if self.is_3d:
            dense_arr = np.zeros([self.n_vars] * 3)
            for (i, j, k, val) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
                dense_arr[i, j, k] = val
        else:
            dense_arr = np.zeros([self.n_vars] * 2)
            for (i, j, val) in zip(self.coo_i, self.coo_j, self.coo_vals):
                dense_arr[i, j] = val
        return dense_arr

    def gen_dedalus_ops(self, u_list: List) -> List:  # pylint: disable=W0221
        if not self.n_vars == len(u_list):
            raise ValueError("Length of 'u_list' should be equal to 'n_vars'.")
        op_list = [0 for _ in u_list]
        if self.is_3d:
            for (i, j, k, val) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
                op_list[i] = op_list[i] + val * u_list[j] * u_list[k]
        else:
            for (i, j, val) in zip(self.coo_i, self.coo_j, self.coo_vals):
                op_list[i] = op_list[i] + val * u_list[j]
        return op_list

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray]]:
        pad_len = self.max_len - self.coo_len
        data_dict = {prefix + "/coo_len": self.coo_len}

        def add_item(name, data):
            data_dict[prefix + name] = np.pad(data, (0, pad_len))

        add_item("/coo_vals", self.coo_vals)
        add_item("/coo_i", self.coo_i)
        add_item("/coo_j", self.coo_j)
        if self.is_3d:
            add_item("/coo_k", self.coo_k)
        return data_dict


class MultiComponentPDE(common.PDEDataGenBase):
    r"""
    Generate PDE dataset with multiple components. The PDE takes the form
    $\partial_tu_i + \sum_jc_{ij}u_j + s_i
        +\partial_x(\sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_ju_k
            - \kappa_i\partial_xu_i) = 0$,
    where $0 \le i,j,k \le d-1$, $j \le k$, $(t,x)\in[0,1]\times[-1,1]$.
    Periodic boundary condition is employed for simplicity.
    The coefficients $a,b,c$ are sparse arrays, each with at most $2d$
    non-zero entries. Each entry of the source term $s$ is set to zero
    with probability 0.5.
    All non-zero entries in $a,b,c,s$ are independent random numbers.
    Each viscousity coefficient satisfies by default
        $\log\kappa_i\sim U([\log(10^{-3}), \log(1)])$.
    The initial condition is generated in a way similar to the PDEBench
        dataset.
    """
    info_dict = {"version": 4.2, "preprocess_dag": True, "pde_type_id": 3}
    x_coord = x_coord

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.n_vars = args.n_vars
        self.a_term = SparseCOOCoef(
            self.n_vars, coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.b_term = SparseCOOCoef(
            self.n_vars, is_3d=True,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.c_term = SparseCOOCoef(
            self.n_vars, coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.ic_list = [common.InitialCondition(self.x_coord, periodic=True)
                        for _ in range(self.n_vars)]
        self.s_terms = common.RandomValue(
            size=self.n_vars,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.kappa_terms = common.NonNegativeRandomValue(
            size=self.n_vars, min_val=args.kappa_min, max_val=args.kappa_max)

    @property
    def _term_obj_dict(self) -> Dict[str, common.PDETermBase]:
        return {"a": self.a_term, "b": self.b_term, "c": self.c_term,
                "s": self.s_terms, "kappa": self.kappa_terms}

    @staticmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return (f"compn{args.n_vars}"
                f"_c{args.coef_distribution}{args.coef_magnitude:g}"
                f"_k{args.kappa_min:.0e}_{args.kappa_max:g}")

    @classmethod
    def _get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--n_vars", "-v", type=int, default=2,
                            help="number of variables (components) in the PDE")
        SparseCOOCoef.add_cli_args_(parser)
        common.NonNegativeRandomValue.add_cli_args_(parser)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        self.u_sol = None

        # coefficients
        self.a_term.reset(rng)
        self.b_term.reset(rng)
        self.c_term.reset(rng)
        self.s_terms.reset(rng)

        # initial conditions and probability of each kappa being zero
        zero_kappa_prob = np.zeros(self.n_vars)
        for i in range(self.n_vars):
            self.ic_list[i].reset(rng)
            if self.b_term.is_linear_wrt(i):
                zero_kappa_prob[i] = 0.5
        self.kappa_terms.reset(rng, zero_prob=zero_kappa_prob)

    def _get_dedalus_problem(self) -> Tuple:
        # Fields
        u_list = [dist.Field(name=f'u{i}', bases=xbasis)
                  for i in range(self.n_vars)]

        # Problem and Initial conditions
        problem = d3.IVP(u_list)
        a_u_list = self.a_term.gen_dedalus_ops(u_list)
        b_u_list = self.b_term.gen_dedalus_ops(u_list)
        c_u_list = self.c_term.gen_dedalus_ops(u_list)
        s_vals = self.s_terms.gen_dedalus_ops()
        kappa_vals = self.kappa_terms.gen_dedalus_ops()
        for i, ui_op in enumerate(u_list):
            self.ic_list[i].gen_dedalus_ops(ui_op)
            kappa_i = kappa_vals[i]
            lin = d3_dx(a_u_list[i] - kappa_i * d3_dx(ui_op)) + c_u_list[i]
            nonlin = d3_dx(b_u_list[i]) + s_vals[i]
            problem.add_equation([d3.dt(ui_op) + lin, -nonlin])

        return u_list, problem


if __name__ == "__main__":
    my_args = MultiComponentPDE.get_cli_args()
    pde_data_gen = MultiComponentPDE(my_args)
    common.gen_data(my_args, pde_data_gen)
