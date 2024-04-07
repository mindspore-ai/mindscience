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
r"""
generate inverse problem dataset of the PDE with polynomial (and optionally
sinusoidal) terms
"""
import argparse
import numpy as np
import common
from custom_sinus import SinusoidalPDE


class InverseSinusPDE(SinusoidalPDE, common.PDEInverseDataGenBase):
    r"""
    Generating inverse problem dataset for the PDE with polynomial (and
    optionally sinusoidal) terms. The PDE takes the form
    $u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x)u_x)_x=0$, $(t,x)\in[0,1]\times[-1,1]$,
    where $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                   + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$.
    For each PDE, solutions using multiple initial conditions are generated.
    """
    info_dict = {"version": 4.23, "preprocess_dag": False, "pde_type_id": 2}

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.const_src = args.const_src
        self.const_kappa = args.const_kappa

    @staticmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        bc_type = "circ" if args.periodic else "robin"
        field_coefs = ""
        if not args.const_src:
            field_coefs += "S"
        if not args.const_kappa:
            field_coefs += "K"
        return (f"sinus{args.num_sinusoid}_{bc_type}_f{field_coefs}"
                f"_c{args.coef_distribution}{args.coef_magnitude:g}"
                f"_k{args.kappa_min:.0e}_{args.kappa_max:g}")

    @classmethod
    def _get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = super()._get_cli_args_parser()
        parser.add_argument("--const_src", action="store_true",
                            help="use constant (spatially independent) source "
                            "terms $s(x)$")
        parser.add_argument("--const_kappa", action="store_true",
                            help="use constant (spatially independent) "
                            "diffusion terms $\\kappa(x)$")
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        if self.const_src:
            self.s_term.reset(rng, field_prob=0)
        else:
            self.s_term.reset(rng, zero_prob=0, scalar_prob=0)
        if self.const_kappa:
            allow_zero_kappa = self.periodic and self.f1_term.is_linear
            self.kappa_term.reset(
                rng, zero_prob=float(allow_zero_kappa), field_prob=0)
        else:
            self.kappa_term.reset(rng, zero_prob=0, scalar_prob=0)
        # note: For the boundary condition, the case bc_val_type==FROM_IC_VAL
        # will no longer work as expected, as the initial condition will be
        # regenerated for each data sample.

    def reset_sample(self, rng: np.random.Generator) -> None:
        super().reset_sample(rng)
        self.u_ic.reset(rng)


if __name__ == "__main__":
    my_args = InverseSinusPDE.get_cli_args()
    pde_data_gen = InverseSinusPDE(my_args)
    common.gen_data(my_args, pde_data_gen)
