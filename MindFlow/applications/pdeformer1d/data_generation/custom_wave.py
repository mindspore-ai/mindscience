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
r"""Generate dataset of the wave equation."""
import argparse
from typing import Optional, Callable, Dict, Union, Tuple
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import dedalus.public as d3

import common
from custom_sinus import SinusoidalTermFi

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


class TimeDependentRandomFieldCoef(common.RandomFieldCoef):
    r"""
    Coefficient term involved in a PDE, which can be a zero, a real number
    (scalar) $c$, a spatially-varying random field $c(x)$, a time-dependent
    scalar $c(t)$, or a time-dependent spatially-varying field $c(t,x)$. For
    the last case, we assume the space and time variables are separable, i.e.
    $c(t,x)=c_T(t)c_X(x)$.
    """
    VARYING_SCALAR_COEF = 3
    VARYING_FIELD_COEF = 4
    temporal_coef: NDArray[float]

    # for running updata_dedalus_ops on-the-fly
    current_t: float
    current_t_idx: int

    def __init__(self,
                 x_coord: NDArray[float],
                 periodic: bool = True,
                 t_min: float = 0.,
                 t_max: float = 1.,
                 n_t_grid: int = 256,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(x_coord, periodic, coef_distribution, coef_magnitude)
        self.t_coord = np.linspace(t_min, t_max, n_t_grid)

    def __str__(self) -> str:
        if self.coef_type == self.VARYING_SCALAR_COEF:
            return "varying scalar " + self._arr_to_str(self.temporal_coef)
        if self.coef_type == self.VARYING_FIELD_COEF:
            return ("varying field, spatial " + self._arr_to_str(self.field)
                    + ", temporal " + self._arr_to_str(self.temporal_coef))
        return super().__str__()

    @property
    def is_const(self) -> bool:
        return self.coef_type in [self.ZERO_COEF, self.SCALAR_COEF]

    @property
    def is_field(self) -> bool:
        r""" Whether the current term is spatially-varying. """
        return self.coef_type in [self.FIELD_COEF, self.VARYING_FIELD_COEF]

    @staticmethod
    def _enveloped_wave_signal(t_coord: NDArray[float],
                               rng: np.random.Generator,
                               min_time_ratio: float = 0.1,
                               max_time_ratio: float = 3.) -> NDArray[float]:
        r"""
        An enveloped wave signal in time domain that is commonly used in full
        wave inversion (FWI). The signal takes the form
            $$f(t)=B\phi(B(t-t_0))\cos(\omega_0t+\alpha),$$
        where $\phi(t)=\exp(-t^2/2)/\sqrt(2\pi)$ is the envelop function.
        Denote $T$ to be the time span, the bandwidth $B$ is randomly generated
        according to $\log(1/B)\sim U([\log(r_0T),\log(r_1T)])$, and the
        central modulation frequency $\omega_0=2\pi/T_0$,
        $\log(T_0)\sim U([\log(r_0T),\log(r_1T)])$. The central time $t_0$ is
        uniformly generated from the temporal interval, and modulation phase
        $\alpha\sim U([-\pi,\pi])$.

        Args:
            t_coord (NDArray[float]): Temporal grid points with shape
                `(n_t_grid,)`. The difference of its maximum and minimum values
                is taken to be the time span $T$.
            rng (np.random.Generator): Random number generator instance
                provided by NumPy.
            min_time_ratio (float, optional): The value of $r_0$. Default: 0.1.
            max_time_ratio (float, optional): The value of $r_1$. Default: 3.

        Output:
            signal (NDArray[float]): Resulting signal with shape `(n_t_grid,)`.
        """
        time_span = t_coord.max() - t_coord.min()  # T
        if max_time_ratio <= min_time_ratio or min_time_ratio <= 0:
            raise ValueError(f"'max_time_ratio' should be greater than "
                             f"'min_time_ratio' and both should be positive, but got "
                             f"max_time_ratio={max_time_ratio} and min_time_ratio="
                             f"{min_time_ratio}.")
        if time_span == 0:
            raise ValueError("'t_coord' should have at least two different "
                             "values.")

        log_min = np.log(min_time_ratio * time_span)  # \log(r_0T)
        log_max = np.log(max_time_ratio * time_span)  # \log(r_1T)

        # reshaped envelop function B\phi(B(t-t_0))
        band_width = np.exp(-rng.uniform(log_min, log_max))  # B
        t_0 = rng.uniform(t_coord.min(), t_coord.max())
        coef = band_width / np.sqrt(2 * np.pi)
        envelop = coef * np.exp(-(band_width * (t_coord - t_0))**2 / 2)

        # frequency modulation \cos(\omega_0t+\alpha)
        modulation_period = np.exp(rng.uniform(log_min, log_max))  # T_0
        omega0 = 2 * np.pi / modulation_period
        alpha = rng.uniform(-np.pi, np.pi)
        modulation = np.cos(omega0 * t_coord + alpha)

        signal = envelop * modulation
        return signal

    @staticmethod
    def _chirped_signal(t_coord: NDArray[float],
                        rng: np.random.Generator,
                        min_time_ratio: float = 0.1,
                        max_time_ratio: float = 3.) -> NDArray[float]:
        r"""
        A chirped signal (long signal whose frequency increase with time) that
        is frequently used in long-range imaging with radar. The signal takes
        the form
            $$f(t)=\cos(\omega(t)t+\alpha),$$
        where $\omega(t)$ is a linear function increasing from
        $\omega_{start}=2\pi/T_s$ to $\omega_{end}=2\pi/T_e$. We take
        $[\log(T_e),\log(T_s)]$ to be a random sub-interval of
        $[\log(r_0T),\log(r_1T)]$. The phase is generated as
        $\alpha\sim U([-\pi,\pi])$.

        Args:
            t_coord (NDArray[float]): Temporal grid points with shape
                `(n_t_grid,)`. The difference of its maximum and minimum values
                is taken to be the time span $T$.
            rng (np.random.Generator): Random number generator instance
                provided by NumPy.
            min_time_ratio (float, optional): The value of $r_0$. Default: 0.1.
            max_time_ratio (float, optional): The value of $r_1$. Default: 3.

        Output:
            signal (NDArray[float]): Resulting signal with shape `(n_t_grid,)`.
        """
        time_span = t_coord.max() - t_coord.min()  # T
        if max_time_ratio <= min_time_ratio or min_time_ratio <= 0:
            raise ValueError(f"'max_time_ratio' should be greater than "
                             f"'min_time_ratio' and both should be positive, but got "
                             f"max_time_ratio={max_time_ratio} and min_time_ratio="
                             f"{min_time_ratio}.")
        if time_span == 0:
            raise ValueError("'t_coord' should have at least two different "
                             "values.")

        log_min = np.log(min_time_ratio * time_span)  # \log(r_0T)
        log_max = np.log(max_time_ratio * time_span)  # \log(r_1T)
        # random partition of [0,1], satisfying r_low + r_mid + _ == 1
        (r_low, r_mid, _) = rng.dirichlet([1, 1, 1])
        # \log(T_e) and \log(T_s)
        log_period_e = log_min + r_low * (log_max - log_min)
        log_period_s = log_min + (r_low + r_mid) * (log_max - log_min)
        omega_s = 2 * np.pi / np.exp(log_period_s)
        omega_e = 2 * np.pi / np.exp(log_period_e)
        k_omega = (omega_e - omega_s) / time_span  # slope of \omega(t)
        omega = omega_s + k_omega * (t_coord - t_coord.min())  # \omega(t)

        alpha = rng.uniform(-np.pi, np.pi)
        signal = np.cos(omega * t_coord + alpha)
        return signal

    def reset(self,  # pylint: disable=W0221
              rng: np.random.Generator,
              coef_type: Optional[int] = None) -> None:
        if coef_type is None:
            coef_type = rng.choice(5)
            self.coef_type = coef_type
        else:
            self.coef_type = coef_type

        # spatially-varying part
        if coef_type in [self.ZERO_COEF, self.SCALAR_COEF, self.FIELD_COEF]:
            super().reset(rng, coef_type=coef_type)
        elif coef_type == self.VARYING_SCALAR_COEF:
            self.field = np.ones_like(self.x_coord)
        elif coef_type == self.VARYING_FIELD_COEF:
            super().reset(rng, coef_type=self.FIELD_COEF)
            # change the overwritten value back
            self.coef_type = self.VARYING_FIELD_COEF

        # time-dependent part
        if coef_type in [self.ZERO_COEF, self.SCALAR_COEF, self.FIELD_COEF]:
            self.current_t_idx = -1
            self.current_t = np.inf  # disabling 'updata_dedalus_ops'
            self.temporal_coef = np.ones_like(self.t_coord)
        elif coef_type in [self.VARYING_SCALAR_COEF, self.VARYING_FIELD_COEF]:
            self.current_t_idx = 0
            self.current_t = self.t_coord[self.current_t_idx]

            # temporal_coef
            temporal_coef_type = rng.choice(2, p=[0.8, 0.2])
            if temporal_coef_type == 0:
                self.temporal_coef = self._enveloped_wave_signal(
                    self.t_coord, rng)
            elif temporal_coef_type == 1:
                self.temporal_coef = self._chirped_signal(self.t_coord, rng)

            if coef_type == self.VARYING_SCALAR_COEF:
                value = self._random_value(rng)
                self.temporal_coef *= value

    def reset_debug(self) -> None:
        super().reset_debug()
        self.current_t_idx = -1
        self.current_t = np.inf  # disabling 'updata_dedalus_ops'
        self.temporal_coef = np.ones_like(self.t_coord)

    def gen_dedalus_ops(self, coef_op):  # pylint: disable=W0221
        if self.coef_type == self.VARYING_SCALAR_COEF:
            coef_op["g"] = self.temporal_coef[0]
            return coef_op
        if self.coef_type == self.VARYING_FIELD_COEF:
            coef_op["g"] = self.temporal_coef[0] * self.field
            return coef_op
        return super().gen_dedalus_ops(coef_op)

    def updata_dedalus_ops(self, sim_time: float, coef_op) -> None:
        r"""
        Update the values of the Dedalus operator `coef_op` according to the
        current simulator time-step.
        """
        if sim_time <= self.current_t:
            return
        self.current_t_idx += 1
        current_t_idx = min(self.current_t_idx, self.t_coord.shape[0] - 1)
        self.current_t = self.t_coord[current_t_idx]
        current_temporal_coef = self.temporal_coef[current_t_idx]
        if self.coef_type == self.VARYING_SCALAR_COEF:
            coef_op["g"] = current_temporal_coef
        elif self.coef_type == self.VARYING_FIELD_COEF:
            coef_op.change_scales(1)
            coef_op["g"] = current_temporal_coef * self.field

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {prefix + "/coef_type": self.coef_type,
                prefix + "/field": self.field,
                prefix + "/temporal_coef": self.temporal_coef}

    def prepare_plot(self, title: str = "") -> None:
        coef_type = self.coef_type
        if coef_type in [self.VARYING_SCALAR_COEF, self.VARYING_FIELD_COEF]:
            plt.figure()
            plt.plot(self.t_coord, self.temporal_coef)
            plt.title(title + " temporal_coef")
        if coef_type in [self.FIELD_COEF, self.VARYING_FIELD_COEF]:
            plt.figure()
            plt.plot(self.x_coord, self.field)
            plt.title(title + " field")


class WaveTerm(common.NonNegativeCoefField):
    r"""
    The wave term in the wave equation, whose form is randomly selected from
    the non-divergence form $Lu=-c(x)^2u_{xx}$, the factored form
    $Lu=-c(x)(c(x)u_x)_x$, and the divergence form $Lu=-(c(x)^2u_x)_x$ with
    equal probability. The wave speed $c(x)=\sqrt{\kappa(x)}$ is taken to be a
    random scalar or a random field.

    Note: We call the second form factored since
        $$u_{tt}-c(x)(c(x)u_x)_x =
        (\partial_t-c(x)\partial_x)(\partial_t+c(x)\partial_x)u.$$
    """
    NON_DIV_FORM = 0
    FACTORED_FORM = 1
    DIV_FORM = 2
    wave_type: int

    def __str__(self) -> str:
        if self.wave_type == self.NON_DIV_FORM:
            wave_str = r"non-divergence $-c(x)^2u_{xx}$"
        if self.wave_type == self.FACTORED_FORM:
            wave_str = r"factored $-c(x)(c(x)u_x)_x$"
        if self.wave_type == self.DIV_FORM:
            wave_str = r"divergence $-(c(x)^2u_x)_x$"
        return wave_str + ", c(x)^2: " + super().__str__()

    def reset(self,  # pylint: disable=W0221
              rng: np.random.Generator,
              zero_prob: float = 0.,
              scalar_prob: float = 1.,
              field_prob: float = 1.,
              coef_type: Optional[int] = None,
              wave_type: Optional[int] = None) -> None:
        super().reset(rng, zero_prob, scalar_prob, field_prob, coef_type)
        if wave_type is None:
            self.wave_type = rng.choice(3)
        else:
            self.wave_type = wave_type

    def reset_debug(self,  # pylint: disable=W0221
                    zero: bool = False, wave_type: int = 0) -> None:
        super().reset_debug(zero=zero)
        self.wave_type = wave_type

    def boundary_values(self) -> Tuple[float]:
        r""" Get the signed wave speed values at the boundary points. """
        c_l = -np.sqrt(self.field[0])
        c_r = np.sqrt(self.field[-1])
        return c_l, c_r

    def gen_dedalus_ops(self, field_op,  # pylint: disable=W0221
                        dx_u, dx_fn: Callable):
        if self.coef_type == self.ZERO_COEF:
            return 0

        if self.coef_type == self.SCALAR_COEF:
            c_or_c2 = self.field[0]  # scalar
        elif self.coef_type == self.FIELD_COEF:
            c_or_c2 = self.field  # np.ndarray

        if self.wave_type == self.FACTORED_FORM:
            c_or_c2 = np.sqrt(c_or_c2)  # c(x) rather than c(x)^2

        # replace np.ndarray by Dedalus field operator
        if self.coef_type == self.FIELD_COEF:
            field_op["g"] = c_or_c2
            c_or_c2 = field_op

        # compute the wave term
        if self.wave_type == self.NON_DIV_FORM:
            return -c_or_c2 * dx_fn(dx_u)
        if self.wave_type == self.FACTORED_FORM:
            return -c_or_c2 * dx_fn(c_or_c2 * dx_u)
        if self.wave_type == self.DIV_FORM:
            return -dx_fn(c_or_c2 * dx_u)
        raise NotImplementedError

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {prefix + "/coef_type": self.coef_type,
                prefix + "/wave_type": self.wave_type,
                prefix + "/field": self.field}


class RandomBCwithMur(common.RandomBoundaryCondition):
    r"""
    Non-periodic boundary condition of a PDE, which is taken to be Dirichlet,
    Neumann, Robin or Mur with equal probability. The Robin boundary condition
    takes the form $(\alpha u+\beta u_x)(t,x_{L/R})=\gamma$, where the
    coefficients $\alpha,\beta$ are random numbers satisfying
    $\alpha^2+\beta^2=1$ and $\beta>0$. The Mur boundary condition follows the
    form $(u_t+c u_x)(t,x_{L/R})=\gamma$. The specific boundary value $\gamma$
    (`bc_val`) is set to be zero, to be a random number, or according to the
    initial condition with equal probability.
    """
    MUR = 3

    def __str__(self) -> str:
        if self.bc_type != self.MUR:
            return super().__str__()
        type_str = f"Mur, u_t + {self.dx_u_coef:.4g} u_x"
        val_str = f"{self.bc_val:.4g}"
        if self.bc_val_type == self.FROM_IC_VAL:
            val_str += " (from IC)"
        return f"({type_str}) {val_str}"

    def reset(self,  # pylint: disable=W0221
              rng: np.random.Generator,
              c_val: float,
              ic_val: float,
              dx_ic_val: float,
              dt_ic_val: float,
              bc_type: Optional[int] = None,
              bc_val_type: Optional[int] = None) -> None:
        r"""
        Args:
            rng (np.random.Generator): Random number generator instance.
            c_val (float): The signed sound speed at this boundary point $x_0$
                that appears in the Mur boundary condition. Equal to $c(x_0)$
                at the right endpoint, and $-c(x_0)$ at left.
            ic_val (float): The initial value of $u$ at $x_0$, i.e. $g(x_0)$.
            dx_ic_val (float): The value of $g'(x_0)$.
            dt_ic_val (float): The initial value of $u_t$ at $x_0$.
            bc_type (Optional[int]): Type of the boundary condition. Choices:
                0 (Dirichlet), 1 (Neumann), 2 (Robin), 3 (Mur).
            bc_val_type (Optional[int]): Type of the boundary value. Choices:
                0 (zero), 1 (random), 2 (according to the initial condition).
        """
        if bc_type is None:
            self.bc_type = rng.choice(4)
        else:
            self.bc_type = bc_type

        if self.bc_type != self.MUR:
            super().reset(rng, ic_val, dx_ic_val, bc_type=self.bc_type,
                          bc_val_type=bc_val_type)
            return  # accept the current boundary condition
        super().reset(rng, ic_val, dx_ic_val, bc_type=self.NEUMANN,
                      bc_val_type=bc_val_type)
        self.bc_type = self.MUR  # change the overwritten value back
        self.dx_u_coef = c_val
        if self.bc_val_type == self.FROM_IC_VAL:
            self.bc_val = dt_ic_val + self.dx_u_coef * dx_ic_val

    def gen_dedalus_ops(self, u_op, dx_u, dt_u) -> Tuple:  # pylint: disable=W0221
        if self.bc_type != self.MUR:
            return super().gen_dedalus_ops(u_op, dx_u)
        bc_op = dt_u + self.dx_u_coef * dx_u
        return bc_op, self.bc_val


class WaveEquation(common.PDEDataGenBase):
    r"""
    Generate dataset of 1D time-dependent PDE solutions with Dedalus-v3.
    ======== Wave Equation ========
    The PDE takes the form $u_{tt}+\mu u_t+Lu+bu_x+f(u)+s_T(t)s_X(x)=0$,
        $(t,x)\in[0,1]\times[-1,1]$.
    Here, the wave term is randomly selected from the non-divergence form
    $Lu=-c(x)^2u_{xx}$, the factored form $Lu=-c(x)(c(x)u_x)_x$, and the
    divergence form $Lu=-(c(x)^2u_x)_x$ with equal probability, where
    $c(x)^2=\kappa(x)$.
    We take $f(u) = \sum_{k=1}^3c_{0k}u^k
                   + \sum_{j=1}^{J}c_{j0}h_{j}(c_{j1}u+c_{j2}u^2)$.
    Unless periodic boundary is specified, the boundary condition at each
    endpoint is randomly selected from Dirichlet, Neumann, Robin and Mur
    (absorbing) types.
    """
    info_dict = {"version": 4.2, "preprocess_dag": True, "pde_type_id": 4}
    # strangely, the solver fails for smaller time steps
    T_SAVE_STEPS = 4  # for varying scalars/fields; otherwise 1 is enough
    TIMESTEP = 1e-2 / T_SAVE_STEPS

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.periodic = args.periodic

        # Dedalus Bases
        basis_cls = d3.RealFourier if self.periodic else d3.Chebyshev
        self.xbasis = basis_cls(d3_xcoord, size=N_X_GRID, bounds=(X_L, X_R),
                                dealias=DEALIAS)
        self.x_coord = dist.local_grid(self.xbasis)

        # PDE terms
        self.u_ic = common.InitialCondition(self.x_coord, self.periodic)
        self.ut_ic = common.InitialCondition(self.x_coord, self.periodic)
        self.mu_term = common.NonNegativeRandomValue(
            min_val=args.kappa_min, max_val=args.kappa_max)
        self.wave_term = WaveTerm(
            self.x_coord, self.periodic,
            min_val=args.kappa_min, max_val=args.kappa_max)
        self.b_term = common.RandomValue(
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.f_term = SinusoidalTermFi(
            num_sinusoid=args.num_sinusoid,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        self.s_term = TimeDependentRandomFieldCoef(
            self.x_coord,
            self.periodic,
            t_max=self.STOP_SIM_TIME,
            n_t_grid=N_X_GRID,
            coef_distribution=args.coef_distribution,
            coef_magnitude=args.coef_magnitude)
        if not self.periodic:
            self.bc_l = RandomBCwithMur(
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)
            self.bc_r = RandomBCwithMur(
                coef_distribution=args.coef_distribution,
                coef_magnitude=args.coef_magnitude)

    @property
    def _term_obj_dict(self) -> Dict[str, common.PDETermBase]:
        coef_obj_dict = {"ut_ic": self.ut_ic, "mu": self.mu_term,
                         "Lu": self.wave_term, "b": self.b_term,
                         "f": self.f_term, "s": self.s_term}
        if not self.periodic:
            coef_obj_dict.update({"bc_l": self.bc_l, "bc_r": self.bc_r})
        return coef_obj_dict

    @staticmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        domain_type = "circ" if args.periodic else "interval"
        sinus_type = f"s{args.num_sinusoid}_" if args.num_sinusoid > 0 else ""
        return (f"wave_{sinus_type}{domain_type}"
                f"_c{args.coef_distribution}{args.coef_magnitude:g}"
                f"_k{args.kappa_min:.0e}_{args.kappa_max:g}")

    @classmethod
    def _get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        parser.add_argument("--periodic", action="store_true",
                            help="use periodic (circular) boundary conditions")
        SinusoidalTermFi.add_cli_args_(parser)
        WaveTerm.add_cli_args_(parser)
        return parser

    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)
        self.u_ic.reset(rng)
        self.ut_ic.reset(rng)
        self.mu_term.reset(rng, zero_prob=0.8)
        zero_c_prob = 0.2 if self.periodic else 0.
        self.wave_term.reset(rng, zero_prob=zero_c_prob)
        self.b_term.reset(rng)
        self.f_term.reset(rng)
        self.s_term.reset(rng)

        if not self.periodic:
            (ic_l, ic_r, dx_ic_l, dx_ic_r) = self.u_ic.boundary_values()
            (dt_ic_l, dt_ic_r, _, _) = self.ut_ic.boundary_values()
            (c_l, c_r) = self.wave_term.boundary_values()
            self.bc_l.reset(rng, c_l, ic_l, dx_ic_l, dt_ic_l)
            self.bc_r.reset(rng, c_r, ic_r, dx_ic_r, dt_ic_r)

    def gen_solution(self) -> None:
        u_list, problem, s_op = self._get_dedalus_problem()

        # Solver
        solver = problem.build_solver(self.TIMESTEPPER)
        solver.stop_sim_time = self.STOP_SIM_TIME

        # Main loop
        u_sol, t_coord = [], []

        def record():
            for ui_op in u_list:
                ui_op.change_scales(1)
            u_current = [np.copy(ui_op['g']) for ui_op in u_list]
            u_sol.append(np.stack(u_current, axis=-1))
            t_coord.append(solver.sim_time)

        record()
        while solver.proceed:
            solver.step(self.TIMESTEP)
            # the only difference from 'gen_solution' of the base class
            self.s_term.updata_dedalus_ops(solver.sim_time, s_op)
            if solver.iteration % self.T_SAVE_STEPS == 0:
                record()

        self.t_coord = np.array(t_coord)
        self.u_sol = np.array(u_sol, dtype=np.float32)

    def _get_dedalus_problem(self) -> Tuple:
        # Fields
        u_op = dist.Field(name="u", bases=self.xbasis)
        dt_u = dist.Field(name="dt_u", bases=self.xbasis)
        c_or_c2_field = dist.Field(name="c", bases=self.xbasis)
        if self.s_term.is_field:
            s_op = dist.Field(name="s", bases=self.xbasis)
        else:
            s_op = dist.Field(name="s")
        u_list = [u_op]

        # Initial condition
        u_op = self.u_ic.gen_dedalus_ops(u_op)
        dt_u = self.ut_ic.gen_dedalus_ops(dt_u)

        # PDE Terms
        dx_u = d3_dx(u_op)
        mu_op = self.mu_term.gen_dedalus_ops()
        wave_op = self.wave_term.gen_dedalus_ops(
            c_or_c2_field, dx_u, dx_fn=d3_dx)
        b_op = self.b_term.gen_dedalus_ops()
        f_lin, f_nonlin = self.f_term.gen_dedalus_ops(u_op)
        s_op = self.s_term.gen_dedalus_ops(s_op)

        # \mu u_t + Lu + bu_x + f(u)_lin
        lin = mu_op * dt_u + wave_op + b_op * dx_u + f_lin
        nonlin = f_nonlin + s_op  # f(u)_nlin + s(t,x)

        # Periodic case
        if self.periodic:
            problem = d3.IVP([u_op, dt_u])
            problem.add_equation([d3.dt(u_op) - dt_u, 0])
            problem.add_equation([d3.dt(dt_u) + lin, -nonlin])
            return u_list, problem, s_op

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

        problem = d3.IVP([u_op, dt_u, tau1, tau2])
        problem.add_equation([d3.dt(u_op) - dt_u, 0])
        problem.add_equation([d3.dt(dt_u) + lin_with_tau, -nonlin])

        # boundary conditions
        op_l, bc_val_l = self.bc_l.gen_dedalus_ops(u_op, dx_u, dt_u)
        problem.add_equation([op_l(x="left"), bc_val_l])
        op_r, bc_val_r = self.bc_r.gen_dedalus_ops(u_op, dx_u, dt_u)
        problem.add_equation([op_r(x="right"), bc_val_r])

        return u_list, problem, s_op


class WaveEquationUnitTest(WaveEquation):
    r""" Unit test of wave equation data generation. """

    @staticmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "debug_wave"

    def reset_pde(self, rng: np.random.Generator) -> None:
        super().reset_pde(rng)

        # random fields
        self.u_ic.reset_debug()
        self.ut_ic.reset_debug()
        self.mu_term.reset_debug(zero=True)
        self.wave_term.reset_debug()
        self.b_term.reset_debug()
        self.f_term.reset_debug()

        if not self.periodic:
            (ic_l, ic_r, _, _) = self.u_ic.boundary_values()
            (_, _, _, _) = self.ut_ic.boundary_values()
            (_, _) = self.wave_term.boundary_values()
            self.bc_l.reset_debug(ic_l)
            self.bc_r.reset_debug(ic_r)


if __name__ == "__main__":
    my_args = WaveEquation.get_cli_args()
    pde_data_gen = WaveEquation(my_args)
    common.gen_data(my_args, pde_data_gen)
