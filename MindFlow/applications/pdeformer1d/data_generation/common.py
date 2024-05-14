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
r"""Common utilities utilized during data generation."""
import time
import logging
import argparse
import os
import sys
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, Optional, Callable
import h5py
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import dedalus.public as d3


class PDETermBase(ABC):
    r""" abstract class for special PDE terms used in PDEDataGenBase class """

    def __str__(self) -> str:
        data_dict = self.get_data_dict(" ")
        str_list = [f"{key}: {data}" for key, data in data_dict.items()]
        return " ".join(str_list)

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser) -> None:
        r""" Add command-line arguments for this PDE term. """

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        r"""
        Reset the random parameters in the current term.
        Args:
            rng (numpy.random.Generator): Random number generator instance.
        """

    @abstractmethod
    def reset_debug(self) -> None:
        r""" Reset the parameters in the current term to its default value. """

    @abstractmethod
    def gen_dedalus_ops(self):
        r""" Generate the operator for the Dedalus solver. """
        return 0

    @abstractmethod
    def get_data_dict(self, prefix: str
                      ) -> Dict[str, Union[int, float, NDArray]]:
        r"""
        Returning a dictionary containing the current parameters in this PDE
        term.
        """
        return {}

    def prepare_plot(self, title: str = "") -> None:
        r"""
        Create a matplotlib figure showing the current parameters in this PDE
        term. Use `plt.show()` afterwards to see the results.
        """


class PDERandomCoefBase(PDETermBase):
    r"""
    abstract class for special PDE terms with random scalar coefficients
    """
    distribution: str
    magnitude: float

    def __init__(self,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__()
        self.distribution = coef_distribution
        self.magnitude = coef_magnitude

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--coef_magnitude", "-m", type=float, default=1.,
                            help="magnitude of randomly generate PDE coefficients")
        parser.add_argument("--coef_distribution", "-D", type=str, default="U",
                            choices=["U", "N", "L", "C"], help=r"""
                            distribution type of random PDE coefficients.
                            choices: 'U' (uniform U([-m, m]); default),
                                     'N' (normal N(0, m^2)),
                                     'L' (Laplace with mean 0 and scale m),
                                     'C' (Cauchy with mean 0 and scale m),
                            """)

    @abstractmethod
    def reset(self, rng: np.random.Generator) -> None:
        r"""
        Reset the random parameters in the current term.
        Args:
            rng (numpy.random.Generator): Random number generator instance.
        """

    def _random_value(self,
                      rng: np.random.Generator,
                      size: Union[None, int, Tuple[int]] = None,
                      ) -> NDArray[float]:
        r"""Generate random values."""
        if self.distribution == "U":
            array = rng.uniform(-1, 1, size=size)
        elif self.distribution == "N":
            array = rng.normal(size=size)
        elif self.distribution == "L":
            array = rng.laplace(size=size)
        elif self.distribution == "C":
            array = rng.standard_cauchy(size=size)
        else:
            raise NotImplementedError
        return self.magnitude * array


class RandomValue(PDERandomCoefBase):
    r"""
    Random coefficient(s) in a PDE, each entry set to zero with certain
    probability.
    """
    size: Union[None, int, Tuple[int]]
    value: NDArray[float]

    def __init__(self,
                 size: Union[None, int, Tuple[int]] = None,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(coef_distribution, coef_magnitude)
        self.size = size

    def reset(self, rng: np.random.Generator,  # pylint: disable=W0221
              zero_prob: float = 0.5) -> None:
        value = self._random_value(rng, size=self.size)
        mask = rng.choice([False, True], p=[zero_prob, 1 - zero_prob],
                          size=self.size)
        self.value = np.where(mask, value, 0.)

    def reset_debug(self) -> None:
        self.value = np.zeros(self.size)

    def gen_dedalus_ops(self) -> Union[float, NDArray[float]]:
        if self.size is None:
            return self.value.item()
        return self.value

    def get_data_dict(self, prefix: str) -> Dict[str, NDArray[float]]:
        return {prefix: self.value}


class NonNegativeRandomValue(PDETermBase):
    r"""
    Random non-negative coefficient(s) in the PDE, which may appear in
    diffusion, damping or wave propagation, etc.
    """
    log_min: float
    log_max: float
    size: Union[None, int, Tuple[int]]
    value: NDArray[float]

    def __init__(self,
                 min_val: float = 1e-3,
                 max_val: float = 1.,
                 size: Union[None, int, Tuple[int]] = None,
                 ) -> None:
        super().__init__()
        self.log_min = np.log(min_val)
        self.log_max = np.log(max_val)
        self.size = size

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--kappa_min", type=float, default=1e-3,
                            help="Lower bound for the non-negative "
                            "coefficient. Must be positive.")
        parser.add_argument("--kappa_max", type=float, default=1,
                            help="Similar to --kappa_min, but for the upper "
                            "bound.")

    def reset(self,    # pylint: disable=W0221
              rng: np.random.Generator,
              zero_prob: Union[float, NDArray[float]] = 0.) -> None:
        value = rng.uniform(self.log_min, self.log_max, size=self.size)
        value = np.exp(value)
        mask = rng.random(self.size) >= zero_prob
        self.value = np.where(mask, value, 0.)

    def reset_debug(self, zero: bool = False) -> None:  # pylint: disable=W0221
        if zero:
            self.value = np.zeros(self.size)
        else:
            value = (self.log_min + self.log_max) / 2
            self.value = np.full(self.size, np.exp(value))

    def gen_dedalus_ops(self) -> Union[float, NDArray[float]]:
        if self.size is None:
            return self.value.item()
        return self.value

    def get_data_dict(self, prefix: str) -> Dict[str, NDArray[float]]:
        return {prefix: self.value}


class RandomFieldCoef(PDERandomCoefBase):
    r"""
    Coefficient term involved in a PDE, which can be a zero, a real number
    (scalar) or a spatially-varying random field.
    """
    ZERO_COEF = 0
    SCALAR_COEF = 1
    FIELD_COEF = 2
    coef_type: int
    field: NDArray[float]
    x_coord: NDArray[float]
    periodic: bool

    def __init__(self,
                 x_coord: NDArray[float],
                 periodic: bool = True,
                 coef_distribution: str = "U",
                 coef_magnitude: float = 1.) -> None:
        super().__init__(coef_distribution, coef_magnitude)
        self.x_coord = x_coord
        self.periodic = periodic
        if not np.all(np.diff(x_coord) > 0):
            raise ValueError("'x_coord' should be strictly increasing.")

    def __str__(self) -> str:
        if self.coef_type == self.ZERO_COEF:
            return "0"
        if self.coef_type == self.SCALAR_COEF:
            return f"{self.field[0]:.4g}"
        return "field " + self._arr_to_str(self.field)

    @property
    def is_const(self) -> bool:
        r""" Whether the current coefficient is a constant. """
        return self.coef_type != self.FIELD_COEF

    @staticmethod
    def _rand_frequencies(rng: np.random.Generator,  # pylint: disable=W0221
                          x_len: int,
                          numbers: int,
                          k_tot: int,
                          num_choice_k: int,
                          periodic: bool,
                          ) -> NDArray[float]:
        r"""Generate random frequencies."""
        if x_len <= 0:
            raise ValueError(f"'x_len' should be positive, but got {x_len}.")
        if periodic:
            selected = rng.integers(
                size=[numbers, num_choice_k], low=0, high=k_tot)

            def one_hot(class_id: NDArray[int],
                        num_classes: int) -> NDArray[float]:
                return np.eye(num_classes)[class_id]
            # [numbers, num_choice_k] -> [numbers, num_choice_k, k_tot]
            # -> [numbers, k_tot]
            selected = one_hot(selected, k_tot).sum(axis=1)
            # Shape is [numbers, k_tot].
            angular_freq = np.pi * 2. * np.arange(1, k_tot + 1) * \
                selected / x_len
        else:
            k_max = np.pi * 2. * k_tot / x_len
            angular_freq = rng.uniform(0, k_max, size=[numbers, num_choice_k])
            # [numbers, num_choice_k] -> [numbers, k_tot]
            angular_freq = np.pad(
                angular_freq, ((0, 0), (0, k_tot - num_choice_k)))

        return angular_freq

    @staticmethod
    def _apply_abs_fn(rng: np.random.Generator,
                      u_: NDArray[float],
                      numbers: int) -> NDArray[float]:
        r"""
        Randomly mask the absolute value of u and randomly flip the signature.
        """
        # perform absolute value function
        cond = rng.choice([False, True], p=np.array([0.9, 0.1]), size=numbers)
        u_[cond] = np.abs(u_[cond])

        # random flip of signature
        sgn = rng.choice(a=np.array([1, -1]), size=[numbers, 1])
        u_ *= sgn
        return u_  # [numbers, n_x_grid]

    @staticmethod
    def _apply_window_restriction(rng: np.random.Generator,
                                  u_: NDArray[float],
                                  x_coord: NDArray[float],
                                  numbers: int) -> NDArray[float]:
        r"""Apply window restriction"""
        # perform window function
        cond = rng.choice([False, True], p=np.array([0.9, 0.1]), size=numbers)
        # [n_x_grid] -> [numbers, n_x_grid]
        xc_ = np.repeat(x_coord[np.newaxis, :], numbers, axis=0)
        x_l = rng.uniform(size=([numbers, 1]), low=0.1, high=0.45)
        x_r = rng.uniform(size=([numbers, 1]), low=0.55, high=0.9)
        trns = 0.01
        mask = 0.5 * (np.tanh((xc_ - x_l) / trns) -
                      np.tanh((xc_ - x_r) / trns))
        mask[np.logical_not(cond)] = 1.

        u_ *= mask
        return u_  # [numbers, n_x_grid]

    @staticmethod
    def _arr_to_str(array: NDArray[float]) -> str:
        data = array.flat
        return f"[{data[0]:.4g} ... {data[-1]:.4g}], shape {array.shape}"

    @classmethod
    def _random_field(cls,
                      x_coord: NDArray[float],
                      rng: np.random.Generator,
                      numbers: int = 1,
                      k_tot: int = 4,
                      num_choice_k: int = 2,
                      normalized: bool = False,
                      periodic: bool = True,
                      smooth: bool = False) -> NDArray[float]:
        r"""
        Generate random fields.
        :param x_coord: cell center coordinate
        :param mode: initial condition
        :return: 1D scalar function u at cell center
        code adapted from PDEBench.
        """
        # angular frequency
        x_len = x_coord[-1] - x_coord[0]
        angular_freq = cls._rand_frequencies(
            rng, x_len, numbers, k_tot, num_choice_k, periodic)

        # sum of sine waves
        amp = rng.uniform(size=[numbers, k_tot, 1])
        phs = 2. * np.pi * rng.uniform(size=[numbers, k_tot, 1])
        u_ = amp * np.sin(angular_freq[:, :, np.newaxis] *
                          x_coord[np.newaxis, np.newaxis, :] + phs)
        # [numbers, k_tot, n_x_grid] -> [numbers, n_x_grid]
        u_ = np.sum(u_, axis=1)

        if not smooth:
            u_ = cls._apply_abs_fn(rng, u_, numbers)
            u_ = cls._apply_window_restriction(rng, u_, x_coord, numbers)

        # normalize value between [0, 1] for reaction-diffusion eq.
        if normalized:
            u_ -= np.min(u_, axis=1, keepdims=True)  # positive value
            u_ /= np.max(u_, axis=1, keepdims=True)  # normalize

        return u_

    def reset(self,  # pylint: disable=W0221
              rng: np.random.Generator,
              zero_prob: float = 1.,
              scalar_prob: float = 1.,
              field_prob: float = 1.,
              coef_type: Optional[int] = None,
              ) -> None:
        if coef_type is None:
            probs = np.array([zero_prob, scalar_prob, field_prob])
            if probs.sum() == 0:
                raise ValueError("At least one of the probs should be positive.")
            self.coef_type = rng.choice(3, p=probs/probs.sum())
        else:
            self.coef_type = coef_type

        x_coord = self.x_coord
        if self.coef_type == self.ZERO_COEF:
            self.field = np.zeros_like(x_coord)
        elif self.coef_type == self.SCALAR_COEF:
            value = self._random_value(rng)
            self.field = np.zeros_like(x_coord) + value
        elif self.coef_type == self.FIELD_COEF:
            field = self._random_field(x_coord, rng, periodic=self.periodic)
            self.field, = field  # [1, n_x_grid] -> [n_x_grid]

    def reset_debug(self) -> None:
        self.coef_type = self.ZERO_COEF
        self.field = np.zeros_like(self.x_coord)

    def gen_dedalus_ops(self, field_op):  # pylint: disable=W0221
        if self.coef_type == self.ZERO_COEF:
            return 0
        if self.coef_type == self.SCALAR_COEF:
            return self.field[0]
        field_op["g"] = self.field
        return field_op

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, NDArray[float]]]:
        return {prefix + "/coef_type": self.coef_type,
                prefix + "/field": self.field}

    def prepare_plot(self, title: str = "") -> None:
        if self.is_const:
            return  # no need to plot constants
        plt.figure()
        plt.plot(self.x_coord, self.field)
        plt.title(title)


class InitialCondition(RandomFieldCoef):
    r""" Generating random field initial conditions of a PDE. """

    def __init__(self, x_coord: NDArray[float], periodic: bool = True) -> None:
        super().__init__(x_coord, periodic=periodic)

    @staticmethod
    def add_cli_args_(parser: argparse.ArgumentParser) -> None:
        pass  # no args to be added

    def reset(self, rng: np.random.Generator) -> None:  # pylint: disable=W0221
        super().reset(rng, coef_type=self.FIELD_COEF)

    def reset_debug(self) -> None:
        self.coef_type = self.FIELD_COEF
        self.field = np.zeros_like(self.x_coord)

    def boundary_values(self) -> Tuple[float]:
        r"""
        Get the values and spatial derivatives of the initial condition at the
        boundary points.
        """
        field = self.field  # [n_x_grid]
        x_coord = self.x_coord
        ic_l = field[0]
        ic_r = field[-1]
        dx_ic_l = (field[1] - field[0]) / (x_coord[1] - x_coord[0])
        dx_ic_r = (field[-1] - field[-2]) / (x_coord[-1] - x_coord[-2])
        bv = (ic_l, ic_r, dx_ic_l, dx_ic_r)
        return bv


class NonNegativeCoefField(RandomFieldCoef):
    r"""
    A non-negative coefficient in the PDE, which may appear in diffusion,
    damping or wave propagation, etc. Can be a zero, a real number (scalar) or
    a spatially-varying random field.
    """

    def __init__(self,
                 x_coord: NDArray[float],
                 periodic: bool = True,
                 min_val: float = 1e-3,
                 max_val: float = 1.) -> None:
        super().__init__(x_coord, periodic=periodic)
        self.log_min = np.log(min_val)
        self.log_max = np.log(max_val)

    def reset(self,
              rng: np.random.Generator,
              zero_prob: float = 1.,
              scalar_prob: float = 1.,
              field_prob: float = 1.,
              coef_type: Optional[int] = None) -> None:
        if coef_type is None:
            probs = np.array([zero_prob, scalar_prob, field_prob])
            if probs.sum() == 0:
                raise ValueError("At least one of the probs should be positive.")
            self.coef_type = rng.choice(3, p=probs/probs.sum())
        else:
            self.coef_type = coef_type

        x_coord = self.x_coord
        if self.coef_type == self.ZERO_COEF:
            self.field = np.zeros_like(x_coord)
        elif self.coef_type == self.SCALAR_COEF:
            value = rng.uniform(self.log_min, self.log_max)
            self.field = np.zeros_like(x_coord) + np.exp(value)
        elif self.coef_type == self.FIELD_COEF:
            # field values normalized to [0,1]
            field = self._random_field(x_coord, rng, periodic=self.periodic,
                                       normalized=True, smooth=True)
            field, = field  # [1, n_x_grid] -> [n_x_grid]

            # let the range of field span a random sub-interval of
            # the full interval [log_min,log_max]
            tmp = rng.dirichlet([1, 1, 1]) * (self.log_max - self.log_min)
            (margin_bottom, span, _) = tmp
            # we have margin_bottom + span + _ == log_max - log_min
            field = self.log_min + margin_bottom + span * field

            field = field.clip(self.log_min, self.log_max)
            self.field = np.exp(field)

    def reset_debug(self, zero: bool = False) -> None:  # pylint: disable=W0221
        if zero:
            self.coef_type = self.ZERO_COEF
            self.field = np.zeros_like(self.x_coord)
        else:
            self.coef_type = self.SCALAR_COEF
            value = (self.log_min + self.log_max) / 2
            self.field = np.zeros_like(self.x_coord) + np.exp(value)

    def prepare_plot(self, title: str = "") -> None:
        if self.is_const:
            return  # no need to plot constants
        super().prepare_plot(title=title)
        plt.yscale("log")
        plt.ylim(np.exp(self.log_min), np.exp(self.log_max))

    add_cli_args_ = NonNegativeRandomValue.add_cli_args_


class RandomBoundaryCondition(PDERandomCoefBase):
    r"""
    Non-periodic boundary condition of a PDE, which is taken to be Dirichlet,
    Neumann or Robin with equal probability. The Robin boundary condition takes
    the form $(\alpha u+\beta u_x)(t,x_{L/R})=\gamma$, where the coefficients
    $\alpha,\beta$ are random numbers satisfying $\alpha^2+\beta^2=1$ and
    $\beta>0$. The specific boundary value $\gamma$ (`bc_val`) is set to be
    zero, to be a random number, or according to the initial condition with
    equal probability.
    """
    DIRICHLET = 0
    NEUMANN = 1
    ROBIN = 2
    bc_type: int
    u_coef: float
    dx_u_coef: float

    ZERO_VAL = 0
    RAND_VAL = 1
    FROM_IC_VAL = 2
    bc_val_type: int
    bc_val: float

    def __str__(self) -> str:
        if self.bc_type == self.DIRICHLET:
            type_str = "D"
        elif self.bc_type == self.NEUMANN:
            type_str = "N"
        elif self.bc_type == self.ROBIN:
            type_str = f"R, {self.u_coef:.4g} u + {self.dx_u_coef:.4g} u_x"
        val_str = f"{self.bc_val:.4g}"
        if self.bc_val_type == self.FROM_IC_VAL:
            val_str += " (from IC)"
        return f"({type_str}) {val_str}"

    def reset(self,  # pylint: disable=W0221
              rng: np.random.Generator,
              ic_val: float,
              dx_ic_val: float,
              bc_type: Optional[int] = None,
              bc_val_type: Optional[int] = None) -> None:
        r"""
        Args:
            rng (np.random.Generator): Random number generator instance.
            ic_val (float): Value of the initial condition g(x) at this
                boundary point x_0, i.e. g(x_0).
            dx_ic_val (float): Value of g'(x_0).
            bc_type (Optional[int]): Type of the boundary condition. Choices:
                0 (Dirichlet), 1 (Neumann), 2 (Robin).
            bc_val_type (Optional[int]): Type of the boundary value. Choices:
                0 (zero), 1 (random), 2 (according to the initial condition).
        """
        if bc_type is None:
            self.bc_type = rng.choice(3)
        else:
            self.bc_type = bc_type

        if self.bc_type == self.DIRICHLET:
            self.u_coef = 1
            self.dx_u_coef = 0
        elif self.bc_type == self.NEUMANN:
            self.u_coef = 0
            self.dx_u_coef = 1
        elif self.bc_type == self.ROBIN:
            alpha = rng.uniform(0, np.pi)
            self.u_coef = np.cos(alpha)
            self.dx_u_coef = np.sin(alpha)

        if bc_val_type is None:
            self.bc_val_type = rng.choice(3)
        else:
            self.bc_val_type = bc_val_type

        if self.bc_val_type == self.ZERO_VAL:
            self.bc_val = 0
        elif self.bc_val_type == self.RAND_VAL:
            self.bc_val = self._random_value(rng)
        elif self.bc_val_type == self.FROM_IC_VAL:
            self.bc_val = self.u_coef * ic_val + self.dx_u_coef * dx_ic_val

    def reset_debug(self,  # pylint: disable=W0221
                    ic_val: Optional[float] = None) -> None:
        self.bc_type = self.DIRICHLET
        self.u_coef = 1
        self.dx_u_coef = 0

        if ic_val is None:
            self.bc_val_type = self.ZERO_VAL
            self.bc_val = 0
        else:
            self.bc_val_type = self.FROM_IC_VAL
            self.bc_val = ic_val

    def gen_dedalus_ops(self, u_op, dx_u) -> Tuple:  # pylint: disable=W0221
        if self.bc_type == self.DIRICHLET:
            bc_op = u_op
        elif self.bc_type == self.NEUMANN:
            bc_op = dx_u
        elif self.bc_type == self.ROBIN:
            bc_op = self.u_coef * u_op + self.dx_u_coef * dx_u
        return bc_op, self.bc_val

    def get_data_dict(self, prefix: str) -> Dict[str, Union[int, float]]:
        return {prefix + "/bc_type": self.bc_type,
                prefix + "/u_coef": self.u_coef,
                prefix + "/dx_u_coef": self.dx_u_coef,
                prefix + "/bc_val_type": self.bc_val_type,
                prefix + "/bc_val": self.bc_val}


class PDEDataGenBase(ABC):
    r"""
    Generate dataset of 1D time-dependent PDE solutions using Dedalus-v3.
    Abstract base class.
    """
    info_dict = {"version": 4.2, "preprocess_dag": False, "pde_type_id": 0}
    n_vars: int = 1  # number of components in the PDE

    # Dedalus Parameters
    STOP_SIM_TIME = 1.
    TIMESTEPPER = d3.SBDF2
    T_SAVE_STEPS = 25
    TIMESTEP = 1e-2 / T_SAVE_STEPS

    # solution data
    t_coord: NDArray[float]
    x_coord: NDArray[float]
    u_sol: Union[NDArray[float], None]

    def __init__(self, args: argparse.Namespace) -> None:
        self.u_bound = args.u_bound
        self.num_pde = args.num_pde
        self.print_coef_level = args.print_coef_level
        self.plot_u = args.plot_u
        self.u_sol_all = []
        self.coef_all_dict = {}

    @property
    def current_coef_dict(self) -> Dict[str, Union[int, float, NDArray]]:
        r""" A dictionary containing the current PDE coefficients. """
        coef_dict = {}
        for prefix, term_obj in self._term_obj_dict.items():
            coef_dict.update(term_obj.get_data_dict(prefix))
        return coef_dict

    @property
    def current_coef_str_dict(self) -> Dict[str, str]:
        r"""
        A dictionary containing the string representation of the current PDE
        coefficients.
        """
        str_dict = {prefix: str(term_obj)
                    for prefix, term_obj in self._term_obj_dict.items()}
        return str_dict

    @property
    def n_data(self) -> int:
        r""" Current number of generated PDE solution samples. """
        return len(self.u_sol_all)

    @property
    @abstractmethod
    def _term_obj_dict(self) -> Dict[str, PDETermBase]:
        r"""
        A dictionary containing the PDE terms, as instances of `PDETermBase`.
        """
        return {}

    @staticmethod
    @abstractmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "base"

    @classmethod
    def _get_cli_args_parser(cls) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=cls.__doc__)
        return parser

    @classmethod
    def get_cli_args(cls) -> argparse.Namespace:
        r""" Parse the command-line arguments for this PDE. """
        parser = cls._get_cli_args_parser()
        parser.add_argument("--num_pde", "-n", type=int, default=10000,
                            help="number of PDEs to generate")
        parser.add_argument("--np_seed", "-r", type=int, default=-1,
                            help="NumPy random seed")
        parser.add_argument("--u_bound", type=float, default=10,
                            help="accept only solutions with |u| <= u_bound")
        parser.add_argument("--print_coef_level", type=int, default=0,
                            choices=[0, 1, 2, 3], help=r"""
                            print coefficients of each generated PDE.
                            0: do not print. 1: print human-readable results.
                            2: print raw data. 3: print both human-readable
                            results and raw data. """)
        parser.add_argument("--plot_u", action="store_true",
                            help="show image(s) of each accepted solution")
        parser.add_argument("--h5_file_dir", type=str, default="results")
        args = parser.parse_args()
        return args

    @classmethod
    def get_hdf5_file_name(cls, args: argparse.Namespace) -> str:
        r"""
        Get the name of the target HDF5 file where the generated data is
        stored.
        """
        if 'version' not in cls.info_dict:
            raise KeyError("subclass of PDEDataGenBase should provide "
                           "'info_dict' with 'version' key.")
        version = cls.info_dict['version']
        fname = f"custom_v{version:g}_"
        fname += cls._get_hdf5_file_prefix(args)
        if args.num_pde != 10000:
            fname += f"_num{args.num_pde}"
        if args.np_seed == -1:
            fname += time.strftime('_%Y-%m-%d-%H-%M-%S')
        else:
            fname += f"_seed{args.np_seed}"
        return fname

    @abstractmethod
    def reset_pde(self, rng: np.random.Generator) -> None:
        r""" Randomly reset the terms in the current PDE. """
        self.u_sol = None

    def print_current_coef(self, print_fn: Union[None, Callable] = print) -> None:
        r""" Print the current coefficients of the PDE. """
        if not (self.print_coef_level > 0 and callable(print_fn)):
            return
        if self.print_coef_level in [1, 3]:
            print_str = "coefs: "
            for prefix, coef_str in self.current_coef_str_dict.items():
                print_str += f"\n  {prefix}: {coef_str}"
            print_fn(print_str)
        if self.print_coef_level in [2, 3]:
            print_str = "raw coefs: "
            for key, data in self.current_coef_dict.items():
                if np.size(data) < 20:
                    print_str += f"\n  {key}: {data}"
            print_fn(print_str)

    def gen_solution(self) -> None:
        r"""
        Generate the PDE solution corresponding to the current parameters.
        """
        u_list, problem = self._get_dedalus_problem()

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
            if solver.iteration % self.T_SAVE_STEPS == 0:
                record()

        self.t_coord = np.array(t_coord)
        self.u_sol = np.array(u_sol, dtype=np.float32)

    def plot(self,
             save_pdf_prefix: Optional[str] = None,
             plot_coef: bool = True) -> None:
        r"""
        Plot the current PDE solution as well as the coefficients (optional).
        """
        # plot coefficients
        if plot_coef:
            for prefix, term_obj in self._term_obj_dict.items():
                term_obj.prepare_plot(prefix)

        # plot current solution
        for i in range(self.n_vars):
            self._prepare_plot_2d(self.u_sol[:, :, i], f"u_sol component {i}")
            if save_pdf_prefix is not None:
                plt.savefig(f"{save_pdf_prefix}_{i}.pdf")

        plt.show()

    def gen_data_all(self,
                     rng: np.random.Generator,
                     print_fn: Optional[Callable] = None) -> None:
        r"""
        Generate the PDE solution data, until the target number of samples is
        reached.
        """
        i_pde = 0
        while self.n_data < self.num_pde:
            i_pde += 1
            if callable(print_fn):
                print_fn(f"trial No. {i_pde}, "
                         f"PDE generated {self.n_data}/{self.num_pde}")
            self.reset_pde(rng)
            self.print_current_coef(print_fn)
            self.gen_solution()
            if self._accept_u_sol(print_fn):
                if self.plot_u:
                    self.plot()
                self._record_solution(self.u_sol)

    def save_hdf5(self, args: argparse.Namespace) -> None:
        r""" Save the generated data to the target HDF5 file. """
        os.makedirs(args.h5_file_dir, exist_ok=True)
        fname = self.get_hdf5_file_name(args)
        h5_filepath = os.path.join(args.h5_file_dir, fname + ".hdf5")

        with h5py.File(h5_filepath, "w") as h5_file:
            h5_file.create_dataset("t_coord", data=self.t_coord)
            h5_file.create_dataset("x_coord", data=self.x_coord)
            h5_file.create_dataset("u_sol_all", data=np.array(self.u_sol_all))
            for key, data_list in self.coef_all_dict.items():
                h5_file.create_dataset("coef/" + key + "_all",
                                       data=np.array(data_list))
            for key, value in self.info_dict.items():
                h5_file.create_dataset("pde_info/" + key, data=value)
            for key, value in vars(args).items():
                if isinstance(value, (int, float, np.ndarray)):
                    h5_file.create_dataset("pde_info/args/" + key, data=value)

    @abstractmethod
    def _get_dedalus_problem(self) -> Tuple:
        pass

    def _accept_u_sol(self, print_fn: Optional[Callable] = None) -> bool:
        if not np.isfinite(self.u_sol).all():
            return False
        u_max = np.max(np.abs(self.u_sol))
        if u_max > self.u_bound:
            if callable(print_fn):
                print_fn(f"rejected: u_max {u_max:.2f}")
            return False
        return True

    def _prepare_plot_2d(self,  # pylint: disable=W0221
                         field_2d: NDArray[float],
                         title: str = "") -> None:
        r""" Prepare a 2D plot of the field. """
        plt.figure(figsize=(6, 5))
        plt.pcolormesh(self.x_coord, self.t_coord, field_2d, shading='gouraud',
                       cmap=plt.get_cmap("jet"), rasterized=True)
        plt.ylim(0, self.STOP_SIM_TIME)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(title)
        plt.colorbar()
        plt.tight_layout()

    def _record_solution(self, u_sol: NDArray[float]) -> None:
        coef_dict = self.current_coef_dict
        if self.n_data == 0:
            for key in coef_dict:
                self.coef_all_dict[key] = []
        for key, value in coef_dict.items():
            self.coef_all_dict[key].append(value)
        self.u_sol_all.append(u_sol)


class PDEInverseDataGenBase(PDEDataGenBase):
    r"""
    Abstract base class for generating inverse problem dataset of 1D
    time-dependent PDEs using Dedalus-v3. For each PDE, we generate multiple
    solutions samples, which may have different initial conditions, source
    terms, etc.

    You may first implement a class to generate data for the forward problem,
    and then obtain the class for inverse problem data generation via class
    inheritance.

    Examples:
        >>> MyPDE(PDEDataGenBase):
        ...     pass  # your own implementation
        >>>
        >>> MyPDEInverse(MyPDE, PDEInverseDataGenBase):
        ...     def reset_sample(self, rng):
        ...         pass  # your own implementation
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.num_sample_per_pde = args.num_sample_per_pde

    @staticmethod
    def _get_hdf5_file_prefix(args: argparse.Namespace) -> str:
        return "inverse"

    @classmethod
    def get_cli_args(cls) -> argparse.Namespace:
        parser = cls._get_cli_args_parser()
        parser.add_argument("--num_pde", "-n", type=int, default=100,
                            help="number of PDEs to generate")
        parser.add_argument("--num_sample_per_pde", "-s", type=int, default=100,
                            help="number of samples for each PDE")
        parser.add_argument("--np_seed", "-r", type=int, default=-1,
                            help="NumPy random seed")
        parser.add_argument("--u_bound", type=float, default=10,
                            help="accept only solutions with |u| <= u_bound")
        parser.add_argument("--print_coef_level", type=int, default=0,
                            choices=[0, 1, 2, 3], help=r"""
                            print coefficients of each generated PDE.
                            0: do not print. 1: print human-readable results.
                            2: print raw data. 3: print both human-readable
                            results and raw data. """)
        parser.add_argument("--plot_u", action="store_true",
                            help="show image(s) of each accepted solution")
        parser.add_argument("--h5_file_dir", type=str, default="results")
        args = parser.parse_args()
        return args

    @classmethod
    def get_hdf5_file_name(cls, args: argparse.Namespace) -> str:
        if 'version' not in cls.info_dict:
            raise KeyError("subclass of PDEDataGenBase must define "
                           "'info_dict' with 'version' key.")
        version = cls.info_dict['version']
        fname = f"custom_v{version:g}_inv_"
        fname += cls._get_hdf5_file_prefix(args)
        if args.num_pde != 100:
            fname += f"_num{args.num_pde}"
        if args.num_sample_per_pde != 100:
            fname += f"_samples{args.num_sample_per_pde}"
        if args.np_seed == -1:
            fname += time.strftime('_%Y-%m-%d-%H-%M-%S')
        else:
            fname += f"_seed{args.np_seed}"
        return fname

    @abstractmethod
    def reset_sample(self, rng: np.random.Generator) -> None:
        r""" Reset the current sample of the current PDE. """
        self.u_sol = None

    def gen_data_all(self,
                     rng: np.random.Generator,
                     print_fn: Optional[Callable] = None) -> None:
        i_pde = 0
        num_sample_per_pde = self.num_sample_per_pde
        max_trial_sample = 2 * num_sample_per_pde
        while self.n_data < self.num_pde:
            i_pde += 1
            self.reset_pde(rng)
            self.print_current_coef(print_fn)

            u_sol_pde_all = []
            for i_sample in range(max_trial_sample):
                if (1 + len(u_sol_pde_all)) / (1 + i_sample) < 0.25:
                    break  # discard current PDE with low acceptance ratio
                samples_required_left = num_sample_per_pde - len(u_sol_pde_all)
                samples_trial_left = max_trial_sample - i_sample
                if samples_required_left > samples_trial_left:
                    break  # discard current PDE
                if callable(print_fn):
                    print_fn(f"PDE {i_pde} "
                             f"({self.n_data}/{self.num_pde}), "
                             f"sample {i_sample} "
                             f"({len(u_sol_pde_all)}/{num_sample_per_pde})")

                self.reset_sample(rng)
                self.gen_solution()
                if self._accept_u_sol(print_fn):
                    if self.plot_u:
                        self.plot()
                    u_sol_pde_all.append(self.u_sol)
                if len(u_sol_pde_all) == num_sample_per_pde:
                    u_sol_pde_all = np.array(u_sol_pde_all)
                    self._record_solution(u_sol_pde_all)
                    break  # next PDE


def create_logger(path: str = "./log.log") -> logging.RootLogger:
    r""" save the logger information to a file specified by `path` """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = path
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d]"
                                  " - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)

    return logger


def gen_data(args: argparse.Namespace, pde_data_gen: PDEDataGenBase) -> None:
    r""" main data generation process """
    # random number generator
    if args.np_seed == -1:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(args.np_seed)
        np.random.seed(args.np_seed)  # for spm1d.rft1d

    # logger
    os.makedirs("log", exist_ok=True)
    fname = pde_data_gen.get_hdf5_file_name(args)
    logger = create_logger(os.path.join("log", fname + ".log"))
    logger.info("target file: %s.hdf5", fname)

    # generate data and save
    pde_data_gen.gen_data_all(rng, print_fn=logger.info)
    pde_data_gen.save_hdf5(args)
    logger.info("file saved: %s.hdf5", fname)
