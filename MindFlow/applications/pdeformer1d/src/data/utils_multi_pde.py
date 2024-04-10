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
r"""Utilities for the custom multi_pde dataset."""
import os
import time
from functools import lru_cache
from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, List, Dict

import numpy as np
from numpy.typing import NDArray
import h5py
from omegaconf import DictConfig

from .env import float_dtype, int_dtype
from .pde_dag import PDENodesCollector, PDENode, NULL_NODE

DAG_INFO_DIR = "dag_info_v4.1"


def dag_info_file_path(config: DictConfig, filename: str) -> str:
    r"""
    The custom multi_pde dataset requires preprocessing, in which we construct
    the DAG representation of each PDE in the dataset, and save the graph data
    into a separate HDF5 file. This function returns the path of this file
    containing DAG information.
    """
    uf_num_mod = config.model.inr.num_layers - 1
    max_n_scalar_nodes = config.data.pde_dag.max_n_scalar_nodes
    max_n_function_nodes = config.data.pde_dag.max_n_function_nodes
    function_num_branches = config.model.function_encoder.num_branches
    suffix = (f"_inr{uf_num_mod}sc{max_n_scalar_nodes}"
              f"func{max_n_function_nodes}x{function_num_branches}.hdf5")
    return os.path.join(config.data.path, DAG_INFO_DIR, filename + suffix)


class PDETermBase(ABC):
    r""" Abstract basic class of a PDE term. """

    @abstractmethod
    def __init__(self, hdf5_group, idx_pde: int, keep_zero_coef: bool) -> None:
        self.keep_zero_coef = keep_zero_coef

    @abstractmethod
    def nodes(self, pde: PDENodesCollector) -> Union[float, PDENode]:
        r"""Generate the node representation for this PDE term."""
        return 0.

    @abstractmethod
    def latex(self) -> Tuple:
        r"""
        Generate the mathematical representation (a list of terms represented
        in LaTeX) as well as a list of coefficient values for this PDE term.
        """
        term_list = []  # List[str]
        coef_list = []  # List[Concatenate[str, float]]
        return term_list, coef_list


class FieldCoef(PDETermBase):
    r"""Coefficient term involved in a PDE, which can be a zero, a real number (scalar) or a spatial-varying field."""
    ZERO_COEF = 0
    SCALAR_COEF = 1
    FIELD_COEF = 2

    def __init__(self, hdf5_group, idx_pde: int, keep_zero_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_zero_coef)
        self.coef_type = hdf5_group["coef_type_all"][idx_pde]
        self.field = hdf5_group["field_all"][idx_pde]

    def nodes(self,  # pylint: disable=W0221
              pde: PDENodesCollector,
              x_coord: NDArray[float]) -> PDENode:
        if self.coef_type == self.ZERO_COEF:
            if self.keep_zero_coef:
                return pde.new_coef(0.)
            return NULL_NODE
        if self.coef_type == self.SCALAR_COEF:
            return pde.new_coef(self.field[0])
        if self.coef_type == self.FIELD_COEF:
            return pde.new_coef_field(x_coord, self.field)
        raise NotImplementedError

    def latex(self, symbol: str) -> Tuple:  # pylint: disable=W0221
        term_list = []
        coef_list = []
        if self.coef_type == self.ZERO_COEF and self.keep_zero_coef:
            term_list.append(symbol)
            coef_list = [(symbol, 0.)]
        elif self.coef_type == self.SCALAR_COEF:
            term_list.append(symbol)
            coef_list = [(symbol, self.field[0])]
        elif self.coef_type == self.FIELD_COEF:
            term_list.append(symbol + "(x)")
        return term_list, coef_list


class BoundaryCondition(PDETermBase):
    r"""Non-periodic boundary condition of a PDE, which can be Dirichlet, Neumann or Robin."""
    DIRICHLET = 0
    NEUMANN = 1
    ROBIN = 2

    def __init__(self, hdf5_group, idx_pde: int, keep_zero_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_zero_coef)
        self.bc_type = hdf5_group["bc_type_all"][idx_pde]
        self.u_coef = hdf5_group["u_coef_all"][idx_pde]
        self.dx_u_coef = hdf5_group["dx_u_coef_all"][idx_pde]
        self.bc_val = hdf5_group["bc_val_all"][idx_pde]

    def nodes(self, u_node: PDENode, get_dx_u: Callable) -> PDENode:  # pylint: disable=W0221
        if self.bc_type == self.DIRICHLET:
            return u_node
        if self.bc_type == self.NEUMANN:
            return get_dx_u()
        if self.bc_type == self.ROBIN:
            return self.u_coef * u_node + self.dx_u_coef * get_dx_u()
        raise NotImplementedError

    def latex(self, suffix: str, u_symbol: str = "u") -> Tuple:  # pylint: disable=W0221
        coef_list = []
        if self.bc_type == self.DIRICHLET:
            lhs = u_symbol
        elif self.bc_type == self.NEUMANN:
            lhs = u_symbol + "_x"
        elif self.bc_type == self.ROBIN:
            lhs = rf"(\alpha{suffix}{u_symbol}+\beta{suffix}{u_symbol}_x)"
            coef_list.append((rf"\alpha{suffix}", self.u_coef))
            coef_list.append((rf"\beta{suffix}", self.dx_u_coef))

        if self.bc_val != 0 or self.keep_zero_coef:
            rhs = rf"\gamma{suffix}"
            coef_list.append((rhs, self.bc_val))
        else:
            rhs = "0"

        term_list = [lhs, rhs]
        return term_list, coef_list


class SinusoidalTermFi(PDETermBase):
    r"""
    The term $f_0(u)$ or $f_1(u)$ in the PDE, in the form
    $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
             + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$.
    """

    def __init__(self, hdf5_group, idx_pde: int, keep_zero_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_zero_coef)
        self.coef = hdf5_group[idx_pde]

    def nodes(self,  # pylint: disable=W0221
              pde: PDENodesCollector,
              u_node: PDENode,
              get_u2: Callable,
              get_u3: Callable) -> List[PDENode]:
        coef = self.coef
        sum_list = []

        def u_pow(k):
            if k == 1:
                return u_node
            if k == 2:
                return get_u2()
            if k == 3:
                return get_u3()
            raise NotImplementedError

        # polynomial part
        for k in range(1, 4):
            if coef[0, k] != 0 or self.keep_zero_coef:
                sum_list.append(coef[0, k] * u_pow(k))

        # sinusoidal part
        for j in range(1, coef.shape[0]):
            if coef[j, 0] == 0:
                # ignore this sinusoidal term even when keep_zero_coef=True
                continue
            sinus_sum_list = []
            for k in [1, 2]:
                if coef[j, k] != 0 or self.keep_zero_coef:
                    sinus_sum_list.append(coef[j, k] * u_pow(k))
            op_j = pde.sum(*sinus_sum_list)
            if coef[j, 3] > 0:
                gj_op = pde.sin(op_j)
            else:
                gj_op = pde.cos(op_j)
            sum_list.append(coef[j, 0] * gj_op)

        return sum_list

    def latex(self, i: Union[int, str] = "",  # pylint: disable=W0221
              u_symbol: str = "u") -> Tuple:
        coef = self.coef
        sum_list = []
        coef_list = []

        def u_pow(k):
            if k == 1:
                return u_symbol
            return f"{u_symbol}^{k}"

        # polynomial part
        j = 0 if coef.shape[0] > 1 else ""
        for k in range(1, 4):
            if coef[0, k] != 0 or self.keep_zero_coef:
                coef_latex = f"c_{{{i}{j}{k}}}"
                sum_list.append(coef_latex + u_pow(k))
                coef_list.append((coef_latex, coef[0, k]))

        # sinusoidal part
        for j in range(1, coef.shape[0]):
            if coef[j, 0] == 0:
                continue
            sinus_sum_list = []
            for k in [1, 2]:
                if coef[j, k] != 0 or self.keep_zero_coef:
                    coef_latex = f"c_{{{i}{j}{k}}}"
                    sinus_sum_list.append(coef_latex + u_pow(k))
                    coef_list.append((coef_latex, coef[j, k]))
            op_j = "+".join(sinus_sum_list)
            gj_latex = r"\sin(" if coef[j, 3] > 0 else r"\cos("
            coef_latex = f"c_{{{i}{j}0}}"
            sum_list.append(coef_latex + gj_latex + op_j + ")")
            coef_list.append((coef_latex, coef[j, 0]))

        return sum_list, coef_list


def sinusoidal_pde_nodes(h5file_in,
                         idx_pde: int,
                         keep_zero_coef: bool,
                         inverse_problem: bool = False) -> PDENodesCollector:
    r"""
    Generate DAG nodes for the PDE with sinusoidal terms.
    The PDE takes the form $u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x)u_x)_x=0$,
        $(t,x)\in[0,1]\times[-1,1]$,
    where $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                   + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$.
    Here, the sinusoidal function $h_{ij}\in\{\sin,\cos\}$ is selected
        with equal probability, $J_0+J_1=J$ with $J_0\in\{0,1,\dots,J\}$
        selected with equal probability.
    Unless when periodic boundary is specified, the boundary condition at each
    endpoint is randomly selected from Dirichlet, Neumann and Robin types.
    """
    pde = PDENodesCollector()
    u_node = pde.new_uf()

    # These nodes will be created only when used, and at most once.
    @lru_cache(maxsize=1)
    def get_dx_u():
        return pde.dx(u_node)

    @lru_cache(maxsize=1)
    def get_u2():
        return pde.square(u_node)

    @lru_cache(maxsize=1)
    def get_u3():
        return u_node * get_u2()

    # initial condition
    x_coord = h5file_in["x_coord"][:]
    if inverse_problem:
        # the real initial condition will be introduced in 'load_inverse_data'
        ic_field = np.zeros_like(x_coord) + np.nan
    else:
        ic_field = h5file_in["u_sol_all"][idx_pde, 0, :, 0]
    pde.set_ic(u_node, x_coord, ic_field)

    # diffusion term
    kappa = FieldCoef(h5file_in["coef/kappa"], idx_pde, keep_zero_coef)
    kappa_node = kappa.nodes(pde, x_coord)
    flux_sum_list = []
    if kappa_node is not NULL_NODE:
        flux_sum_list.append(-(kappa_node * get_dx_u()))
    elif inverse_problem:  # keep the kappa node even with zero value
        flux_sum_list.append(-(pde.new_coef(0) * get_dx_u()))

    # PDE term f_0(u) and f_1(u)
    f0_term = SinusoidalTermFi(
        h5file_in["coef/f0_all"], idx_pde, keep_zero_coef)
    f1_term = SinusoidalTermFi(
        h5file_in["coef/f1_all"], idx_pde, keep_zero_coef)
    f0_sum_list = f0_term.nodes(pde, u_node, get_u2, get_u3)
    f1_sum_list = f1_term.nodes(pde, u_node, get_u2, get_u3)
    flux_sum_list.extend(f1_sum_list)

    # source term
    s_term = FieldCoef(h5file_in["coef/s"], idx_pde, keep_zero_coef)
    s_node = s_term.nodes(pde, x_coord)

    # main PDE terms
    flux = pde.sum(flux_sum_list)  # may be NULL_NODE when flux_sum_list == []
    sum_list = [pde.dt(u_node), pde.dx(flux), s_node] + f0_sum_list
    pde.sum_eq0(sum_list)

    # boundary condition
    periodic = np.array(h5file_in["pde_info/args/periodic"])
    if not periodic:
        bc_l = BoundaryCondition(
            h5file_in["coef/bc_l"], idx_pde, keep_zero_coef)
        pde.set_bv_l(bc_l.nodes(u_node, get_dx_u), bc_l.bc_val)
        bc_r = BoundaryCondition(
            h5file_in["coef/bc_r"], idx_pde, keep_zero_coef)
        pde.set_bv_r(bc_r.nodes(u_node, get_dx_u), bc_r.bc_val)

    return pde


def sinusoidal_pde_latex(h5file_in, idx_pde: int, keep_zero_coef: bool) -> Tuple:
    r"""
    Generate LaTeX expression for the PDE with sinusoidal terms.
    The PDE takes the form $u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x)u_x)_x=0$,
        $(t,x)\in[0,1]\times[-1,1]$,
    where $f_i(u) = \sum_{k=1}^3c_{i0k}u^k
                   + \sum_{j=1}^{J_i}c_{ij0}h_{ij}(c_{ij1}u+c_{ij2}u^2)$.
    Here, the sinusoidal function $h_{ij}\in\{\sin,\cos\}$ is selected
        with equal probability, $J_0+J_1=J$ with $J_0\in\{0,1,\dots,J\}$
        selected with equal probability.
    Unless when periodic boundary is specified, the boundary condition at each
    endpoint is randomly selected from Dirichlet, Neumann and Robin types.
    """
    # diffusion term
    kappa = FieldCoef(h5file_in["coef/kappa"], idx_pde, keep_zero_coef)
    kappa_term, kappa_coefs = kappa.latex(r"\kappa")
    flux_term_list = []
    if len(kappa_term) == 1:
        flux_term_list.append(rf"-{kappa_term[0]} u_x")

    # PDE term f_0(u) and f_1(u)
    f0_term = SinusoidalTermFi(
        h5file_in["coef/f0_all"], idx_pde, keep_zero_coef)
    f1_term = SinusoidalTermFi(
        h5file_in["coef/f1_all"], idx_pde, keep_zero_coef)
    f0_term_list, f0_coefs = f0_term.latex(0)
    f1_term_list, f1_coefs = f1_term.latex(1)
    flux_term_list.extend(f1_term_list)

    # source term
    s_term = FieldCoef(h5file_in["coef/s"], idx_pde, keep_zero_coef)
    s_term, s_coefs = s_term.latex("s")

    term_list = ["u_t"] + f0_term_list + s_term
    if len(flux_term_list) > 0:  # pylint: disable=C1801
        flux_term = "+".join(flux_term_list).replace("+-", "-")
        term_list.append(rf"({flux_term})_x")
    pde_latex = "$" + "+".join(term_list) + "=0$"
    coef_list = kappa_coefs + f0_coefs + f1_coefs + s_coefs

    # boundary condition
    if not np.array(h5file_in["pde_info/args/periodic"]):
        bc_l = BoundaryCondition(
            h5file_in["coef/bc_l"], idx_pde, keep_zero_coef)
        (bcll, bclr), bcl_coef = bc_l.latex(r"_\mathrm{L}")
        bc_r = BoundaryCondition(
            h5file_in["coef/bc_r"], idx_pde, keep_zero_coef)
        (bcrl, bcrr), bcr_coef = bc_r.latex(r"_\mathrm{R}")
        pde_latex += "\n"rf"${bcll}(t,-1)={bclr},\ {bcrl}(t,1)={bcrr}$"
        coef_list = bcl_coef + bcr_coef + coef_list

    return pde_latex, coef_list


class SparseCOOCoef(PDETermBase):
    r"""Sparse coefficient matrices/tensors in the COOrdinate format for the multi-component PDEs."""

    def __init__(self, hdf5_group, idx_pde: int, is_3d: bool = False) -> None:
        super().__init__(hdf5_group, idx_pde, keep_zero_coef=False)
        self.is_3d = is_3d
        coo_len = hdf5_group["coo_len_all"][idx_pde]
        self.coo_i = hdf5_group["coo_i_all"][idx_pde, :coo_len]
        self.coo_j = hdf5_group["coo_j_all"][idx_pde, :coo_len]
        self.coo_vals = hdf5_group["coo_vals_all"][idx_pde, :coo_len]
        if is_3d:
            self.coo_k = hdf5_group["coo_k_all"][idx_pde, :coo_len]

    def nodes(self, pde: PDENodesCollector,  # pylint: disable=W0221
              u_list: List[PDENode]) -> List[List[PDENode]]:
        node_lists = [[] for _ in u_list]
        if self.is_3d:
            u_square_list = [NULL_NODE for _ in u_list]
            for (i, j, k, value) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
                if j == k:
                    if u_square_list[j] is NULL_NODE:
                        u_square_list[j] = pde.square(u_list[j])
                    node_lists[i].append(value * u_square_list[j])
                else:
                    node_lists[i].append(pde.prod(value, u_list[j], u_list[k]))
        else:
            for (i, j, value) in zip(self.coo_i, self.coo_j, self.coo_vals):
                node_lists[i].append(value * u_list[j])
        return node_lists

    def latex(self, symbol: str, n_vars: int) -> Tuple:  # pylint: disable=W0221
        sum_lists = [[] for _ in range(n_vars)]
        coef_list = []
        if self.is_3d:
            for (i, j, k, value) in zip(self.coo_i, self.coo_j, self.coo_k, self.coo_vals):
                coef_latex = f"{symbol}_{{{i}{j}{k}}}"
                if j == k:
                    sum_lists[i].append(f"{coef_latex}u_{j}^2")
                else:
                    sum_lists[i].append(f"{coef_latex}u_{j}u_{k}")
                coef_list.append((coef_latex, value))
        else:
            for (i, j, value) in zip(self.coo_i, self.coo_j, self.coo_vals):
                coef_latex = f"{symbol}_{{{i}{j}}}"
                sum_lists[i].append(f"{coef_latex}u_{j}")
                coef_list.append((coef_latex, value))
        return sum_lists, coef_list


def multi_component_pde_nodes(h5file_in,
                              idx_pde: int,
                              keep_zero_coef: bool) -> PDENodesCollector:
    r"""
    Generate DAG nodes for the PDE with multiple components.
    The PDE takes the form
    $\partial_tu_i + \sum_jc_{ij}u_j + s_i
        +\partial_x(\sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_ju_k
            - \kappa_i\partial_xu_i) = 0$,
    where $0 \le i,j,k \le d-1$, $j \le k$, $(t,x)\in[0,1]\times[-1,1]$.
    Periodic boundary condition is employed for simplicity.
    """
    # read h5file_in
    coef_a = SparseCOOCoef(h5file_in["coef/a"], idx_pde)
    coef_b = SparseCOOCoef(h5file_in["coef/b"], idx_pde, is_3d=True)
    coef_c = SparseCOOCoef(h5file_in["coef/c"], idx_pde)
    coef_s = h5file_in["coef/s_all"][idx_pde]  # [n_vars]
    coef_kappa = h5file_in["coef/kappa_all"][idx_pde]  # [n_vars]
    x_coord = h5file_in["x_coord"][:]
    n_vars = h5file_in["u_sol_all"].shape[-1]

    # create nodes
    pde = PDENodesCollector()
    u_list = [pde.new_uf() for _ in range(n_vars)]
    a_u_lists = coef_a.nodes(pde, u_list)
    b_u_lists = coef_b.nodes(pde, u_list)
    c_u_lists = coef_c.nodes(pde, u_list)
    for i in range(n_vars):
        ui_node = u_list[i]

        # initial condition
        ic_field = h5file_in["u_sol_all"][idx_pde, 0, :, i]
        pde.set_ic(ui_node, x_coord, ic_field)

        # flux function
        flux_i_sum_list = a_u_lists[i] + b_u_lists[i]
        if coef_kappa[i] != 0 or keep_zero_coef:
            flux_i_sum_list.append(-(coef_kappa[i] * pde.dx(ui_node)))
        flux_i = pde.sum(flux_i_sum_list)  # may be NULL_NODE

        # remaining terms
        sum_i_list = c_u_lists[i] + [pde.dt(ui_node), pde.dx(flux_i)]
        if coef_s[i] != 0 or keep_zero_coef:
            sum_i_list.append(pde.new_coef(coef_s[i]))
        pde.sum_eq0(sum_i_list)

    return pde


def multi_component_pde_latex(h5file_in,
                              idx_pde: int,
                              keep_zero_coef: bool,
                              idx_var: int = 0) -> Tuple:
    r"""
    Generate LaTeX expression for the PDE with multiple components.
    The PDE takes the form
    $\partial_tu_i + \sum_jc_{ij}u_j + s_i
        +\partial_x(\sum_ja_{ij}u_j + \sum_{j,k}b_{ijk}u_ju_k
            - \kappa_i\partial_xu_i) = 0$,
    where $0 \le i,j,k \le d-1$, $j \le k$, $(t,x)\in[0,1]\times[-1,1]$.
    Periodic boundary condition is employed for simplicity.
    """
    # read h5file_in
    coef_a = SparseCOOCoef(h5file_in["coef/a"], idx_pde)
    coef_b = SparseCOOCoef(h5file_in["coef/b"], idx_pde, is_3d=True)
    coef_c = SparseCOOCoef(h5file_in["coef/c"], idx_pde)
    coef_s = h5file_in["coef/s_all"][idx_pde]  # [n_vars]
    coef_kappa = h5file_in["coef/kappa_all"][idx_pde]  # [n_vars]
    n_vars = h5file_in["u_sol_all"].shape[-1]

    a_u_lists, a_coefs = coef_a.latex("a", n_vars)
    b_u_lists, b_coefs = coef_b.latex("b", n_vars)
    c_u_lists, c_coefs = coef_c.latex("c", n_vars)
    coef_list = a_coefs + b_coefs + c_coefs
    pde_latex_list = []
    for i in range(n_vars):
        # flux function
        flux_i_sum_list = a_u_lists[i] + b_u_lists[i]
        if coef_kappa[i] != 0 or keep_zero_coef:
            flux_i_sum_list.append(rf"-\kappa_{i}\partial_xu_{i}")
            coef_list.append((rf"-\kappa_{i}", coef_kappa[i]))

        # remaining terms
        sum_i_list = [rf"\partial_tu_{i}"] + c_u_lists[i]
        if coef_s[i] != 0 or keep_zero_coef:
            sum_i_list.append(rf"s_{i}")
            coef_list.append((rf"s_{i}", coef_s[i]))
        if len(flux_i_sum_list) > 0:  # pylint: disable=C1801
            flux_i_term = "+".join(flux_i_sum_list).replace("+-", "-")
            sum_i_list.append(rf"\partial_x({flux_i_term})")
        pde_i_latex = "$" + "+".join(sum_i_list) + "=0$"
        pde_latex_list.append(pde_i_latex)

    # The equations is ordered so that the current variable comes the first.
    pde_latex_list = pde_latex_list[idx_var:] + pde_latex_list[:idx_var]
    pde_latex = "\n".join(pde_latex_list)
    return pde_latex, coef_list


class TimeDependentFieldCoef(FieldCoef):
    r"""
    Coefficient term involved in a PDE, which can be a zero, a real number
    (scalar) $c$, a spatially-varying random field $c(x)$, a time-dependent
    scalar $c(t)$, or a time-dependent spatially-varying field $c(t,x)$. For
    the last case, we assume the space and time variables are separable, i.e.
    $c(t,x)=c_T(t)c_X(x)$.
    """
    VARYING_SCALAR_COEF = 3
    VARYING_FIELD_COEF = 4

    def __init__(self,
                 hdf5_group,
                 idx_pde: int,
                 keep_zero_coef: bool,
                 t_min: float = 0.,
                 t_max: float = 1.,
                 n_t_grid: int = 256) -> None:
        super().__init__(hdf5_group, idx_pde, keep_zero_coef)
        self.temporal_coef = hdf5_group["temporal_coef_all"][idx_pde]
        self.t_coord = np.linspace(t_min, t_max, n_t_grid)

    def nodes(self, pde: PDENodesCollector, x_coord: NDArray[float]) -> PDENode:
        if self.coef_type == self.VARYING_SCALAR_COEF:
            return pde.new_varying_coef(self.t_coord, self.temporal_coef)
        if self.coef_type == self.VARYING_FIELD_COEF:
            t_node = pde.new_varying_coef(self.t_coord, self.temporal_coef)
            x_node = pde.new_coef_field(x_coord, self.field)
            return t_node * x_node
        return super().nodes(pde, x_coord)

    def latex(self, symbol: str) -> Tuple:
        coef_list = []
        term_list = []
        if self.coef_type == self.VARYING_SCALAR_COEF:
            term_list.append(symbol + "(t)")
            return term_list, coef_list
        if self.coef_type == self.VARYING_FIELD_COEF:
            term_list.append(symbol + r"^\mathrm{T}(t)" +
                             symbol + r"^\mathrm{X}(x)")
            return term_list, coef_list
        return super().latex(symbol)


class WaveTerm(FieldCoef):
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

    def __init__(self, hdf5_group, idx_pde: int, keep_zero_coef: bool) -> None:
        super().__init__(hdf5_group, idx_pde, keep_zero_coef)
        self.wave_type = hdf5_group["wave_type_all"][idx_pde]

    def nodes(self,  # pylint: disable=W0221
              pde: PDENodesCollector,
              x_coord: NDArray[float],
              get_dx_u: Callable) -> PDENode:
        coef_type = self.coef_type
        if coef_type == self.ZERO_COEF:
            if self.keep_zero_coef:
                # keep this term, treating zero as a special scalar
                coef_type = self.SCALAR_COEF
            else:
                return NULL_NODE

        if coef_type == self.SCALAR_COEF:
            c_or_c2_val = self.field[0]  # scalar
        elif coef_type == self.FIELD_COEF:
            c_or_c2_val = self.field  # np.ndarray

        if self.wave_type == self.FACTORED_FORM:
            c_or_c2_val = np.sqrt(c_or_c2_val)  # c(x) rather than c(x)^2

        # convert float/np.ndarray to PDENode
        if coef_type == self.SCALAR_COEF:
            c_or_c2_node = pde.new_coef(c_or_c2_val)
        elif coef_type == self.FIELD_COEF:
            c_or_c2_node = pde.new_coef_field(x_coord, c_or_c2_val)

        # compute the wave term
        dx_u = get_dx_u()
        if self.wave_type == self.NON_DIV_FORM:
            return -(c_or_c2_node * pde.dx(dx_u))
        if self.wave_type == self.FACTORED_FORM:
            return -(c_or_c2_node * pde.dx(c_or_c2_node * dx_u))
        if self.wave_type == self.DIV_FORM:
            return -pde.dx(c_or_c2_node * dx_u)
        raise NotImplementedError

    def latex(self, symbol: str = "c", u_symbol: str = "u") -> Tuple:  # pylint: disable=W0221
        term_list = []
        coef_list = []

        coef_type = self.coef_type
        if coef_type == self.ZERO_COEF:
            if self.keep_zero_coef:
                # keep this term, treating zero as a special scalar
                coef_type = self.SCALAR_COEF
            else:
                return term_list, coef_list

        if coef_type == self.FIELD_COEF:
            symbol = symbol + r"(x)"
        if self.wave_type != self.FACTORED_FORM:
            symbol = symbol + r"^2"
            if coef_type == self.SCALAR_COEF:
                coef_list.append((symbol, self.field[0]))
        elif coef_type == self.SCALAR_COEF:
            coef_list.append((symbol, np.sqrt(self.field[0])))

        # compute the wave term
        if self.wave_type == self.NON_DIV_FORM:
            term_list.append("-" + symbol + u_symbol + r"_{xx}")
        elif self.wave_type == self.FACTORED_FORM:
            term_list.append(rf"-{symbol}({symbol}{u_symbol}_x)_x")
        elif self.wave_type == self.DIV_FORM:
            term_list.append(rf"-({symbol}{u_symbol}_x)_x")
        return term_list, coef_list


class BoundaryConditionWithMur(BoundaryCondition):
    r"""
    Non-periodic boundary condition of a PDE, which is taken to be Dirichlet,
    Neumann, Robin or Mur with equal probability. The Robin boundary condition
    takes the form $(\alpha u+\beta u_x)(t,x_{L/R})=\gamma$, where the
    coefficients $\alpha,\beta$ are random numbers satisfying
    $\alpha^2+\beta^2=1$ and $\beta>0$. The Mur boundary condition follows the
    form $(u_t+c u_x)(t,x_{L/R})=\gamma$.
    """
    MUR = 3

    def nodes(self,  # pylint: disable=W0221
              u_node: PDENode,
              dt_u: PDENode,
              get_dx_u: Callable) -> PDENode:
        if self.bc_type == self.MUR:
            return dt_u + self.dx_u_coef * get_dx_u()
        return super().nodes(u_node, get_dx_u)

    def latex(self, suffix: str,  # pylint: disable=W0221
              u_symbol: str = "u", c_symbol: str = "c") -> Tuple:
        if self.bc_type != self.MUR:
            return super().latex(suffix, u_symbol=u_symbol)
        c_symbol = c_symbol + suffix
        lhs = rf"({u_symbol}_t+{c_symbol}{u_symbol}_x)"
        coef_list = [(c_symbol, self.dx_u_coef)]

        if self.bc_val != 0 or self.keep_zero_coef:
            rhs = rf"\gamma{suffix}"
            coef_list.append((rhs, self.bc_val))
        else:
            rhs = "0"

        term_list = [lhs, rhs]
        return term_list, coef_list


def wave_eqn_nodes(h5file_in,
                   idx_pde: int,
                   keep_zero_coef: bool,
                   inverse_problem: bool = False) -> PDENodesCollector:
    r"""
    Generate DAG nodes for the wave equation.
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
    pde = PDENodesCollector()
    u_node = pde.new_uf()

    # These nodes will be created only when used, and at most once.
    @lru_cache(maxsize=1)
    def get_dx_u():
        return pde.dx(u_node)

    @lru_cache(maxsize=1)
    def get_u2():
        return pde.square(u_node)

    @lru_cache(maxsize=1)
    def get_u3():
        return u_node * get_u2()

    x_coord = h5file_in["x_coord"][:]
    dt_u = pde.dt(u_node)
    sum_list = [pde.dt(dt_u)]

    # initial condition
    if inverse_problem:
        # the real initial condition will be introduced in 'load_inverse_data'
        empty_field = np.zeros_like(x_coord) + np.nan
        pde.set_ic(u_node, x_coord, empty_field)
        pde.set_ic(dt_u, x_coord, empty_field)
    else:
        ic_field = h5file_in["u_sol_all"][idx_pde, 0, :, 0]
        pde.set_ic(u_node, x_coord, ic_field)
        ut_ic = FieldCoef(h5file_in["coef/ut_ic"], idx_pde, keep_zero_coef)
        pde.set_ic(dt_u, x_coord, ut_ic.field)

    # damping term
    mu_value = h5file_in["coef/mu_all"][idx_pde]
    if mu_value != 0 or keep_zero_coef:
        sum_list.append(mu_value * dt_u)

    # wave term
    wave_term = WaveTerm(h5file_in["coef/Lu"], idx_pde, keep_zero_coef)
    sum_list.append(wave_term.nodes(pde, x_coord, get_dx_u))

    # first-order advection
    b_value = h5file_in["coef/b_all"][idx_pde]
    if b_value != 0 or keep_zero_coef:
        sum_list.append(b_value * get_dx_u())

    # f(u) terms
    f_term = SinusoidalTermFi(h5file_in["coef/f_all"], idx_pde, keep_zero_coef)
    sum_list.extend(f_term.nodes(pde, u_node, get_u2, get_u3))

    # source term
    n_x_grid, = x_coord.shape
    s_term = TimeDependentFieldCoef(
        h5file_in["coef/s"], idx_pde, keep_zero_coef, n_t_grid=n_x_grid)
    sum_list.append(s_term.nodes(pde, x_coord))

    # main PDE terms
    pde.sum_eq0(sum_list)

    # boundary condition
    periodic = np.array(h5file_in["pde_info/args/periodic"])
    if not periodic:
        bc_l = BoundaryConditionWithMur(
            h5file_in["coef/bc_l"], idx_pde, keep_zero_coef)
        pde.set_bv_l(bc_l.nodes(u_node, dt_u, get_dx_u), bc_l.bc_val)
        bc_r = BoundaryConditionWithMur(
            h5file_in["coef/bc_r"], idx_pde, keep_zero_coef)
        pde.set_bv_r(bc_r.nodes(u_node, dt_u, get_dx_u), bc_r.bc_val)

    return pde


def wave_eqn_latex(h5file_in, idx_pde: int, keep_zero_coef: bool) -> Tuple:
    r"""
    Generate LaTeX expression for the wave equation.
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
    term_list = [r"u_{tt}"]
    coef_list = []

    # damping term
    mu_value = h5file_in["coef/mu_all"][idx_pde]
    if mu_value != 0 or keep_zero_coef:
        term_list.append(r"\mu u_t")
        coef_list.append((r"\mu", mu_value))

    # wave term
    wave_term = WaveTerm(h5file_in["coef/Lu"], idx_pde, keep_zero_coef)
    new_terms, new_coefs = wave_term.latex()
    term_list.extend(new_terms)
    coef_list.extend(new_coefs)

    # first-order advection
    b_value = h5file_in["coef/b_all"][idx_pde]
    if b_value != 0 or keep_zero_coef:
        term_list.append(r"bu_x")
        coef_list.append((r"b", b_value))

    # f(u) terms
    f_term = SinusoidalTermFi(h5file_in["coef/f_all"], idx_pde, keep_zero_coef)
    new_terms, new_coefs = f_term.latex()
    term_list.extend(new_terms)
    coef_list.extend(new_coefs)

    # source term
    n_x_grid, = h5file_in["x_coord"].shape
    s_term = TimeDependentFieldCoef(
        h5file_in["coef/s"], idx_pde, keep_zero_coef, n_t_grid=n_x_grid)
    new_terms, new_coefs = s_term.latex("s")
    term_list.extend(new_terms)
    coef_list.extend(new_coefs)

    # main PDE terms
    pde_latex = "+".join(term_list).replace("+-", "-")
    pde_latex = "$" + pde_latex + "=0$"

    # boundary condition
    periodic = np.array(h5file_in["pde_info/args/periodic"])
    if not periodic:
        bc_l = BoundaryConditionWithMur(
            h5file_in["coef/bc_l"], idx_pde, keep_zero_coef)
        (bcll, bclr), bcl_coefs = bc_l.latex(r"_\mathrm{L}")
        bc_r = BoundaryConditionWithMur(
            h5file_in["coef/bc_r"], idx_pde, keep_zero_coef)
        (bcrl, bcrr), bcr_coefs = bc_r.latex(r"_\mathrm{R}")
        pde_latex += "\n"rf"${bcll}(t,-1)={bclr},\ {bcrl}(t,1)={bcrr}$"
        coef_list.extend(bcl_coefs + bcr_coefs)

    return pde_latex, coef_list


def gen_pde_nodes(h5file_in,
                  idx_pde: int,
                  keep_zero_coef: bool = False) -> PDENodesCollector:
    r"""Generate DAG nodes for the PDE."""
    pde_type_id = np.array(h5file_in["pde_info/pde_type_id"]).item()
    if pde_type_id == 1:
        return sinusoidal_pde_nodes(
            h5file_in, idx_pde, keep_zero_coef, inverse_problem=False)
    if pde_type_id == 2:
        return sinusoidal_pde_nodes(
            h5file_in, idx_pde, keep_zero_coef, inverse_problem=True)
    if pde_type_id == 3:
        return multi_component_pde_nodes(h5file_in, idx_pde, keep_zero_coef)
    if pde_type_id == 4:
        return wave_eqn_nodes(
            h5file_in, idx_pde, keep_zero_coef, inverse_problem=False)
    if pde_type_id == 5:
        return wave_eqn_nodes(
            h5file_in, idx_pde, keep_zero_coef, inverse_problem=True)
    raise NotImplementedError


def get_pde_latex(h5file_in,
                  idx_pde: int,
                  idx_var: int = 0,
                  keep_zero_coef: bool = False) -> Tuple:
    r"""Generate LaTeX expression for the PDE."""
    pde_type_id = np.array(h5file_in["pde_info/pde_type_id"]).item()
    if pde_type_id in [1, 2]:
        return sinusoidal_pde_latex(h5file_in, idx_pde, keep_zero_coef)
    if pde_type_id == 3:
        return multi_component_pde_latex(
            h5file_in, idx_pde, keep_zero_coef, idx_var)
    if pde_type_id in [4, 5]:
        return wave_eqn_latex(h5file_in, idx_pde, keep_zero_coef)
    raise NotImplementedError


def gen_dag_info(filename: str, config: DictConfig) -> None:
    r"""Precompute Graphormer input (PDE DAG info) for a single data file, and save the results to a HDF5 file."""
    n_scalar_nodes = config.data.pde_dag.max_n_scalar_nodes
    n_function_nodes = config.data.pde_dag.max_n_function_nodes
    function_num_branches = config.model.function_encoder.num_branches
    n_node = n_scalar_nodes + n_function_nodes * function_num_branches

    # h5file_in
    u_filepath = os.path.join(config.data.path, filename + ".hdf5")
    h5file_in = h5py.File(u_filepath, "r")

    preprocess_dag = np.array(h5file_in["pde_info/preprocess_dag"]).item()

    # target DAG data file
    dag_filepath = dag_info_file_path(config, filename)
    if os.path.exists(dag_filepath) or not preprocess_dag:
        # no need to (re)generate DAG file
        h5file_in.close()
        return
    print(time.strftime("%H:%M:%S") + " generating " + dag_filepath)

    # data to be saved
    # Shape is [n_pde, n_t_grid, n_x_grid, n_vars].
    n_pde, _, n_x_grid, _ = h5file_in["u_sol_all"].shape
    node_type_all = np.zeros([n_pde, n_node, 1], dtype=int_dtype)
    node_scalar_all = np.zeros([n_pde, n_scalar_nodes, 1], dtype=float_dtype)
    node_function_all = np.zeros([n_pde, n_function_nodes, n_x_grid, 2],
                                 dtype=float_dtype)
    spatial_pos_all = np.zeros([n_pde, n_node, n_node], dtype=np.uint8)
    in_degree_all = np.zeros([n_pde, n_node], dtype=int_dtype)
    out_degree_all = np.zeros([n_pde, n_node], dtype=int_dtype)

    for idx_pde in range(n_pde):
        pde = gen_pde_nodes(h5file_in, idx_pde)
        pde_dag = pde.gen_dag(config)

        node_type_all[idx_pde] = pde_dag.node_type
        node_scalar_all[idx_pde] = pde_dag.node_scalar
        node_function_all[idx_pde] = pde_dag.node_function
        spatial_pos_all[idx_pde] = pde_dag.spatial_pos
        in_degree_all[idx_pde] = pde_dag.in_degree
        out_degree_all[idx_pde] = pde_dag.out_degree

    h5file_in.close()
    with h5py.File(dag_filepath, "w") as h5file_out:
        h5file_out.create_dataset("node_type_all", data=node_type_all)
        h5file_out.create_dataset("node_scalar_all", data=node_scalar_all)
        h5file_out.create_dataset("node_function_all", data=node_function_all)
        h5file_out.create_dataset("spatial_pos_all", data=spatial_pos_all)
        h5file_out.create_dataset("in_degree_all", data=in_degree_all)
        h5file_out.create_dataset("out_degree_all", data=out_degree_all)


def datafiles_from_dict(datafile_dict: Dict[str, List[str]]) -> List[str]:
    r"""Get a list consisting of all the data files in a dict."""
    # {pde_type: [file]} -> [file]
    file_all = []
    for file_list in datafile_dict.values():
        file_all.extend(file_list)
    return file_all


def gen_dag_info_all(config: DictConfig) -> None:
    r"""
    Precompute Graphormer input (PDE DAG info) for all custom multi_pde data
    files, and save the results to the corresponding HDF5 files.
    """
    os.makedirs(os.path.join(config.data.path, DAG_INFO_DIR), exist_ok=True)
    file_list = (datafiles_from_dict(config.data.multi_pde.train)
                 + datafiles_from_dict(config.data.multi_pde.get("test", {})))
    for filename in file_list:
        gen_dag_info(filename, config)
    print("All auxiliary DAG data generated.")
