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
r"""PDEformer inference on a given PDE."""
from typing import List, Union
import numpy as np
from numpy.typing import NDArray
from omegaconf import DictConfig
from mindspore import Tensor

from .data.pde_dag import PDEAsDAG, PDENodesCollector
from .utils.visual import plot_infer_result
from .cell import PDEformer


def inference_pde(model: PDEformer,
                  pde_dag: PDEAsDAG,
                  t_coord: NDArray[float],
                  x_coord: NDArray[float]) -> NDArray[float]:
    r"""
    Use the PDEformer model to predict the solution of the PDE as specified in
    `pde_dag`.
    """
    # coordinate
    n_t_grid, = t_coord.shape
    n_x_grid, = x_coord.shape
    x_extend, t_extend = np.meshgrid(x_coord, t_coord)
    # We have t_x_coord[idx_t, idx_x, :] == [t[idx_t], x[idx_x]]
    t_x_coord = np.dstack((t_extend, x_extend)).astype(np.float32)
    coordinate = t_x_coord.reshape((n_t_grid * n_x_grid, 2))

    def as_tensor(array):
        return Tensor(array).expand_dims(0)  # [*] -> [1, *]

    # inference the first PDE component
    pred = model(as_tensor(pde_dag.node_type), as_tensor(pde_dag.node_scalar),
                 as_tensor(pde_dag.node_function), as_tensor(pde_dag.in_degree),
                 as_tensor(pde_dag.out_degree), as_tensor(pde_dag.attn_bias),
                 as_tensor(pde_dag.spatial_pos), as_tensor(coordinate))
    pred = pred.asnumpy().astype(np.float32)  # [1, n_pts, 1]

    # multi-component case, inference the rest components
    if pde_dag.n_vars > 1:
        pred_all = [pred]
        # iterate over all remaining components
        for idx_var in range(1, pde_dag.n_vars):
            spatial_pos, attn_bias = pde_dag.get_spatial_pos_attn_bias(idx_var)
            pred = model(
                as_tensor(pde_dag.node_type), as_tensor(pde_dag.node_scalar),
                as_tensor(pde_dag.node_function), as_tensor(pde_dag.in_degree),
                as_tensor(pde_dag.out_degree), as_tensor(attn_bias),
                as_tensor(spatial_pos), as_tensor(coordinate))
            pred = pred.asnumpy().astype(np.float32)  # [1, n_pts, 1]
            pred_all.append(pred)
        pred = np.concatenate(pred_all, axis=-1)  # [1, n_pts, n_vars]

    pred = pred.reshape((n_t_grid, n_x_grid, pde_dag.n_vars))
    return pred  # [n_t_grid, n_x_grid, n_vars]


def generate_plot(model: PDEformer,
                  config: DictConfig,
                  u_ic: NDArray[float],
                  x_coord: NDArray[float],
                  t_coord: NDArray[float],
                  c_list: List[float],  # shape (6,), range [-3, 3]
                  kappa: Union[float, NDArray[float]],
                  source: Union[float, NDArray[float]] = 0.,
                  periodic: bool = True,
                  robin_theta_l: float = 0.,  # range [0, np.pi]
                  bc_value_l: float = 0.,  # range [-3, 3]
                  robin_theta_r: float = 0.,
                  bc_value_r: float = 0.,
                  figure=None,
                  canvas=None,
                  ic_latex: str = None,) -> None:
    r"""
    Generate a plot for the predicted solution, in which the PDE takes the form
    $u_t+f_0(u)+s(x)+(f_1(u)-\kappa(x) u_x)_x=0$, $(t,x)\in[0,1]\times[-1,1]$,
    $u(0,x)=g(x)$, where $f_i(u) = c_{i1}u + c_{i2}u^2 + c_{i3}u^3$.
    When 'periodic' is False, the boundary condition is specified as
    $(\cos(\pi \theta_L)u + \sin(\pi \theta_R)u_x)|_{x_L} = \gamma_L$,
    $(\cos(\pi \theta_R)u + \sin(\pi \theta_R)u_x)|_{x_R} = \gamma_R$.
    """
    c_01, c_02, c_03, c_11, c_12, c_13 = c_list

    pde = PDENodesCollector()
    u_node = pde.new_uf()

    # specify the initial condition g(x)
    pde.set_ic(u_node, x_coord, u_ic)

    # specify the PDE form
    u_square = pde.square(u_node)
    u_cubic = u_node * u_square
    u_dx = u_node.dx

    if np.isscalar(kappa):
        kappa_node = kappa
    else:
        kappa_node = pde.new_coef_field(x_coord, kappa)
    if np.isscalar(source):
        source_node = source
    else:
        source_node = pde.new_coef_field(x_coord, source)

    f1_node = pde.sum(c_11 * u_node,
                      c_12 * u_square,
                      c_13 * u_cubic,
                      pde.neg(kappa_node * u_node.dx))
    pde.sum_eq0(pde.dt(u_node),
                pde.dx(f1_node),
                c_01 * u_node,
                c_02 * u_square,
                c_03 * u_cubic,
                source_node)

    if not periodic:
        if np.abs(np.sin(np.pi * robin_theta_l)) < 1e-5:
            pde.set_bv_l(u_node, bc_value_l)
        elif np.abs(np.cos(np.pi * robin_theta_l)) < 1e-5:
            pde.set_bv_l(u_dx, bc_value_l)
        else:
            pde.set_bv_l(np.cos(np.pi * robin_theta_l) * u_node + np.sin(np.pi * robin_theta_l) * u_dx,
                         bc_value_l)
        if np.abs(np.sin(np.pi * robin_theta_r)) < 1e-5:
            pde.set_bv_r(u_node, bc_value_r)
        elif np.abs(np.cos(np.pi * robin_theta_r)) < 1e-5:
            pde.set_bv_r(u_dx, bc_value_r)
        else:
            pde.set_bv_r(np.cos(np.pi * robin_theta_r) * u_node + np.sin(np.pi * robin_theta_r) * u_dx,
                         bc_value_r)

    # generate predicted solution using PDEformer
    pde_dag = pde.gen_dag(config)
    u_pred = inference_pde(model, pde_dag, t_coord, x_coord)

    # plot the results
    def format_term(coeff, power, is_kappa=False, is_source=False):
        if coeff == 0 and not is_kappa and not is_source:
            return ""
        formatted_coeff = f"{abs(coeff):.3g}"
        if coeff < 0:
            sign = "-"
        else:
            sign = "+" if not is_kappa else ""
        if is_kappa or is_source:
            return f"{sign}{formatted_coeff}"
        if power == 1:
            return f" {sign} {formatted_coeff}u"
        return f" {sign} {formatted_coeff}u^{power}"

    def format_title(f0_terms, f1_terms, kappa_term, source_term):
        f0_formatted = f"({f0_terms})" if f0_terms.strip() else ""
        kappa_formatted = f"- {kappa_term} u_{{x}}" if kappa_term != "0" else ""
        f1_formatted = f"({f1_terms} {kappa_formatted})" if f1_terms.strip() else ""

        title_parts = list(
            filter(None, [f0_formatted, f"{f1_formatted}_x" if f1_formatted else ""]))

        title = " + ".join(title_parts)

        # Adjust here to include s(x) in the equation
        equation = f"{title} {source_term} = 0" if any(title_parts) else f"{kappa_formatted} {source_term}= 0"

        return rf"$u_t + {equation}$"

    f0_components = [format_term(c_01, 1),
                     format_term(c_02, 2),
                     format_term(c_03, 3)]
    f1_components = [format_term(c_11, 1),
                     format_term(c_12, 2),
                     format_term(c_13, 3)]
    f0_terms = ''.join(filter(None, f0_components)).lstrip(" +")
    f1_terms = ''.join(filter(None, f1_components)).lstrip(" +")

    if np.isscalar(source):
        source_term = format_term(source, 1, is_source=True)
    else:
        source_term = r"+s(x)"
    if np.isscalar(kappa):
        kappa_term = format_term(kappa, 1, is_kappa=True)
    else:
        kappa_term = r"\kappa(x)"
    title = format_title(f0_terms, f1_terms, kappa_term, source_term)

    if ic_latex is not None:
        ic_title = "$u(x,0) = " + ic_latex + "$"

    if periodic:
        plot_infer_result(u_pred, x_coord, t_coord, figure=figure, canvas=canvas,
                          periodic=periodic, title_list=[title, ic_title] if ic_latex is not None else [title])
    else:
        if np.abs(np.sin(np.pi * robin_theta_l)) < 1e-5:
            bc_l_title = "$u|_{x=-1} = " + f"{bc_value_l / np.cos(np.pi * robin_theta_l)}$"
        elif np.abs(np.cos(np.pi * robin_theta_l)) < 1e-5:
            bc_l_title = "$u_x|_{x=-1} = " + f"{bc_value_l / np.sin(np.pi * robin_theta_l)}$"
        else:
            bc_l_title = f"$({np.cos(np.pi * robin_theta_l):.3f}u + " \
                + f"{np.sin(np.pi * robin_theta_l):.3f}u_x)|_" \
                + "{x=-1} = " \
                + f"{bc_value_l}" \
                + "$"
        if np.abs(np.sin(np.pi * robin_theta_r)) < 1e-5:
            bc_r_title = "$u|_{x=1} = " + f"{bc_value_r / np.cos(np.pi * robin_theta_r)}$"
        elif np.abs(np.cos(np.pi * robin_theta_r)) < 1e-5:
            bc_r_title = "$u_x|_{x=1} = " + f"{bc_value_r / np.sin(np.pi * robin_theta_r)}$"
        else:
            bc_r_title = f"$({np.cos(np.pi * robin_theta_r):.3f}u + " \
                + f"{np.sin(np.pi * robin_theta_r):.3f}u_x)|_" \
                + "{x=1} = " \
                + f"{bc_value_r}" \
                + "$"
        bc_title = f"{bc_l_title}, {bc_r_title}"

        plot_infer_result(u_pred, x_coord, t_coord, figure=figure, canvas=canvas,
                          periodic=periodic, title_list=[title, ic_title, bc_title]
                          if ic_latex is not None else [title])



def generate_plot_wave(model: PDEformer,
                       config: DictConfig,
                       u_ic: NDArray[float],
                       ut_ic: NDArray[float],
                       x_coord: NDArray[float],
                       t_coord: NDArray[float],
                       coef_list: List[float],  # shape (4,), range [-3, 3]
                       wave_speed: Union[float, NDArray[float]],  # c(x), range [0, 2]
                       wave_type: int = 0,  # choices: {0, 1, 2}
                       source: Union[float, NDArray[float]] = 0.,
                       mu_value: float = 0.,  # range [0, 4]
                       periodic: bool = True,
                       use_mur_l: bool = False,
                       robin_theta_l: float = 0.,  # range [0, np.pi]
                       bc_value_l: float = 0.,  # range [-3, 3]
                       use_mur_r: bool = False,
                       robin_theta_r: float = 0.,
                       bc_value_r: float = 0.,
                       figure=None,
                       canvas=None,
                       ic_latex: str = None,
                       ic_speed_latex: str = None,) -> None:
    r"""
    Generate a plot for the predicted solution, in which the PDE takes the form
    $u_{tt}+\mu u_t+Lu+bu_x+c_1u+c_2u^2+c^3u^3+s(x)=0$ for
    $(t,x)\in[0,1]\times[-1,1]$, $u(0,x)=g(x)$, $u_t(0,x)=h(x)$. The wave term
    can be the non-divergence form $Lu=-c(x)^2u_{xx}$ (when 'wave_type' is 0),
    the factored form $Lu=-c(x)(c(x)u_x)_x$ (when 'wave_type' is 1), and the
    divergence form $Lu=-(c(x)^2u_x)_x$ (when 'wave_type' is 2).

    When 'periodic' is False, the boundary condition on the left endpoint is
    specified as $(\cos(\theta_L)u + \sin(\theta_R)u_x)|_{x_L} = \gamma_L$ if
    'use_mur_l' is False, and $(u_t - c(x_L)u_x)|_{x_L} = \gamma_L$ otherwise.
    The right boundary condition is similar (except for the Mur boundary now
    takes the form $(u_t + c(x_R)u_x)|_{x_R} = \gamma_R$, which differs in
    sign).
    """
    b_val, c_1, c_2, c_3 = coef_list
    pde = PDENodesCollector()
    u_node = pde.new_uf()

    # specify the initial condition g(x) and h(x)
    pde.set_ic(u_node, x_coord, u_ic)
    pde.set_ic(u_node.dt, x_coord, ut_ic)

    # wave term Lu
    if wave_type not in [0, 1, 2]:
        raise ValueError(f"'wave_type' supports 0, 1, 2, but got {wave_type}.")
    u_dx = u_node.dx
    c_or_c2_val = wave_speed if wave_type == 1 else wave_speed**2
    if np.isscalar(c_or_c2_val):
        c_or_c2_node = pde.new_coef(c_or_c2_val)
    else:
        c_or_c2_node = pde.new_coef_field(x_coord, c_or_c2_val)
    if wave_type == 0:
        wave_node = -(c_or_c2_node * pde.dx(u_dx))
    elif wave_type == 1:
        wave_node = -(c_or_c2_node * pde.dx(c_or_c2_node * u_dx))
    elif wave_type == 2:
        wave_node = -pde.dx(c_or_c2_node * u_dx)  # pylint: disable=E1130

    # specify the PDE form
    u_square = pde.square(u_node)
    u_cubic = u_node * u_square
    u_dt = u_node.dt

    if np.isscalar(source):
        source_node = pde.new_coef(source)
    else:
        source_node = pde.new_coef_field(x_coord, source)

    pde.sum_eq0(u_dt.dt,
                mu_value * u_dt,
                wave_node,
                b_val * u_dx,
                c_1 * u_node,
                c_2 * u_square,
                c_3 * u_cubic,
                source_node)

    # boundary conditions
    def add_bc_nodes(use_mur, robin_theta, bc_value, location="L"):
        if periodic:
            return
        if use_mur:
            c_arr = wave_speed + np.zeros_like(x_coord)  # float -> array
            c_val = -c_arr[0] if location == "L" else c_arr[-1]
            bc_node = u_node.dt + c_val * u_dx
        else:
            if np.abs(np.sin(np.pi * robin_theta)) < 1e-5:
                bc_node = u_node
            elif np.abs(np.cos(np.pi * robin_theta)) < 1e-5:
                bc_node = u_dx
            else:
                bc_node = np.cos(robin_theta) * u_node + np.sin(robin_theta) * u_dx

        if location == "L":
            pde.set_bv_l(bc_node, bc_value)
        else:
            pde.set_bv_r(bc_node, bc_value)

    add_bc_nodes(use_mur_l, robin_theta_l, bc_value_l, "L")
    add_bc_nodes(use_mur_r, robin_theta_r, bc_value_r, "R")

    # generate predicted solution using PDEformer
    pde_dag = pde.gen_dag(config)
    u_pred = inference_pde(model, pde_dag, t_coord, x_coord)

    def format_term(coeff, power, var='u', is_c=False):
        if coeff == 0 and not is_c:
            return ""
        formatted_coeff = f"{abs(coeff):.3g}"
        if coeff < 0:
            sign = "-" if is_c else " -"  # Adjusted here
        else:
            sign = "+" if not is_c else ""
        if var == 'u':
            if power == 1:
                return f"{sign} {formatted_coeff}{var}"
            return f"{sign} {formatted_coeff}{var}^{power}"
        return f"{sign} {formatted_coeff}{var}"

    def format_title(coef_list, mu_value, c_sq_term, s_term):
        b_term, c_terms = coef_list[0], coef_list[1:]
        mu_formatted = format_term(mu_value, 1, var='u_t', is_c=False)
        b_formatted = format_term(b_term, 1, var='u_x', is_c=False)
        if wave_type == 0:
            c_sq_formatted = f"- {c_sq_term}^2u_{{xx}}"
        elif wave_type == 1:
            c_sq_formatted = f"- {c_sq_term}({c_sq_term}u_{{x}})_{{x}}"
        else:
            c_sq_formatted = f"- ({c_sq_term}^2u_{{x}})_{{x}}"
        c_formatted = " ".join([format_term(c_terms[p], p+1) for p in range(len(c_terms))])
        s_formatted = f"+ {s_term}" if s_term.strip() else ""
        title = f"$u_{{tt}} {mu_formatted}{c_sq_formatted}{b_formatted} {c_formatted} {s_formatted}=0$"
        return title

    c_sq_term = 'c(x)'  # For c(x)^2
    if np.isscalar(source):
        if source == 0:
            source_term = ""
        else:
            source_term = str(source)
    else:
        source_term = r"s(x)"

    title = format_title(coef_list, mu_value, c_sq_term, source_term)

    if ic_latex is not None:
        ic_title = "$u(x,0) = " + ic_latex + "$"
    if ic_speed_latex is not None:
        ic_speed_title = "$u_t(x,0) = " + ic_speed_latex + "$"

    if periodic:
        plot_infer_result(u_pred=u_pred,
                          x_coord=x_coord,
                          t_coord=t_coord,
                          figure=figure,
                          canvas=canvas,
                          periodic=periodic,
                          title_list=[title, ic_title, ic_speed_title])
    else:
        if np.abs(np.sin(np.pi * robin_theta_l)) < 1e-5:
            bc_l_title = "$u|_{x=-1} = " + f"{bc_value_l / np.cos(np.pi * robin_theta_l)}$"
        elif np.abs(np.cos(np.pi * robin_theta_l)) < 1e-5:
            bc_l_title = "$u_x|_{x=-1} = " + f"{bc_value_l / np.sin(np.pi * robin_theta_l)}$"
        else:
            bc_l_title = f"$({np.cos(np.pi * robin_theta_l):.3f}u + " \
                + f"{np.sin(np.pi * robin_theta_l):.3f}u_x)|_" \
                + "{x=-1} = " \
                + f"{bc_value_l}" \
                + "$"
        if np.abs(np.sin(np.pi * robin_theta_r)) < 1e-5:
            bc_r_title = "$u|_{x=1} = " + f"{bc_value_r / np.cos(np.pi * robin_theta_r)}$"
        elif np.abs(np.cos(np.pi * robin_theta_r)) < 1e-5:
            bc_r_title = "$u_x|_{x=1} = " + f"{bc_value_r / np.sin(np.pi * robin_theta_r)}$"
        else:
            bc_r_title = f"$({np.cos(np.pi * robin_theta_r):.3f}u + " \
                + f"{np.sin(np.pi * robin_theta_r):.3f}u_x)|_" \
                + "{x=1} = " \
                + f"{bc_value_r}" \
                + "$"
        if use_mur_l:
            bc_l_title = f"$(u_t - c(x)u_x)|_{{x=-1}}={robin_theta_l}$"
        if use_mur_r:
            bc_r_title = f"$(u_t + c(x)u_x)|_{{x=1}}={robin_theta_r}$"
        bc_title = f"{bc_l_title}, {bc_r_title}"

        plot_infer_result(u_pred=u_pred,
                          x_coord=x_coord,
                          t_coord=t_coord,
                          figure=figure,
                          canvas=canvas,
                          periodic=periodic,
                          title_list=[title, ic_title, ic_speed_title, bc_title])
