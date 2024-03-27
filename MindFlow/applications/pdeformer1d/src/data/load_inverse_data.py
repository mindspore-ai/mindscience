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
r"""Loading datasets for inverse problems."""
import os
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import h5py
from omegaconf import DictConfig
from mindspore import Tensor

from .env import float_dtype
from .utils_multi_pde import gen_pde_nodes, get_pde_latex


def get_inverse_data(config: DictConfig, idx_pde: int) -> Tuple:
    r"""
    Generate the data tuple of one PDE for the inverse problems (recovery of
    scalar coefficients or coefficient fields), including multiple initial
    conditions at a time.
    """
    num_ic = config.inverse.num_ic_per_pde  # int
    keep_zero_coef = config.inverse.get("system_identification", False)  # bool
    filename = config.inverse.data_file  # str
    u_filepath = os.path.join(config.data.path, filename + ".hdf5")
    with h5py.File(u_filepath, "r") as h5_file_u:
        x_coord = h5_file_u["x_coord"][:]
        t_coord = h5_file_u["t_coord"][:]
        u_sol_pde_all = h5_file_u["u_sol_all"][idx_pde, :num_ic]

        pde_type_id = np.array(h5_file_u["pde_info/pde_type_id"]).item()
        if pde_type_id == 5:  # inverse problem data for wave equation
            persample_function = h5_file_u["coef/persample_field_all"]
            # Shape is [num_ic, 3, n_x_grid].
            persample_function = persample_function[idx_pde, :num_ic]

        pde_latex, coef_list = get_pde_latex(
            h5_file_u, idx_pde, keep_zero_coef=keep_zero_coef)
        pde = gen_pde_nodes(h5_file_u, idx_pde, keep_zero_coef)

    pde_dag = pde.gen_dag(config)

    def repeat(array):
        # [*] -> [1, *] -> [num_ic, *]
        array = np.expand_dims(array, axis=0)
        return np.repeat(array, num_ic, axis=0)

    # node_function
    # Shape is [num_ic, n_function_nodes, n_x_grid, 2].
    node_function = repeat(pde_dag.node_function)
    ic_all = u_sol_pde_all[:, 0:1, :, 0:1]  # [num_ic, 1, n_x_grid, 1]
    ic_noisy = add_noise_to_label(
        ic_all,
        config.inverse.observation.ic_noise.type.lower(),
        config.inverse.observation.ic_noise.level)
    node_function[:, 0:1, :, 1:2] = ic_noisy
    if pde_type_id == 5:  # inverse problem data for wave equation
        # second axis of node_function: [u_ic, ut_ic, c(x), s^T(t), s^X(x)]
        # initial value of u_t, [num_ic, 1, n_x_grid, 1]
        ut_ic_all = persample_function[:, 0:1, :, np.newaxis]
        ut_ic_noisy = add_noise_to_label(
            ut_ic_all,
            config.inverse.observation.ic_noise.type.lower(),
            config.inverse.observation.ic_noise.level)
        node_function[:, 1:2, :, 1:2] = ut_ic_noisy
        # source term, including temporal_coef s^T(t) and field s^X(x)
        node_function[:, 3:5, :, 1] = persample_function[:, 1:3, :]

    # coordinate, u_label
    x_extend, t_extend = np.meshgrid(x_coord, t_coord)
    # We have t_x_coord[idx_t, idx_x, :] == [t[idx_t], x[idx_x]]
    t_x_coord = np.dstack((t_extend, x_extend))
    n_t_grid, n_x_grid, _ = t_x_coord.shape
    # [n_t_grid, n_x_grid, 2] -> [n_t_grid * n_x_grid, 2]
    coordinate = t_x_coord.reshape((-1, 2)).astype(float_dtype)
    # [num_ic, n_t_grid, n_x_grid, 1] -> [num_ic, n_t_grid * n_x_grid, 1]
    u_label = u_sol_pde_all.reshape((num_ic, -1, 1)).astype(float_dtype)

    # data_tuple
    data_tuple = (repeat(pde_dag.node_type),  # [num_ic, n_node, 1]
                  repeat(pde_dag.node_scalar),  # [num_ic, n_scalar_node, 1]
                  node_function,  # [num_ic, n_function_node, n_x_grid, 2]
                  repeat(pde_dag.in_degree),  # [num_ic, n_node]
                  repeat(pde_dag.out_degree),  # [num_ic, n_node]
                  repeat(pde_dag.attn_bias),  # [num_ic, n_node, n_node]
                  repeat(pde_dag.spatial_pos),  # [num_ic, n_node, n_node]
                  repeat(coordinate),  # [num_ic, n_t_grid * n_x_grid, 2]
                  u_label,  # [num_ic, n_t_grid * n_x_grid, 1]
                  )
    data_tuple = tuple(Tensor(array) for array in data_tuple)

    # data_info
    data_info = {"pde_latex": pde_latex, "coef_list": coef_list,
                 "n_t_grid": n_t_grid, "n_x_grid": n_x_grid,
                 "n_tx_pts": n_t_grid * n_x_grid}
    return data_tuple, data_info


def add_noise_to_label(u_label: NDArray[float],
                       noise_type: str,
                       noise_level: float) -> NDArray[float]:
    r"""Add noise to observation for inverse problems."""
    # [batch_size, n_t_grid, n_x_grid, 1] -> [batch_size, 1, 1, 1]
    u_max = np.abs(u_label).max(axis=(1, 2, 3), keepdims=True)
    if noise_type == 'none':
        noise = 0
    elif noise_type == 'uniform':
        noise = np.random.uniform(low=-u_max, high=u_max, size=u_label.shape)
    elif noise_type == 'normal':
        noise = u_max * np.random.normal(size=u_label.shape)
    else:
        raise NotImplementedError(
            "The noise_type must be in ['none', 'uniform', 'normal'], but got"
            f"' {noise_type}'")
    u_noisy = u_label + noise_level * noise
    return u_noisy


def get_observed_indices(batch_size: int,
                         n_t_grid: int,
                         n_x_grid: int,
                         x_obs_type: str = "all",
                         n_x_obs_pts: int = 10,
                         t_obs_type: str = "all",
                         n_t_obs_pts: int = 10) -> NDArray[int]:
    r"""Generate the indices of the spatial-temporal observation points for inverse problems."""
    obs_inds = np.arange(batch_size * n_t_grid * n_x_grid).reshape(
        (batch_size, n_t_grid, n_x_grid))

    # x locations
    # [bsz, n_t_grid, n_x_grid] -> [bsz, n_t_grid, n_x_obs_pts]
    if x_obs_type == "all":
        n_x_obs_pts = n_x_grid
    elif x_obs_type == "equispaced":
        stride = n_x_grid // n_x_obs_pts
        obs_inds = obs_inds[:, :, ::stride]
        obs_inds = obs_inds[:, :, :n_x_obs_pts]
    elif x_obs_type == "last":
        obs_inds = obs_inds[:, :, -n_x_obs_pts:]
    elif x_obs_type == "random":
        x_inds = np.random.choice(n_x_grid, size=n_x_obs_pts, replace=False)
        obs_inds = obs_inds[:, :, x_inds]
    else:
        raise NotImplementedError(f"Unknown x_obs_type {x_obs_type}.")

    # t locations
    # [bsz, n_t_grid, n_x_obs_pts] -> [bsz, n_t_obs_pts, n_x_obs_pts]
    if t_obs_type == "all":
        n_t_obs_pts = n_t_grid
    elif t_obs_type == "equispaced":
        stride = n_t_grid // n_t_obs_pts
        obs_inds = obs_inds[:, ::stride, :]
        obs_inds = obs_inds[:, -n_t_obs_pts:, :]
    elif t_obs_type == "last":
        obs_inds = obs_inds[:, -n_t_obs_pts:, :]
    elif t_obs_type == "t_random":
        t_inds = np.random.choice(n_t_grid, size=n_t_obs_pts, replace=False)
        obs_inds = obs_inds[:, t_inds, :]
    elif t_obs_type == "all_random":
        obs_inds_old = obs_inds.reshape((batch_size, n_t_grid * n_x_obs_pts))
        n_tx_obs_pts = n_t_obs_pts * n_x_obs_pts
        obs_inds = []
        for i in range(batch_size):
            inds_i = np.random.choice(
                obs_inds_old.shape[1], size=n_tx_obs_pts, replace=False)
            obs_inds.append(obs_inds_old[i, inds_i])  # [n_tx_obs_pts]
        obs_inds = np.array(obs_inds)  # [bsz, n_tx_obs_pts]
    else:
        raise NotImplementedError(f"Unknown t_obs_type {t_obs_type}.")

    return obs_inds.flat  # [bsz * n_t_obs_pts * n_x_obs_pts]


def inverse_observation(observe_config: DictConfig,
                        u_label: NDArray[float],
                        coord_gt: NDArray[float]) -> Tuple[NDArray[float]]:
    r"""Apply additive noise and restriction of spatial-temporal observation points for inverse problems."""
    u_noisy = add_noise_to_label(u_label, observe_config.noise.type.lower(),
                                 observe_config.noise.level)

    # obs_inds
    batch_size, n_t_grid, n_x_grid, _ = u_label.shape
    obs_inds = get_observed_indices(
        batch_size, n_t_grid, n_x_grid,
        x_obs_type=observe_config.x_location.type.lower(),
        n_x_obs_pts=observe_config.x_location.num_pts,
        t_obs_type=observe_config.t_location.type.lower(),
        n_t_obs_pts=observe_config.t_location.num_pts)

    # u_obs_plot
    mask_obs = np.ones(u_label.shape, dtype=bool).flatten()
    mask_obs[obs_inds] = False
    mask_obs = mask_obs.reshape(u_label.shape)
    u_obs_plot = np.ma.masked_array(u_noisy, mask=mask_obs)

    # u_obs
    u_obs = u_noisy.reshape((-1, 1))  # [bsz * n_t_grid * n_x_grid, 1]
    u_obs = u_obs[obs_inds, :]  # [bsz * n_t_obs_pts * n_x_obs_pts, 1]
    # Shape is [bsz, n_t_obs_pts * n_x_obs_pts, 1].
    u_obs = u_obs.reshape((batch_size, -1, 1))

    # coord_obs
    coord_obs = coord_gt.reshape((-1, 2))  # [bsz * n_t_grid * n_x_grid, 2]
    coord_obs = coord_obs[obs_inds, :]  # [bsz * n_t_obs_pts * n_x_obs_pts, 2]
    # Shape is [bsz, n_t_obs_pts * n_x_obs_pts, 2].
    coord_obs = coord_obs.reshape((batch_size, -1, 2))

    return u_noisy, u_obs_plot, u_obs, coord_obs
