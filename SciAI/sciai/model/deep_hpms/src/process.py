
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

"""process"""
import argparse
import os

import yaml
import numpy as np
import scipy.io
from pyDOE import lhs

from sciai.utils import parse_arg, flatten_add_dim, to_tensor
from .network_burgers_different import DeepHPMBurgersDiff
from .network_kdv_same import DeepHPMKdvSame


def prepare(problem=None):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    if problem is not None:
        config_dict["problem"] = problem
    args_, problem_ = generate_args(config_dict)
    return args_, problem_


def generate_args(config):
    """generate arguments"""
    common_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
    problem_name = find_problem(config)
    problem_config = config.get(problem_name)
    concat_config = {**common_config, **problem_config}
    args = parse_arg(concat_config)
    problem = {
        "burgers_different": DeepHPMBurgersDiff,
        "kdv_same": DeepHPMKdvSame
    }.get(problem_name, DeepHPMBurgersDiff)
    return args, problem


def find_problem(config):
    """find problem according to config case"""
    parser = argparse.ArgumentParser(description=config.get("case"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "burgers_different"))
    args = parser.parse_known_args()
    return args[0].problem


def load_data(args, dtype):
    """load data"""
    # Domain bounds
    lb_idn, ub_idn, lb_sol, ub_sol = \
        np.array(args.lb_idn), np.array(args.ub_idn), np.array(args.lb_sol), np.array(args.ub_sol)

    # identification
    data_idn = scipy.io.loadmat(args.load_data_idn_path)
    t_idn, x_idn = flatten_add_dim(data_idn['t'], data_idn['x'])
    t_idn_, x_idn_ = np.meshgrid(t_idn, x_idn)
    exact_idn_ = np.real(data_idn['usol'])
    index = int(2 / 3 * t_idn.shape[0])
    t_idn_, x_idn_, exact_idn_ = t_idn_[:, :index], x_idn_[:, :index], exact_idn_[:, :index]
    t_idn_star, x_idn_star, u_idn_star = flatten_add_dim(t_idn_, x_idn_, exact_idn_)
    n_train = 10000
    idx = np.random.choice(t_idn_star.shape[0], n_train, replace=False)
    t_train, x_train, u_train = t_idn_star[idx, :], x_idn_star[idx, :], u_idn_star[idx, :]
    noise = 0.00
    u_train = u_train + noise * np.std(u_train) * np.random.randn(*u_train.shape)

    # solution
    data_sol = scipy.io.loadmat(args.load_data_sol_path)
    t_sol, x_sol = flatten_add_dim(data_sol['t'], data_sol['x'])
    t_sol_, x_sol_ = np.meshgrid(t_sol, x_sol)
    exact_sol_ = np.real(data_sol['usol'])
    t_sol_star, x_sol_star, u_sol_star = flatten_add_dim(t_sol_, x_sol_, exact_sol_)
    x_sol_star_ = np.hstack((t_sol_star, x_sol_star))
    n0, n_b = exact_sol_.shape
    idx_x = np.random.choice(x_sol.shape[0], n0, replace=False)
    idx_t = np.random.choice(t_sol.shape[0], n_b, replace=False)
    tb_train, x0_train, u0_train = t_sol[idx_t, :], x_sol[idx_x, :], exact_sol_[idx_x, :1]
    n_f = 20000
    x_f_train_ = lb_sol + (ub_sol - lb_sol) * lhs(2, n_f)

    tensors = to_tensor((lb_idn, ub_idn, lb_sol, ub_sol, t_train, x_train, u_train, t_idn_star, x_idn_star, u_idn_star,
                         x_f_train_, tb_train, x0_train, u0_train, t_sol_star, x_sol_star, u_sol_star), dtype)

    return x_sol_star_, exact_sol_, t_sol_, x_sol_, tensors
