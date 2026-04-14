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

"""cpinns process"""
import os
from collections import defaultdict

import yaml
import numpy as np
import scipy.io
import scipy.stats
from pyDOE import lhs

from sciai.utils import parse_arg
from sciai.utils.ms_utils import to_tensor


def prepare():
    """parse the configs and prepare environment"""
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def get_model_inputs(total_dict, dtype):
    """get model inputs"""
    x_u_train_total = total_dict["x_u_train_total"]
    u_train_total = total_dict["u_train_total"]
    x_f_train_total = total_dict["x_f_train_total"]
    x_f_inter_total = total_dict["x_f_inter_total"]
    # solutions along common interfaces
    u1, u2, u3, u4 = to_tensor((u_train_total[0], u_train_total[1], u_train_total[2], u_train_total[3]), dtype)

    # boundary/initial training points
    x_u1, t_u1 = to_tensor((x_u_train_total[0][:, 0:1], x_u_train_total[0][:, 1:2]), dtype)
    x_u2, t_u2 = to_tensor((x_u_train_total[1][:, 0:1], x_u_train_total[1][:, 1:2]), dtype)
    x_u3, t_u3 = to_tensor((x_u_train_total[2][:, 0:1], x_u_train_total[2][:, 1:2]), dtype)
    x_u4, t_u4 = to_tensor((x_u_train_total[3][:, 0:1], x_u_train_total[3][:, 1:2]), dtype)

    # residual points
    x_f1, t_f1 = to_tensor((x_f_train_total[0][:, 0:1], x_f_train_total[0][:, 1:2]), dtype)
    x_f2, t_f2 = to_tensor((x_f_train_total[1][:, 0:1], x_f_train_total[1][:, 1:2]), dtype)
    x_f3, t_f3 = to_tensor((x_f_train_total[2][:, 0:1], x_f_train_total[2][:, 1:2]), dtype)
    x_f4, t_f4 = to_tensor((x_f_train_total[3][:, 0:1], x_f_train_total[3][:, 1:2]), dtype)

    # interface points
    x_fi1, t_fi1 = to_tensor((x_f_inter_total[0][:, 0:1], x_f_inter_total[0][:, 1:2]), dtype)
    x_fi2, t_fi2 = to_tensor((x_f_inter_total[1][:, 0:1], x_f_inter_total[1][:, 1:2]), dtype)
    x_fi3, t_fi3 = to_tensor((x_f_inter_total[2][:, 0:1], x_f_inter_total[2][:, 1:2]), dtype)

    return u1, u2, u3, u4, x_u1, t_u1, x_u2, t_u2, x_u3, t_u3, x_u4, t_u4, \
           x_f1, t_f1, x_f2, t_f2, x_f3, t_f3, x_f4, t_f4, x_fi1, t_fi1, x_fi2, t_fi2, x_fi3, t_fi3


def get_star_inputs(np_dtype, total_dict):
    """get star inputs"""
    stars = tuple([total_dict["x_star_total"][i] for i in range(4)] + [total_dict["u_star_total"][i] for i in range(4)])
    stars_cast = [_.astype(np_dtype) for _ in stars]
    return stars_cast


def get_data(args, np_dtype):
    """get data"""
    data = scipy.io.loadmat(f'{args.load_data_path}/burgers_shock.mat')
    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    exact = np.real(data['usol']).T
    x_mesh, t_mesh = np.meshgrid(x, t)
    u_star = exact.flatten()[:, None]
    x_star = np.hstack((x_mesh.flatten()[:, None], t_mesh.flatten()[:, None]))
    x_interface = np.array([-1, -0.6, 0.2, 0.5, 1])
    idx_x_interface = np.floor((x_interface + 1) / 2 * len(x)).astype(int)
    num_interface, num_subdomain = len(x_interface) - 2, len(x_interface) - 1
    nn_layers_total = get_nn_layers_total(args.nn_depth, args.nn_width, num_subdomain)
    total_dict = get_total_dict(exact, t, num_interface, num_subdomain, t_mesh, x_mesh, data, idx_x_interface, x,
                                x_interface)
    x_interface = [_.astype(np_dtype) for _ in x_interface]
    return nn_layers_total, t_mesh, x_mesh, x_star, u_star, x_interface, total_dict


def get_nn_layers_total(nn_depth, nn_width, num_subdomain):
    """get neural network layers"""
    nn_layers_total = []
    for sd in range(num_subdomain):
        nn_layer_sd = [2] + [nn_width[sd]] * nn_depth[sd] + [1]
        nn_layers_total.append(nn_layer_sd)
    return nn_layers_total


def get_total_dict(*args):
    """get total data dictionary"""
    exact, t, num_interface, num_subdomain, t_mesh, x_mesh, data, idx_x_interface, x, x_interface = args
    total_dict = defaultdict(list)
    n_u_boundary = min(len(t), 4)
    n_u_subdomain_total = idx_x_interface[1:] - idx_x_interface[:-1] - 25
    for subdomain in range(num_subdomain):
        t_sd = data['t'].flatten()[:, None]
        x_sd = x[idx_x_interface[subdomain]:idx_x_interface[subdomain + 1]].flatten()
        total_dict["u_star_total"].append(
            exact[:, idx_x_interface[subdomain]:idx_x_interface[subdomain + 1]].flatten()[:, None])
        x_sd_mesh, t_sd_mesh = np.meshgrid(x_sd, t_sd)
        total_dict["x_sd_total"].append(x_sd_mesh)
        total_dict["t_sd_total"].append(t_sd_mesh)
        total_dict["x_star_total"].append(np.hstack((x_sd_mesh.flatten()[:, None], t_sd_mesh.flatten()[:, None])))
        xx_sd_init = np.hstack((x_sd_mesh[:1, :].T, t_sd_mesh[:1, :].T))
        uu_sd_init = exact[:1, idx_x_interface[subdomain]:idx_x_interface[subdomain + 1]].T
        if subdomain and subdomain != num_subdomain - 1:
            idx_sd = np.random.choice(xx_sd_init.shape[0], n_u_subdomain_total[subdomain], replace=False)
            x_u_sd_train, u_sd_train = xx_sd_init[idx_sd, :], uu_sd_init[idx_sd, :]
        elif not subdomain:
            xx1bdy, uu1bdy = np.hstack((x_mesh[:, :1], t_mesh[:, :1])), exact[:, :1]
            x_u1_train, u1_train = np.vstack([xx_sd_init, xx1bdy]), np.vstack([uu_sd_init, uu1bdy])
            idx_sd = np.random.choice(x_u1_train.shape[0], n_u_subdomain_total[subdomain] + n_u_boundary + 100,
                                      replace=False)
            x_u_sd_train, u_sd_train = x_u1_train[idx_sd, :], u1_train[idx_sd, :]
        elif subdomain == num_subdomain - 1:
            xx2bdy, uu2bdy = np.hstack((x_mesh[:, -1:], t_mesh[:, -1:])), exact[:, -1:]
            x_u2_train, u2_train = np.vstack([xx_sd_init, xx2bdy]), np.vstack([uu_sd_init, uu2bdy])
            idx_sd = np.random.choice(x_u2_train.shape[0], n_u_subdomain_total[subdomain] + n_u_boundary + 100,
                                      replace=False)
            x_u_sd_train, u_sd_train = x_u2_train[idx_sd, :], u2_train[idx_sd, :]
        else:
            continue
        total_dict["x_u_train_total"].append(x_u_sd_train)
        total_dict["u_train_total"].append(u_sd_train)
    total_dict["x_f_train_total"] = get_x_f_train_total(num_subdomain, x_interface)
    total_dict["x_f_inter_total"] = get_x_f_inter_total(num_interface, data, idx_x_interface, x)
    return total_dict


def get_x_f_train_total(num_subdomain, x_interface):
    """get x_f_train_total"""
    n_f = 3000
    n_f_total = num_subdomain * [n_f]
    x_f_train_total = []
    for sd in range(num_subdomain):
        x_f_sd_train_temp = lhs(2, n_f_total[sd])
        x_f_sd_train_x = x_interface[sd] + (x_interface[sd + 1] - x_interface[sd]) * x_f_sd_train_temp[:, 0]
        x_f_sd_train_t = x_f_sd_train_temp[:, 1]
        x_f_sd_train = np.hstack([x_f_sd_train_x[:, None], x_f_sd_train_t[:, None]])
        x_f_train_total.append(x_f_sd_train)
    return x_f_train_total


def get_x_f_inter_total(num_interface, data, idx_x_interface, x):
    """get x_f_inter_total"""
    n_f_interface = 99
    n_f_interface_total = num_interface * [n_f_interface]
    x_f_inter_total = []
    for inter in range(num_interface):
        t_inter = data['t'].flatten()[:, None]
        x_inter = x[idx_x_interface[inter + 1]:idx_x_interface[inter + 1] + 1].flatten()

        x_inter, t_inter = np.meshgrid(x_inter, t_inter)
        x_star_inter = np.hstack((x_inter.flatten()[:, None], t_inter.flatten()[:, None]))
        idx_inter = np.random.choice(x_star_inter.shape[0], n_f_interface_total[inter], replace=False)
        x_f_inter_train = x_star_inter[idx_inter, :]
        x_f_inter_total.append(x_f_inter_train)
    return x_f_inter_total
