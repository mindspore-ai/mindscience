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

"""Data processing"""
import os
import time

import yaml
import numpy as np
import scipy.io
import mindspore as ms
from mindspore import ops, Tensor

from sciai.utils import print_log, amp2datatype, to_tensor, parse_arg, flatten_add_dim
from .network import HFM


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def relative_error(pred, exact):
    """Compute relative error"""
    if isinstance(pred, np.ndarray):
        return np.sqrt(np.mean(np.square(pred - exact)) / np.mean(np.square(exact - np.mean(exact))))
    return ops.sqrt(
        ops.reduce_mean(ops.square(pred - exact)) / ops.reduce_mean(ops.square(exact - ops.reduce_mean(exact))))


def generate_simple_data(args):
    """Generate simple data"""
    n = args.n
    t = args.t

    data = scipy.io.loadmat(f'{args.load_data_path}/Cylinder2D_flower_original.mat')
    x_star = data.get('x_star')
    y_star = data.get('y_star')
    c_star = data.get('C_star')
    u_star = data.get('U_star')
    v_star = data.get('V_star')
    p_star = data.get('P_star')
    t_star = data.get('t_star')

    n = min(n, x_star.shape[0])
    t = min(t, t_star.shape[0])
    idx_n = np.random.choice(x_star.shape[0], n, replace=False)
    idx_t = np.random.choice(t_star.shape[0], t, replace=False)

    x_star = x_star[idx_n, :]
    y_star = y_star[idx_n, :]
    c_star = c_star[idx_n, :][:, idx_t]
    u_star = u_star[idx_n, :][:, idx_t]
    v_star = v_star[idx_n, :][:, idx_t]
    p_star = p_star[idx_n, :][:, idx_t]
    t_star = t_star[idx_t, :]

    scipy.io.savemat(f'{args.save_data_path}/Cylinder2D_flower_%d_%d.mat' % (t, n),
                     {'C_star': c_star, 'U_star': u_star, 'V_star': v_star,
                      'P_star': p_star, 'x_star': x_star, 'y_star': y_star, 't_star': t_star})


def tile_data(t_star, x_star, y_star):
    """Tile data"""
    t_num = t_star.shape[0]
    x_num = x_star.shape[0]
    t_star_tile = np.tile(t_star, (1, x_num)).T  # x_num x t_num
    x_star_tile = np.tile(x_star, (1, t_num))  # x_num x t_num
    y_star_tile = np.tile(y_star, (1, t_num))  # x_num x t_num
    return t_star_tile, x_star_tile, y_star_tile


def full_evaluation(*inputs):
    """Full evaluation"""
    args, model, c_star, p_star, u_star, v_star, t_star, x_star, y_star = inputs
    dtype = amp2datatype(args.amp_level)
    t_star_tile, x_star_tile, y_star_tile = tile_data(t_star, x_star, y_star)

    c_pred_container = Tensor(0 * c_star)
    u_pred_container = Tensor(0 * u_star)
    v_pred_container = Tensor(0 * v_star)
    p_pred_container = Tensor(0 * p_star)

    for snap in range(0, t_star.shape[0]):
        t_test = ms.Tensor(t_star_tile[:, snap:snap + 1].astype(float), dtype=dtype)
        x_test = ms.Tensor(x_star_tile[:, snap:snap + 1].astype(float), dtype=dtype)
        y_test = ms.Tensor(y_star_tile[:, snap:snap + 1].astype(float), dtype=dtype)

        c_pred, u_pred, v_pred, p_pred = model.net_cuvp(t_test, x_test, y_test)

        c_test = ms.Tensor(c_star[:, snap:snap + 1].astype(float), dtype=dtype)
        u_test = ms.Tensor(u_star[:, snap:snap + 1].astype(float), dtype=dtype)
        v_test = ms.Tensor(v_star[:, snap:snap + 1].astype(float), dtype=dtype)
        p_test = ms.Tensor(p_star[:, snap:snap + 1].astype(float), dtype=dtype)

        error_c = relative_error(c_pred, c_test)
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_p = relative_error(p_pred - p_pred.mean(), p_test - p_test.mean())

        print_log('Error c: %e, Error u: %e, Error v: %e, Error p: %e' % (error_c, error_u, error_v, error_p))

        c_pred_container[:, snap:snap + 1] = c_pred
        u_pred_container[:, snap:snap + 1] = u_pred
        v_pred_container[:, snap:snap + 1] = v_pred
        p_pred_container[:, snap:snap + 1] = p_pred

    if args.save_result:
        os.makedirs(f"{args.save_data_path}/results", exist_ok=True)
        scipy.io.savemat(f'{args.save_data_path}/results/Cylinder2D_flower_results_%s.mat'
                         % time.strftime('%d_%m_%Y'),
                         {'C_pred': c_pred_container.asnumpy(), 'U_pred': u_pred_container.asnumpy(),
                          'V_pred': v_pred_container.asnumpy(), 'P_pred': p_pred_container.asnumpy()})


def simple_evaluate(*inputs):
    """Simple evaluate"""
    args, model, c_star, p_star, u_star, v_star, t_star, x_star, y_star = inputs
    dtype = amp2datatype(args.amp_level)
    t_star_tile, x_star_tile, y_star_tile = tile_data(t_star, x_star, y_star)

    snap = np.array([100])
    t_test = ms.Tensor(t_star_tile[:, snap].astype(float), dtype=dtype)
    x_test = ms.Tensor(x_star_tile[:, snap].astype(float), dtype=dtype)
    y_test = ms.Tensor(y_star_tile[:, snap].astype(float), dtype=dtype)

    c_pred, u_pred, v_pred, p_pred = model.net_cuvp(t_test, x_test, y_test)

    c_test = ms.Tensor(c_star[:, snap].astype(float), dtype=dtype)
    u_test = ms.Tensor(u_star[:, snap].astype(float), dtype=dtype)
    v_test = ms.Tensor(v_star[:, snap].astype(float), dtype=dtype)
    p_test = ms.Tensor(p_star[:, snap].astype(float), dtype=dtype)

    error_c = relative_error(c_pred, c_test)
    error_u = relative_error(u_pred, u_test)
    error_v = relative_error(v_pred, v_test)
    error_p = relative_error(p_pred - p_pred.mean(), p_test - p_test.mean())

    print_log('Error c: %e, Error u: %e, Error v: %e, Error p: %e' % (error_c, error_u, error_v, error_p))


def obtain_data(args):
    """Obtain data"""
    data = scipy.io.loadmat(f'{args.load_data_path}/Cylinder2D_flower.mat')
    t_star = data.get('t_star')  # t_num x 1
    x_star = data.get('x_star')  # x_num x 1
    y_star = data.get('y_star')  # x_num x 1
    c_star = data.get('C_star')  # x_num x t_num
    u_star = data.get('U_star')  # x_num x t_num
    v_star = data.get('V_star')  # x_num x t_num
    p_star = data.get('P_star')  # x_num x t_num
    del data
    return c_star, p_star, t_star, u_star, v_star, x_star, y_star


def prepare_training_data(args, c_star, t_star, x_star, y_star):
    """Prepare training data"""
    t_num = t_star.shape[0]
    x_num = x_star.shape[0]

    t_star_tile = np.tile(t_star, (1, x_num)).T  # x_num x t_num
    x_star_tile = np.tile(x_star, (1, t_num))  # x_num x t_num
    y_star_tile = np.tile(y_star, (1, t_num))  # x_num x t_num

    t_data_num = min(args.t, t_num)
    x_data_num = min(args.n, x_num)

    print_log("t_data_num:%d, x_data_num:%d" % (t_data_num, x_data_num))

    idx_t = np.concatenate([np.array([0]),
                            np.random.choice(t_num - 2, t_data_num - 2, replace=False) + 1, np.array([t_num - 1])])
    idx_x = np.random.choice(x_num, x_data_num, replace=False)
    t_data = flatten_add_dim(t_star_tile[:, idx_t][idx_x, :])
    x_data = flatten_add_dim(x_star_tile[:, idx_t][idx_x, :])
    y_data = flatten_add_dim(y_star_tile[:, idx_t][idx_x, :])
    c_data = flatten_add_dim(c_star[:, idx_t][idx_x, :])

    t_eqns_num = t_num
    x_eqns_num = x_num

    idx_t = np.concatenate([np.array([0]),
                            np.random.choice(t_num - 2, t_eqns_num - 2, replace=False) + 1, np.array([t_num - 1])])
    idx_x = np.random.choice(x_num, x_eqns_num, replace=False)
    t_eqns = flatten_add_dim(t_star_tile[:, idx_t][idx_x, :])
    x_eqns = flatten_add_dim(x_star_tile[:, idx_t][idx_x, :])
    y_eqns = flatten_add_dim(y_star_tile[:, idx_t][idx_x, :])

    return t_data, x_data, y_data, c_data, t_eqns, x_eqns, y_eqns


def prepare_network(args, dtype, t_data, x_data, y_data):
    """Prepare network"""
    t_data_tensor, x_data_tensor, y_data_tensor = to_tensor((t_data, x_data, y_data), dtype=dtype)

    model = HFM(t_data_tensor, x_data_tensor, y_data_tensor, args.layers, pec=100, rey=100)
    return model
