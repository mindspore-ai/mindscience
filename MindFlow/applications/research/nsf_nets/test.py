# Copyright 2024 Huawei Technologies Co., Ltd
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
"""ICNet train"""
# pylint: disable=C0103
# pylint: disable=E1121
import os
import argparse

import numpy as np
import scipy.io
import mindspore as ms
from mindspore import set_seed, context
from mindflow.utils import load_yaml_config
from src.network import VPNSFNets
from src.datasets import read_training_data, ThreeBeltramiflow

# Adam loss history
loss_history_adam_pretrain = np.empty([0])
loss_b_history_adam_pretrain = np.empty([0])
loss_i_history_adam_pretrain = np.empty([0])
loss_f_history_adam_pretrain = np.empty([0])
np.random.seed(123456)
set_seed(123456)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="./config/NSFNet.yaml")
    # args = parser.parse_args()
    args = parser.parse_known_args()[0]

    config = load_yaml_config(args.config_file_path)
    params = config["params"]

    model_name = params['model_name']
    case = params['case']
    device = params['device']
    device_id = params['device_id']
    network_size = params['network_size']
    learning_rate = params['learning_rate']
    epochs = params['epochs']
    batch_size = params['batch_size']
    load_params = params['load_params_test']
    second_path = params['second_path1']
    xmin = params['xmin']
    xmax = params['xmax']
    ymin = params['ymin']
    ymax = params['ymax']
    zmin = params['zmin']
    zmax = params['zmax']
    tmin = params['tmin']
    tmax = params['tmax']
    n_x = params['n_x']
    n_y = params['n_y']
    n_z = params['n_z']
    n_t = params['n_t']

    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=device)

    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    if use_ascend:
        msfloat_type = ms.float16
        npfloat_type = np.float16
    else:
        msfloat_type = ms.float32
        npfloat_type = np.float32

    x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, u_b, v_b, w_b, \
        u_i, v_i, w_i, x_f, y_f, z_f, t_f, Re, X_min, X_max = read_training_data()

    model = VPNSFNets(x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, u_b, v_b, w_b, \
                      u_i, v_i, w_i, x_f, y_f, z_f, t_f, network_size, \
                        Re, X_min, X_max, use_ascend, msfloat_type, npfloat_type, load_params, second_path)

    x_ts_sum = np.arange(xmin, xmax+1e-5, (xmax-xmin)/(n_x-1.))
    y_ts_sum = np.arange(ymin, ymax+1e-5, (ymax-ymin)/(n_y-1.))
    z_ts_sum = np.arange(zmin, zmax+1e-5, (zmax-zmin)/(n_z-1.))

    x_ts, y_ts, z_ts = np.meshgrid(x_ts_sum, y_ts_sum, z_ts_sum)
    x_ts = x_ts.reshape([-1, 1])
    y_ts = y_ts.reshape([-1, 1])
    z_ts = z_ts.reshape([-1, 1])
    x_ts = ms.Tensor(np.array(x_ts, npfloat_type))
    y_ts = ms.Tensor(np.array(y_ts, npfloat_type))
    z_ts = ms.Tensor(np.array(z_ts, npfloat_type))

    # t1
    t1 = tmin*np.ones([x_ts.shape[0], 1])
    t1 = ms.Tensor(np.array(t1, npfloat_type))
    U_ts = model.net_u(x_ts, y_ts, z_ts, t1)
    u_ts, v_ts, w_ts, p_ts = U_ts[:, 0:1], U_ts[:, 1:2], U_ts[:, 2:3], U_ts[:, 3:4]
    u_ts_real, v_ts_real, w_ts_real, p_ts_real = \
        ThreeBeltramiflow(x_ts.numpy(), y_ts.numpy(), z_ts.numpy(), t1.numpy(), Re)
    scipy.io.savemat('test_result1.mat', {'xts': x_ts.numpy(), 'yts': y_ts.numpy(), 'zts': z_ts.numpy(),
                                          'uts': u_ts.numpy(), 'vts': v_ts.numpy(),
                                          'wts': w_ts.numpy(), 'pts': p_ts.numpy(),
                                          'ureal': u_ts_real, 'vreal': v_ts_real,
                                          'wreal': w_ts_real, 'preal': p_ts_real})
    # t2
    t2 = (tmin + 0.25*(tmax-tmin))*np.ones([x_ts.shape[0], 1])
    t2 = ms.Tensor(np.array(t2, npfloat_type))
    U_ts = model.net_u(x_ts, y_ts, z_ts, t2)
    u_ts, v_ts, w_ts, p_ts = U_ts[:, 0:1], U_ts[:, 1:2], U_ts[:, 2:3], U_ts[:, 3:4]
    u_ts_real, v_ts_real, w_ts_real, p_ts_real = ThreeBeltramiflow(x_ts.numpy(), y_ts.numpy(),
                                                                   z_ts.numpy(), t2.numpy(), Re)
    scipy.io.savemat('test_result2.mat', {'xts': x_ts.numpy(), 'yts': y_ts.numpy(),
                                          'zts': z_ts.numpy(), 'uts': u_ts.numpy(),
                                          'vts': v_ts.numpy(), 'wts': w_ts.numpy(),
                                          'pts': p_ts.numpy(), 'ureal': u_ts_real,
                                          'vreal': v_ts_real, 'wreal': w_ts_real,
                                          'preal': p_ts_real})

    # t3
    t3 = (tmin + 0.5*(tmax-tmin))*np.ones([x_ts.shape[0], 1])
    t3 = ms.Tensor(np.array(t3, npfloat_type))
    U_ts = model.net_u(x_ts, y_ts, z_ts, t3)
    u_ts, v_ts, w_ts, p_ts = U_ts[:, 0:1], U_ts[:, 1:2], U_ts[:, 2:3], U_ts[:, 3:4]
    u_ts_real, v_ts_real, w_ts_real, p_ts_real = ThreeBeltramiflow(x_ts.numpy(), y_ts.numpy(),
                                                                   z_ts.numpy(), t3.numpy(), Re)
    scipy.io.savemat('test_result3.mat', {'xts': x_ts.numpy(), 'yts': y_ts.numpy(),
                                          'zts': z_ts.numpy(), 'uts': u_ts.numpy(),
                                          'vts': v_ts.numpy(), 'wts': w_ts.numpy(),
                                          'pts': p_ts.numpy(), 'ureal': u_ts_real,
                                          'vreal': v_ts_real, 'wreal': w_ts_real,
                                          'preal': p_ts_real})

    # t4
    t4 = (tmin + 0.75*(tmax-tmin))*np.ones([x_ts.shape[0], 1])
    t4 = ms.Tensor(np.array(t4, npfloat_type))
    U_ts = model.net_u(x_ts, y_ts, z_ts, t4)
    u_ts, v_ts, w_ts, p_ts = U_ts[:, 0:1], U_ts[:, 1:2], U_ts[:, 2:3], U_ts[:, 3:4]
    u_ts_real, v_ts_real, w_ts_real, p_ts_real = ThreeBeltramiflow(x_ts.numpy(), y_ts.numpy(),
                                                                   z_ts.numpy(), t4.numpy(), Re)
    scipy.io.savemat('test_result3.mat', {'xts': x_ts.numpy(), 'yts': y_ts.numpy(),
                                          'zts': z_ts.numpy(), 'uts': u_ts.numpy(),
                                          'vts': v_ts.numpy(), 'wts': w_ts.numpy(),
                                          'pts': p_ts.numpy(), 'ureal': u_ts_real,
                                          'vreal': v_ts_real, 'wreal': w_ts_real,
                                          'preal': p_ts_real})
    # t5
    t5 = tmax*np.ones([x_ts.shape[0], 1])
    t5 = ms.Tensor(np.array(t5, npfloat_type))
    U_ts = model.net_u(x_ts, y_ts, z_ts, t5)
    u_ts, v_ts, w_ts, p_ts = U_ts[:, 0:1], U_ts[:, 1:2], U_ts[:, 2:3], U_ts[:, 3:4]
    u_ts_real, v_ts_real, w_ts_real, p_ts_real = ThreeBeltramiflow(x_ts.numpy(), y_ts.numpy(),
                                                                   z_ts.numpy(), t5.numpy(), Re)
    scipy.io.savemat('test_result3.mat', {'xts': x_ts.numpy(), 'yts': y_ts.numpy(),
                                          'zts': z_ts.numpy(), 'uts': u_ts.numpy(),
                                          'vts': v_ts.numpy(), 'wts': w_ts.numpy(),
                                          'pts': p_ts.numpy(), 'ureal': u_ts_real,
                                          'vreal': v_ts_real, 'wreal': w_ts_real,
                                          'preal': p_ts_real})
