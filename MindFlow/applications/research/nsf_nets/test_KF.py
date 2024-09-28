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
"""NSFNet train"""
# pylint: disable=C0103
import os
import argparse

import numpy as np
import scipy.io
import mindspore as ms
from mindspore import set_seed, context
from mindflow.utils import load_yaml_config
from src.network_kf import VPNSFNets
from src.datasets_kf import read_training_data

# Adam loss history
loss_history_adam_pretrain = np.empty([0])
loss_b_history_adam_pretrain = np.empty([0])
loss_f_history_adam_pretrain = np.empty([0])
np.random.seed(123456)
set_seed(123456)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, default="./config/NSFNet_KF.yaml")
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
    load_params = params['load_params_test']
    second_path = params['second_path1']
    re = params['re']

    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=device)

    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    if use_ascend:
        msfloat_type = ms.float16
        npfloat_type = np.float16
    else:
        msfloat_type = ms.float32
        npfloat_type = np.float32

    xb_train, yb_train, ub_train, vb_train, x_train, y_train = read_training_data()

    model = VPNSFNets(xb_train, yb_train, ub_train, vb_train, x_train, y_train, \
                      network_size, use_ascend, msfloat_type, npfloat_type, load_params, second_path)

    x_test = (np.random.rand(1000, 1) - 1 / 3) * 3 / 2
    y_test = (np.random.rand(1000, 1) - 1 / 4) * 2
    lam = 0.5 * re - np.sqrt(0.25 * (re ** 2) + 4 * (np.pi ** 2))
    u_test = 1 - np.exp(lam * x_test) * np.cos(2 * np.pi * y_test)
    v_test = (lam / (2 * np.pi)) * np.exp(lam * x_test) * np.sin(2 * np.pi * y_test)
    p_test = 0.5 * (1 - np.exp(2 * lam * x_test))
    # to tensor
    x_test = ms.Tensor(np.array(x_test, npfloat_type))
    y_test = ms.Tensor(np.array(y_test, npfloat_type))

    U_pred = model.net_u(x_test, y_test)
    u_pred, v_pred, p_pred = U_pred[:, 0:1], U_pred[:, 1:2], U_pred[:, 2:3]
    scipy.io.savemat('test_result.mat', {'xts': x_test.numpy(), 'yts': y_test.numpy(), \
                                         'uts': u_pred.numpy(), 'vts': v_pred.numpy(), 'pts': p_pred.numpy(), \
                                            'ureal': u_test, 'vreal': v_test, 'preal': p_test})
