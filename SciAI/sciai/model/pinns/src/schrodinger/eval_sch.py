# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Evaluate PINNs for Schrodinger equation scenario"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, load_checkpoint, load_param_into_net
from sciai.utils import print_log
from src.schrodinger.dataset import get_eval_data
from src.schrodinger.net import PINNs


def eval_pinns_sch(ckpoint_name, num_neuron=100, path='./data/NLS.mat'):
    """
    Evaluation of PINNs for Schrodinger equation scenario.

    Args:
        ckpoint_name (str): model checkpoint file name
        num_neuron (int): number of neurons for fully connected layer in the network
        path (str): path of the dataset for Schrodinger equation
    """
    layers = [2, num_neuron, num_neuron, num_neuron, num_neuron, 2]
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    n = PINNs(layers, lb, ub)
    param_dict = load_checkpoint(ckpoint_name)
    load_param_into_net(n, param_dict)

    x_star, _, _, h_star = get_eval_data(path)

    x_tensor = Tensor(x_star, mstype.float32)
    pred = n(x_tensor)
    u_pred = pred[0].asnumpy()
    v_pred = pred[1].asnumpy()
    h_pred = np.sqrt(u_pred**2 + v_pred**2)

    error_h = np.linalg.norm(h_star-h_pred, 2)/np.linalg.norm(h_star, 2)
    print_log(f'evaluation error is: {error_h}')

    return error_h
