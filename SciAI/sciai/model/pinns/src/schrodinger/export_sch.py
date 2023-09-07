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
"""Export PINNs (Schrodinger) model"""
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import (Tensor, context, export, load_checkpoint,
                       load_param_into_net)
from src.schrodinger.net import PINNs


def export_sch(num_neuron, n0, nb, nf, ck_file, export_format, export_name):
    """
    export PINNs for Schrodinger model

    Args:
        num_neuron (int): number of neurons for fully connected layer in the network
        n0 (int): number of data points sampled from the initial condition,
            0<n0<=256 for the default NLS dataset
        nb (int): number of data points sampled from the boundary condition,
            0<nb<=201 for the default NLS dataset. Size of training set = n0+2*nb
        nf (int): number of collocation points, collocation points are used
            to calculate regularizer for the network from Schoringer equation.
            0<nf<=51456 for the default NLS dataset
        ck_file (str): path for checkpoint file
        export_format (str): file format to export
        export_name (str): name of exported file
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    layers = [2, num_neuron, num_neuron, num_neuron, num_neuron, 2]
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    n = PINNs(layers, lb, ub)
    param_dict = load_checkpoint(ck_file)
    load_param_into_net(n, param_dict)

    batch_size = n0 + 2*nb + nf
    inputs = Tensor(np.ones((batch_size, 2)), mstype.float32)
    export(n, inputs, file_name=export_name, file_format=export_format)
