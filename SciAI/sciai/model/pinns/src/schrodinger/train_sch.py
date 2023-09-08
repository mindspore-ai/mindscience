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
"""Train PINNs for Schrodinger equation scenario"""
import numpy as np
from mindspore.common import set_seed
from mindspore import nn, Model
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)
from sciai.utils import print_log
from src.schrodinger.dataset import generate_pinns_training_set
from src.schrodinger.net import PINNs
from src.schrodinger.loss import PINNsLoss


def train_sch(epoch=50000, lr=0.0001, n0=50, nb=50, nf=20000, num_neuron=100, seed=None,
              path='./data/NLS.mat', ck_path='./ckpoints/'):
    """
    Train PINNs network for Schrodinger equation

    Args:
        epoch (int): number of epochs
        lr (float): learning rate
        n0 (int): number of data points sampled from the initial condition,
            0<n0<=256 for the default NLS dataset
        nb (int): number of data points sampled from the boundary condition,
            0<nb<=201 for the default NLS dataset. Size of training set = n0+2*nb
        nf (int): number of collocation points, collocation points are used
            to calculate regularizer for the network from Schoringer equation.
            0<nf<=51456 for the default NLS dataset
        num_neuron (int): number of neurons for fully connected layer in the network
        seed (int): random seed
        path (str): path of the dataset for Schrodinger equation
        ck_path (str): path to store checkpoint files (.ckpt)
    """
    if seed is not None:
        np.random.seed(seed)
        set_seed(seed)

    layers = [2, num_neuron, num_neuron, num_neuron, num_neuron, 2]

    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi/2])

    training_set = generate_pinns_training_set(n0, nb, nf, lb, ub, path=path)

    n = PINNs(layers, lb, ub)
    opt = nn.Adam(n.trainable_params(), learning_rate=lr)
    loss = PINNsLoss(n0, nb, nf)

    # call back configuration
    loss_print_num = 1  # print loss per loss_print_num epochs
    # save model
    config_ck = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=50)
    ckpoint = ModelCheckpoint(prefix="checkpoint_PINNs_Schrodinger", directory=ck_path, config=config_ck)

    model = Model(network=n, loss_fn=loss, optimizer=opt)

    model.train(epoch=epoch, train_dataset=training_set,
                callbacks=[LossMonitor(loss_print_num), ckpoint, TimeMonitor(1)], dataset_sink_mode=True)
    print_log('Training complete')
