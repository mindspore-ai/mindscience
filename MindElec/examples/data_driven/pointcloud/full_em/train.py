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
# ==============================================================================
"""
train
"""
import os
import argparse
import datetime
import numpy as np

import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore.common import set_seed
from mindspore import Tensor
from mindspore import context
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindelec.solver import Solver
from src.dataset import create_dataset
from src.loss import MyMSELoss
from src.maxwell_model import Maxwell3D
from src.config import config


set_seed(0)
np.random.seed(0)

print("pid:", os.getpid())
print(datetime.datetime.now())
parser = argparse.ArgumentParser(description='Electromagnetic Simulation')
parser.add_argument('--device_id', type=int, default=1)
parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')
parser.add_argument('--data_path', default='data path')

opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=opt.device_id)


def get_lr(lr_init, steps_per_epoch, total_epochs):
    """get lr"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    for i in range(total_steps):
        epoch = i // steps_per_epoch
        lr_local = lr_init
        if epoch <= 15:
            lr_local = lr_init
        elif epoch <= 45:
            lr_local = lr_init * 0.5
        elif epoch <= 300:
            lr_local = lr_init * 0.25
        elif epoch <= 600:
            lr_local = lr_init * 0.125
        lr_each_step.append(lr_local)
    learning_rate = np.array(lr_each_step).astype(np.float32)
    print(learning_rate)
    return learning_rate


def init_weight(net):
    """init weight"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv3d, nn.Dense)):
            cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))


def train():
    """train"""
    dataset, _ = create_dataset(opt.data_path, batch_size=config.batch_size, shuffle=True)
    model_net = Maxwell3D(6)
    init_weight(net=model_net)
    step_size = dataset.get_dataset_size()
    lr = get_lr(config.lr, step_size, config.epochs)
    optimizer = nn.Adam(model_net.trainable_params(), learning_rate=Tensor(lr))
    loss_net = MyMSELoss()
    loss_scale = DynamicLossScaleManager()

    solver = Solver(model_net,
                    optimizer=optimizer,
                    loss_scale_manager=loss_scale,
                    amp_level="O2",
                    keep_batchnorm_fp32=False,
                    loss_fn=loss_net)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=config["save_checkpoint_epochs"] * step_size,
                                   keep_checkpoint_max=config["keep_checkpoint_max"])
    ckpt_cb = ModelCheckpoint(prefix='Maxwell3d', directory=opt.checkpoint_dir, config=ckpt_config)
    solver.model.train(config.epochs, dataset, callbacks=[LossMonitor(), TimeMonitor(), ckpt_cb],
                       dataset_sink_mode=False)


if __name__ == '__main__':
    train()
