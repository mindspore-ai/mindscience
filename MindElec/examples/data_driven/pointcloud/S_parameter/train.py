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
"""s parameter prediction model train"""
import os
import argparse
import datetime
import numpy as np

import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore.common import set_seed
from mindspore import context
from mindspore import save_checkpoint
from mindspore.train.callback import LossMonitor, TimeMonitor

from mindelec.solver import Solver

from src.dataset import create_dataset
from src.model import S11Predictor
from src.lr_generator import step_lr_generator
from src.config import config
from src.metric import MyMSELoss

set_seed(0)
np.random.seed(0)

print("pid:", os.getpid())
print(datetime.datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument('--train_input_path', type=str)
parser.add_argument('--train_label_path', type=str)
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')

opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=opt.device_num)

def init_weight(net):
    """init_weight"""
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv3d, nn.Dense)):
            cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))

def train():
    """training"""
    model_net = S11Predictor(input_dim=config["input_channels"])
    init_weight(net=model_net)

    train_dataset = create_dataset(input_path=opt.train_input_path,
                                   label_path=opt.train_label_path,
                                   batch_size=config["batch_size"],
                                   shuffle=True)

    step_size = train_dataset.get_dataset_size()
    milestones, learning_rates = step_lr_generator(step_size,
                                                   config["epochs"],
                                                   config["lr"],
                                                   config["lr_decay_milestones"])
    optimizer = nn.Adam(model_net.trainable_params(),
                        learning_rate=nn.piecewise_constant_lr(milestones, learning_rates))

    loss_net = MyMSELoss()

    solver = Solver(model_net,
                    train_input_map={'train': ['train_input_data']},
                    test_input_map={'test': ['test_input_data']},
                    optimizer=optimizer,
                    amp_level="O2",
                    loss_fn=loss_net)

    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)

    solver.model.train(config["epochs"],
                       train_dataset,
                       callbacks=[LossMonitor(), TimeMonitor()],
                       dataset_sink_mode=True)

    save_checkpoint(model_net, os.path.join(opt.checkpoint_dir, 'model_best.ckpt'))
    print("model saved in ", opt.checkpoint_dir)


if __name__ == '__main__':
    train()
