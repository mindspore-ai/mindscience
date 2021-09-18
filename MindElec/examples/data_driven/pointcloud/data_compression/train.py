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
"""ae_train"""

import os
import argparse
import datetime

import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from mindspore import context
from mindspore import save_checkpoint
from mindspore.train.callback import LossMonitor, TimeMonitor

from mindelec.solver import Solver

from src.dataset import create_dataset
from src.model import EncoderDecoder
from src.lr_generator import step_lr_generator
from src.metric import MyMSELoss, EvalMetric
from src.config import config


print("pid:", os.getpid())
print(datetime.datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument('--train_input_path')
parser.add_argument('--test_input_path')
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')

opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=opt.device_num)


def init_weight(net):
    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv3d, nn.Dense)):
            cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(),
                                                         cell.weight.shape,
                                                         cell.weight.dtype))

def train():
    """training"""

    model_net = EncoderDecoder(config["input_channels"], config["patch_shape"], config["base_channels"], decoding=True)
    init_weight(net=model_net)

    train_dataset = create_dataset(input_path=opt.train_input_path,
                                   label_path=opt.train_input_path,
                                   batch_size=config["batch_size"],
                                   shuffle=True)

    eval_dataset = create_dataset(input_path=opt.test_input_path,
                                  label_path=opt.test_input_path,
                                  batch_size=config["batch_size"],
                                  shuffle=False)


    step_size = train_dataset.get_dataset_size()
    milestones, learning_rates = step_lr_generator(step_size,
                                                   config["epochs"],
                                                   config["lr"],
                                                   config["lr_decay_milestones"])

    optimizer = nn.Adam(model_net.trainable_params(),
                        learning_rate=nn.piecewise_constant_lr(milestones, learning_rates))

    loss_net = MyMSELoss()
    eval_step_size = eval_dataset.get_dataset_size() * config["batch_size"]
    evl_error_mrc = EvalMetric(eval_step_size)

    solver = Solver(model_net,
                    train_input_map={'train': ['train_input_data']},
                    test_input_map={'test': ['test_input_data']},
                    optimizer=optimizer,
                    metrics={'evl_mrc': evl_error_mrc,},
                    amp_level="O2",
                    loss_fn=loss_net)

    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)

    min_loss, min_l1_error = float('inf'), float('inf')

    for epoch in range(config["epochs"] // config["eval_interval"]):
        solver.model.train(config["eval_interval"],
                           train_dataset,
                           callbacks=[LossMonitor(), TimeMonitor()],
                           dataset_sink_mode=True)

        res_test = solver.model.eval(eval_dataset, dataset_sink_mode=True)
        error_mean_l1_error = res_test['evl_mrc']['mean_l1_error']
        min_l1_error = min(min_l1_error, error_mean_l1_error)
        print(f'epoch: {(epoch + 1) * config["eval_interval"]}  ',
              f'error_mean_l1_error: {error_mean_l1_error:.5f}  ',)

        # save eval results
        if (epoch+1)*config["eval_interval"] % config["save_epoch"] == 0:
            save_checkpoint(model_net, os.path.join(opt.checkpoint_dir, 'model_last.ckpt'))
            # save best model (which has min loss)
            if min_loss > min_l1_error:
                min_loss = min_l1_error
                save_checkpoint(model_net, os.path.join(opt.checkpoint_dir, 'model_best.ckpt'))

if __name__ == '__main__':
    train()
