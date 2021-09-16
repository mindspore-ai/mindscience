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
import numpy as np
import mindspore.nn as nn
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore import context, save_checkpoint
from mindspore.train.callback import TimeMonitor

from mindelec.solver import Solver
from mindelec.vision import MonitorTrain, MonitorEval

from src.dataset import create_dataset
from src.maxwell_model import S11Predictor
from src.loss import EvalMetric

set_seed(123456)
np.random.seed(123456)

parser = argparse.ArgumentParser(description='Parametrization S11 Simulation')
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--print_interval', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--input_dim', type=int, default=3)
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--device_target', type=str, default="Ascend")
parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')
parser.add_argument('--save_graphs_path', default='./graph_result/', help='checkpoint directory')
parser.add_argument('--input_path', default='./dataset/Butterfly_antenna/data_input.npy')
parser.add_argument('--label_path', default='./dataset/Butterfly_antenna/data_label.npy')
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=True,
                    save_graphs_path=opt.save_graphs_path,
                    device_target=opt.device_target,
                    device_id=opt.device_num)


def get_lr(data):
    """get learning rate"""
    num_milestones = 10
    if data['train_data_length'] % opt.batch_size == 0:
        iter_number = int(data['train_data_length'] / opt.batch_size)
    else:
        iter_number = int(data['train_data_length'] / opt.batch_size) + 1
    iter_number = opt.epochs * iter_number
    milestones = [int(iter_number * i / num_milestones) for i in range(1, num_milestones)]
    milestones.append(iter_number)
    learning_rates = [opt.lr * 0.5 ** i for i in range(0, num_milestones - 1)]
    learning_rates.append(opt.lr * 0.5 ** (num_milestones - 1))
    return milestones, learning_rates


def train():
    """train model"""
    data, config_data = create_dataset(opt)

    print("scale_input: ", config_data["scale_input"])
    print("scale_s11: ", config_data["scale_S11"])

    model_net = S11Predictor(opt.input_dim)
    model_net.to_float(mstype.float16)

    milestones, learning_rates = get_lr(data)

    optim = nn.Adam(model_net.trainable_params(),
                    learning_rate=nn.piecewise_constant_lr(milestones, learning_rates))

    eval_error_mrc = EvalMetric(scale_s11=config_data["scale_S11"],
                                length=data["eval_data_length"],
                                frequency=data["frequency"],
                                show_pic_number=4,
                                file_path='./eval_res')

    solver = Solver(network=model_net,
                    mode="Data",
                    optimizer=optim,
                    metrics={'eval_mrc': eval_error_mrc},
                    loss_fn=nn.MSELoss())

    monitor_train = MonitorTrain(per_print_times=1,
                                 summary_dir='./summary_dir_train')

    monitor_eval = MonitorEval(summary_dir='./summary_dir_eval',
                               model=solver,
                               eval_ds=data["eval_loader"],
                               eval_interval=opt.print_interval,
                               draw_flag=True)

    time_monitor = TimeMonitor()
    callbacks_train = [monitor_train, time_monitor, monitor_eval]

    solver.model.train(epoch=opt.epochs,
                       train_dataset=data["train_loader"],
                       callbacks=callbacks_train,
                       dataset_sink_mode=True)

    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
    save_checkpoint(model_net, os.path.join(opt.checkpoint_dir, 'model.ckpt'))


if __name__ == '__main__':
    train()
