# ============================================================================
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
"""
train_2d
"""
import argparse
import os
import random
import shutil

import numpy as np

import mindspore.nn as nn
from mindspore import Tensor, Model, DynamicLossScaleManager, load_checkpoint, \
    load_param_into_net, context, set_seed, set_auto_parallel_context, ParallelMode

from mindspore.common import dtype as mstype
from mindspore.common import initializer
from mindspore.communication import init, get_rank, get_group_size
from mindflow.utils import load_yaml_config

from src import Callback2D
from src import data_parallel_2d
from src import LossFunc2D, LossToEval2D, CustomWithLossCell2D
from src import StepLR
from src import MLP
from src import plt_loss_func


# 设置全局随机数种子
set_seed(2333)
np.random.seed(2333)
random.seed(2333)


def train(config):
    """train"""
    # 训练数据和测试数据，数据并行
    rank_id = get_rank()
    rank_size = get_group_size()
    train_path = config["data_path"] + '/train.txt'
    test_path = config["data_path"] + '/val.txt'
    train_dataset = data_parallel_2d(config, train_path, rank_id, rank_size,
                                     is_train=True)
    test_dataset_batch = data_parallel_2d(config, test_path, rank_id, rank_size,
                                          is_train=False)

    # 定义网络 + 网络初始化
    net = MLP(config["MLP"])

    prefix = os.path.join(config["data_path"], "2d_network_example")
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    model_file = "2d_net_best_" + os.environ["DEVICE_ID"] + ".ckpt"
    model_path = os.path.join(prefix, model_file)
    if not os.path.exists(model_path):
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer.initializer(initializer.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
    else:
        print("Load existed Checkpoint!")
        param_dict = load_checkpoint(model_path)
        load_param_into_net(net, param_dict)
    net_with_loss = CustomWithLossCell2D(net, LossFunc2D())
    # 动态学习率
    steps_per_epoch = train_dataset.get_dataset_size()
    lr_config = config["lr_scheduler"]
    lr_scheduler = StepLR(lr=lr_config["learning_rate"], epoch_size=lr_config["epoch_size"],
                          gamma=lr_config["gamma"], steps_per_epoch=steps_per_epoch,
                          max_epoch=config["epochs"], warmup_epochs=lr_config["warmup_epochs"])
    lr_tensor = Tensor(lr_scheduler.get_lr(), mstype.float32)

    # 优化器
    optim = nn.Adam(net.trainable_params(), learning_rate=lr_tensor)
    # 损失函数
    eval_loss = LossToEval2D()
    eval_net = CustomWithLossCell2D(net, eval_loss)
    eval_net.set_train(False)

    my_call = Callback2D(model_path, network=net, eval_network=eval_net,
                         eval_1=train_dataset, eval_2=test_dataset_batch)

    loss_scale = DynamicLossScaleManager()
    model = Model(network=net_with_loss, optimizer=optim,
                  amp_level="O2", loss_scale_manager=loss_scale)

    model.train(config["epochs"], train_dataset, callbacks=my_call, dataset_sink_mode=False)
    shutil.copy(model_path, config["model_path"])

    # loss曲线
    plt_loss_func(config["epochs"], my_call.train_loss_log, "train_loss.png", prefix=prefix)
    plt_loss_func(config["epochs"], my_call.val_loss_log, "test_loss.png", is_train=False,
                  prefix=prefix)


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default="./configs/TurbAI_2D_MLP.yaml")
    input_args = parser.parse_args()
    return input_args


if __name__ == '__main__':
    # 参数设置
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_id=int(os.environ["DEVICE_ID"]),
                        device_target="Ascend")
    init()
    device_num = int(os.getenv('RANK_SIZE'))
    set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                              device_num=device_num, global_rank=0)
    train_config = load_yaml_config(args.config_file_path)
    train(train_config)
