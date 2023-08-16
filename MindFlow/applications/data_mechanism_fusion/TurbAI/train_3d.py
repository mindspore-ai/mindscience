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
train_3d
"""
import argparse
import os
import random
import time
import shutil

import numpy as np

from mindspore import context, Model, Tensor, DynamicLossScaleManager
from mindspore import nn, load_checkpoint, load_param_into_net, set_seed
from mindspore.common import initializer
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindflow.utils import load_yaml_config

from src import Callback3D
from src import data_parallel_3d
from src import LossFunc3D, CustomWithLossCell3D, LossToEval3D
from src import StepLR
from src import ResMLP
from src import plt_loss_func

#%% 设置全局随机数种子
set_seed(2333)
np.random.seed(2333)
random.seed(2333)


def train(config):
    """train"""
    sca_min = np.loadtxt(config["data_path"] + '/3d_min.dat')[-1]
    sca_max = np.loadtxt(config["data_path"] + '/3d_max.dat')[-1]

    # 数据并行模式加载数据集
    rank_id = get_rank()  # 获取rank_id
    rank_size = get_group_size()  # 获取rank_size
    train_dataset = data_parallel_3d(config["data_path"] + '/train_data_3d.npy',
                                     rank_id, rank_size, config["batch_size"], is_train=True)
    val_dataset = data_parallel_3d(config["data_path"] + '/val_data_3d.npy',
                                   rank_id, rank_size, config["batch_size"], is_train=False)

    device_num = int(os.getenv('RANK_SIZE'))
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True, device_num=device_num,
                                      global_rank=0, parameter_broadcast=True)

    # 实例化前向网络
    net_config = config["ResMLP"]
    net = ResMLP(input_num=net_config["input_num"], width=net_config["width"],
                 depth=net_config["depth"], output_num=net_config["output_num"])

    # 网络初始化 若已存在权重
    prefix = os.path.join(config["data_path"], "3d_network_example")
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    model_file = "3d_net_best_" + os.environ["DEVICE_ID"] + ".ckpt"
    model_path = os.path.join(prefix, model_file)

    if not os.path.exists(model_path):
        print("Init Weight by XavierUniform!")
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(initializer.initializer(initializer.XavierUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
    else:
        print("Load existed Checkpoint!")
        param_dict = load_checkpoint(model_path)
        load_param_into_net(net, param_dict)
    # 定义损失函数
    loss_fn = LossFunc3D(sca_min, sca_max)
    # 定义带损失的model
    loss_net = CustomWithLossCell3D(net, loss_fn)
    # 定义学习率
    steps_per_epoch = train_dataset.get_dataset_size()
    lr_config = config["lr_scheduler"]
    lr_scheduler = StepLR(lr=lr_config["learning_rate"], epoch_size=lr_config["epoch_size"],
                          gamma=lr_config["gamma"], steps_per_epoch=steps_per_epoch,
                          max_epoch=config["epochs"], warmup_epochs=lr_config["warmup_epochs"])
    learning_rate = Tensor(lr_scheduler.get_lr())

    # 定义优化器
    optim = nn.Adam(net.trainable_params(), learning_rate=learning_rate)

    eval_loss = LossToEval3D(sca_min, sca_max)  # 构建评估损失
    eval_net = CustomWithLossCell3D(net, eval_loss)  # 构建评估网络
    eval_net.set_train(False)

    my_call = Callback3D(ckpt_path=model_path, eval_network=eval_net,
                         eval_1=train_dataset, eval_2=val_dataset)
    loss_scale_manager = DynamicLossScaleManager()
    model = Model(network=loss_net, optimizer=optim, amp_level="O2",
                  loss_scale_manager=loss_scale_manager)

    time1 = time.time()
    model.train(config["epochs"], train_dataset, callbacks=my_call, dataset_sink_mode=False)
    shutil.copy(model_path, config["model_path"])
    time2 = time.time()
    print('----------model.train time----------')
    print(time2-time1)

    plt_loss_func(config["epochs"], my_call.train_loss_log, "train_loss.png", prefix=prefix)
    plt_loss_func(config["epochs"], my_call.val_loss_log, "val_loss.png",
                  is_train=False, prefix=prefix)


def parse_args():
    """parse_args"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file_path', type=str, default="./configs/TurbAI_3D_ResMLP.yaml")
    input_args = parser.parse_args()
    return input_args


if __name__ == '__main__':
    args = parse_args()
    # 超参数设置
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(mode=context.GRAPH_MODE, device_id=device_id, device_target="Ascend")
    init()  # 使能HCCL通讯，并完成分布式训练初始化操作
    train_config = load_yaml_config(args.config_file_path)
    train(train_config)
