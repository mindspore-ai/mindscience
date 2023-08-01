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
"""sed-_ann train"""

import argparse
import os
import shutil

import numpy as np
import pandas as pd
from mindspore import context, set_seed, ParallelMode, nn
from mindspore.communication import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor

from mindelec.data import Dataset, ExistedDataConfig
from mindelec.solver import Solver


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Parametrization sed_AI Simulation')
    parser.add_argument('--epochs', type=int, default=1500)
    parser.add_argument('--print_interval', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--device_id', type=int, default=None)
    parser.add_argument('--device_target', type=str, default=None)
    parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')
    parser.add_argument('--save_graphs_path', default='./graph_result/', help='checkpoint directory')
    parser.add_argument('--input_path', default='./dataset/Butterfly_antenna/data_input.npy')
    parser.add_argument('--label_path', default='./dataset/Butterfly_antenna/data_label.npy')
    opt = parser.parse_args()
    return opt


def custom_normalize(data):
    """
    get normalize data
    """
    print("Custom normalization is called")
    ori_shape = data.shape
    data = data.values.reshape(ori_shape[0], -1)
    print("data reshape", data.shape)
    data = np.transpose(data)
    mean = np.mean(data, axis=1)
    print("data mean", mean.shape)
    data = data - mean[:, None]
    std = np.std(data, axis=1)
    print("data std", std.shape)
    std += (np.abs(std) < 0.0000001)
    data = data / std[:, None]
    data = np.transpose(data)
    data = data.reshape(ori_shape)
    return data


class Model(nn.Cell):
    """model"""

    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Dense(8, 128)
        self.fc2 = nn.Dense(128, 128)
        self.fc3 = nn.Dense(128, 128)
        self.fc4 = nn.Dense(128, 128)
        self.fc5 = nn.Dense(128, 256)
        self.fc6 = nn.Dense(256, 256)
        self.fc7 = nn.Dense(256, 256)
        self.fc8 = nn.Dense(256, 512)
        self.fc9 = nn.Dense(512, 512)
        self.fc10 = nn.Dense(512, 512)
        self.fc11 = nn.Dense(512, 128)
        self.fc12 = nn.Dense(128, 128)
        self.fc13 = nn.Dense(128, 128)
        self.fc14 = nn.Dense(128, 64)
        self.fc15 = nn.Dense(64, 64)
        self.fc16 = nn.Dense(64, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.bn5 = nn.BatchNorm1d(num_features=256)
        self.bn6 = nn.BatchNorm1d(num_features=256)
        self.bn7 = nn.BatchNorm1d(num_features=256)
        self.bn8 = nn.BatchNorm1d(num_features=512)
        self.bn9 = nn.BatchNorm1d(num_features=512)
        self.bn10 = nn.BatchNorm1d(num_features=512)
        self.bn11 = nn.BatchNorm1d(num_features=128)
        self.bn12 = nn.BatchNorm1d(num_features=128)
        self.bn13 = nn.BatchNorm1d(num_features=128)
        self.bn14 = nn.BatchNorm1d(num_features=64)
        self.bn15 = nn.BatchNorm1d(num_features=64)

    def construct(self, x):
        """forward"""
        x0 = x
        x1 = self.relu(self.bn1(self.fc1(x0)))
        x2 = self.relu(self.bn2(self.fc2(x1)))
        x3 = self.relu(self.bn3(self.fc3(x1 + x2)))
        x4 = self.relu(self.bn4(self.fc4(x1 + x2 + x3)))
        x5 = self.relu(self.bn5(self.fc5(x1 + x2 + x3 + x4)))
        x6 = self.relu(self.bn6(self.fc6(x5)))
        x7 = self.relu(self.bn7(self.fc7(x5 + x6)))
        x8 = self.relu(self.bn8(self.fc8(x5 + x6 + x7)))
        x9 = self.relu(self.bn9(self.fc9(x8)))
        x10 = self.relu(self.bn10(self.fc10(x8 + x9)))
        x11 = self.relu(self.bn11(self.fc11(x8 + x9 + x10)))
        x12 = self.relu(self.bn12(self.fc12(x11)))
        x13 = self.relu(self.bn13(self.fc13(x11 + x12)))
        x14 = self.relu(self.bn14(self.fc14(x11 + x12 + x13)))
        x15 = self.relu(self.bn15(self.fc15(x14)))
        x = self.fc16(x14 + x15)
        return x


# @ac.aicc_monitor
def train(args):
    """train function"""
    aicc_path = './aicc_tools-0.1.7-py3-none-any.whl'
    os.system(f'pip install {aicc_path}')
    os.system('pip install fsspec')
    os.system('pip list')
    print("success")
    init()
    import aicc_tools as ac

    cfts = ac.CFTS(obs_path="obs://sed-ann/train/train1/cc/", upload_frequence=1, keep_last=True)

    x_path = cfts.get_dataset(dataset_path="obs://sed-ann/train/train1/cc/data_cc/Input_cc_v5.csv")
    y_path = cfts.get_dataset(dataset_path="obs://sed-ann/train/train1/cc/data_cc/MP_SED_cc.csv")
    x_train = pd.read_csv(x_path, header=None)
    y_train = pd.read_csv(y_path, header=None, usecols=[0])
    print("x_train shape ", x_train.shape)
    print("y_train shape ", y_train.shape)

    if not os.path.exists('./data_prepare'):
        os.mkdir('./data_prepare')
    else:
        shutil.rmtree('./data_prepare')
        os.mkdir('./data_prepare')

    x_train = x_train.astype(np.float32)
    np.save('./data_prepare/x_train', x_train)
    y_train = y_train.astype(np.float32)
    np.save('./data_prepare/y_train', y_train)

    context.set_auto_parallel_context(parallel_mode=ParallelMode.AUTO_PARALLEL, gradients_mean=True)

    electromagnetic_train = ExistedDataConfig(name="electromagnetic_train",
                                              data_dir=['./data_prepare/x_train.npy',
                                                        './data_prepare/y_train.npy'],
                                              columns_list=["inputs", "label"],
                                              data_format="npy")

    rank_id = get_rank()
    rank_size = get_group_size()
    print("rank_id=", rank_id)
    print("rank_size=", rank_size)

    train_dataset = Dataset(existed_data_list=[electromagnetic_train])
    train_batch_size = 8000
    train_loader = train_dataset.create_dataset(batch_size=train_batch_size, shuffle=True, num_shards=rank_size,
                                                shard_id=rank_id)
    model_net = Model()

    milestones = [131000, 262000, 393000]
    learning_rates = [0.01, 0.005, 0.001]
    optim = nn.Adam(model_net.trainable_params(),
                    learning_rate=nn.piecewise_constant_lr(milestones, learning_rates))

    solver = Solver(network=model_net,
                    mode="Data",
                    optimizer=optim,
                    loss_fn=nn.MSELoss())

    ckpt_cb = cfts.checkpoint_monitor(prefix="model", save_checkpoint_steps=32, keep_checkpoint_max=1500,
                                      integrated_save=False)

    time_monitor = TimeMonitor()
    loss_monitor = LossMonitor(1)
    obs_cb = cfts.obs_monitor()
    callbacks_train = [time_monitor, loss_monitor, ckpt_cb, obs_cb]

    solver.train(epoch=args.epochs,
                 train_dataset=train_loader,
                 callbacks=callbacks_train,
                 dataset_sink_mode=True)


if __name__ == '__main__':
    args_ = parse_args()
    if args_.device_target is not None:
        context.set_context(device_targer=args_.device_target)
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)
    set_seed(123)
    train(args_)
