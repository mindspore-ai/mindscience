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
"""train process"""
import os
import argparse
import shutil
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.common import set_seed
from mindspore import context, save_checkpoint
from mindspore.train.callback import TimeMonitor
import mindspore.common.dtype as mstype
from mindelec.solver import Solver
from mindelec.vision import MonitorTrain
from mindelec.data import Dataset, ExistedDataConfig
from metric import EvalMetric
from monitoreval import MonitorEval

set_seed(123456)
np.random.seed(123456)

parser = argparse.ArgumentParser(description='Electromagnetic Inversion for Ground Penetrating Radar')
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--device_target', type=str, default="GPU")
parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')
parser.add_argument('--save_graphs_path', default='./graph_result/', help='checkpoint directory')
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=True,
                    save_graphs_path=opt.save_graphs_path,
                    device_target=opt.device_target,
                    device_id=opt.device_num)


def custom_normalize(data):
    """
    get normalize data
    """
    print("Custom normalization is called")
    ori_shape = data.shape
    data = np.transpose(data)
    mean = np.mean(data, axis=1)
    data = data - mean[:, None]
    std = np.std(data, axis=1)
    std += (np.abs(std) < 0.0000001)
    data = data / std[:, None]
    data = np.transpose(data)
    data = data.reshape(ori_shape)
    return data


class Model(nn.Cell):
    """
    Maxwell inversion model definition
    """
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(3, 10, 3)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(10, 32, 3)
        self.tanh = nn.Tanh()
        self.conv3 = nn.Conv1d(32, 64, 3)
        self.fc1 = nn.Dense(5056, 1200)
        self.fc2 = nn.Dense(1200, 100)
        self.fc3 = nn.Dense(100, 2)
        self.reshape = ops.Reshape()

    def construct(self, x):
        """forward"""
        x1 = self.relu(self.conv1(x))
        x2 = self.maxpool(x1)
        x3 = self.tanh(self.conv2(x2))
        x3 = self.maxpool(x3)
        x4 = self.relu(self.conv3(x3))
        x4 = self.maxpool(x4)
        x5 = self.reshape(x4, (x.shape[0], -1))
        x6 = self.relu(self.fc1(x5))
        x7 = self.fc2(x6)
        output = self.fc3(x7)
        return output


def data_process():
    """
    load data
    """
    ez_input = np.load('./dataset/Ez.npy')
    pos_input = np.load('./dataset/Label.npy')
    hx_input = np.load('./dataset/Hx.npy')
    hy_input = np.load('./dataset/Hy.npy')
    ez_input = custom_normalize(ez_input)
    hx_input = custom_normalize(hx_input)
    hy_input = custom_normalize(hy_input)
    eh_input = np.array([ez_input, hx_input, hy_input])
    eh_input = eh_input.transpose(1, 0, 2)
    train_number = int(eh_input.shape[0] * 0.8)
    train_data = eh_input[0:train_number]
    train_label = pos_input[0:train_number]
    eval_data = eh_input[train_number:]
    eval_label = pos_input[train_number:]
    if not os.path.exists('./data_prepare'):
        os.mkdir('./data_prepare')
    else:
        shutil.rmtree('./data_prepare')
        os.mkdir('./data_prepare')
    np.save('./data_prepare/train_data', train_data)
    np.save('./data_prepare/train_label', train_label)
    np.save('./data_prepare/eval_data', eval_data)
    np.save('./data_prepare/eval_label', eval_label)
    return eval_data


def train():
    """train model"""
    eval_data = data_process()
    electromagnetic_train = ExistedDataConfig(name="electromagnetic_train",
                                              data_dir=['./data_prepare/train_data.npy',
                                                        './data_prepare/train_label.npy'],
                                              columns_list=["inputs", "label"],
                                              data_format="npy")
    electromagnetic_eval = ExistedDataConfig(name="electromagnetic_eval",
                                             data_dir=['./data_prepare/eval_data.npy',
                                                       './data_prepare/eval_label.npy'],
                                             columns_list=["inputs", "label"],
                                             data_format="npy")
    train_dataset = Dataset(existed_data_list=[electromagnetic_train])
    train_batch_size = opt.batch_size
    train_loader = train_dataset.create_dataset(batch_size=train_batch_size, shuffle=True)

    eval_dataset = Dataset(existed_data_list=[electromagnetic_eval])
    eval_batch_size = len(eval_data)
    eval_loader = eval_dataset.create_dataset(batch_size=eval_batch_size, shuffle=False)
    model_net = Model()
    if opt.device_target == "Ascend":
        model_net.to_float(mstype.float16)

    optim = nn.Adam(model_net.trainable_params(), learning_rate=opt.lr)
    eval_error_mrc = EvalMetric(length=eval_batch_size, file_path='./eval_res')

    solver = Solver(network=model_net,
                    mode="Data",
                    optimizer=optim,
                    metrics={'eval_mrc': eval_error_mrc},
                    loss_fn=nn.MSELoss())

    monitor_train = MonitorTrain(per_print_times=1,
                                 summary_dir='./summary_dir_train')

    monitor_eval = MonitorEval(summary_dir='./summary_dir_eval',
                               model=solver,
                               eval_ds=eval_loader,
                               eval_interval=opt.print_interval,
                               draw_flag=False)

    time_monitor = TimeMonitor()
    callbacks_train = [monitor_train, time_monitor, monitor_eval]
    solver.model.train(epoch=opt.epochs,
                       train_dataset=train_loader,
                       callbacks=callbacks_train,
                       dataset_sink_mode=True)

    if not os.path.exists(opt.checkpoint_dir):
        os.mkdir(opt.checkpoint_dir)
    save_checkpoint(model_net, os.path.join(opt.checkpoint_dir, 'model.ckpt'))


if __name__ == '__main__':
    train()
