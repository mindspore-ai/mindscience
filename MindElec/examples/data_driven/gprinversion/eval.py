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
eval
"""

import os
import argparse
import numpy as np
from mindspore import nn
from mindspore import ops
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindelec.solver import Solver
from mindelec.data import Dataset, ExistedDataConfig
from metric import EvalMetric


set_seed(123456)
np.random.seed(123456)

parser = argparse.ArgumentParser(description='Electromagnetic Inversion for Ground Penetrating Radar')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--device_target', type=str, default="GPU")
parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target=opt.device_target,
                    device_id=opt.device_num)


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


def eval_position():
    """eval model"""
    eval_data = np.load('./data_prepare/eval_data.npy')
    electromagnetic_eval = ExistedDataConfig(name="electromagnetic_eval",
                                             data_dir=['./data_prepare/eval_data.npy',
                                                       './data_prepare/eval_label.npy'],
                                             columns_list=["inputs", "label"],
                                             data_format="npy")

    eval_dataset = Dataset(existed_data_list=[electromagnetic_eval])
    eval_batch_size = len(eval_data)
    eval_loader = eval_dataset.create_dataset(batch_size=eval_batch_size, shuffle=False)
    model_net = Model()
    model_net.to_float(mstype.float32)

    param_dict = load_checkpoint(os.path.join(opt.checkpoint_dir, 'model.ckpt'))
    load_param_into_net(model_net, param_dict)
    eval_error_mrc = EvalMetric(length=eval_batch_size, file_path='./eval_res')

    solver = Solver(network=model_net,
                    mode="Data",
                    optimizer=nn.Adam(model_net.trainable_params(), 0.001),
                    metrics={'eval_mrc': eval_error_mrc},
                    loss_fn=nn.MSELoss())

    res_eval = solver.model.eval(valid_dataset=eval_loader, dataset_sink_mode=True)

    loss_mse, l2_pos = res_eval["eval_mrc"]["loss_error"], res_eval["eval_mrc"]["l2_error"]
    print(f'Loss_mse: {loss_mse:.10f}  ',
          f'L2_pos: {l2_pos:.10f}')


if __name__ == '__main__':
    eval_position()
