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

"""network for phylstm_2"""
import mindspore as ms
from mindspore import nn, ops
from mindspore import load_checkpoint, load_param_into_net

from sciai.utils import print_log

def prepare_network(args):
    """prepare network and loss"""
    network = Network()
    loss = Loss()

    if args.load_ckpt:
        param_dict = load_checkpoint(args.load_ckpt_path)
        load_param_into_net(network, param_dict)
        print_log('Successfully loaded ckpt from {}'.format(args.load_ckpt_path))

    return network, loss

class LSTM1(nn.Cell):
    """deep lstm network 1"""
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(1, 100, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(100, 100, num_layers=1, batch_first=True)
        self.relu = nn.ReLU()
        self.dense = nn.Dense(100, 3, weight_init="xavier_uniform")
    def construct(self, ag):
        z = self.dense(self.relu(self.lstm2(self.relu(self.lstm1(ag)[0]))[0]))
        return z

class LSTM2(nn.Cell):
    """deep lstm network 2"""
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(3, 100, num_layers=1, batch_first=True)
        self.lstm2 = nn.LSTM(100, 100, num_layers=1, batch_first=True)
        self.relu = nn.ReLU()
        self.dense = nn.Dense(100, 1, weight_init="xavier_uniform")
    def construct(self, ag):
        z = self.dense(self.relu(self.lstm2(self.relu(self.lstm1(ag)[0]))[0]))
        return z

class Network(nn.Cell):
    """Physics-informed double-LSTM Network"""
    def __init__(self):
        super().__init__()
        self.lstm_1 = LSTM1()
        self.lstm_2 = LSTM2()

    def net(self, ag, phi):
        z = self.lstm_1(ag)
        z_1, z_2, z_3 = z[:, :, :1], z[:, :, 1:2], z[:, :, 2:]

        z_1_dot = ops.matmul(phi, z_1)
        z_2_dot = ops.matmul(phi, z_2)
        return z_1, z_1_dot, z_2, z_2_dot, z_3

    def net_c(self, ag_c, phi):
        z_1_c, z_1_dot_c, z_2_c, z_2_dot_c, z_3_c = self.net(ag_c, phi)
        g = self.lstm_2(ops.concat((z_1_c, z_2_c, z_3_c), 2))
        lift = z_2_dot_c + g
        return z_1_dot_c, z_2_c, lift

    def construct(self, ag, phi, ag_c, phi_c):
        """Network forward pass"""
        a = self.net(ag, phi)
        b = self.net_c(ag_c, phi_c)
        return a, b

    def predict(self, ag, phi):
        z = self.lstm_1(ag)
        z_1_dot = ops.matmul(phi, z[:, :, :1])
        z_2_dot = ops.matmul(phi, z[:, :, 1:2])
        g = self.lstm2(z)
        lift = z_2_dot + g
        return z[:, :, :1], z_1_dot, z[:, :, 1:2], z_2_dot, z[:, :, 2:], lift

class Loss(nn.Cell):
    """loss for training phylstm_2"""
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.reduce_mean = ops.ReduceMean()

    def construct(self, z_1, u_train, z_2, ut_train, z_1_dot_c, z_2_c, lift, ag_c_train, lift_train):
        loss_u = self.reduce_mean(self.mse(z_1, u_train))
        loss_ut = self.reduce_mean(self.mse(z_2, ut_train))
        loss_e = self.reduce_mean(self.mse(z_1_dot_c, z_2_c))
        gamma_ag = ops.matmul(lift_train, ops.ones((lift_train.shape[0], 1, ag_c_train.shape[2]), ms.float32))
        loss_g = self.reduce_mean(self.mse(gamma_ag, lift))
        return loss_u + loss_ut + loss_e + loss_g
