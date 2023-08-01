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
"""define neural network"""

import yaml
import numpy as np
import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import TruncatedNormal

with open('src/default_config.yaml', 'r') as y:
    cfg = yaml.full_load(y)

dx = cfg['ax_spec'] / cfg['nx']
dz = cfg['az_spec'] / cfg['nz']
ax = cfg['xsf'] - cfg['n_absx'] * dx
az = cfg['az_spec'] - cfg['n_absz'] * dz
ub = np.array([ax / cfg['Lx'], az / cfg['Lz'], (cfg['t_m'] - cfg['t_st'])]).reshape(-1, 1).T
ub0 = np.array([ax / cfg['Lx'], az / cfg['Lz']]).reshape(-1, 1).T


def xavier_init(in_dim, out_dim):
    xavier_stddev = np.sqrt(2.0 / (in_dim + out_dim))
    return xavier_stddev


class Net(nn.Cell):
    """parameterize the wave ponential"""

    def __init__(self, layers):
        super(Net, self).__init__()

        self.layer1 = nn.Dense(layers[0], layers[1], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[0], out_dim=layers[1])), activation=nn.Tanh())
        self.layer2 = nn.Dense(layers[1], layers[2], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[1], out_dim=layers[2])), activation=nn.Tanh())
        self.layer3 = nn.Dense(layers[2], layers[3], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[2], out_dim=layers[3])), activation=nn.Tanh())
        self.layer4 = nn.Dense(layers[3], layers[4], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[3], out_dim=layers[4])), activation=nn.Tanh())
        self.layer5 = nn.Dense(layers[4], layers[5], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[4], out_dim=layers[5])), activation=nn.Tanh())
        self.layer6 = nn.Dense(layers[5], layers[6], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[5], out_dim=layers[6])), activation=nn.Tanh())
        self.layer7 = nn.Dense(layers[6], layers[7], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[6], out_dim=layers[7])), activation=nn.Tanh())
        self.layer8 = nn.Dense(layers[7], layers[8], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[7], out_dim=layers[8])), activation=nn.Tanh())
        self.layer9 = nn.Dense(layers[8], layers[9], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[8], out_dim=layers[9])))

        self.op_concat = ops.Concat(1)
        self.ub = ms.Tensor(ub, dtype=ms.float32)

    def construct(self, x, z, t):
        """parameterize the wave ponential"""

        xzt = self.op_concat((x, z, t))
        out = 2 * (xzt / self.ub) - 1
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        return out


class Net0(nn.Cell):
    """parameterize the wave speed"""

    def __init__(self, layers):
        super(Net0, self).__init__()

        self.mix_coe = ms.Tensor(
            np.arange(1, layers[1] + 1)[np.newaxis, :], dtype=ms.float32)

        self.layer1 = nn.Dense(layers[0], layers[1], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[0], out_dim=layers[1])), has_bias=False)
        self.layer2 = nn.Dense(layers[1], layers[2], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[1], out_dim=layers[2])), activation=nn.Tanh())
        self.layer3 = nn.Dense(layers[2], layers[3], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[2], out_dim=layers[3])), activation=nn.Tanh())
        self.layer4 = nn.Dense(layers[3], layers[4], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[3], out_dim=layers[4])), activation=nn.Tanh())
        self.layer5 = nn.Dense(layers[4], layers[5], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[4], out_dim=layers[5])), activation=nn.Tanh())
        self.layer6 = nn.Dense(layers[5], layers[6], weight_init=TruncatedNormal(
            xavier_init(in_dim=layers[5], out_dim=layers[6])))

        self.op_concat = ops.Concat(1)
        self.tanh = nn.Tanh()
        self.add_bias = Parameter(
            Tensor(np.zeros((1, layers[1])), dtype=ms.float32), name='add_bias')

        self.ub0 = ms.Tensor(ub0, dtype=ms.float32)

    def construct(self, x, z):
        """parameterize the wave speed"""

        x_z = self.op_concat((x, z))
        out = 2 * (x_z / self.ub0) - 1

        out = self.tanh(self.layer1(out) * self.mix_coe) + self.add_bias

        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)

        return out
