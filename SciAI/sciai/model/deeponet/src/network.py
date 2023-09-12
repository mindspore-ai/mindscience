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

"""deeponet network"""
import mindspore as ms
from mindspore import ops, nn

from sciai.architecture import MLP


class DeepONet(nn.Cell):
    """DeepONet"""
    def __init__(self, layers_u, layers_y):
        super(DeepONet, self).__init__()
        self.net_u = MLP(layers_u, weight_init="xavier_normal", bias_init="zeros", activation="tanh")
        self.net_y = MLP(layers_y, weight_init="xavier_normal", bias_init="zeros", activation="tanh",
                         last_activation="tanh")
        self.b0 = ms.Parameter(ms.Tensor(0.0, dtype=ms.float32))
        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def construct(self, x_u, x_y, y):
        net_u = self.net_u(x_u)
        net_y = self.net_y(x_y)
        net_o = self.reduce_sum(net_u * net_y, 1) + self.b0
        loss = ops.reduce_mean(ops.square(net_o - y)) / ops.reduce_mean(ops.square(y))
        return loss, net_o


class SampleNet(nn.Cell):
    def __init__(self, net):
        super(SampleNet, self).__init__(auto_prefix=False)
        self.net = net

    def construct(self, x_u_train, x_y_train, y_train, indexes_interval):
        x_u = x_u_train[indexes_interval, :]
        x_y = x_y_train[indexes_interval, :]
        y = y_train[indexes_interval, :]
        return self.net(x_u, x_y, y)
