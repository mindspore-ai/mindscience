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
Model
"""
import mindspore
import mindspore.nn as nn


class MLP(nn.Cell):
    """
    mindspore MLP架构
    """
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Dense(config["input_num"], config["layer1_output"])
        self.fc2 = nn.Dense(config["layer1_output"], config["layer2_output"])
        self.fc3 = nn.Dense(config["layer2_output"], config["layer3_output"])
        self.fc4 = nn.Dense(config["layer3_output"], config["layer4_output"])
        self.fc5 = nn.Dense(config["layer4_output"], config["layer5_output"])
        if config["activation_function"] == "ReLU":
            self.actfunc = nn.ReLU()
        elif config["activation_function"] == "LeakyReLU":
            self.actfunc = nn.LeakyReLU()
        elif config["activation_function"] == "Tanh":
            self.actfunc = nn.Tanh()
        elif config["activation_function"] == "Sigmoid":
            self.actfunc = nn.Sigmoid()
        else:
            self.actfunc = nn.ReLU()

    def construct(self, input_tensor):
        """
        前向传播网络
        """
        input_tensor = self.actfunc(self.fc1(input_tensor))
        input_tensor = self.actfunc(self.fc2(input_tensor))
        input_tensor = self.actfunc(self.fc3(input_tensor))
        input_tensor = self.actfunc(self.fc4(input_tensor))
        output = self.fc5(input_tensor)
        return output


class ResMLP(nn.Cell):
    """
    带残差连接的全连接神经网络
    """
    def __init__(self, input_num, width, depth, output_num):
        super(ResMLP, self).__init__()
        self.linear_input = nn.Dense(input_num, width)
        self.layers = nn.CellList([nn.Dense(width, width) for _ in range(depth)])
        self.linear_out = nn.Dense(width, output_num)
        self.act_func = nn.ReLU()
        self.resw = mindspore.Parameter(mindspore.numpy.zeros(depth), requires_grad=True)

    def construct(self, input_tensor):
        """construct"""
        input_tensor = self.act_func(self.linear_input(input_tensor))
        for i, _ in enumerate(self.layers):
            input_tensor = input_tensor + self.resw[i] * self.act_func(self.layers[i](input_tensor))
        input_tensor = self.linear_out(input_tensor)
        return input_tensor
