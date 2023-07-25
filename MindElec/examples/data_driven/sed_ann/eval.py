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
"""sed-_ann eval"""

import argparse
import os

import numpy as np
import pandas as pd
from mindspore import nn, Tensor, load_checkpoint, load_param_into_net, Model as Model_nn


def parse_args():
    """parse args"""
    parser = argparse.ArgumentParser(description='Parametrization sed_AI Simulation')
    parser.add_argument('--case', default="ic Re")
    parser.add_argument('--param_dict_path', default='./model/model-ic_Re.ckpt')
    parser.add_argument('--predict_input_path', default='./data/Test_ic_50_0.6.csv')
    parser.add_argument('--predict_output_path', default='output_Mag_ic_50_0.6.csv')
    opt = parser.parse_args()
    return opt


def evaluation(opt):
    """evaluation"""
    inputs = pd.read_csv(os.path.join(opt.predict_input_path), header=None)
    print("predict_input shape ", inputs.shape)
    predict_input = inputs.astype(np.float32)
    predict_input = np.array(predict_input)
    predict_input = Tensor(predict_input)
    model_net = Model()
    model_net.set_train(False)
    param_dict = load_checkpoint(opt.param_dict_path)
    load_param_into_net(model_net, param_dict)
    model = Model_nn(model_net)
    result = model.predict(predict_input)
    result = result.asnumpy()
    print(result)
    output_file = "./output/" + opt.predict_output_path
    np.savetxt(output_file, result, delimiter=',')


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


if __name__ == '__main__':
    opt_ = parse_args()
    evaluation(opt_)
