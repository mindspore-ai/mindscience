# Copyright 2025 Huawei Technologies Co., Ltd
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
"""model utils"""
from collections import OrderedDict

from mindspore import nn


def activation_func():
    """activation function"""
    return nn.ELU()


def build_net(layers, activation_end=False):
    """build net"""
    net = nn.SequentialCell()
    layer_n = len(layers)

    assert layer_n >= 2

    for i in range(layer_n - 2):
        net.append(nn.Dense(layers[i], layers[i + 1]))
        net.append(activation_func())
    net.append(nn.Dense(layers[layer_n - 2], layers[layer_n - 1]))
    if activation_end:
        net.append(activation_func())
    return net


def build_dict_net(layers, activation_end=False):
    """build dict net"""
    layer_n = len(layers)

    assert layer_n >= 2
    d = OrderedDict()
    for i in range(layer_n - 2):
        d['dense' + str(i)] = nn.Dense(layers[i], layers[i + 1])
        d['activation' + str(i)] = activation_func()
    d['dense' + str(layer_n - 2)] = nn.Dense(layers[layer_n - 2], layers[layer_n - 1])
    if activation_end:
        d['activation' + str(layer_n - 2)] = activation_func()

    net = nn.SequentialCell(d)
    return net
