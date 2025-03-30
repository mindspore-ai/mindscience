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
"""encoder and decoder module"""
from mindspore import nn

from .utils import build_net


class Encoder(nn.Cell):
    "Encoder"
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.net = build_net(layers)

    def construct(self, inputs):
        return self.net(inputs)


class Decoder(nn.Cell):
    "Decoder"
    def __init__(self, layers):
        super().__init__()
        self.layers = layers
        self.net = build_net(layers)

    def construct(self, h):
        return self.net(h)
