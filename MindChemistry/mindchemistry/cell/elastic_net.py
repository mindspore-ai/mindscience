# Copyright 2024 Huawei Technologies Co., Ltd
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
"""define models"""
import mindspore as ms


class ElasticNet(ms.nn.Cell):
    """elastic net"""

    def __init__(self, n_inputs: int = 1, n_outputs: int = 1, save_type=None):
        """init"""
        super().__init__()
        self.save_type = save_type
        if self.save_type == 'density':
            self.hidden_size1 = 512
            self.hidden_size2 = 1024
            self.hidden_size3 = 2048
        else:
            self.hidden_size1 = 256
            self.hidden_size2 = 64
            self.hidden_size3 = 16
        self.hidden_layer1 = ms.nn.Dense(n_inputs, self.hidden_size1)
        self.hidden_layer2 = ms.nn.Dense(self.hidden_size1, self.hidden_size2)
        self.hidden_layer3 = ms.nn.Dense(self.hidden_size2, self.hidden_size3)
        self.output_layer = ms.nn.Dense(self.hidden_size3, n_outputs)
        self.output_layer1 = ms.nn.Dense(self.hidden_size3, n_outputs)
        self.batch_norm1 = ms.nn.BatchNorm1d(self.hidden_size1)
        self.batch_norm2 = ms.nn.BatchNorm1d(self.hidden_size2)
        self.batch_norm3 = ms.nn.BatchNorm1d(self.hidden_size3)
        self.relu = ms.nn.ReLU()
        self.optimizer = None

    def construct(self, x):
        """construct"""
        if self.save_type == 'density':
            x = self.relu(self.batch_norm1(self.hidden_layer1(x)))
            x = self.relu(self.batch_norm2(self.hidden_layer2(x)))
            x = self.relu(self.batch_norm3(self.hidden_layer3(x)))
            outputs = self.output_layer(x.astype(ms.float32))
        else:
            x = self.relu(self.hidden_layer1(x))
            x = self.relu(self.hidden_layer2(x))
            x = self.relu(self.hidden_layer3(x))
            outputs = self.output_layer1(x.astype(ms.float32))
        return outputs
