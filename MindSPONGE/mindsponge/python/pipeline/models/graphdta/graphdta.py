# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""graphdta"""
import numpy as np

from mindspore import jit, nn
from mindspore import Tensor
from mindspore.common import mutable
from mindspore.nn import TrainOneStepCell

from .nn_arch import Graphdta
from ..model import Model


class GraphDTA(Model):
    """GraphDTA"""
    name = "GraphDTA"
    feature_list = ["x_feature", "x_mask", "edge_feature", "edge_mask", "target_feature", "target_mask", "label",
                    "batch_info", "index_all"]

    def __init__(self, config):
        self.config = config
        self.use_jit = self.config.use_jit
        self.white_list = (nn.Softmax, nn.LayerNorm)
        self.checkpoint_url = self.config.checkpoint_url
        self.network = Graphdta(self.config)

        if self.config.train:
            self.lr = float(self.config.lr)
            optimizer = nn.Adam(params=self.network.trainable_params(),
                                learning_rate=self.lr, weight_decay=self.config.weight_decay)
            self.train_wrapper = TrainOneStepCell(self.network, optimizer=optimizer)
            self.train_wrapper.set_train(True)

        else:
            self.network.set_train(False)

        super().__init__(self.checkpoint_url, self.network, self.name, self.white_list)

    @jit
    def backward(self, data):
        loss = self.train_wrapper(*data)
        return loss

    def forward(self, data):
        feat = []
        for key in self.feature_list:
            feat.append(data[key])
        if self.use_jit:
            out = self._jit_forward(feat)
        else:
            out = self._pynative_forward(feat)
        return out

    def predict(self, data, **kwargs):
        for key in data:
            data[key] = Tensor(data[key])
        logits = self.forward(data)
        return logits

    def loss(self, data):
        pass

    def grad_operations(self, gradient):
        pass

    def train_step(self, data):
        """train step"""

        inputs_feats = np.array(data["x_feature"], np.float32), \
            np.array(data["edge_feature"], np.int32), \
            np.array(data["target_feature"], np.int32), \
            np.array(data["batch_info"], np.int32), \
            np.array(data["label"], np.float32)
        inputs_feat = [Tensor(feat) for feat in inputs_feats]
        inputs_feat = mutable(inputs_feat)
        loss = self.backward(inputs_feat)
        return loss

    def _pynative_forward(self, data):
        out = self.network(*data)
        return out

    @jit
    def _jit_forward(self, data):
        out = self.network(*data)
        return out
