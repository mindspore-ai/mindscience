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
"""pafnucy Model"""
import time

import mindspore as ms
from mindspore import Tensor, context, jit, nn
from mindspore.common import dtype as mstype
from mindspore.common import mutable
from mindspore.nn import TrainOneStepCell

from ..model import Model
from .nn_arch import SBNetWork


class PAFNUCY(Model):
    """pafnucy model"""
    def __init__(self, config):
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        self.config = config
        self.use_jit = self.config.use_jit
        self.is_training = self.config.is_training
        self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/Pafnucy/checkpoint/pafnucy.ckpt'
        self.checkpoint_path = "./pafnucy.ckpt"
        if self.is_training:
            self.network = SBNetWork(in_channel=[19, 64, 128],
                                     out_channel=self.config.conv_channels,
                                     dense_size=self.config.dense_sizes,
                                     lmbda=self.config.lmbda,
                                     isize=self.config.isize, keep_prob=self.config.keep_prob)
            self.lr = Tensor(float(self.config.lr), mstype.float32)
            optimizer = nn.Adam(params=self.network.trainable_params(),
                                learning_rate=self.lr, weight_decay=self.config.weight_decay)
            self.train_wrapper = TrainOneStepCell(self.network, optimizer=optimizer)
            self.network.set_train()
        else:
            self.network = SBNetWork(in_channel=[19, 64, 128],
                                     out_channel=config.conv_channels,
                                     dense_size=config.dense_sizes,
                                     lmbda=config.lmbda,
                                     isize=config.isize, keep_prob=1.0)
            self.network.set_train(False)

        super().__init__(self.checkpoint_url, self.network)


    def forward(self, data):
        if self.use_jit:
            result = self._jit_forward(data)
        else:
            result = self._pynative_forward(data)
        return result

    # pylint: disable=arguments-differ
    def predict(self, data):
        """predict"""
        feat = []
        test_data = {}
        test_data["coords_feature"] = Tensor(data["coords_feature"], ms.float32)
        if len(test_data.get("coords_feature").shape) == 4:
            test_data["coords_feature"] = test_data.get("coords_feature").expand_dims(axis=0)
        test_data["affinity"] = Tensor(data.get("affinity"), ms.float32)

        feat.append(test_data.get("coords_feature"))
        feat.append(test_data.get("affinity"))
        feat = mutable(feat)

        t1 = time.time()
        result = self.forward(feat)
        t2 = time.time()
        print(round(t2 - t1, 2))
        return result


    def loss(self, data):
        pass


    def grad_operations(self, gradient):
        pass


    @jit
    def backward(self, data):
        loss = self.train_wrapper(*data)
        return loss


    def train_step(self, data):
        """train step"""
        feat = []
        feat.append(Tensor(data.get("coords_feature"), ms.float32))
        feat.append(Tensor(data.get("affinity"), ms.float32))
        feat.append(Tensor(data.get("rot")[-1]))
        feat = mutable(feat)
        loss = self.backward(feat)
        return loss


    @jit
    def _jit_forward(self, data):
        result = self.network(*data)
        return result


    def _pynative_forward(self, data):
        result = self.network(*data)
        return result
