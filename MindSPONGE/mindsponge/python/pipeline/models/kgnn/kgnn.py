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
"""kgnn"""
import time

from mindspore import jit, context, nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.common import mutable

from ..model import Model
from .nn_arch import KGCN
from .loss import BinaryCrossEntropyLoss


def get_optimizer(optimizer_name, params, lr, l2_weight):
    """get_optimizer"""
    if optimizer_name == 'sgd':
        return nn.SGD(params=params, learning_rate=lr, weight_decay=l2_weight)
    if optimizer_name == 'rmsprop':
        return nn.RMSProp(params=params, learning_rate=lr, weight_decay=l2_weight)
    if optimizer_name == 'adagrad':
        return nn.Adagrad(params=params, learning_rate=lr, weight_decay=l2_weight)
    if optimizer_name == 'adadelta':
        return nn.Adadelta(params=params, learning_rate=lr, weight_decay=l2_weight)
    if optimizer_name == 'adam':
        return nn.Adam(params=params, learning_rate=lr, weight_decay=l2_weight)
    raise ValueError('Optimizer Not Understood: {}'.format(optimizer_name))


class KGNN(Model):
    """KGNN Model"""
    def __init__(self, config):
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        self.config = config
        self.use_jit = self.config.use_jit
        self.is_training = self.config.is_training
        self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/kgnn/checkpoint/kgnn.ckpt'
        self.checkpoint_path = "./kgnn.ckpt"
        self.network = KGCN(self.config)
        if self.is_training:
            loss_fn = BinaryCrossEntropyLoss()
            optimizer = get_optimizer(self.config.optimizer, self.network.trainable_params(),
                                      self.config.lr, self.config.l2_weight)
            loss_net = WithLossCell(self.network, loss_fn)
            self.train_wrapper = TrainOneStepCell(loss_net, optimizer=optimizer)
            self.train_wrapper.set_train()
        else:
            self.network.set_train(False)
        super().__init__(self.checkpoint_url, self.network)


    def forward(self, data):
        if self.use_jit:
            mse = self._jit_forward(data)
        else:
            mse = self._pynative_forward(data)
        return mse

    # pylint: disable=arguments-differ
    def predict(self, data):
        feat = Tensor(data.get("data"))
        t1 = time.time()
        mse = self.forward(feat)
        t2 = time.time()
        print(round(t2 - t1, 2))
        return mse


    def loss(self, data):
        pass


    def grad_operations(self, gradient):
        pass


    @jit
    def backward(self, data):
        loss = self.train_wrapper(*data)
        return loss


    def train_step(self, data):
        feat = []
        feat.append(Tensor(data.get("data")))
        feat.append(Tensor(data.get("label")))
        feat = mutable(feat)
        loss = self.backward(feat)
        return loss


    @jit
    def _jit_forward(self, data):
        mse = self.network(data)
        return mse


    def _pynative_forward(self, data):
        mse = self.network(data)
        return mse
