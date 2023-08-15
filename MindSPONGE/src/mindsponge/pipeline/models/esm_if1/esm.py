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
"""esm model"""
import mindspore as ms
from mindspore import jit, nn
from mindspore import ops
# pylint: disable=relative-beyond-top-level
from .module.esm_wrapcell import TrainOneStepCell
from .nn_arch import GVPTransformerModel as esm
from ..model import Model
from .module.util import Alphabet


class ESM(Model):
    """ESM Model"""
    name = "ESM"

    def __init__(self, config):
        self.mixed_precision = False
        self.config = config
        self.use_jit = self.config.use_jit
        self.temperature = self.config.temperature
        self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/esm/checkpoint/esm_if1.ckpt'
        self.checkpoint_path = "./esm_if1.ckpt"
        self.alphabet = Alphabet.from_architecture('vt_medium_with_invariant_gvp')
        self.network = esm(self.config, self.alphabet)
        self.is_training = self.config.is_training
        if self.is_training:
            self.feature_list = ['coords', 'confidence', 'padding_mask', 'prev_output_tokens', 'target']
            loss = nn.CrossEntropyLoss()
            net_with_loss = nn.WithLossCell(self.network, loss)
            opt = nn.Adam(net_with_loss.trainable_params(), learning_rate=0.0001, eps=1e-6)
            self.train_net = TrainOneStepCell(net_with_loss, opt)
            self.train_net.set_train()
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name,
                         mixed_precision=self.mixed_precision)

    def forward(self, data):
        if self.use_jit:
            outputs = self._jit_forward(data)
        else:
            outputs = self._pynative_forward(data)
        return outputs

    # pylint: disable=arguments-differ
    def predict(self, inputs):
        sampled_seq = self.forward(inputs)
        return sampled_seq

    def loss(self, data):
        pass

    def grad_operations(self, gradient):
        pass

    # pylint: disable=arguments-differ
    def backward(self, feat):
        loss = self.train_net(feat)
        return loss

    def train_step(self, data):
        result = self.backward(data)
        coord_mask = ops.IsFinite()(data['coords']).all(axis=-1).all(axis=-1)
        coord_mask = coord_mask[:, 1:-1]
        loss = ops.ReduceSum()(result * coord_mask) / ops.ReduceSum()(ops.Cast()(coord_mask, ms.float32))
        print("loss is:", loss)
        return loss

    @jit
    def _jit_forward(self, data):
        sampled_seq = self.network.sample(data, self.temperature)
        return sampled_seq

    def _pynative_forward(self, data):
        sampled_seq = self.network.sample(data, self.temperature)
        return sampled_seq
