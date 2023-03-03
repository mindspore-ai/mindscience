# Copyright 2023 @ Huawei Technologies Co., Ltd
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
"""DeepDR"""
import numpy as np
import mindspore as ms
from mindspore import Tensor, jit, nn, ops
from mindspore.common import mutable

from ..model import Model
from .mda_nn_arch import MDA, MDALoss, MDAWithLossCell, MDACustomTrainOneStepCell
from .vae_nn_arch import VAE, CvaeLoss, CustomTrainOneStepCell


class DeepDR(Model):
    """DeepDR"""
    arch = {}
    arch['mda'] = {}
    arch['mda']['drug'] = {}
    arch['mda']['drug'] = {1: [9 * 100],
                           2: [9 * 1000, 9 * 100, 9 * 1000],
                           3: [9 * 1000, 9 * 500, 9 * 100, 9 * 500, 9 * 1000],
                           4: [9 * 1000, 9 * 500, 9 * 200, 9 * 100, 9 * 200, 9 * 500, 9 * 1000],
                           5: [9 * 1000, 9 * 800, 9 * 500, 9 * 200, 9 * 100, 9 * 200, 9 * 500, 9 * 800, 9 * 1000],
                           }
    arch['ae'] = {}
    arch['ae']['drug'] = {}
    arch['ae']['drug'] = {1: [1000],
                          2: [2000, 1000, 2000],
                          3: [2000, 1500, 1000, 1500, 2000],
                          }

    def __init__(self, config):
        self.config = config
        self.use_jit = self.config.use_jit
        self.is_training = self.config.is_training
        self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/deepdr/checkpoint/cvae.ckpt'
        self.checkpoint_path = "./cvae.ckpt"

        if self.config.model == 'mda':
            self.input_dims = []
            for i in self.config.mda_select_nets:
                self.input_dims.append(self.config.input_dims[i-1])
            if self.is_training:
                encoding_dims = DeepDR.arch.get(self.config.modeltype).get(self.config.ORG).get(self.config.select_arch)
                self.network = MDA(input_dims=self.input_dims, encoding_dims=encoding_dims)
            else:
                encoding_dims = DeepDR.arch.get('mda').get('drug').get(3)
                self.network = MDA(input_dims=self.input_dims, encoding_dims=encoding_dims, train=False)
            sgd = nn.SGD(params=self.network.trainable_params(), learning_rate=0.005,
                         momentum=0.9, weight_decay=0.0, nesterov=False)
            loss = MDALoss()
            net_with_loss = MDAWithLossCell(self.network, loss_fn=loss)
            self.train_wrapper = MDACustomTrainOneStepCell(net_with_loss, sgd)
        elif self.config.model == 'cvae':
            self.vae_encoder_layer_sizes = self.config.vae_encoder_layer_sizes
            self.vae_latent_size = self.config.vae_latent_size
            self.vae_decoder_layer_sizes = self.config.vae_decoder_layer_sizes
            if self.config.rate:
                self.a = self.config.a_rate
                self.b = self.config.b_rate
                loss = CvaeLoss(option=2, alpha=self.a, beta=self.b)
            else:
                self.a = self.config.a
                self.b = self.config.b
                loss = CvaeLoss(option=1, alpha=self.a, beta=self.b)
            self.network = VAE(self.vae_encoder_layer_sizes, self.vae_latent_size, self.vae_decoder_layer_sizes)
            net_with_loss = nn.WithLossCell(self.network, loss)
            if self.config.rate:
                optimizer = nn.Adam(params=self.network.trainable_params(),
                                    learning_rate=self.config.learn_rate, weight_decay=1e-1)
            else:
                optimizer = nn.Adam(params=self.network.trainable_params(), learning_rate=self.config.learn_rate)
            self.train_wrapper = CustomTrainOneStepCell(net_with_loss, optimizer)
        else:
            raise ValueError("Invalid model type!", self.config.model)

        if self.is_training:
            self.train_wrapper.set_train()
        else:
            self.train_wrapper.set_train(False)
        super().__init__(self.checkpoint_url, self.network)


    def forward(self, data):
        if self.use_jit:
            result = self._jit_forward(data)
        else:
            result = self._pynative_forward(data)
        return result


    def predict(self, data, **kwargs):
        feat = []
        if self.config.model == 'mda':
            test_data = Tensor(np.array(data).astype(np.float32))
            feat.append(test_data)
            result = self.forward(feat)
            return result
        feat.append(data)
        result, _, _, _, _ = self.forward(feat)
        return result.asnumpy()


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
        if self.config.model == 'mda':
            feat.append(ops.cast(data.get("data"), ms.float32))
            feat.append(ops.cast(data.get("label"), ms.float32))
        elif self.config.model == 'cvae':
            feat.append(data.get("data"))
            feat.append(data.get("label"))
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
