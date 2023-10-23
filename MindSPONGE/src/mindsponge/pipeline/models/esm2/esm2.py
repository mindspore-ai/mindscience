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
"""esm2 model"""
from mindspore import jit, context
from mindspore.common import mutable
# pylint: disable=relative-beyond-top-level
from .nn_arch import ESM2 as esm2
from ..model import Model


class ESM2(Model):
    """ESM2 Model"""
    name = "ESM"

    def __init__(self, config):
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        self.config = config
        self.use_jit = self.config.use_jit
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
        else:
            self.mixed_precision = True
        self.checkpoint_url = 'https://download.mindspore.cn/mindscience/mindsponge/esm2/checkpoint/esm2.ckpt'
        self.return_contacts = self.config.return_contacts
        self.checkpoint_path = "./esm2.ckpt"
        self.network = esm2(num_layers=self.config.encoder_layers,
                            embed_dim=self.config.encoder_embed_dim,
                            attention_heads=self.config.encoder_attention_heads,
                            alphabet=self.config.alphabet,
                            token_dropout=self.config.token_dropout,
                            return_contacts=self.config.return_contacts)
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name,
                         mixed_precision=self.mixed_precision)

    def forward(self, data):
        if self.return_contacts:
            if self.use_jit:
                x, hidden_representations, attentions, contacts = self._jit_forward(data)
            else:
                x, hidden_representations, attentions, contacts = self._pynative_forward(data)
            result = (x, hidden_representations, attentions, contacts)
        else:
            if self.use_jit:
                x, hidden_representations = self._jit_forward(data)
            else:
                x, hidden_representations = self._pynative_forward(data)
            result = (x, hidden_representations)
        return result

    def predict(self, data, **kwargs):
        batch_tokens = mutable(data)
        forward_data = batch_tokens
        if self.return_contacts:
            x, hidden_representations, attentions, contacts = self.forward(forward_data)
            result = (x, hidden_representations, attentions, contacts)
        else:
            x, hidden_representations = self.network(data)
            result = (x, hidden_representations)
        return result

    def loss(self, data):
        pass

    def grad_operations(self, gradient):
        pass

    def backward(self, data):
        pass

    def train_step(self, data):
        pass

    @jit
    def _jit_forward(self, data):
        if self.return_contacts:
            x, hidden_representations, attentions, contacts = self.network(data)
            result = (x, hidden_representations, attentions, contacts)
        else:
            x, hidden_representations = self.network(data)
            result = (x, hidden_representations)
        return result

    def _pynative_forward(self, data):
        if self.return_contacts:
            x, hidden_representations, attentions, contacts = self.network(data)
            result = (x, hidden_representations, attentions, contacts)
        else:
            x, hidden_representations = self.network(data)
            result = (x, hidden_representations)
        return result
