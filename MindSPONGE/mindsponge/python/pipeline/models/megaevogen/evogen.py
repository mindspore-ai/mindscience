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
"""evogen"""
from mindspore import jit, context
from mindspore.common import mutable
from mindspore import Tensor

from ..model import Model
from .nn_arch import megaevogen


class MEGAEvoGen(Model):
    '''MEGAEvoGen'''
    feature_list = ['seq_mask', 'msa_mask', 'msa_input', 'query_input', 'additional_input', 'evogen_random_data',
                    'evogen_context_mask']

    def __init__(self, config):
        self.name = "MEGAEvoGen"
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
            context.set_context(graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                                   "--composite_op_limit_size=50", enable_graph_kernel=True)
        else:
            self.mixed_precision = True

        self.config = config
        self.use_jit = self.config.use_jit
        self.checkpoint_url = \
            'https://download.mindspore.cn/mindscience/mindsponge/MEGAEvoGen/checkpoint/MEGAEvoGen.ckpt'
        self.network = megaevogen(self.config, self.mixed_precision)
        super().__init__(self.checkpoint_url, self.network, self.name)

    # pylint: disable=invalid-name
    def forward(self, data):
        feat = []
        for key in self.feature_list:
            feat.append(data[key])
        if self.use_jit:
            reconstruct_msa = self._jit_forward(feat)
        else:
            reconstruct_msa = self._pynative_forward(feat)
        return reconstruct_msa

    # pylint: disable=arguments-differ
    @jit
    def backward(self, feat):
        pass

    # pylint: disable=arguments-differ
    def predict(self, inputs):
        for key in inputs:
            inputs[key] = Tensor(inputs[key])
        inputs = mutable(inputs)
        reconstruct_msa = self.forward(inputs)
        return reconstruct_msa

    def loss(self, data):
        pass

    def grad_operations(self, gradient):
        pass

    def train_step(self, data):
        pass

    @jit
    def _jit_forward(self, feat):
        reconstruct_msa = self.network(*feat)
        return reconstruct_msa

    def _pynative_forward(self, feat):
        reconstruct_msa = self.network(*feat)
        return reconstruct_msa
