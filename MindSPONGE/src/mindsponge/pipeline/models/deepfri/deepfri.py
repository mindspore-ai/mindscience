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
"""deepfri"""
from mindspore import jit, context
import mindspore as ms
from mindspore import Tensor

from ..model import Model
from .nn_arch import Predictor


class DeepFri(Model):
    """deepfri"""

    def __init__(self, config):
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
            context.set_context(graph_kernel_flags="--disable_expand_ops=Softmax --disable_cluster_ops=ReduceSum "
                                                   "--composite_op_limit_size=50", enable_graph_kernel=True)
        else:
            self.mixed_precision = True
        self.config = config
        self.checkpoint_url = \
            f"https://download.mindspore.cn/mindscience/mindsponge/DeepFri/checkpoint/" \
            f"DeepFRI_{self.config.prefix}.ckpt"
        self.checkpoint_path = f"./DeepFRI_{self.config.prefix}.ckpt"
        self.use_jit = self.config.use_jit
        self.network = Predictor(self.config.prefix, self.config, gcn=True)
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network,
                         mixed_precision=self.mixed_precision)

    def forward(self, data):
        pass

    @jit
    def backward(self, data):
        pass

    def train_step(self, data):
        pass

    # pylint: disable=arguments-differ
    def predict(self, inputs):
        inputs[0] = Tensor(inputs[0], dtype=ms.float32)
        inputs[1] = Tensor(inputs[1], dtype=ms.float32)
        if self.use_jit:
            outputs = self._jit_forward(inputs)
        else:
            outputs = self._pynative_forward(inputs)
        return outputs

    @jit
    def _jit_forward(self, data):
        outputs = self.network.predict(*data)
        return outputs

    def _pynative_forward(self, data):
        outputs = self.network.predict(*data)
        return outputs
