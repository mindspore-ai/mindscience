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
"""Network definition"""
from mindspore import ops, nn
from sciai.architecture import MLP, Normalize


class PeriodicSphere(nn.Cell):
    """Periodic sphere class"""
    def construct(self, inputs0, inputs1):
        """Network forward pass"""
        return ops.concat((ops.cos(inputs1) * ops.cos(inputs0),
                           ops.cos(inputs1) * ops.sin(inputs0),
                           ops.sin(inputs1)), axis=1)


# Define the network
class Model(nn.Cell):
    """Model definition"""
    def __init__(self, t0, tfinal, layers):
        super().__init__()
        self.normalize = Normalize(t0, tfinal)
        self.periodic_sphere = PeriodicSphere()
        self.concat = ops.Concat(axis=1)
        self.mlp = MLP(layers, weight_init="XavierNormal", bias_init="zeros", activation="tanh")

    def construct(self, inp1, inp2, inp3):
        """Network forward pass"""
        b1 = self.normalize(inp1)
        b23 = self.periodic_sphere(inp2, inp3)
        b = self.concat((b1, b23))
        out = self.mlp(b)
        return out
