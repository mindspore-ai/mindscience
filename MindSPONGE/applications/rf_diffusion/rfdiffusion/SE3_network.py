# Modified from RFdiffusion (https://github.com/RosettaCommons/RFdiffusion)
# Original license: BSD License
#
# Copyright 2025 Huawei Technologies Co., Ltd
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


from mindspore import nn
from mindspore.common.initializer import HeNormal, Zero, initializer
from se3_transformer.model import SE3Transformer
from se3_transformer.model.fiber import Fiber

from .util_module import init_lecun_normal_param


class SE3TransformerWrapper(nn.Cell):
    """SE(3) equivariant GCN with attention"""

    def __init__(
        self,
        num_layers=2,
        num_channels=32,
        num_degrees=3,
        n_heads=4,
        div=4,
        l0_in_features=32,
        l0_out_features=32,
        l1_in_features=3,
        l1_out_features=2,
        num_edge_features=32,
    ):
        super().__init__()
        # Build the network
        self.l1_in = l1_in_features

        fiber_edge = Fiber({0: num_edge_features})
        if l1_out_features > 0:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features, 1: l1_out_features})
        else:
            if l1_in_features > 0:
                fiber_in = Fiber({0: l0_in_features, 1: l1_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features})
            else:
                fiber_in = Fiber({0: l0_in_features})
                fiber_hidden = Fiber.create(num_degrees, num_channels)
                fiber_out = Fiber({0: l0_out_features})

        self.se3 = SE3Transformer(
            num_layers=num_layers,
            fiber_in=fiber_in,
            fiber_hidden=fiber_hidden,
            fiber_out=fiber_out,
            num_heads=n_heads,
            channels_div=div,
            fiber_edge=fiber_edge,
            use_layer_norm=True,
        )

        self.reset_parameter()

    def reset_parameter(self):

        # make sure linear layer before ReLu are initialized with kaiming_normal_
        for n, p in self.se3.parameters_and_names():
            if "bias" in n:
                p.set_data(initializer(Zero(), p.shape, p.dtype))
            elif len(p.shape) == 1:
                continue
            else:
                if "radial_func" not in n:
                    p = init_lecun_normal_param(p)
                else:
                    if "net.6" in n:
                        p.set_data(initializer(Zero(), p.shape, p.dtype))
                    else:
                        p.set_data(
                            initializer(HeNormal(nonlinearity="relu"), p.shape, p.dtype)
                        )

        # make last layers to be zero-initialized
        # self.se3.graph_modules[-1].to_kernel_self['0'] = init_lecun_normal_param(self.se3.graph_modules[-1].to_kernel_self['0'])
        # self.se3.graph_modules[-1].to_kernel_self['1'] = init_lecun_normal_param(self.se3.graph_modules[-1].to_kernel_self['1'])
        p = self.se3.graph_modules[-1].to_kernel_self["0"]
        p.set_data(initializer(Zero(), p.shape, p.dtype))
        p = self.se3.graph_modules[-1].to_kernel_self["1"]
        p.set_data(initializer(Zero(), p.shape, p.dtype))

    def construct(self, G, type_0_features, type_1_features=None, edge_features=None):
        if self.l1_in > 0:
            node_features = {"0": type_0_features, "1": type_1_features}
        else:
            node_features = {"0": type_0_features}
        edge_features = {"0": edge_features}
        return self.se3(G, node_features, edge_features)
