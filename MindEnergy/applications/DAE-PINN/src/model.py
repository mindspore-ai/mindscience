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
"""backbone model for DAE-PINN"""

import numpy as np
from mindspore import nn, ops
from .layer import Fnn, Attention, Conv1D


def dyn_input_feature_layer(x):
    return ops.cat((x, ops.cos(np.pi * x), ops.sin(np.pi * x), ops.cos(2 * np.pi * x), ops.sin(2 * np.pi * x)), axis=-1)


class ThreeBusPN(nn.Cell):
    """ThreeBusPN network"""

    def __init__(self,
                 dynamic,
                 algebraic,
                 use_input_layer=None,
                 stacked=False):
        super().__init__()
        self.stacked = stacked
        self.dim = 4
        self.num_irk_stages = dynamic.num_irk_stages
        if stacked:
            num_layer = self.dim
        else:
            num_layer = 1
        dyn_in_transform = dyn_input_feature_layer if use_input_layer else None
        dyn_out_transform = None
        alg_out_transform = ops.Softplus()
        alg_in_transform = None
        if dynamic.type == "fnn":
            self.y_net = nn.CellList([
                Fnn(
                    dynamic.layer_size,
                    dynamic.activation,
                    dynamic.initializer,
                    dropout_rate=dynamic.dropout_rate,
                    batch_normalization=dynamic.batch_normalization,
                    layer_normalization=dynamic.layer_normalization,
                    input_transform=dyn_in_transform,
                    output_transform=dyn_out_transform
                ) for _ in range(num_layer)
            ])

        elif dynamic.type == "attention":
            self.y_net = nn.CellList([
                Attention(
                    dynamic.layer_size,
                    dynamic.activation,
                    dynamic.initializer,
                    dropout_rate=dynamic.dropout_rate,
                    batch_normalization=dynamic.batch_normalization,
                    layer_normalization=dynamic.layer_normalization,
                    input_transform=dyn_in_transform,
                    output_transform=dyn_out_transform
                ) for _ in range(num_layer)
            ])

        elif dynamic.type == "Conv1D":
            self.y_net = nn.CellList([
                Conv1D(
                    dynamic.layer_size,
                    dynamic.activation,
                    dropout_rate=dynamic.dropout_rate,
                    batch_normalization=dynamic.batch_normalization,
                    layer_normalization=dynamic.layer_normalization,
                    input_transform=dyn_in_transform,
                    output_transform=dyn_out_transform
                ) for _ in range(num_layer)
            ])
        else:
            raise ValueError(f"{dynamic.type} type on NN not implemented")

        if algebraic.type == "fnn":
            self.z_net = Fnn(
                algebraic.layer_size,
                algebraic.activation,
                algebraic.initializer,
                dropout_rate=algebraic.dropout_rate,
                batch_normalization=algebraic.batch_normalization,
                layer_normalization=algebraic.layer_normalization,
                input_transform=alg_in_transform,
                output_transform=alg_out_transform
            )
        elif algebraic.type == "attention":
            self.z_net = Attention(
                algebraic.layer_size,
                algebraic.activation,
                algebraic.initializer,
                dropout_rate=algebraic.dropout_rate,
                batch_normalization=algebraic.batch_normalization,
                layer_normalization=algebraic.layer_normalization,
                input_transform=alg_in_transform,
                output_transform=alg_out_transform
            )
        elif algebraic.type == "Conv1D":
            self.z_net = Conv1D(
                algebraic.layer_size,
                algebraic.activation,
                dropout_rate=algebraic.dropout_rate,
                batch_normalization=algebraic.batch_normalization,
                layer_normalization=algebraic.layer_normalization,
                input_transform=alg_in_transform,
                output_transform=alg_out_transform
            )
        else:
            raise ValueError(f"{algebraic.type} type on NN not implemented")

    def construct(self, inputs):
        """construct for ThreeBusPN"""
        if self.stacked:
            y0 = self.y_net[0](inputs)
            y1 = self.y_net[1](inputs)
            y2 = self.y_net[2](inputs)
            y3 = self.y_net[3](inputs)
        else:
            dim_out = self.num_irk_stages + 1
            y = inputs
            for layer in self.y_net:
                y = layer(y)
            y0 = y[..., :dim_out]
            y1 = y[..., dim_out:2 * dim_out]
            y2 = y[..., 2 * dim_out:3 * dim_out]
            y3 = y[..., 3 * dim_out:4 * dim_out]
        z = self.z_net(inputs)
        return y0, y1, y2, y3, z
