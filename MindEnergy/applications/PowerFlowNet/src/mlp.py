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
"""
Multi-Layer Perceptron (MLP) Implementation for PowerFlowNet
MindSpore Version - CPU and NPU Compatible
"""

from mindspore import nn


class MLPNet(nn.Cell):
    """Simple MLP Network for Power Flow Prediction (baseline)"""

    def __init__(self, nfeature_dim, output_dim, hidden_dim, n_layers, dropout_rate):
        super().__init__()
        self.nfeature_dim = nfeature_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate

        # Build layers
        layers = []
        layers.append(nn.Dense(nfeature_dim, hidden_dim))

        for _ in range(n_layers - 2):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            layers.append(nn.Dense(hidden_dim, hidden_dim))

        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))
        layers.append(nn.Dense(hidden_dim, output_dim))

        self.model = nn.SequentialCell(layers)

    def construct(self, data):
        """Forward pass of the MLP network.

        Args:
            data (Union[Data, Tensor]): Input data containing node features.
                If Data object, uses x attribute. If Tensor, directly uses the tensor.

        Returns:
            Tensor: Output features with shape (num_nodes, output_dim).
        """
        # Handle both Data objects and direct tensors
        if hasattr(data, 'x'):
            x = data.x
        else:
            x = data
        return self.model(x)
