# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
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
"""
chebnet
"""


import copy
from mindspore import ops
from mindspore import nn
from .. import layers


class InfoGraph(nn.Cell):
    """
    InfoGraph proposed in
    `InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information
    Maximization`_.

    .. _InfoGraph:
        Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization:
        https://arxiv.org/pdf/1908.01000.pdf

    Args:
        model (nn.Cell): node & graph representation model
        num_mlp_layer (int, optional): number of MLP layers in mutual information estimators
        activation (str or function, optional): activation function
        loss_weight (float, optional): weight of both unsupervised & transfer losses
        separate_model (bool, optional): separate supervised and unsupervised encoders.
            If true, the unsupervised loss will be applied on a separate encoder,
            and a transfer loss is applied between the two encoders.
    """

    def __init__(self, model, num_mlp_layer=2, activation="relu", loss_weight=1, separate_model=False):
        super().__init__()
        self.model = model
        self.separate_model = separate_model
        self.loss_weight = loss_weight
        self.output_dim = self.model.output_dim

        if separate_model:
            self.unsupervised_model = copy.deepcopy(model)
            self.transfer_mi = layers.MutualInformation(model.output_dim, num_mlp_layer, activation)
        else:
            self.unsupervised_model = model
        self.unsupervised_mi = layers.MutualInformation(model.output_dim, num_mlp_layer, activation)

    def construct(self, graph, inputs, all_loss=None, metric=None):
        """
        Compute the node representations and the graph representation(s).
        Add the mutual information between graph and nodes to the loss.

        Args:
            graph (Graph): :math:`n` graph(s)
            inputs (Tensor): input node representations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict

        Returns:
            dict with ``node`` and ``graph`` fields:
                node representations of shape :math:`(|V|, d)`, graph representations of shape :math:`(n, d)`
        """
        output = self.model(graph, inputs)

        if all_loss is not None:
            if self.separate_model:
                unsupervised_output = self.unsupervised_model(graph, inputs)
                mutual_info = self.transfer_mi(output["graph"], unsupervised_output["graph"])

                metric["distillation mutual information"] = mutual_info
                if self.loss_weight > 0:
                    all_loss -= mutual_info * self.loss_weight
            else:
                unsupervised_output = output

            graph_index = graph.node2graph
            node_index = ops.arange(graph.n_node)
            pair_index = ops.stack([graph_index, node_index], axis=-1)

            mutual_info = self.unsupervised_mi(unsupervised_output.graph_feat,
                                               unsupervised_output.node_feat, pair_index)

            metric["graph-node mutual information"] = mutual_info
            if self.loss_weight > 0:
                all_loss -= mutual_info * self.loss_weight

        return output
