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
# ==============================================================================
"""padding"""


def periodic_padding(features, source_index, target_index):
    """

    Args:
        features (Tensor): shape [n, ...], the origin features
        source_index (Tensor): shape [m,]
        target_index (Tensor): shape [m,]

    Returns:
        features (Tensor): shape [n, ...], the padded features
    """
    features[target_index] = features[source_index]
    return features


def dirichlet_padding(features, padding_index, padding_value):
    """dirichlet padding"""
    if len(features.shape) == 3:
        #  (m, t, d)
        features[padding_index] = padding_value.unsqueeze(1)\
            .repeat(features.shape[1], axis=1)
    else:  # == 2
        features[padding_index] = padding_value
    return features


def neumann_padding(features, source_index, target_index):
    """neumann padding"""
    features[target_index] = features[source_index]
    return features


def graph_padding(graph, clone=False):
    """graph padding"""
    if hasattr(graph, 'dirichlet_index'):
        graph.y = dirichlet_padding(graph.y, graph.dirichlet_index,
                                    graph.dirichlet_value)
    if hasattr(graph, 'inlet_index'):
        graph.y = dirichlet_padding(graph.y, graph.inlet_index,
                                    graph.inlet_value)
    if hasattr(graph, 'periodic_src_index'):
        graph.y = periodic_padding(graph.y, graph.periodic_src_index,
                                   graph.periodic_tgt_index)
    if hasattr(graph, 'neumann_src_index'):
        graph.y = neumann_padding(graph.y, graph.neumann_src_index,
                                  graph.neumann_tgt_index)

    if clone:
        graph.y = graph.y.copy()


def h_padding(h, graph):
    """hidden state padding"""
    if hasattr(graph, 'dirichlet_index'):
        h = dirichlet_padding(h, graph.dirichlet_index,
                              graph.dirichlet_h_value)
    if hasattr(graph, 'inlet_index'):
        h = dirichlet_padding(h, graph.inlet_index,
                              graph.inlet_h_value)
    if hasattr(graph, 'periodic_src_index'):
        h = periodic_padding(h, graph.periodic_src_index,
                             graph.periodic_tgt_index)
    if hasattr(graph, 'neumann_src_index'):
        h = neumann_padding(h, graph.neumann_src_index,
                            graph.neumann_tgt_index)
    return h
