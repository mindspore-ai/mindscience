# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Aggregator"""
import mindspore.nn as nn
import mindspore.ops as ops


class SumAggregator(nn.Cell):
    """SumAggregator"""
    def __init__(self, in_channels, out_channels, weight_init='normal', activation='relu'):
        """

        :param in_channels: ent_embed_dim
        :param out_channels: ent_embed_dim
        :param weight_init:
        :param activation:
        """
        super(SumAggregator, self).__init__()
        self.dense = nn.Dense(in_channels, out_channels, weight_init=weight_init, activation=activation)

    def construct(self, inputs):
        entity, neighbor = inputs
        x = self.dense(entity + neighbor)
        return x


class ConcatAggregator(nn.Cell):
    """ConcatAggregator"""
    def __init__(self, in_channels, out_channels, weight_init='normal', activation='relu'):
        """

        :param in_channels: ent_embed_dim + neighbor_embed_dim
        :param out_channels: ent_embed_dim
        :param weight_init:
        :param activation:
        """
        super(ConcatAggregator, self).__init__()
        self.dense = nn.Dense(in_channels, out_channels, weight_init=weight_init, activation=activation)
        self.concat = ops.Concat(-1)

    def construct(self, inputs):
        entity, neighbor = inputs
        x = self.dense(self.concat((entity, neighbor)))
        return x


class NeighAggregator(nn.Cell):
    """NeighAggregator"""
    def __init__(self, in_channels, out_channels, weight_init='normal', activation='relu'):
        """

        :param in_channels: neighbor_embed_dim
        :param out_channels: ent_embed_dim
        :param weight_init:
        :param activation:
        """
        super(NeighAggregator, self).__init__()
        self.dense = nn.Dense(in_channels, out_channels, weight_init=weight_init, activation=activation)

    def construct(self, inputs):
        _, neighbor = inputs
        x = self.dense(neighbor)
        return x
