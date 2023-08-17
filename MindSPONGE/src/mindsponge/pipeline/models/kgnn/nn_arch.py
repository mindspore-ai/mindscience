# Copyright 2023 @ Shenzhen Bay Laboratory &
#                  Peking University &
#                  Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
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
"""kgnn model"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.initializer import initializer
from .kgnn_data import pickle_load, format_filename

DRUG_VOCAB_TEMPLATE = '{dataset}_drug_vocab.pkl'
ENTITY_VOCAB_TEMPLATE = '{dataset}_entity_vocab.pkl'
RELATION_VOCAB_TEMPLATE = '{dataset}_relation_vocab.pkl'
ADJ_ENTITY_TEMPLATE = '{dataset}_adj_entity.npy'
ADJ_RELATION_TEMPLATE = '{dataset}_adj_relation.npy'


class KGCN(nn.Cell):
    """GCN graph convolution layer."""
    def __init__(self, config):
        super(KGCN, self).__init__()
        self.config = config
        self.n_depth = self.config.n_depth

        self.drug_vocab_size = len(pickle_load(
            format_filename(self.config.DATA_DIR, DRUG_VOCAB_TEMPLATE, dataset=self.config.dataset)))
        self.entity_vocab_size = len(pickle_load(
            format_filename(self.config.DATA_DIR, ENTITY_VOCAB_TEMPLATE, dataset=self.config.dataset)))
        self.relation_vocab_size = len(pickle_load(
            format_filename(self.config.DATA_DIR, RELATION_VOCAB_TEMPLATE, dataset=self.config.dataset)))
        self.adj_entity = np.load(format_filename(
            self.config.DATA_DIR, ADJ_ENTITY_TEMPLATE, dataset=self.config.dataset))
        self.adj_relation = np.load(format_filename(
            self.config.DATA_DIR, ADJ_RELATION_TEMPLATE, dataset=self.config.dataset))

        self.drug_one_embedding = nn.Embedding(vocab_size=self.drug_vocab_size,
                                               embedding_size=self.config.embed_dim,
                                               embedding_table='normal')
        self.entity_embedding = nn.Embedding(vocab_size=self.entity_vocab_size,
                                             embedding_size=self.config.embed_dim,
                                             embedding_table='normal')
        self.relation_embedding = nn.Embedding(vocab_size=self.relation_vocab_size,
                                               embedding_size=self.config.embed_dim,
                                               embedding_table='normal')
        self.get_receptive_field = GetReceptiveField(self.n_depth, self.adj_entity, self.adj_relation)
        self.get_neighbor_info = GetNeighborInfo(self.config.embed_dim, self.config.neighbor_sample_size)
        self.squeeze = ops.Squeeze(axis=1)
        self.reduce_sum = ops.ReduceSum(keep_dims=True)
        self.sigmoid = ops.Sigmoid()

        aggregators_1 = nn.CellList()
        aggregators_2 = nn.CellList()
        for depth in range(self.n_depth):
            activation = 'tanh' if depth == self.n_depth - 1 else 'relu'
            if self.config.aggregator_type == "sum":
                aggregators_1.append(SumAggregator(in_channels=self.config.embed_dim,
                                                   out_channels=self.config.embed_dim,
                                                   activation=activation))
                aggregators_2.append(SumAggregator(in_channels=self.config.embed_dim,
                                                   out_channels=self.config.embed_dim,
                                                   activation=activation))
            elif self.config.aggregator_type == "concat":
                aggregators_1.append(ConcatAggregator(in_channels=self.config.embed_dim + self.config.embed_dim,
                                                      out_channels=self.config.embed_dim,
                                                      activation=activation))
                aggregators_2.append(ConcatAggregator(in_channels=self.config.embed_dim + self.config.embed_dim,
                                                      out_channels=self.config.embed_dim,
                                                      activation=activation))
            elif self.config.aggregator_type == "neigh":
                aggregators_1.append(NeighAggregator(in_channels=self.config.embed_dim,
                                                     out_channels=self.config.embed_dim,
                                                     activation=activation))
                aggregators_2.append(NeighAggregator(in_channels=self.config.embed_dim,
                                                     out_channels=self.config.embed_dim,
                                                     activation=activation))
            else:
                raise ValueError("aggregator type only supports ['sum', 'concat', 'neigh'].")
        self.aggregators_1 = aggregators_1
        self.aggregators_2 = aggregators_2

    def construct(self, inputs):
        """GCN graph convolution layer."""
        input_drug_one, input_drug_two = inputs[:, :1], inputs[:, 1:2]
        drug_embed = self.drug_one_embedding(input_drug_one)  # [batch_size, 1, embed_dim]
        receptive_list_drug_one = self.get_receptive_field(input_drug_one)
        neineigh_ent_list_drug_one = receptive_list_drug_one[:self.n_depth + 1]
        neigh_rel_list_drug_one = receptive_list_drug_one[self.n_depth + 1:]

        neigh_ent_embed_list_drug_one = []
        for neigh_ent in neineigh_ent_list_drug_one:
            neigh_ent_embed_list_drug_one.append(self.entity_embedding(neigh_ent))
        neigh_rel_embed_list_drug_one = []
        for neigh_rel in neigh_rel_list_drug_one:
            neigh_rel_embed_list_drug_one.append(self.relation_embedding(neigh_rel))

        for depth in range(self.n_depth):
            next_neigh_ent_embed_list_drug_one = []
            for hop in range(self.n_depth - depth):
                neighbor_embed = self.get_neighbor_info(drug_embed, neigh_rel_embed_list_drug_one[hop],
                                                        neigh_ent_embed_list_drug_one[hop + 1])
                next_entity_embed = self.aggregators_1[depth](
                    [neigh_ent_embed_list_drug_one[hop], neighbor_embed])
                next_neigh_ent_embed_list_drug_one.append(next_entity_embed)
            neigh_ent_embed_list_drug_one = next_neigh_ent_embed_list_drug_one

        # get receptive field
        receptive_list = self.get_receptive_field(input_drug_two)
        neigh_ent_list = receptive_list[:self.n_depth + 1]
        neigh_rel_list = receptive_list[self.n_depth + 1:]

        neigh_ent_embed_list = []
        for neigh_ent in neigh_ent_list:
            neigh_ent_embed_list.append(self.entity_embedding(neigh_ent))
        neigh_rel_embed_list = []
        for neigh_rel in neigh_rel_list:
            neigh_rel_embed_list.append(self.relation_embedding(neigh_rel))

        for depth in range(self.n_depth):
            next_neigh_ent_embed_list = []
            for hop in range(self.n_depth - depth):
                neighbor_embed = self.get_neighbor_info(drug_embed, neigh_rel_embed_list[hop],
                                                        neigh_ent_embed_list[hop + 1])
                next_entity_embed = self.aggregators_2[depth](
                    [neigh_ent_embed_list[hop], neighbor_embed])
                next_neigh_ent_embed_list.append(next_entity_embed)
            neigh_ent_embed_list = next_neigh_ent_embed_list

        drug1_squeeze_embed = self.squeeze(neigh_ent_embed_list_drug_one[0])
        drug2_squeeze_embed = self.squeeze(neigh_ent_embed_list[0])
        drug_drug_score = self.sigmoid(self.reduce_sum(drug1_squeeze_embed * drug2_squeeze_embed, -1))
        drug_drug_score = self.squeeze(drug_drug_score)
        return drug_drug_score


class GetReceptiveField(nn.Cell):
    """GetReceptiveField architecture"""
    def __init__(self, n_depth, adj_entity, adj_relation):
        super(GetReceptiveField, self).__init__()
        self.n_depth = n_depth
        self.adj_entity = Tensor(adj_entity)
        self.adj_relation = Tensor(adj_relation)
        self.adj_entity_matrix = initializer(self.adj_entity, shape=self.adj_entity.shape, dtype=ms.int64)
        self.adj_relation_matrix = initializer(self.adj_relation, shape=self.adj_relation.shape, dtype=ms.int64)
        self.shape = ops.Shape()
        self.gather = ops.Gather()
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()

    def construct(self, entity):
        """GetReceptiveField architecture"""
        neigh_ent_list = [entity]
        neigh_rel_list = []
        n_neighbor = self.shape(self.adj_entity_matrix)[1]
        for i in range(self.n_depth):
            new_neigh_ent = self.gather(self.adj_entity_matrix, self.cast(
                neigh_ent_list[-1], ms.int64), 0)  # cast function used to transform data type
            new_neigh_rel = self.gather(self.adj_relation_matrix, self.cast(
                neigh_ent_list[-1], ms.int64), 0)
            neigh_ent_list.append(
                self.reshape(new_neigh_ent, (-1, n_neighbor ** (i + 1))))
            neigh_rel_list.append(
                self.reshape(new_neigh_rel, (-1, n_neighbor ** (i + 1))))

        return neigh_ent_list + neigh_rel_list


class GetNeighborInfo(nn.Cell):
    """GetNeighborInfo architecture"""
    def __init__(self, embed_dim, neighbor_sample_size):
        super(GetNeighborInfo, self).__init__()
        self.embed_dim = embed_dim
        self.neighbor_sample_size = neighbor_sample_size

        self.reduce_sum_keep_dims = ops.ReduceSum(keep_dims=True)
        self.reduce_sum = ops.ReduceSum()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def construct(self, drug, rel, ent):
        """Get neighbor representation.

        :param drug: a tensor shaped [batch_size, 1, embed_dim]
        :param rel: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :param ent: a tensor shaped [batch_size, neighbor_size ** hop, embed_dim]
        :return: a tensor shaped [batch_size, neighbor_size ** (hop -1), embed_dim]
        """
        drug_rel_score = self.reduce_sum_keep_dims(drug * rel, -1)
        weighted_ent = drug_rel_score * ent
        weighted_ent = self.reshape(weighted_ent,
                                    (self.shape(weighted_ent)[0], -1,
                                     self.neighbor_sample_size, self.embed_dim))
        neighbor_embed = self.reduce_sum(weighted_ent, 2)
        return neighbor_embed


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
