# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
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
embedding
"""

from mindspore import nn
from mindspore import ops

from ..util import functional as F


class TransE(nn.Cell):
    """
    TransE embedding proposed in `Translating Embeddings for Modeling Multi-relational Data`_.

    .. _Translating Embeddings for Modeling Multi-relational Data:
        https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

    Parameters:
        n_entity (int): number of entities
        n_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        max_score (float, optional): maximal score for triplets
    """

    def __init__(self, n_entity, n_relation, embedding_dim, max_score=12):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.max_score = max_score

        self.entity = nn.Parameter(ops.empty(n_entity, embedding_dim))
        self.relation = nn.Parameter(ops.empty(n_relation, embedding_dim))

        nn.init.uniform_(self.entity, -self.max_score / embedding_dim, self.max_score / embedding_dim)
        nn.init.uniform_(self.relation, -self.max_score / embedding_dim, self.max_score / embedding_dim)

    def construct(self, h_index, t_index, r_index):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = F.transe_score(self.entity, self.relation, h_index, t_index, r_index)
        return self.max_score - score


class DistMult(nn.Cell):
    """
    DistMult embedding proposed in `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf

    Parameters:
        n_entity (int): number of entities
        n_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): weight for l3 regularization
    """

    def __init__(self, n_entity, n_relation, embedding_dim, l3_regularization=0):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.l3_regularization = l3_regularization

        self.entity = nn.Parameter(ops.empty(n_entity, embedding_dim))
        self.relation = nn.Parameter(ops.empty(n_relation, embedding_dim))

        nn.init.uniform_(self.entity, -0.5, 0.5)
        nn.init.uniform_(self.relation, -0.5, 0.5)

    def construct(self, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = F.distmult_score(self.entity, self.relation, h_index, t_index, r_index)

        if all_loss is not None and self.l3_regularization > 0:
            loss = (self.entity.abs() ** 3).sum() + (self.relation.abs() ** 3).sum()
            all_loss += loss * self.l3_regularization
            metric["l3 regularization"] = loss / (self.n_entity + self.n_relation)

        return score


class ComplEx(nn.Cell):
    """
    ComplEx embedding proposed in `Complex Embeddings for Simple Link Prediction`_.

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf

    Parameters:
        n_entity (int): number of entities
        n_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): weight for l3 regularization
    """

    def __init__(self, n_entity, n_relation, embedding_dim, l3_regularization=0):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.l3_regularization = l3_regularization

        self.entity = nn.Parameter(ops.empty(n_entity, embedding_dim))
        self.relation = nn.Parameter(ops.empty(n_relation, embedding_dim))

        nn.init.uniform_(self.entity, -0.5, 0.5)
        nn.init.uniform_(self.relation, -0.5, 0.5)

    def construct(self, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for triplets.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = F.complex_score(self.entity, self.relation, h_index, t_index, r_index)

        if all_loss is not None and self.l3_regularization > 0:
            loss = (self.entity.abs() ** 3).sum() + (self.relation.abs() ** 3).sum()
            all_loss += loss * self.l3_regularization
            metric["l3 regularization"] = loss / (self.n_entity + self.n_relation)

        return score


class RotatE(nn.Cell):
    """
    RotatE embedding proposed in `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_.

    .. _RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space:
        https://arxiv.org/pdf/1902.10197.pdf

    Parameters:
        n_entity (int): number of entities
        n_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        max_score (float, optional): maximal score for triplets
    """

    def __init__(self, n_entity, n_relation, embedding_dim, max_score=12):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.max_score = max_score

        self.entity = nn.Parameter(ops.empty(n_entity, embedding_dim))
        self.relation = nn.Parameter(ops.empty(n_relation, embedding_dim // 2))

        nn.init.uniform_(self.entity, -max_score * 2 / embedding_dim, max_score * 2 / embedding_dim)
        nn.init.uniform_(self.relation, -max_score * 2 / embedding_dim, max_score * 2 / embedding_dim)
        pi = ops.acos(ops.zeros(1)) * 2
        self.relation_scale = pi * embedding_dim / max_score / 2

    def construct(self, h_index, t_index, r_index):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = F.rotate_score(self.entity, self.relation * self.relation_scale,
                               h_index, t_index, r_index)
        return self.max_score - score


class SimplE(nn.Cell):
    """
    SimplE embedding proposed in `SimplE Embedding for Link Prediction in Knowledge Graphs`_.

    .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
        https://papers.nips.cc/paper/2018/file/b2ab001909a8a6f04b51920306046ce5-Paper.pdf

    Parameters:
        n_entity (int): number of entities
        n_relation (int): number of relations
        embedding_dim (int): dimension of embeddings
        l3_regularization (float, optional): maximal score for triplets
    """

    def __init__(self, n_entity, n_relation, embedding_dim, l3_regularization=0):
        super().__init__()
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.l3_regularization = l3_regularization

        self.entity = nn.Parameter(ops.empty(n_entity, embedding_dim))
        self.relation = nn.Parameter(ops.empty(n_relation, embedding_dim))

        nn.init.uniform_(self.entity, -0.5, 0.5)
        nn.init.uniform_(self.relation, -0.5, 0.5)

    def construct(self, h_index, t_index, r_index, all_loss=None, metric=None):
        """
        Compute the score for each triplet.

        Parameters:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = F.simple_score(self.entity, self.relation, h_index, t_index, r_index)

        if all_loss is not None and self.l3_regularization > 0:
            loss = (self.entity.abs() ** 3).sum() + (self.relation.abs() ** 3).sum()
            all_loss += loss * self.l3_regularization
            metric["l3 regularization"] = loss / (self.n_entity + self.n_relation)

        return score
