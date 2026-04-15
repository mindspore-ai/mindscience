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


def transe_score(entity, relation, h_index, t_index, r_index):
    """
    TransE score function from `Translating Embeddings for Modeling Multi-relational Data`_.

    .. _Translating Embeddings for Modeling Multi-relational Data:
        https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

    Args:
        entity (Tensor): entity embeddings of shape :math:`(|V|, d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]
    score = ops.LpNorm(axis=-1, p=1)(h + r - t)
    return score


def distmult_score(entity, relation, h_index, t_index, r_index):
    """
    DistMult score function from `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf

    Args:
        entity (Tensor): entity embeddings of shape :math:`(|V|, d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]
    score = (h * r * t).sum(axis=-1)
    return score


def complex_score(entity, relation, h_index, t_index, r_index):
    """
    ComplEx score function from `Complex Embeddings for Simple Link Prediction`_.

    .. _Complex Embeddings for Simple Link Prediction:
        http://proceedings.mlr.press/v48/trouillon16.pdf

    Args:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, 2d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]
    h_re, h_im = ops.split(h, 2, -1)
    r_re, r_im = ops.split(r, 2, -1)
    t_re, t_im = ops.split(t, 2, -1)

    x_re = h_re * r_re - h_im * r_im
    x_im = h_re * r_im + h_im * r_re
    x = x_re * t_re + x_im * t_im
    score = x.sum(axis=-1)
    return score


def simple_score(entity, relation, h_index, t_index, r_index):
    """
    SimplE score function from `SimplE Embedding for Link Prediction in Knowledge Graphs`_.

    .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
        https://papers.nips.cc/paper/2018/file/b2ab001909a8a6f04b51920306046ce5-Paper.pdf

    Args:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]
    t_flipped = ops.concat(ops.split(t, 2, -1)[::-1], axis=-1)
    score = (h * r * t_flipped).sum(axis=-1)
    return score


def rotate_score(entity, relation, h_index, t_index, r_index):
    """
    RotatE score function from `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_.

    .. _RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space:
        https://arxiv.org/pdf/1902.10197.pdf

    Args:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]

    h_re, h_im = ops.split(h, 2, -1)
    r_re, r_im = ops.cos(r), ops.sin(r)
    t_re, t_im = ops.split(t, 2, -1)

    x_re = h_re * r_re - h_im * r_im - t_re
    x_im = h_re * r_im + h_im * r_re - t_im
    x = ops.stack([x_re, x_im], axis=-1)
    score = ops.LpNorm(axis=-1, p=2)(x).sum(axis=-1)
    return score


class TransE(nn.Cell):
    """
    TransE embedding proposed in `Translating Embeddings for Modeling Multi-relational Data`_.

    .. _Translating Embeddings for Modeling Multi-relational Data:
        https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

    Args:
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

        Args:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = transe_score(self.entity, self.relation, h_index, t_index, r_index)
        return self.max_score - score


class DistMult(nn.Cell):
    """
    DistMult embedding proposed in `Embedding Entities and Relations for Learning and Inference in Knowledge Bases`_.

    .. _Embedding Entities and Relations for Learning and Inference in Knowledge Bases:
        https://arxiv.org/pdf/1412.6575.pdf

    Args:
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

        Args:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = distmult_score(self.entity, self.relation, h_index, t_index, r_index)

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

    Args:
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

        Args:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = complex_score(self.entity, self.relation, h_index, t_index, r_index)

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

    Args:
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

        Args:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = rotate_score(self.entity, self.relation * self.relation_scale,
                             h_index, t_index, r_index)
        return self.max_score - score


class SimplE(nn.Cell):
    """
    SimplE embedding proposed in `SimplE Embedding for Link Prediction in Knowledge Graphs`_.

    .. _SimplE Embedding for Link Prediction in Knowledge Graphs:
        https://papers.nips.cc/paper/2018/file/b2ab001909a8a6f04b51920306046ce5-Paper.pdf

    Args:
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

        Args:
            graph (Graph): fact graph
            h_index (Tensor): indexes of head entities
            t_index (Tensor): indexes of tail entities
            r_index (Tensor): indexes of relations
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
        """
        score = simple_score(self.entity, self.relation, h_index, t_index, r_index)

        if all_loss is not None and self.l3_regularization > 0:
            loss = (self.entity.abs() ** 3).sum() + (self.relation.abs() ** 3).sum()
            all_loss += loss * self.l3_regularization
            metric["l3 regularization"] = loss / (self.n_entity + self.n_relation)

        return score
