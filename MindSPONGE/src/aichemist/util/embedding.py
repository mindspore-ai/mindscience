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
embedding
"""


from mindspore import ops

split = ops.Split(-1, 2)


def transe_score(entity, relation, h_index, t_index, r_index):
    """
    TransE score function from `Translating Embeddings for Modeling Multi-relational Data`_.

    .. _Translating Embeddings for Modeling Multi-relational Data:
        https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf

    Parameters:
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

    Parameters:
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

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, 2d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]
    h_re, h_im = split(h)
    r_re, r_im = split(r)
    t_re, t_im = split(t)

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

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]
    t_flipped = ops.concat(split(t)[::-1], axis=-1)
    score = (h * r * t_flipped).sum(axis=-1)
    return score


def rotate_score(entity, relation, h_index, t_index, r_index):
    """
    RotatE score function from `RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space`_.

    .. _RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space:
        https://arxiv.org/pdf/1902.10197.pdf

    Parameters:
        entity (Tensor): entity embeddings of shape :math:`(|V|, 2d)`
        relation (Tensor): relation embeddings of shape :math:`(|R|, d)`
        h_index (LongTensor): index of head entities
        t_index (LongTensor): index of tail entities
        r_index (LongTensor): index of relations
    """
    h = entity[h_index]
    r = relation[r_index]
    t = entity[t_index]

    h_re, h_im = split(h)
    r_re, r_im = ops.cos(r), ops.sin(r)
    t_re, t_im = split(t)

    x_re = h_re * r_re - h_im * r_im - t_re
    x_im = h_re * r_im + h_im * r_re - t_im
    x = ops.stack([x_re, x_im], axis=-1)
    score = ops.LpNorm(axis=-1, p=2)(x).sum(axis=-1)
    return score
