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
reasoning
"""
import numpy as np
from mindspore import ops
from mindspore import nn
from ..util import functional as F
from ..core import Registry as R
from ..core import args_from_dict


@R.register('scenario.GraphCompletion')
class GraphCompletion(nn.Cell):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, net, criterion="ranking",
                 n_negative=10,
                 margin=6,
                 adversarial_temperature=0,
                 strict_negative=False,
                 filtered_ranking=True,
                 sample_weight=True,
                 full_batch_eval=False,
                 n_entity=None,
                 n_relation=None):
        super().__init__()
        self.net = net
        self.criterion = criterion
        self.n_negative = n_negative
        self.margin = margin
        self.adversarial_temperature = adversarial_temperature
        self.strict_negative = strict_negative
        self.filtered_ranking = filtered_ranking
        self.sample_weight = sample_weight
        self.criterion = criterion
        self.full_batch_eval = full_batch_eval
        self.n_entity = n_entity
        self.n_relation = n_relation

    def preprocess(self, train_set, valid_set=None, test_set=None):
        """_summary_

        Args:
            train_set (_type_): _description_
            valid_set (_type_, optional): _description_. Defaults to None.
            test_set (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.n_entity = train_set.graph.n_node
        self.n_relation = train_set.n_relation
        if self.sample_weight:
            degree_hr = np.zeros((self.n_entity, self.n_relation))
            degree_tr = np.zeros((self.n_entity, self.n_relation))
            for _, h, t, r in train_set:
                degree_hr[h, r] += 1
                degree_tr[t, r] += 1
            setattr(self, 'degree_hr', degree_hr)
            setattr(self, 'degree_tr', degree_tr)
        return train_set, valid_set, test_set

    def predict(self, batch, graph):
        """_summary_

        Args:
            batch (_type_): _description_
            graph (_type_): _description_

        Returns:
            _type_: _description_
        """
        pos_h_index, pos_t_index = batch.edges
        pos_r_index = batch.edge_type
        all_index = ops.arange(self.n_entity)
        t_preds = []
        h_preds = []
        for neg_index in all_index.split(self.n_negative):
            r_index = pos_r_index.expand_dims(-1).repeat(len(neg_index), -1)
            h_index, t_index = ops.meshgrid(pos_h_index, neg_index)
            t_pred = self.net(graph, [h_index, t_index, r_index])
            t_index, h_index = ops.meshgrid(pos_t_index, neg_index)
            h_pred = self.net(graph, [h_index, t_index, r_index])
            t_preds.append(t_pred)
            h_preds.append(h_pred)
        h_pred = ops.concat(h_preds, axis=-1)
        t_pred = ops.concat(t_preds, axis=-1)
        pred = ops.stack([t_pred, h_pred], axis=1)
        return pred

    def construct(self, graph, sources, targets, relations):
        """_summary_

        Args:
            graph (_type_): _description_
            sources (_type_): _description_
            targets (_type_): _description_
            relations (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size = len(sources)
        if self.strict_negative:
            neg_index = self._strict_negative(graph, sources, targets, relations)
        else:
            neg_index = ops.randint(0, self.n_entity, (batch_size, self.n_negative))
        src_idx = sources.expand_dims(-1).repeat(self.n_negative + 1, 1)
        trg_idx = targets.expand_dims(-1).repeat(self.n_negative + 1, 1)
        rel_idx = relations.expand_dims(-1).repeat(self.n_negative + 1, 1)
        trg_idx[:batch_size // 2, 1:] = neg_index[:batch_size // 2]
        src_idx[batch_size // 2:, 1:] = neg_index[batch_size // 2:]
        pred = self.net(graph, [src_idx, trg_idx, rel_idx])
        return pred

    def loss_fn(self, *args, **kwargs):
        """_summary_

        Returns:
            _type_: _description_
        """
        args, kwargs = args_from_dict(*args, **kwargs)
        graph, sources, targets, relations = args
        loss = 0
        pred = self(graph, sources, targets, relations)
        target = ops.zeros(pred.shape)
        target[:, 0] = 1
        if self.criterion == 'bce':
            neg_weight = ops.ones(pred.shape)
            if self.adversarial_temperature > 0:
                neg_weight[:, 1:] = ops.softmax(pred[:, 1:] / self.adversarial_temperature, axis=-1)
            else:
                neg_weight[:, 1:] = 1 / self.n_negative
            self.criterion = nn.BCELoss(weight=neg_weight)
            loss = self.criterion(pred, target)
        elif self.criterion == 'ce':
            target = ops.zeros(len(pred))
            self.criterion = nn.CrossEntropyLoss()
            loss = self.criterion(pred, target)
        elif self.criterion == 'ranking':
            positive = pred[:, :1]
            negative = pred[:, 1:]
            loss = F.margin_ranking_loss(positive, negative, target[:, 1:] + 1, margin=self.margin)
        return loss, (pred, target)

    def _strict_negative(self, graph, sources, targets, relations):
        """_summary_

        Args:
            graph (_type_): _description_
            sources (_type_): _description_
            targets (_type_): _description_
            relations (_type_): _description_

        Returns:
            _type_: _description_
        """
        batch_size = len(sources)
        t_any = -ops.ones(sources.shape)

        pattern = ops.stack([sources, t_any, relations], axis=-1)
        pattern = pattern[:batch_size // 2]
        edge_index, n_trg_truth = graph.match(pattern.asnumpy())
        trg_truth_index = graph.edges[1, edge_index]
        pos_index = F.size_to_index(n_trg_truth)
        trg_mask = ops.ones(len(pattern), self.n_entity)
        trg_mask[pos_index, trg_truth_index] = 0
        neg_t_candidate = trg_mask.nonzero()[:, 1]
        n_trg_candidate = trg_mask.sum(dim=-1)
        neg_trg_index = F.variadic_sample(neg_t_candidate, n_trg_candidate, self.n_negative)

        pattern = ops.stack([t_any, targets, relations], axis=-1)
        pattern = pattern[batch_size // 2:]
        edge_index, n_src_truth = graph.match(pattern.asnumpy())
        src_truth_index = graph.edges[0, edge_index]
        pos_index = F.size_to_index(n_src_truth)
        src_mask = ops.ones(len(pattern), self.n_entity)
        src_mask[pos_index, src_truth_index] = 0
        neg_src_candidate = src_mask.nonzero()[:, 1]
        n_src_candidate = src_mask.sum(dim=-1)
        neg_src_index = F.variadic_sample(neg_src_candidate, n_src_candidate, self.n_negative)

        neg_index = ops.concat([neg_trg_index, neg_src_index])

        return neg_index
