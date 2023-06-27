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
property
"""
import mindspore as ms
from mindspore import nn
from mindspore import ops
from .. import core
from ..core import Registry as R


@R.register('scenario.GraphProperty')
class GraphProperty(core.Cell):
    """_summary_

    Args:
        core (_type_): _description_
    """

    def __init__(self,
                 net,
                 task=(),
                 metrics=None,
                 criterion="BCELoss",
                 activation='Sigmoid',
                 readout='graph',
                 normalizer=None):
        super().__init__()
        self.net = net
        self.task = task
        self.metrics = metrics
        self.readout = readout
        if isinstance(activation, str):
            self.activation = getattr(ops, activation)()
        else:
            self.activation = activation
        if isinstance(criterion, str):
            self.criterion = getattr(nn, criterion)()
        else:
            self.criterion = criterion
        if isinstance(criterion, str):
            self.criterion = getattr(nn, criterion)(reduction='mean')
        else:
            self.criterion = criterion
        self.normalizer = normalizer
        self.out = None

    def preprocess(self, train_set, valid_set=None, test_set=None):
        """
        Compute the mean and derivation for each task on the training set.
        """
        if self.normalizer == 'stand':
            alpha = []
            beta = []
            for col in train_set.targets.columns:
                label = train_set.targets[col]
                mean = label.mean()
                alpha.append(mean)
                std = label.std()
                beta.append(std)
                for ds in [train_set, valid_set, test_set]:
                    if ds is not None:
                        ds.targets[col] = (ds.targets[col] - mean) / (std + 1e-6)

            setattr(self, 'mean', ms.Tensor(alpha))
            setattr(self, 'std', ms.Tensor(beta))
        elif self.normalizer == 'minmax':
            alpha = []
            beta = []
            for col in train_set.targets.columns:
                label = train_set.targets[col]
                xmax = label.max()
                alpha.append(xmax)
                xmin = label.min()
                beta.append(xmin)
                for ds in [train_set, valid_set, test_set]:
                    if ds is not None:
                        ds.targets[col] = (ds.targets[col] - xmin) / (xmax + 1e-6 - xmin)
            setattr(self, 'min', ms.Tensor(alpha))
            setattr(self, 'max', ms.Tensor(beta))
        self.task = list(train_set.task_list)
        self.out = nn.Dense(self.net.output_dim, len(self.task))

    def construct(self, graph, inputs):
        output = self.net(graph, inputs)[self.readout]
        output = self.out(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def loss_fn(self, *args, **kwargs):
        args, kwargs = core.args_from_dict(*args, **kwargs)
        graph, label = args
        ix = ~label.isnan()
        output = self(graph, graph.node_feat)
        output = output.masked_select(ix)
        label = label.masked_select(ix)
        loss = self.criterion(output, label)
        return loss, (output, label)

    def eval(self, *batch):
        graph, label = batch
        loss = self.loss_fn(graph, label)
        return loss

    def predict(self, molset):
        out = []
        for batch in molset:
            graph = batch[0]
            label_ = self.net(graph.tensor().to_dict())
            out.append(label_)
        out = ops.concat(out)
        return out
