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
from ..configs import Registry as R


@R.register('scenario.GraphProperty')
class GraphProperty(nn.Cell):
    """Prediction of Graph property

    Args:
        net (nn.Cell): The cell based neural network model
        task (tuple, optional): The names of prediction tasks. Defaults to ().
        criterion (str or Callable, , optional): criterion of loss function. Defaults to "BCELoss".
        activation (str or Callable, optional): activation function. Defaults to 'Sigmoid'.
        readout (str or Callable, optional): Readout method for the last layer of net. Defaults to 'graph'.
        normalizer (str or Callable, optional): Data normalization methods. Defaults to None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 net,
                 task=(),
                 criterion="BCELoss",
                 activation='Sigmoid',
                 readout='graph',
                 normalizer=None):
        super().__init__()
        self.net = net
        self.task = task
        self.readout = readout
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
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
        self.out = nn.Dense(self.net.output_dim, len(self.task), weight_init='xavier_uniform')

    def construct(self, graph, inputs):
        """
        Args:
            graph (Graph): The input graph data
            inputs (ms.Tensor): The inputs feature representations.

        Returns:
            outputs (ms.Tensor): output data.
        """
        output = self.net(graph, inputs)[self.readout]
        output = self.out(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def loss_fn(self, **kwargs):
        """loss function"""
        args = core.obj_from_dict(kwargs)
        graph, label = args['graph'], args['label']
        output = self(graph, graph.node_feat)
        ix = ~label.isnan()
        output = output.masked_select(ix)
        label = label.masked_select(ix)
        loss = self.criterion(output, label)
        return loss, (output, label)

    def predict(self, molset):
        out = []
        for batch in molset:
            args = core.obj_from_dict(**batch)
            graph = args['graph']
            label_ = self(graph)
            out.append(label_)
        out = ops.concat(out)
        return out
