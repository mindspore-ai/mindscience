# Copyright 2023 @ Huawei Technologies Co., Ltd
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
"""MDA model"""
from collections import OrderedDict
from mindspore import nn
import mindspore.ops as ops


class MDA(nn.Cell):
    """MDA model"""

    def __init__(self, input_dims, encoding_dims, train=True):
        super(MDA, self).__init__()
        self.train = train
        self.input_dims = input_dims
        self.encoding_dims = encoding_dims
        self.cell_layer = nn.CellList()
        for i, _ in enumerate(range(len(input_dims))):
            self.cell_layer.append(nn.Dense(input_dims[i], int(encoding_dims[0] / len(input_dims)),
                                            activation='sigmoid'))
        if len(encoding_dims) == 1:
            self.concatenate = ops.Concat(axis=1)
            self.concatenate.name = "middle_layer"
        else:
            self.concatenate = ops.Concat(axis=1)
        self.concat_output = ops.Concat(axis=1)
        self.test_concatenate = ops.Concat(axis=3)
        self.squeeze = ops.Squeeze()
        self.expand_dims = ops.ExpandDims()
        self.encoder_dicts = OrderedDict()
        self.decoder_dicts = OrderedDict()
        for i in range(0, len(encoding_dims) // 2):
            self.encoder_dicts.update({'encoder_layer_{}'.format(i): nn.Dense(encoding_dims[i], encoding_dims[i + 1],
                                                                              activation='sigmoid')})
        self.encoder_layers = nn.SequentialCell(self.encoder_dicts)
        if self.train:
            for i in range(len(encoding_dims) // 2, len(encoding_dims) - 1):
                self.decoder_dicts.update({'decoder_layer_{}'.format(i - (len(encoding_dims) // 2))
                                           : nn.Dense(encoding_dims[i], encoding_dims[i + 1], activation='sigmoid')})
            self.decoder_layers = nn.SequentialCell(self.decoder_dicts)
            for i in range(len(input_dims)):
                self.cell_layer.append(nn.Dense(encoding_dims[0], int(encoding_dims[-1] / len(input_dims)),
                                                activation='sigmoid'))
            # output layers
            for i, _ in enumerate(range(len(input_dims))):
                self.cell_layer.append(nn.Dense(int(encoding_dims[-1] / len(input_dims)), input_dims[i],
                                                activation='sigmoid'))
        self.hidden_bias = len(input_dims)
        self.output_bias = 2 * len(input_dims)

    def construct(self, x):
        """construct"""
        if not self.train:
            x = self.expand_dims(x, 0)
        split = ops.Split(1, 9)
        x0, x1, x2, x3, x4, x5, x6, x7, x8 = split(x)
        xsplit = [x0, x1, x2, x3, x4, x5, x6, x7, x8]
        for i, xi in enumerate(xsplit):
            xsplit[i] = self.cell_layer[i](xi)
        if self.train:
            x = self.concatenate(
                (xsplit[0], xsplit[1], xsplit[2], xsplit[3], xsplit[4], xsplit[5], xsplit[6], xsplit[7], xsplit[8]))
            mid_output = self.encoder_layers(x)
            x = self.decoder_layers(mid_output)
            output_list = []
            for i in range(9):
                output_list.append(
                    self.expand_dims(self.cell_layer[i + self.output_bias](self.cell_layer[i + self.hidden_bias](x)),
                                     1))
            output = self.concat_output((output_list[0], output_list[1], output_list[2], output_list[3],
                                         output_list[4], output_list[5], output_list[6], output_list[7],
                                         output_list[8]))
            return self.squeeze(mid_output), output
        x = self.test_concatenate(
            (xsplit[0], xsplit[1], xsplit[2], xsplit[3], xsplit[4], xsplit[5], xsplit[6], xsplit[7], xsplit[8]))
        x = self.squeeze(x)
        mid_output = self.encoder_layers(x)
        return self.squeeze(mid_output)


class MDAWithLossCell(nn.Cell):
    """MDAWithLossCell"""
    def __init__(self, backbone, loss_fn):
        super(MDAWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    @property
    def backbone_network(self):
        return self._backbone

    def construct(self, data, label):
        _, out = self._backbone(data)
        return self._loss_fn(out, label)


class MDACustomTrainOneStepCell(nn.Cell):
    """MDACustomTrainOneStepCell"""

    def __init__(self, network, optimizer):
        super(MDACustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.optimizer = optimizer
        self.weights = self.optimizer.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        loss = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*inputs)
        self.optimizer(grads)
        return loss


class MDALoss(nn.Cell):
    """MDALoss"""
    def __init__(self):
        super(MDALoss, self).__init__()
        self.loss = ops.BinaryCrossEntropy()

    def construct(self, logits, labels):
        labels = labels.view((logits.shape[0], 9, -1))
        output = self.loss(logits, labels)
        return output
