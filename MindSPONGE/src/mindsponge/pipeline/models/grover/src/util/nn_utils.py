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
"""
The utility function for model construction.
This implementation is adapted from
https://github.com/chemprop/chemprop/blob/master/chemprop/nn_utils.py
"""
from mindspore import nn, ops
from mindspore.common.initializer import initializer


class AggregateNeighbor(nn.Cell):
    """
    Aggregate neighbor.
    """

    def __init__(self):
        super(AggregateNeighbor, self).__init__()
        self.select_index = SelectIndex()

    def index_select_nd(self, source, index):
        """
        Selects the message features from source corresponding to the atom or bond indices in index.

        :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
        :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
        indices to select from source.
        :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
        features corresponding to the atoms/bonds specified in index.
        """
        index_size = index.shape  # (num_atoms/num_bonds, max_num_bonds)
        suffix_dim = source.shape[1:]  # (hidden_size,)
        final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

        o_type = source.dtype
        target = self.select_index(source, index)

        if target.dtype != o_type:
            target = self.cast(target, o_type)

        target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

        return target

    def construct(self, feature, index):
        neighbor = self.index_select_nd(feature, index)
        return neighbor.sum(1)


class SelectIndex(nn.Cell):
    """
    Select neighbor index.
    """

    def __init__(self):
        super(SelectIndex, self).__init__()
        self.dim = 0

    def construct(self, source, index):
        indices = index.view(-1)
        target = source.take(indices, self.dim)

        return target


def index_select_nd(source, index):
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.shape  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.shape[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.take(index.view(-1), 0)  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


def get_activation_function(activation):
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    if activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    if activation == 'PReLU':
        return nn.PReLU()
    if activation == 'tanh':
        return nn.Tanh()
    if activation == 'SELU':
        return ops.SELU()
    if activation == 'ELU':
        return nn.ELU()
    if activation == "Linear":
        return lambda x: x

    raise ValueError(f'Activation "{activation}" not supported.')


def initialize_weights(model: nn.Cell, distinct_init=False, model_idx=0):
    """
    Initializes the weights of a model in place.

    :param model: An nn.Cell.
    """

    init_fns = [initializer.HeNormal(nonlinearity='relu'),
                initializer.HeUniform,
                initializer.Normal,
                initializer.XavierUniform]
    for param in model.trainable_params():
        if param.dim == 1:
            initializer.Constant(value=0)(param)
        else:
            if distinct_init:
                init_fn = init_fns[model_idx % 4]
                init_fn(param)
            else:
                initializer.XavierUniform(param)


def select_neighbor_and_aggregate(feature, index):
    """
    The basic operation in message passing.

    :param feature: the candidate feature for aggregate. (n_nodes, hidden)
    :param index: the selected index (neighbor indexes).
    :return:
    """
    neighbor = index_select_nd(feature, index)
    return neighbor.sum(1)
