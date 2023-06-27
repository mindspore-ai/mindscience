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
Cybertron
"""
from typing import Union
from mindspore import nn
from .. import core


class Cybertron(core.Cell):
    """Cybertron: An architecture to perform deep molecular model for molecular modeling.

    Args:

        model (Cell):           Deep molecular model.

        readout (Cell):         Readout function.

        dim_output (int):       Output dimension. Default: 1.

        num_atoms (int):        Maximum number of atoms in system. Default: None.

        atom_type (Tensor):    Tensor of shape (B, A). Data type is int.
                                Index of atom types.
                                Default: None,

        bond_types (Tensor):    Tensor of shape (B, A, N). Data type is int.
                                Index of bond types. Default: None.

        num_atom_types (int):   Maximum number of atomic types. Default: 64

        pbc_box (Tensor):       Tensor of shape (B, D).
                                Box size of periodic boundary condition. Default: None

        use_pbc (bool):         Whether to use periodic boundary condition. Default: None

        length_unit (str):      Unit of position coordinate. Default: None

        energy_unit (str):      Unit of output energy. Default: None.

        hyper_param (dict):     Hyperparameters of Cybertron. Default: None.

    Symbols:

        B:  Number of simulation walker.

        A:  Number of atoms in system.

        N:  Number of neighbour atoms.

        D:  Dimension of position coordinates, usually is 3.

        O:  Output dimension of the predicted properties.

    """

    def __init__(self,
                 net: Union[core.Cell, dict, str],
                 criterion=nn.MSELoss(),
                 activation=None, task_list=None):
        super().__init__()
        self.net = net
        self.cutoff = None
        self.output_dim = (2, 3)
        self.criterion = criterion
        if isinstance(activation, str):
            self.activation = getattr(nn, activation)()
        else:
            self.activation = activation
        self.task_list = task_list
        if self.task_list is not None:
            self.out = nn.Dense(self.net.hidden_channels, len(self.task_list))

    def preprocess(self, train_set): # , valid_set=None, test_set=None
        """_summary_

        Args:
            train_set (_type_): _description_
        """
        self.task_list = list(train_set.task_list)
        self.out = nn.Dense(self.net.hidden_channels, len(self.task_list))

    def construct(self, graph):
        output = self.net(graph.coord(), graph.atom_type, graph.n_nodes)
        output = self.out(output)
        return output

    def loss_fn(self, *args, **kwargs):
        (graph, label), kwargs = core.args_from_dict(*args, **kwargs)
        output = self(graph)
        ix = ~label.isnan()
        output = output[ix]
        label = label[ix]
        loss = self.criterion(output, label)
        return loss, (output, label)

    def eval(self, *args, **kwargs):
        loss, (output, label) = self.loss_fn(*args, **kwargs)
        return loss, (output, label)
