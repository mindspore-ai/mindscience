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
Cell for training
"""

from typing import Union, List

import mindspore as ms
from mindspore.ops import functional as F
from mindspore.numpy import count_nonzero
from mindspore.nn.loss.loss import LossBase

from ..configs import Config

from .wrapper import MoleculeWrapper
from ..scenarios.potential import Cybertron


class MolWithLossCell(MoleculeWrapper):
    r"""Basic cell to combine the network and the loss/evaluate function.

    Args:
        datatypes (str):        Data types of the inputs.

        network (AIchemist):    Neural network of AIchemist

        loss_fn (Cell):         Loss function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 data_keys: List[str],
                 network: Cybertron,
                 loss_fn: Union[LossBase, List[LossBase]],
                 calc_force: bool = False,
                 energy_key: str = 'energy',
                 force_key: str = 'force',
                 loss_weights: List[float] = 1,
                 weights_normalize: bool = True,
                 **kwargs
                 ):
        super().__init__(
            network=network,
            loss_fn=loss_fn,
            data_keys=data_keys,
            calc_force=calc_force,
            energy_key=energy_key,
            force_key=force_key,
            loss_weights=loss_weights,
            weights_normalize=weights_normalize,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

        self._loss_fn = self._check_loss(loss_fn)
        self._loss_weights = self._check_weights(loss_weights)
        self._molecular_loss = self._set_molecular_loss()
        self._any_atomwise = any(self._molecular_loss)
        self._set_atomwise()

        self._network.set_train()

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        super().print_info(num_retraction, num_gap, char)
        print('-'*80)

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *inputs: Tuple of Tensor

        Returns:
            loss (Tensor):  Tensor of shape (B, 1). Data type is float.

        """
        inputs = inputs + (None,)

        atom_type = inputs[self.atom_type_id]
        coordinate = inputs[self.coordinate_id]
        pbc_box = inputs[self.pbc_box_id]
        bonds = inputs[self.bonds_id]
        bond_mask = inputs[self.bond_mask_id]

        labels = [inputs[self.labels_id[i]] for i in range(self.num_labels)]

        outputs = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            bonds=bonds,
            bond_mask=bond_mask
        )
        if self.num_outputs == 1:
            outputs = (outputs,)

        if self.calc_force:
            force_predict = -1 * self.grad_op(self._network)(
                coordinate,
                atom_type,
                pbc_box,
                bonds,
                bond_mask
            )

            if self.num_labels == 1:
                outputs = (force_predict,)
            else:
                outputs += (force_predict,)

        num_atoms = None
        atom_mask = None
        if self._any_atomwise:
            if atom_type is None:
                atom_type = self.atom_type
            atom_mask = atom_type > 0
            num_atoms = count_nonzero(F.cast(atom_mask, ms.int16), axis=-1, keepdims=True)

        loss = 0
        for i in range(self.num_labels):
            if self._molecular_loss[i]:
                loss_ = self._loss_fn[i](outputs[i], labels[i], num_atoms, atom_mask)
            else:
                loss_ = self._loss_fn[i](outputs[i], labels[i])

            loss += loss_ * self._loss_weights[i]

        return loss
