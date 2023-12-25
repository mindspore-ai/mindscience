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
Cell for evaluation
"""

from typing import Union, List
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore.numpy import count_nonzero
from mindspore.nn.loss.loss import LossBase

from ..configs import Config

from .wrapper import MoleculeWrapper
from ..scenarios.potential import Cybertron


class MolWithEvalCell(MoleculeWrapper):
    r"""Basic cell to combine the network and the loss/evaluate function.

    Args:
        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 data_keys: List[str],
                 network: Cybertron,
                 loss_fn: Union[LossBase, List[LossBase]] = None,
                 loss_weights: List[Union[float, Tensor, ndarray]] = 1,
                 calc_force: bool = False,
                 energy_key: str = 'energy',
                 force_key: str = 'force',
                 weights_normalize: bool = True,
                 normed_evaldata: bool = False,
                 add_cast_fp32: bool = False,
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

        self.add_cast_fp32 = add_cast_fp32

        self._normed_evaldata = normed_evaldata

        self._force_id = self.get_index(self.force_key, self.label_keys)

        if loss_fn is not None:
            self._loss_fn = self._check_loss(loss_fn)
            self._loss_weights = self._check_weights(loss_weights)
            self._molecular_loss = self._set_molecular_loss()
            self._any_atomwise = any(self._molecular_loss)
            self._set_atomwise()

        self.zero = Tensor(0, ms.float32)

    def print_info(self, num_retraction: int = 3, num_gap: int = 3, char: str = ' '):
        super().print_info(num_retraction, num_gap, char)
        ret = char * num_retraction
        print(f'{ret} Using normalized dataset: {self._normed_evaldata}')
        print('-'*80)

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *input: Tuple of Tensor

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

        if atom_type is None:
            atom_type = self.atom_type
        atom_mask = atom_type > 0
        num_atoms = count_nonzero(atom_mask.astype(ms.int16), axis=-1, keepdims=True)

        normed_labels = None
        if self._normed_evaldata:
            normed_labels = labels
            labels = [self.scaleshift[i](normed_labels[i], atom_type, num_atoms)
                      for i in range(self.num_readouts)]
            if self.calc_force:
                labels += [self.scaleshift[0].scale_force(normed_labels[-1])]
        elif self._loss_fn is not None:
            normed_labels = [self.scaleshift[i].normalize(labels[i], atom_type, num_atoms)
                             for i in range(self.num_readouts)]
            if self.calc_force:
                force_label = self.scaleshift[0].normalize_force(labels[-1])
                normed_labels.append(force_label)

        outputs = self._network(
            coordinate=coordinate,
            atom_type=atom_type,
            pbc_box=pbc_box,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        if self.num_readouts == 1:
            outputs = [outputs]
        else:
            outputs = list(outputs)
        outputs, normed_outputs = self._normalize_outputs(outputs, atom_type, num_atoms)

        if self.calc_force:
            force = -1 * self.grad_op(self._network)(
                coordinate,
                atom_type,
                pbc_box,
                bonds,
                bond_mask,
            )
            outputs, normed_outputs = self._normalize_force(force, outputs, normed_outputs)
        if self.add_cast_fp32:
            for i in range(self.num_readouts):
                outputs[i] = outputs[i].astype(ms.float32)
                labels[i] = labels[i].astype(ms.float32)
                normed_outputs[i] = normed_outputs[i].astype(ms.float32)
                normed_labels[i] = normed_labels[i].astype(ms.float32)

        loss = self.zero
        if self._loss_fn:
            for i in range(self.num_labels):
                if self._molecular_loss[i]:
                    loss_ = self._loss_fn[i](normed_outputs[i], normed_labels[i], num_atoms, atom_mask)
                else:
                    loss_ = self._loss_fn[i](normed_outputs[i], normed_labels[i])

                loss += loss_ * self._loss_weights[i]
        out = loss, outputs, labels, atom_mask
        return out

    def _normalize_outputs(self, outputs, atom_type, num_atoms):
        """_summary_

        Args:
            outputs (_type_): _description_
            scaled_outputs (_type_): _description_
            atom_type (_type_): _description_
            num_atoms (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.scaled_outputs:
            normed_outputs = []
            if self._loss_fn is not None:
                for i in range(self.num_readouts):
                    normed_outputs += [self.scaleshift[i].normalize(outputs[i], atom_type, num_atoms)]
        else:
            normed_outputs = outputs
            outputs = []
            for i in range(self.num_readouts):
                outputs += [self.scaleshift[i](normed_outputs[i], atom_type, num_atoms)]

        return outputs, normed_outputs

    def _normalize_force(self, force, outputs, normed_outputs):
        """
        Normalize force

        Args:
            force (_type_): _description_

        Returns:
            _type_: _description_
        """
        if self.scaled_outputs:
            normed_force = None
            if self._loss_fn is not None:
                normed_force = self.scaleshift[0].normalize_force(force)
        else:
            normed_force = force
            force = self.scaleshift[0].scale_force(normed_force)

        if self.num_labels == 1:
            outputs = [force]
            normed_outputs = [normed_force]
        else:
            outputs += [force]
            normed_outputs += [normed_force]
        return outputs, normed_outputs
