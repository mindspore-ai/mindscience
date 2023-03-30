# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of Cybertron package.
#
# The Cybertron is open-source software based on the AI-framework:
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
Modules for training
"""

import os
from shutil import copyfile
from collections import deque
import numpy as np
from numpy import ndarray
import mindspore as ms
import mindspore.numpy as msnp
from mindspore import Model
from mindspore import Tensor
from mindspore import context
from mindspore.nn import Cell
from mindspore.nn import TrainOneStepCell
from mindspore.nn.metrics import Metric
from mindspore.nn.loss.loss import LossBase
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.dataset import Dataset
from mindspore.train import save_checkpoint
from mindspore.train.callback import Callback, RunContext
from mindspore.train.callback._callback import InternalCallbackParam
from mindspore.train._utils import _make_directory
from mindspore._checkparam import Validator as validator

from .cybertron import Cybertron

_cur_dir = os.getcwd()

__all__ = [
    "WithForceLossCell",
    "WithLabelLossCell",
    "WithForceEvalCell",
    "WithLabelEvalCell",
    "TrainMonitor",
    "MAE",
    "MSE",
    "MLoss",
    "TransformerLR",
]


class OutputScaleShift(Cell):
    r"""A network to scale and shift the label of dataset or prediction.

    Args:

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        axis (int):                 Axis to summation the reference value of molecule. Default: -2

    Symbols:

        B:  Batch size

        A:  Number of atoms

        T:  Number of total atom types

        E:  Number of labels

    """

    def __init__(self,
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: bool = None,
                 axis: int = -2,
                 ):

        super().__init__()

        self.scale = Tensor(scale, ms.float32)
        self.shift = Tensor(shift, ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = Tensor(type_ref, ms.float32)

        self.atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
        self.all_atomwsie = False
        if self.atomwise_scaleshift.all():
            self.all_atomwsie = True

        self.all_graph = False
        if not self.atomwise_scaleshift.any():
            self.all_graph = True

        if (not self.all_atomwsie) and (not self.all_graph):
            self.atomwise_scaleshift = F.reshape(
                self.atomwise_scaleshift, (1, -1))

        self.axis = axis

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(True)

    def construct(self, outputs: Tensor, num_atoms: Tensor, atom_types: Tensor = None):
        """Scale and shift output.

        Args:
            outputs (Tensor):       Tensor with shape (B, E). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_types (Tensor):    Tensor with shape (B, A). Data type is float.
                                    Default: None

        Returns:
            outputs (Tensor):       Tensor with shape (B,E). Data type is float.

        """
        ref = 0
        if self.type_ref is not None:
            # (B,A,E)
            ref = F.gather(self.type_ref, atom_types, 0)
            # (B,E)
            ref = self.reduce_sum(ref, self.axis)

        # (B,E) + (B,E)
        outputs = outputs * self.scale + ref
        if self.all_atomwsie:
            # (B,E) + (B,1)
            return outputs + self.shift * num_atoms
        if self.all_graph:
            # (B,E)
            return outputs + self.shift

        atomwise_output = outputs + self.shift * num_atoms
        graph_output = outputs + self.shift
        return msnp.where(self.atomwise_scaleshift, atomwise_output, graph_output)


class DatasetNormalization(Cell):
    r"""A network to normalize the label of dataset or prediction.

    Args:

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        axis (int):                 Axis to summation the reference value of molecule. Default: -2

    Symbols:

        B:  Batch size

        A:  Number of atoms

        T:  Number of total atom types

        E:  Number of labels

    """

    def __init__(self,
                 scale: float = 1,
                 shift: float = 0,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: bool = None,
                 axis: int = -2,
                 ):

        super().__init__()

        self.scale = Tensor(scale, ms.float32)
        self.shift = Tensor(shift, ms.float32)

        self.type_ref = None
        if type_ref is not None:
            self.type_ref = Tensor(type_ref, ms.float32)

        self.atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
        self.all_atomwsie = False
        if self.atomwise_scaleshift.all():
            self.all_atomwsie = True

        self.all_graph = False
        if not self.atomwise_scaleshift.any():
            self.all_graph = True

        if (not self.all_atomwsie) and (not self.all_graph):
            self.atomwise_scaleshift = F.reshape(
                self.atomwise_scaleshift, (1, -1))

        self.axis = axis

        self.reduce_sum = P.ReduceSum()
        self.keep_sum = P.ReduceSum(True)

    def construct(self, label: Tensor, num_atoms: Tensor, atom_types: Tensor = None):
        """Normalize outputs.

        Args:
            outputs (Tensor):       Tensor with shape (B, E). Data type is float.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
            atom_types (Tensor):    Tensor with shape (B, A). Data type is float.
                                    Default: None

        Returns:
            outputs (Tensor):       Tensor with shape (B,E). Data type is float.

        """
        ref = 0
        if self.type_ref is not None:
            ref = F.gather(self.type_ref, atom_types, 0)
            ref = self.reduce_sum(ref, self.axis)

        label -= ref
        if self.all_atomwsie:
            return (label - self.shift * num_atoms) / self.scale
        if self.all_graph:
            return (label - self.shift) / self.scale

        atomwise_norm = (label - self.shift * num_atoms) / self.scale
        graph_norm = (label - self.shift) / self.scale
        return msnp.where(self.atomwise_scaleshift, atomwise_norm, graph_norm)


class LossWithEnergyAndForces(LossBase):
    r"""Loss function of the energy and force of molecule.

    Args:

        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: True

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    """
    def __init__(self,
                 ratio_energy: float = 1,
                 ratio_forces: float = 100,
                 force_dis: float = 1,
                 ratio_normlize: bool = True,
                 reduction: str = 'mean',
                 ):

        super().__init__(reduction)

        self.force_dis = Tensor(force_dis, ms.float32)
        self.ratio_normlize = ratio_normlize

        self.ratio_energy = ratio_energy
        self.ratio_forces = ratio_forces

        self.norm = 1
        if self.ratio_normlize:
            self.norm = ratio_energy + ratio_forces

        self.reduce_mean = P.ReduceMean()
        self.reduce_sum = P.ReduceSum()

    def _calc_loss(self, diff: Tensor) -> Tensor:
        """calculate loss function"""
        return diff

    def construct(self,
                  pred_energy: Tensor,
                  label_energy: Tensor,
                  pred_forces: Tensor = None,
                  label_forces: Tensor = None,
                  num_atoms: Tensor = 1,
                  atom_mask: Tensor = None
                  ):
        """calculate loss function

        Args:
            pred_energy (Tensor):   Tensor with shape (B, E). Data type is float.
                                    Predicted energy.
            label_energy (Tensor):  Tensor with shape (B, E). Data type is float.
                                    Label energy.
            pred_forces (Tensor):   Tensor with shape (B, A, D). Data type is float.
                                    Predicted force.
            label_forces (Tensor):  Tensor with shape (B, A, D). Data type is float.
                                    Label energy.
            num_atoms (Tensor):     Tensor with shape (B, 1). Data type is int.
                                    Number of atoms in each molecule.
                                    Default: 1
            atom_mask (Tensor):     Tensor with shape (B, A). Data type is bool.
                                    Mask of atoms in each molecule.
                                    Default: None

        Symbols:
            B:  Batch size
            A:  Number of atoms
            D:  Dimension of position coordinate. Usually is 3.
            E:  Number of labels

        Returns:
            loss (Tensor):  Tensor with shape (B, 1). Data type is float.
                            Loss function.

        """

        if pred_forces is None:
            loss = self._calc_loss(pred_energy - label_energy)
            return self.get_loss(loss)

        eloss = 0
        if self.ratio_forces > 0:
            ediff = (pred_energy - label_energy) / num_atoms
            eloss = self._calc_loss(ediff)

        floss = 0
        if self.ratio_forces > 0:
            # (B,A,D)
            fdiff = (pred_forces - label_forces) * self.force_dis
            fdiff = self._calc_loss(fdiff)
            # (B,A)
            fdiff = self.reduce_sum(fdiff, -1)

            if atom_mask is None:
                floss = self.reduce_mean(fdiff, -1)
            else:
                fdiff = fdiff * atom_mask
                floss = self.reduce_sum(fdiff, -1)
                floss = floss / num_atoms

        y = (eloss * self.ratio_energy + floss * self.ratio_forces) / self.norm

        natoms = F.cast(num_atoms, pred_energy.dtype)
        weights = natoms / self.reduce_mean(natoms)

        return self.get_loss(y, weights)


class MAELoss(LossWithEnergyAndForces):
    r"""Mean-absolute-error-type Loss function for energy and force.

    Args:

        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: True

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    """
    def __init__(self,
                 ratio_energy: float = 1,
                 ratio_forces: float = 0,
                 force_dis: float = 1,
                 ratio_normlize: bool = True,
                 reduction: str = 'mean',
                 ):

        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_dis=force_dis,
            ratio_normlize=ratio_normlize,
            reduction=reduction,
        )

        self.abs = P.Abs()

    def _calc_loss(self, diff: Tensor) -> Tensor:
        return self.abs(diff)


class MSELoss(LossWithEnergyAndForces):
    r"""Mean-square-error-type Loss function for energy and force.

    Args:

        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: True

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    """
    def __init__(self,
                 ratio_energy: float = 1,
                 ratio_forces: float = 0,
                 force_dis: float = 1,
                 ratio_normlize: bool = True,
                 reduction: str = 'mean',
                 ):

        super().__init__(
            ratio_energy=ratio_energy,
            ratio_forces=ratio_forces,
            force_dis=force_dis,
            ratio_normlize=ratio_normlize,
            reduction=reduction,
        )

        self.square = P.Square()

    def _calc_loss(self, diff: Tensor) -> Tensor:
        return self.square(diff)


class CrossEntropyLoss(LossBase):
    r"""Cross entropy Loss function for positive and negative samples.

    Args:

        reduction (str):    Method to reduction the output Tensor. Default: 'mean'

        use_sigmoid (bool): Whether to use sigmoid function for output. Default: False

    """
    def __init__(self,
                 reduction: str = 'mean',
                 use_sigmoid: bool = False
                 ):

        super().__init__(reduction)

        self.sigmoid = None
        if use_sigmoid:
            self.sigmoid = P.Sigmoid()

        self.cross_entropy = P.BinaryCrossEntropy(reduction)

    def construct(self, pos_pred: Tensor, neg_pred: Tensor):
        """calculate cross entropy loss function

        Args:
            pos_pred (Tensor):  Positive samples
            neg_pred (Tensor):  Negative samples

        Returns:
            loss (Tensor):      Loss function with same shape of samples

        """
        if self.sigmoid is not None:
            pos_pred = self.sigmoid(pos_pred)
            neg_pred = self.sigmoid(neg_pred)

        pos_loss = self.cross_entropy(pos_pred, F.ones_like(pos_pred))
        neg_loss = self.cross_entropy(neg_pred, F.zeros_like(neg_pred))

        return pos_loss + neg_loss


class WithCell(Cell):
    r"""Basic cell to combine  the network and the loss/evaluate function.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

        fulltypes (str):        Full list of data types. Default: RZCDNnBbE'

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell,
                 fulltypes: str = 'RZCDNnBbE',
                 ):

        super().__init__(auto_prefix=False)

        #pylint: disable=invalid-name

        self.fulltypes = fulltypes
        self.datatypes = datatypes

        if not isinstance(self.datatypes, str):
            raise TypeError('Type of "datatypes" must be str')

        for datatype in self.datatypes:
            if self.fulltypes.count(datatype) == 0:
                raise ValueError('Unknown datatype: ' + datatype)

        for datatype in self.fulltypes:
            num = self.datatypes.count(datatype)
            if num > 1:
                raise ValueError('There are '+str(num)+' "' + datatype +
                                 '" in datatype "' + self.datatypes + '".')

        self.R = self.datatypes.find('R')  # positions
        self.Z = self.datatypes.find('Z')  # atom_types
        self.C = self.datatypes.find('C')  # pbcbox
        self.D = self.datatypes.find('D')  # distances
        self.N = self.datatypes.find('N')  # neighbours
        self.n = self.datatypes.find('n')  # neighbour_mask
        self.B = self.datatypes.find('B')  # bonds
        self.b = self.datatypes.find('b')  # bond_mask
        self.E = self.datatypes.find('E')  # energy

        if self.E < 0:
            raise TypeError('The datatype "E" must be included!')

        self._network = network
        self._loss_fn = loss_fn

        self.hyper_param = None
        if 'hyper_param' in self._network.__dict__.keys():
            self.hyper_param = self._network.hyper_param

        self.atom_types = None
        if (context.get_context("mode") == context.PYNATIVE_MODE and
                'atom_types' in self._network.__dict__['_tensor_list'].keys()) or \
                (context.get_context("mode") == context.GRAPH_MODE and
                 'atom_types' in self._network.__dict__.keys()):
            self.atom_types = self._network.atom_types

        print(self.cls_name + ' with input type: ' + self.datatypes)

        self.keep_sum = P.ReduceSum(True)


class WithForceLossCell(WithCell):
    r"""Cell to combine the network and the loss function with force.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes='RZCDNnBbFE'
        )

        #pylint: disable=invalid-name

        self.F = self.datatypes.find('F')  # force
        if self.F < 0:
            raise TypeError(
                'The datatype "F" must be included in WithForceLossCell!')

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):  Tensor of shape (B, 1). Data type is float.

        """
        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbc_box = inputs[self.C]
        distances = inputs[self.D]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        energy = inputs[self.E]
        out = self._network(
            positions=positions,
            atom_types=atom_types,
            pbc_box=pbc_box,
            distances=distances,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        forces = inputs[self.F]
        fout = -1 * self.grad_op(self._network)(
            positions,
            atom_types,
            pbc_box,
            distances,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types > 0, out.dtype)
        num_atoms = self.keep_sum(num_atoms, -1)

        if atom_types is None:
            return self._loss_fn(out, energy, fout, forces)
        atom_mask = atom_types > 0
        return self._loss_fn(out, energy, fout, forces, num_atoms, atom_mask)

    @property
    def backbone_network(self):
        return self._network


class WithLabelLossCell(WithCell):
    r"""Cell to combine the network and the loss function with label.

    Args:

        datatypes (str):        Data types of the inputs.

        network (Cybertron):    Neural network of Cybertron

        loss_fn (Cell):         Loss function.

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes='RZCDNnBbE'
        )

    def construct(self, *inputs):
        """calculate loss function

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):  Tensor of shape (B, 1). Data type is float.

        """
        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbc_box = inputs[self.C]
        distances = inputs[self.D]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        out = self._network(
            positions=positions,
            atom_types=atom_types,
            pbc_box=pbc_box,
            distances=distances,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        label = inputs[self.E]

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types > 0, out.dtype)
        num_atoms = self.keep_sum(num_atoms, -1)

        return self._loss_fn(out, label)


class WithEvalCell(WithCell):
    r"""Basic cell to combine the network and the evaluate function.

    Args:

        datatypes (str):            Data types of the inputs.

        network (Cybertron):        Neural network of Cybertron

        loss_fn (Cell):             Loss function.

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        eval_data_is_normed (bool): Whether the evaluate dataset is normalized. Default: False

        add_cast_fp32 (bool):       Whether cast the dataset to 32-bit. Default: False

        fulltypes (str):            Full list of data types. Default: RZCDNnBbE'

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell = None,
                 scale: float = None,
                 shift: float = None,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: Tensor = None,
                 eval_data_is_normed: bool = True,
                 add_cast_fp32: bool = False,
                 fulltypes: str = 'RZCDNnBbE'
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            fulltypes=fulltypes
        )

        self.scale = scale
        self.shift = shift

        if atomwise_scaleshift is None:
            atomwise_scaleshift = self._network.atomwise_scaleshift
        else:
            atomwise_scaleshift = Tensor(atomwise_scaleshift, ms.bool_)
        self.atomwise_scaleshift = atomwise_scaleshift

        self.scaleshift = None
        self.normalization = None
        self.scaleshift_eval = eval_data_is_normed
        self.normalize_eval = False
        self.type_ref = None
        if scale is not None or shift is not None:
            if scale is None:
                scale = 1
            if shift is None:
                shift = 0

            if type_ref is not None:
                self.type_ref = Tensor(type_ref, ms.float32)

            self.scaleshift = OutputScaleShift(
                scale=scale,
                shift=shift,
                type_ref=self.type_ref,
                atomwise_scaleshift=atomwise_scaleshift
            )

            if self._loss_fn is not None:
                self.normalization = DatasetNormalization(
                    scale=scale,
                    shift=shift,
                    type_ref=self.type_ref,
                    atomwise_scaleshift=atomwise_scaleshift
                )
                if not eval_data_is_normed:
                    self.normalize_eval = True

            self.scale = self.scaleshift.scale
            self.shift = self.scaleshift.shift

            scale = self.scale.asnumpy().reshape(-1)
            shift = self.shift.asnumpy().reshape(-1)
            atomwise_scaleshift = self.scaleshift.atomwise_scaleshift.asnumpy().reshape(-1)
            print('   with scaleshift for training ' +
                  ('and evaluate ' if eval_data_is_normed else ' ')+'dataset:')
            if atomwise_scaleshift.size == 1:
                print('   Scale: '+str(scale))
                print('   Shift: '+str(shift))
                print('   Scaleshift mode: ' +
                      ('atomwise' if atomwise_scaleshift else 'graph'))
            else:
                print('   {:>6s}. {:>16s}{:>16s}{:>12s}'.format(
                    'Output', 'Scale', 'Shift', 'Mode'))
                for i, m in enumerate(atomwise_scaleshift):
                    scale_ = scale if scale.size == 1 else scale[i]
                    shift_ = scale if shift.size == 1 else shift[i]
                    mode = 'Atomwise' if m else 'graph'
                    print('   {:<6s}{:>16.6e}{:>16.6e}{:>12s}'.format(
                        str(i)+': ', scale_, shift_, mode))
            if type_ref is not None:
                print('   with reference value for atom types:')
                info = '   Type '
                for i in range(self.type_ref.shape[-1]):
                    info += '{:>10s}'.format('Label'+str(i))
                print(info)
                for i, ref in enumerate(self.type_ref):
                    info = '   {:<7s} '.format(str(i)+':')
                    for r in ref:
                        info += '{:>10.2e}'.format(r.asnumpy())
                    print(info)

        self.add_cast_fp32 = add_cast_fp32
        self.reducesum = P.ReduceSum(True)


class WithLabelEvalCell(WithEvalCell):
    r"""Cell to combine the network and the evaluate function with label.

    Args:

        datatypes (str):            Data types of the inputs.

        network (Cybertron):        Neural network of Cybertron

        loss_fn (Cell):             Loss function.

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        eval_data_is_normed (bool): Whether the evaluate dataset is normalized. Default: False

        add_cast_fp32 (bool):       Whether cast the dataset to 32-bit. Default: False

    """
    def __init__(self,
                 datatypes: str,
                 network: Cybertron,
                 loss_fn: Cell = None,
                 scale: float = None,
                 shift: float = None,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: Tensor = None,
                 eval_data_is_normed: bool = True,
                 add_cast_fp32: bool = False,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            eval_data_is_normed=eval_data_is_normed,
            add_cast_fp32=add_cast_fp32,
            fulltypes='RZCDNnBbE',
        )

    def construct(self, *inputs):
        """calculate evaluate data

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):      Tensor of shape (B, 1). Data type is float.
                                Loss function of evaluate data.
            output (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Predicted results of network.
            label (Tensor):     Tensor of shape (B, 1). Data type is float.
                                Label of evaluate data.
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms in each molecule.

        """
        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbc_box = inputs[self.C]
        distances = inputs[self.D]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        output = self._network(
            positions=positions,
            atom_types=atom_types,
            pbc_box=pbc_box,
            distances=distances,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        label = inputs[self.E]
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(ms.float32, label)
            output = F.cast(output, ms.float32)

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types > 0, ms.int32)
        num_atoms = msnp.sum(atom_types > 0, -1, keepdims=True)

        loss = 0
        if self._loss_fn is not None:
            if self.normalize_eval:
                normed_label = self.normalization(label, num_atoms, atom_types)
                loss = self._loss_fn(output, normed_label)
            else:
                loss = self._loss_fn(output, label)

        if self.scaleshift is not None:
            output = self.scaleshift(output, num_atoms, atom_types)
            if self.scaleshift_eval:
                label = self.scaleshift(label, num_atoms, atom_types)

        return loss, output, label, num_atoms


class WithForceEvalCell(WithEvalCell):
    r"""Cell to combine the network and the evaluate function with force.

    Args:

        datatypes (str):            Data types of the inputs.

        network (Cybertron):        Neural network of Cybertron

        loss_fn (Cell):             Loss function.

        scale (float):              Scale value. Default: 1

        shift (float):              Shift value. Default: 0

        type_ref (Tensor):          Tensor of shape (T, E). Data type is float
                                    Reference values of label for each atom type. Default: None

        atomwise_scaleshift (bool): Whether to do atomwise scale and shift. Default: None

        eval_data_is_normed (bool): Whether the evaluate dataset is normalized. Default: False

        add_cast_fp32 (bool):       Whether cast the dataset to 32-bit. Default: False

    """
    def __init__(self,
                 datatypes,
                 network: Cybertron,
                 loss_fn: Cell = None,
                 scale: float = None,
                 shift: float = None,
                 type_ref: Tensor = None,
                 atomwise_scaleshift: Tensor = None,
                 eval_data_is_normed: bool = True,
                 add_cast_fp32: bool = False,
                 ):

        super().__init__(
            datatypes=datatypes,
            network=network,
            loss_fn=loss_fn,
            scale=scale,
            shift=shift,
            type_ref=type_ref,
            atomwise_scaleshift=atomwise_scaleshift,
            eval_data_is_normed=eval_data_is_normed,
            add_cast_fp32=add_cast_fp32,
            fulltypes='RZCDNnBbFE',
        )
        #pylint: disable=invalid-name

        self.F = self.datatypes.find('F')  # force

        if self.F < 0:
            raise TypeError(
                'The datatype "F" must be included in WithForceEvalCell!')

        self.grad_op = C.GradOperation()

    def construct(self, *inputs):
        """calculate evaluate data

        Args:
            *input: Tuple of Tensor

        Returns:
            loss (Tensor):      Tensor of shape (B, 1). Data type is float.
                                Loss function of evaluate data.
            output (Tensor):    Tensor of shape (B, 1). Data type is float.
                                Predicted results of network.
            label (Tensor):     Tensor of shape (B, 1). Data type is float.
                                Label of evaluate data.
            num_atoms (Tensor): Tensor of shape (B, 1). Data type is int.
                                Number of atoms in each molecule.

        """
        inputs = inputs + (None,)

        positions = inputs[self.R]
        atom_types = inputs[self.Z]
        pbc_box = inputs[self.C]
        distances = inputs[self.D]
        neighbours = inputs[self.N]
        neighbour_mask = inputs[self.n]
        bonds = inputs[self.B]
        bond_mask = inputs[self.b]

        output_energy = self._network(
            positions=positions,
            atom_types=atom_types,
            pbc_box=pbc_box,
            distances=distances,
            neighbours=neighbours,
            neighbour_mask=neighbour_mask,
            bonds=bonds,
            bond_mask=bond_mask,
        )

        output_forces = -1 * self.grad_op(self._network)(
            positions,
            atom_types,
            pbc_box,
            distances,
            neighbours,
            neighbour_mask,
            bonds,
            bond_mask,
        )

        label_forces = inputs[self.F]
        label_energy = inputs[self.E]

        if self.add_cast_fp32:
            label_forces = F.mixed_precision_cast(ms.float32, label_forces)
            label_energy = F.mixed_precision_cast(ms.float32, label_energy)
            output_energy = F.cast(output_energy, ms.float32)

        if atom_types is None:
            atom_types = self.atom_types

        num_atoms = F.cast(atom_types > 0, ms.int32)
        num_atoms = msnp.sum(atom_types > 0, -1, keepdims=True)

        loss = 0
        if self._loss_fn is not None:
            atom_mask = atom_types > 0
            if self.normalize_eval:
                normed_label_energy = self.normalization(
                    label_energy, num_atoms, atom_types)
                normed_label_forces = label_forces / self.scale
                loss = self._loss_fn(output_energy, normed_label_energy,
                                     output_forces, normed_label_forces, num_atoms, atom_mask)
            else:
                loss = self._loss_fn(
                    output_energy, label_energy, output_forces, label_forces, num_atoms, atom_mask)

        if self.scaleshift is not None:
            output_energy = self.scaleshift(
                output_energy, num_atoms, atom_types)
            output_forces = output_forces * self.scale
            if self.scaleshift_eval:
                label_energy = self.scaleshift(
                    label_energy, num_atoms, atom_types)
                label_forces = label_forces * self.scale

        return loss, output_energy, label_energy, output_forces, label_forces, num_atoms


class WithAdversarialLossCell(Cell):
    r"""Adversarial network.

    Args:

        network (Cell): Neural network.

        loss_fn (Cell): Loss function.

    """
    def __init__(self,
                 network: Cell,
                 loss_fn: Cell,
                 ):

        super().__init__(auto_prefix=False)
        self._network = network
        self._loss_fn = loss_fn

    def construct(self, pos_samples: Tensor, neg_samples: Tensor):
        """calculate the loss function of adversarial network

        Args:
            pos_pred (Tensor):  Positive samples
            neg_pred (Tensor):  Negative samples

        Returns:
            loss (Tensor):      Loss function with same shape of samples

        """
        pos_pred = self._network(pos_samples)
        neg_pred = self._network(neg_samples)
        return self._loss_fn(pos_pred, neg_pred)

    @property
    def backbone_network(self):
        return self._network


class TrainMonitor(Callback):
    r"""A callback to show and record the information during training process.

    Args:

        model (Model):              Mindspore model.

        file_name (str):            Name of the file to record the training information.

        directory (str):            Name of output directory. Default: None

        per_epoch (int):            The epoch interval for outputting training information. Default: 1

        per_step (int):             The step interval for outputting training information. Default: 0

        avg_steps (int):            Number of step for the moving average of loss function.
                                    If 0 is given, the loss will be averaged over all previous steps.
                                    Default: 0

        eval_dataset (Dataset):     Evaluate dataset. Default: None

        best_ckpt_metrics (str):    Reference metric to record the best parameters. Default: None

    """
    def __init__(self,
                 model: Model,
                 file_name: str,
                 directory: str = None,
                 per_epoch: int = 1,
                 per_step: int = 0,
                 avg_steps: int = 0,
                 eval_dataset: Dataset = None,
                 best_ckpt_metrics: str = None,
                 ):

        super().__init__()
        if not isinstance(per_epoch, int) or per_epoch < 0:
            raise ValueError("per_epoch must be int and >= 0.")
        if not isinstance(per_step, int) or per_step < 0:
            raise ValueError("per_step must be int and >= 0.")

        self.avg_steps = avg_steps
        self.loss_record = 0
        self.train_num = 0
        if avg_steps > 0:
            self.train_num = deque(maxlen=avg_steps)
            self.loss_record = deque(maxlen=avg_steps)

        if per_epoch * per_step != 0:
            if per_epoch == 1:
                per_epoch = 0
            else:
                raise ValueError(
                    "per_epoch and per_step cannot larger than 0 at same time.")
        self.model = model
        self._per_epoch = per_epoch
        self._per_step = per_step
        self.eval_dataset = eval_dataset

        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir

        self._filename = file_name + '-info.data'
        self._ckptfile = file_name + '-best'
        self._ckptdata = file_name + '-ckpt.data'

        self.num_ckpt = 1
        self.best_value = 5e4
        self.best_ckpt_metrics = best_ckpt_metrics

        self.last_loss = 0
        self.record = []

        self.hyper_param = None

        self.output_title = True
        filename = os.path.join(self._directory, self._filename)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    os.remove(filename)

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        #pylint: disable=protected-access
        cb_params: InternalCallbackParam = run_context.original_args()
        train_network: TrainOneStepCell = cb_params.train_network
        cells = train_network._cells
        if 'network' in cells.keys() and 'hyper_param' in cells['network'].__dict__.keys():
            self.hyper_param = cells['network'].hyper_param

    def _write_ckpt_file(self, filename: str, info: str, network: TrainOneStepCell):
        """write checkpoint (.ckpt) file"""
        ckptfile = os.path.join(self._directory, filename + '.ckpt')
        ckptbck = os.path.join(self._directory, filename + '.bck.ckpt')
        ckptdata = os.path.join(self._directory, self._ckptdata)

        if os.path.exists(ckptfile):
            os.rename(ckptfile, ckptbck)

        save_checkpoint(network, ckptfile, append_dict=self.hyper_param)
        with open(ckptdata, "a") as f:
            f.write(info + os.linesep)

    def _output_data(self, cb_params: InternalCallbackParam):
        """output data"""
        cur_epoch = cb_params.cur_epoch_num

        opt = cb_params.optimizer
        if opt is None:
            opt = cb_params.train_network.optimizer

        if opt.dynamic_lr:
            step = opt.global_step
            if not isinstance(step, int):
                step = step.asnumpy()[0]
        else:
            step = cb_params.cur_step_num

        if self.avg_steps > 0:
            mov_avg = sum(self.loss_record) / sum(self.train_num)
        else:
            mov_avg = self.loss_record / self.train_num

        title = "#! FIELDS step"
        info = 'Epoch: ' + str(cur_epoch) + ', Step: ' + str(step)
        outdata = '{:>10d}'.format(step)

        lr = opt.learning_rate
        if opt.dynamic_lr:
            step = F.cast(step, ms.int32)
            if opt.is_group_lr:
                lr = ()
                for learning_rate in opt.learning_rate:
                    current_dynamic_lr = learning_rate(step-1)
                    lr += (current_dynamic_lr,)
            else:
                lr = opt.learning_rate(step-1)
        lr = lr.asnumpy()

        title += ' learning_rate'
        info += ', Learning_rate: ' + str(lr)
        outdata += '{:>15e}'.format(lr)

        title += " last_loss avg_loss"
        info += ', Last_Loss: ' + \
            str(self.last_loss) + ', Avg_loss: ' + str(mov_avg)
        outdata += '{:>15e}'.format(self.last_loss) + '{:>15e}'.format(mov_avg)

        _make_directory(self._directory)

        if self.eval_dataset is not None:
            eval_metrics = self.model.eval(
                self.eval_dataset, dataset_sink_mode=False)
            for k, v in eval_metrics.items():
                info += ', '
                info += k
                info += ': '
                info += str(v)

                if isinstance(v, ndarray) and v.size > 1:
                    for i in range(v.size):
                        title += (' ' + k + str(i))
                        outdata += '{:>15e}'.format(v[i])
                else:
                    title += (' ' + k)
                    outdata += '{:>15e}'.format(v)

            if self.best_ckpt_metrics in eval_metrics.keys():
                best_value = eval_metrics[self.best_ckpt_metrics]
                self._write_best_ckpt(best_value, info, cb_params.train_network)

        print(info, flush=True)
        filename = os.path.join(self._directory, self._filename)
        if self.output_title:
            with open(filename, "a") as f:
                f.write(title + os.linesep)
            self.output_title = False
        with open(filename, "a") as f:
            f.write(outdata + os.linesep)

    def _write_best_ckpt(self, best_value: ndarray, info: str, network: Cell):
        """write the best parameter of checkpoint file"""
        if isinstance(best_value, ndarray) and len(best_value) > 1:
            output_ckpt = best_value < self.best_value
            num_best = np.count_nonzero(output_ckpt)
            if num_best > 0:
                self._write_ckpt_file(
                    self._ckptfile, info, network)
                source_ckpt = os.path.join(
                    self._directory, self._ckptfile + '.ckpt')
                for i in range(len(best_value)):
                    if output_ckpt[i]:
                        dest_ckpt = os.path.join(
                            self._directory, self._ckptfile + '-' + str(i) + '.ckpt')
                        bck_ckpt = os.path.join(
                            self._directory, self._ckptfile + '-' + str(i) + '.ckpt.bck')
                        if os.path.exists(dest_ckpt):
                            os.rename(dest_ckpt, bck_ckpt)
                        copyfile(source_ckpt, dest_ckpt)
                self.best_value = np.minimum(best_value, self.best_value)
        else:
            if best_value < self.best_value:
                self._write_ckpt_file(
                    self._ckptfile, info, network)
                self.best_value = best_value
        return self

    def step_end(self, run_context: RunContext):
        """step end"""
        cb_params: InternalCallbackParam = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        nbatch = len(cb_params.train_dataset_element[0])
        batch_loss = loss * nbatch

        self.last_loss = loss
        if self.avg_steps > 0:
            self.loss_record.append(batch_loss)
            self.train_num.append(nbatch)
        else:
            self.loss_record += batch_loss
            self.train_num += nbatch

        if self._per_step > 0 and cb_params.cur_step_num % self._per_step == 0:
            self._output_data(cb_params)

    def epoch_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if self._per_epoch > 0 and cur_epoch % self._per_epoch == 0:
            self._output_data(cb_params)


class MaxError(Metric):
    r"""Metric to calcaulte the max error.

    Args:

        indexes (tuple):        Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool): Whether to summation the data of all atoms in molecule. Default: True

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True
                 ):

        super().__init__()
        self.clear()
        self._indexes = indexes
        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

    def clear(self):
        self._max_error = 0

    def update(self, *inputs):
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])
        diff = y.reshape(y_pred.shape) - y_pred
        max_error = diff.max() - diff.min()
        if max_error > self._max_error:
            self._max_error = max_error

    def eval(self):
        return self._max_error


class Error(Metric):
    r"""Metric to calcaulte the error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__()
        self.clear()
        self._indexes = indexes
        self.read_num_atoms = False
        if len(self._indexes) > 2:
            self.read_num_atoms = True

        self.reduce_all_dims = reduce_all_dims

        if atom_aggregate.lower() not in ('mean', 'sum'):
            raise ValueError(
                'aggregate_by_atoms method must be "mean" or "sum"')
        self.atom_aggregate = atom_aggregate.lower()

        if reduce_all_dims:
            self.axis = None
        else:
            self.axis = 0

        if averaged_by_atoms and not self.read_num_atoms:
            raise ValueError(
                'When to use averaged_by_atoms, the index of atom number must be set at "indexes".')

        self.averaged_by_atoms = averaged_by_atoms

        self._error_sum = 0
        self._samples_num = 0

    def clear(self):
        self._error_sum = 0
        self._samples_num = 0

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        """calculate error"""
        return y.reshape(y_pred.shape) - y_pred

    def update(self, *inputs):
        """update metric"""
        y_pred = self._convert_data(inputs[self._indexes[0]])
        y = self._convert_data(inputs[self._indexes[1]])

        error = self._calc_error(y, y_pred)
        if len(error.shape) > 2:
            axis = tuple(range(2, len(error.shape)))
            if self.atom_aggregate == 'mean':
                error = np.mean(error, axis=axis)
            else:
                error = np.sum(error, axis=axis)

        tot = y.shape[0]
        if self.read_num_atoms:
            natoms = self._convert_data(inputs[self._indexes[2]])
            if self.averaged_by_atoms:
                error /= natoms
            elif self.reduce_all_dims:
                tot = np.sum(natoms)
                if natoms.shape[0] != y.shape[0]:
                    tot *= y.shape[0]
        elif self.reduce_all_dims:
            tot = error.size

        self._error_sum += np.sum(error, axis=self.axis)
        self._samples_num += tot

    def eval(self) -> float:
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self._error_sum / self._samples_num


class MAE(Error):
    r"""Metric to calcaulte the mean absolute error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        return np.abs(y.reshape(y_pred.shape) - y_pred)

class MSE(Error):
    r"""Metric to calcaulte the mean square error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        return np.square(y.reshape(y_pred.shape) - y_pred)


class MNE(Error):
    r"""Metric to calcaulte the mean norm error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        diff = y.reshape(y_pred.shape) - y_pred
        return np.linalg.norm(diff, axis=-1)

class RMSE(Error):
    r"""Metric to calcaulte the root mean square error.

    Args:

        indexes (tuple):            Indexes for label and predicted data. Default: (1, 2)

        reduce_all_dims (bool):     Whether to summation the data of all atoms in molecule. Default: True

        averaged_by_atoms (bool):   Whether to averaged the data by the number of atoms in molecule.
                                    Default: True

        atom_aggregate (str):       The way to aggregate the data of each atom. Default: 'mean'

    """
    def __init__(self,
                 indexes: tuple = (1, 2),
                 reduce_all_dims: bool = True,
                 averaged_by_atoms: bool = False,
                 atom_aggregate: str = 'mean',
                 ):

        super().__init__(
            indexes=indexes,
            reduce_all_dims=reduce_all_dims,
            averaged_by_atoms=averaged_by_atoms,
            atom_aggregate=atom_aggregate,
        )

    def _calc_error(self, y: ndarray, y_pred: ndarray) -> ndarray:
        return np.square(y.reshape(y_pred.shape) - y_pred)

    def eval(self):
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return np.sqrt(self._error_sum / self._samples_num)


class MLoss(Metric):
    r"""Metric to calcaulte the loss function.

    Args:

        indexes (int):            Index for loss function. Default: 0

    """
    def __init__(self, index: int = 0):
        super().__init__()
        self.clear()
        self._index = index

    def clear(self):
        self._sum_loss = 0
        self._total_num = 0

    def update(self, *inputs):
        """update metric"""
        loss = self._convert_data(inputs[self._index])

        if loss.ndim == 0:
            loss = loss.reshape(1)

        if loss.ndim != 1:
            raise ValueError(
                "Dimensions of loss must be 1, but got {}".format(loss.ndim))

        loss = loss.mean(-1)
        self._sum_loss += loss
        self._total_num += 1

    def eval(self):
        if self._total_num == 0:
            raise RuntimeError('Total number can not be 0.')
        return self._sum_loss / self._total_num


class TransformerLR(LearningRateSchedule):
    r"""A transformer type dynamic learning rate schedule.

    Args:

        learning_rate (float):  Reference learning rate. Default: 1.0

        warmup_steps (int):     Warm up steps. Default: 4000

        dimension (int):        Dimension of output Tensor. Default: 1

    """
    def __init__(self,
                 learning_rate: float = 1.0,
                 warmup_steps: int = 4000,
                 dimension: int = 1,
                 ):

        super().__init__()
        if not isinstance(learning_rate, float):
            raise TypeError("learning_rate must be float.")
        validator.check_non_negative_float(
            learning_rate, "learning_rate", self.cls_name)
        validator.check_positive_int(
            warmup_steps, 'warmup_steps', self.cls_name)

        self.learning_rate = learning_rate

        self.pow = P.Pow()
        self.warmup_steps = F.cast(warmup_steps, ms.float32)
        # self.warmup_scale = self.pow(F.cast(warmup_steps,ms.float32),-1.5)
        self.dimension = F.cast(dimension, ms.float32)
        # self.dim_scale = self.pow(F.cast(dimension,ms.float32),-0.5)

        self.min = P.Minimum()

    def construct(self, global_step: int):
        """Calculate the learning rate at current step.

        Args:
            global_step (int):  Global training step.

        Returns:
            lr (Tensor):   Current learning rate.

        """
        step_num = F.cast(global_step, ms.float32)
        warmup_scale = self.pow(self.warmup_steps, -1.5)
        dim_scale = self.pow(self.dimension, -0.5)
        lr1 = self.pow(step_num, -0.5)
        lr2 = step_num*warmup_scale
        lr_percent = dim_scale * self.min(lr1, lr2)
        return self.learning_rate * lr_percent
