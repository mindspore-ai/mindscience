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
Loss functions
"""

from typing import Union
from numpy import ndarray

import mindspore as ms
from mindspore import Tensor
from mindspore.nn.loss.loss import LossBase
from mindspore import ops
from mindspore.ops import functional as F


from ..configs import Config

__all__ = [
    'MolecularLoss',
    'MAELoss',
    'MSELoss',
    'CrossEntropyLoss',
]


class MolecularLoss(LossBase):
    r"""Loss function of the energy and force of molecule.

    Args:
        force_dis (float): A average norm value of force, which used to scale the force. Default: 1

        atomwise (bool): Whether to average over each atom when calculating the loss function.
            Default: ``None``.

        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            Default: "mean".

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 force_dis: Union[float, Tensor, ndarray] = 1,
                 atomwise: bool = None,
                 reduction: str = 'mean',
                 **kwargs
                 ):
        super().__init__(reduction)
        self._kwargs = kwargs

        self._atomwise = atomwise
        self._force_dis = ms.Tensor(force_dis, ms.float32)

    def set_atomwise(self, atomwise: bool = True):
        """set whether to use atomwise """
        if self._atomwise is None:
            self._atomwise = atomwise
        return self

    def construct(self,
                  predict: Tensor,
                  label: Tensor,
                  num_atoms: Tensor = 1,
                  atom_mask: Tensor = None,
                  **kwargs,
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
                                    Default: ``None``.

        Note:
            B:  Batch size
            A:  Number of atoms
            D:  Dimension of position coordinate. Usually is 3.
            E:  Number of labels

        Returns:
            loss (Tensor):  Tensor with shape (B, 1). Data type is float.
                            Loss function.
        """
        # pylint: disable=unused-argument

        if (not self._atomwise) or predict.ndim > 3 or predict.ndim < 2:
            loss = self._calc_loss(predict - label)
            return self.get_loss(loss)

        if predict.ndim == 3:
            # The shape looks like (B, A, X)
            diff = (predict - label) * self._force_dis
            diff = self._calc_loss(diff)
            # The shape looks like (B, A)
            diff = F.reduce_sum(diff, -1)

            if atom_mask is None:
                # The shape looks like (B, 1) <- (B, A)
                loss = ops.mean(diff, -1, keep_dims=True)
            else:
                # The shape looks like (B, A) * (B, A)
                diff = diff * atom_mask
                # The shape looks like (B, 1) <- (B, A)
                loss = diff.sum(-1, keepdims=True)
                # The shape looks like (B, 1) / (B, 1)
                loss = loss / num_atoms
        else:
            # The shape looks like (B, Y)
            diff = (predict - label) / num_atoms
            loss = self._calc_loss(diff)

        # The shape looks like (B, 1)
        num_atoms = F.cast(num_atoms, predict.dtype)
        weights = num_atoms / F.reduce_mean(num_atoms)

        return self.get_loss(loss, weights)

    def _calc_loss(self, diff: Tensor) -> Tensor:
        """calculate loss function"""
        raise NotImplementedError


class MAELoss(MolecularLoss):
    r"""Mean-absolute-error-type Loss function for energy and force.

    Args:
        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: ``True``.

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 force_dis: Union[float, Tensor, ndarray] = 1,
                 atomwise: bool = None,
                 reduction: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            force_dis=force_dis,
            atomwise=atomwise,
            reduction=reduction,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def _calc_loss(self, diff: Tensor) -> Tensor:
        return F.abs(diff)


class MSELoss(MolecularLoss):
    r"""Mean-square-error-type Loss function for energy and force.

    Args:
        ratio_energy (float):   Ratio of energy in loss function. Default: 1

        ratio_forces (float):   Ratio of forces in loss function. Default: 100

        force_dis (float):      A average norm value of force, which used to scale the force.
                                Default: 1

        ratio_normlize (bool):  Whether to do normalize the ratio of energy and force. Default: ``True``.

        reduction (str):        Method to reduction the output Tensor. Default: 'mean'

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 force_dis: Union[float, Tensor, ndarray] = 1,
                 atomwise: bool = None,
                 reduction: str = 'mean',
                 **kwargs
                 ):

        super().__init__(
            force_dis=force_dis,
            atomwise=atomwise,
            reduction=reduction,
        )
        self._kwargs = Config.get_arguments(locals(), kwargs)

    def _calc_loss(self, diff: Tensor) -> Tensor:
        return F.square(diff)


class CrossEntropyLoss(LossBase):
    r"""Cross entropy Loss function for positive and negative samples.

    Args:
        reduction (str):    Method to reduction the output Tensor. Default: 'mean'

        use_sigmoid (bool): Whether to use sigmoid function for output. Default: ``False``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 reduction: str = 'mean',
                 use_sigmoid: bool = False
                 ):

        super().__init__(reduction)

        self.sigmoid = None
        self.use_sigmoid = use_sigmoid

        self.cross_entropy = ops.BinaryCrossEntropy(reduction)

    def construct(self, pos_pred: Tensor, neg_pred: Tensor):
        """calculate cross entropy loss function

        Args:
            pos_pred (Tensor):  Positive samples
            neg_pred (Tensor):  Negative samples

        Returns:
            loss (Tensor):      Loss function with same shape of samples

        """
        if self.use_sigmoid:
            pos_pred = F.sigmoid(pos_pred)
            neg_pred = F.sigmoid(neg_pred)

        pos_loss = self.cross_entropy(pos_pred, F.ones_like(pos_pred))
        neg_loss = self.cross_entropy(neg_pred, F.zeros_like(neg_pred))

        return pos_loss + neg_loss
