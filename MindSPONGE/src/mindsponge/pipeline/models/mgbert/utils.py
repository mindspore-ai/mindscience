# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Utils of MG-BERT"""

from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
import numpy as np
import openbabel as ob
import mindspore.nn as nn
import mindspore as ms
from mindspore.ops.function.nn_func import _innner_log_softmax, _get_cache_prim
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore import _checkparam as Validator
from mindspore.nn.layer.activation import get_activation
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops.primitive import Primitive


def obsmitosmile(smi):
    """obsmitosmile"""
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = ob.OBMol()
    conv.ReadString(mol, smi)
    smile = conv.WriteString(mol)
    smile = smile.replace('\t\n', '')
    return smile


def smiles2adjoin(smiles, explicit_hydrogens=True, canonical_atom_order=False):
    """Smiles to adjoin"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('error')
        mol = Chem.MolFromSmiles(obsmitosmile(smiles))
        if mol is None:
            raise ValueError(f'{smiles} if not vaild')

    if explicit_hydrogens:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)

    if canonical_atom_order:
        new_order = rdmolfiles.CanonicalRankAtoms(mol)
        mol = rdmolops.RenumberAtoms(mol, new_order)
    num_atoms = mol.GetNumAtoms()
    atoms_list = []
    for i in range(num_atoms):
        atom = mol.GetAtomWithIdx(i)
        atoms_list.append(atom.GetSymbol())

    adjoin_matrix = np.eye(num_atoms)
    # Add edges
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0
    return atoms_list, adjoin_matrix


def data_rebuild_pre(x, adjoin_matrix, y, char_weight):
    """Data rebuild"""
    x = ms.ops.Cast()(x, ms.int32)
    adjoin_matrix = ms.ops.Cast()(adjoin_matrix, ms.float32)
    output = [x, adjoin_matrix, y, char_weight]
    return output


def data_rebuild_c_r(x, adjoin_matrix):
    """Data rebuild"""
    x = ms.ops.Cast()(x, ms.int32)
    adjoin_matrix = ms.ops.Cast()(adjoin_matrix, ms.float32)
    return x, adjoin_matrix


def cross_entropy(inputs, target, sample_weight, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    """cross_entropy"""
    class_dim = 0 if inputs.ndim == 1 else 1

    return nll_loss(_innner_log_softmax(inputs, class_dim), target, sample_weight, weight, ignore_index, reduction,
                    label_smoothing)


def nll_loss(inputs, target, sample_weight, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0):
    """Loss"""
    ndim = inputs.ndim
    if ndim == 2:
        ret = _nll_loss(inputs, target, sample_weight, -1, weight, ignore_index, reduction, label_smoothing)
    elif ndim == 4:
        ret = _nll_loss(inputs, target, sample_weight, 1, weight, ignore_index, reduction, label_smoothing)
    elif ndim == 1:
        ret = _nll_loss(inputs, target, sample_weight, 0, weight, ignore_index, reduction, label_smoothing)
    else:
        n = inputs.shape[0]
        c = inputs.shape[1]
        out_size = (n,) + inputs.shape[2:]
        inputs = inputs.view(n, c, 1, -1)
        target = target.view(n, 1, -1)
        if reduction != 'none':
            ret = _nll_loss(inputs, target, sample_weight, 1, weight, ignore_index, reduction, label_smoothing)
        else:
            ret = _nll_loss(inputs, target, sample_weight, 1, weight, ignore_index, label_smoothing=label_smoothing)
            ret = ret.view(out_size)
    return ret


def _nll_loss(inputs, target, sample_weight, target_dim=-1, weight=None, ignore_index=None, reduction='none',
              label_smoothing=0.0):
    """Loss"""
    l_neg = _get_cache_prim(P.Neg)()
    l_gather_d = _get_cache_prim(P.GatherD)()
    l_gather = _get_cache_prim(P.Gather)()
    l_ones_like = _get_cache_prim(P.OnesLike)()
    l_equal = _get_cache_prim(P.Equal)()

    if target.ndim == inputs.ndim - 1:
        target = target.expand_dims(target_dim)
    loss = l_neg(l_gather_d(inputs, target_dim, target))
    smooth_loss = l_neg(inputs.sum(axis=target_dim, keepdims=True))
    if weight is not None:
        loss_weights = l_gather(weight, target, 0)
        loss = loss * loss_weights
    else:
        loss_weights = l_ones_like(loss)
    if ignore_index is not None:
        non_pad_mask = l_equal(target, ignore_index)
        loss = loss.masked_fill(non_pad_mask, 0.)
        loss_weights = loss_weights.masked_fill(non_pad_mask, 0.)
        smooth_loss = smooth_loss.masked_fill(non_pad_mask, 0.)

    loss = loss.squeeze(target_dim)
    smooth_loss = smooth_loss.squeeze(target_dim)
    loss = loss * sample_weight

    if reduction == 'sum':
        loss = loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == 'mean':
        loss = loss.sum() / loss_weights.sum()
        smooth_loss = smooth_loss.mean()

    eps_i = label_smoothing / inputs.shape[target_dim]
    loss = (1. - label_smoothing) * loss + eps_i * smooth_loss

    return loss


class SampleLoss(nn.LossBase):
    """Sample loss"""

    def __init__(self, weight=None, ignore_index=-100, reduction='mean',
                 label_smoothing=0.0):
        super().__init__(reduction)

        self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def construct(self, logits, labels, sample_weight):
        '''
        :param logits: batch_size * length * num_classes  #b*n*c
        :param labels: batch_size * length                #b*n
        '''
        return cross_entropy(logits, labels, sample_weight, self.weight, self.ignore_index, self.reduction,
                             self.label_smoothing)


class Dense(nn.Cell):
    """
    preprocess input of each layer.
    """

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None):
        super(Dense, self).__init__()
        self.in_channels = Validator.check_positive_int(in_channels, "in_channels", self.cls_name)
        self.out_channels = Validator.check_positive_int(out_channels, "out_channels", self.cls_name)
        self.has_bias = Validator.check_bool(has_bias, "has_bias", self.cls_name)
        self.reshape = P.Reshape()
        self.shape_op = P.Shape()

        if isinstance(weight_init, Tensor):
            if weight_init.ndim != 2 or weight_init.shape[0] != out_channels or \
                    weight_init.shape[1] != in_channels:
                raise ValueError(f"For '{self.cls_name}', weight init shape error. The ndim of 'weight_init' must "
                                 f"be equal to 2, and the first dim must be equal to 'out_channels', and the "
                                 f"second dim must be equal to 'in_channels'. But got 'weight_init': {weight_init}, "
                                 f"'out_channels': {out_channels}, 'in_channels': {in_channels}.")
        self.weight = Parameter(initializer(weight_init, [out_channels, in_channels]), name="weight")

        self.bias = None
        if self.has_bias:
            if isinstance(bias_init, Tensor):
                if bias_init.ndim != 1 or bias_init.shape[0] != out_channels:
                    raise ValueError(f"For '{self.cls_name}', bias init shape error. The ndim of 'bias_init' must "
                                     f"be equal to 1, and the first dim must be equal to 'out_channels'. But got "
                                     f"'bias_init': {bias_init}, 'out_channels': {out_channels}.")
            self.bias = Parameter(initializer(bias_init, [out_channels]), name="bias")
            self.bias_add = P.BiasAdd()

        self.matmul = P.MatMul(transpose_b=True)
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (nn.Cell, Primitive)):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, but got "
                            f"{type(activation).__name__}.")
        self.activation_flag = self.activation is not None

        self.cast = ms.ops.Cast()
        self.get_dtype = ms.ops.DType()

    def construct(self, x):
        """dense construction"""
        x = self.cast(x, ms.float16)

        x_shape = self.shape_op(x)
        if len(x_shape) != 2:
            x = self.reshape(x, (-1, x_shape[-1]))
        x = self.matmul(x, self.cast(self.weight, x.dtype))
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, x.dtype))
        if self.activation_flag:
            x = self.activation(x)
        if len(x_shape) != 2:
            out_shape = x_shape[:-1] + (-1,)
            x = self.reshape(x, out_shape)

        x = self.cast(x, ms.float32)
        return x
