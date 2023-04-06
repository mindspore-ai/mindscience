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
"""Some functions used in transformer network"""

from typing import Sequence, Tuple, List
import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.tensor import Tensor
from mindspore.common.initializer import initializer
from mindspore.ops import operations as P
from mindspore.ops.primitive import Primitive
from mindspore.common.parameter import Parameter
from mindspore import _checkparam as Validator
from mindspore.nn.layer.activation import get_activation
from src.data import BatchConverter


def ms_sum(x, dim, keep_dims=False):
    """Sum"""
    op = ms.ops.ReduceSum(keep_dims=keep_dims)
    return op(x, dim)


def ms_padding(x, num, val):
    """Padding"""
    num_shape = len(x.shape)
    if num == -3:
        a = np.ones(shape=x.shape[num + 1:]).astype(np.float32)
        a[:] = val
        x_pad = ms.Tensor(a)
    elif num == -1:
        x_pad = ms.Tensor(val)
    else:
        print("wrong with num, it should be -1 or -3")
    pad_tuple = list((0, 0) for i in range(num_shape))
    pad_tuple[num] = (1, 1)
    pad_op = ms.nn.Pad(paddings=tuple(pad_tuple))
    output = pad_op(x)
    if num == -3:
        output[..., 0, :, :] = x_pad
        output[..., -1, :, :] = x_pad
    elif num == -1:
        output[..., 0] = x_pad
        output[..., -1] = x_pad
    else:
        output[..., -1] = x_pad
    return output


def load_structure(fpath, chain=None):
    """Load structure"""
    if fpath.endswith('cif'):
        with open(fpath) as fin:
            pdbxf = pdbx.PDBxFile.read(fin)
        structure = pdbx.get_structure(pdbxf, model=1)
    elif fpath.endswith('pdb'):
        with open(fpath) as fin:
            pdbf = pdb.PDBFile.read(fin)
        structure = pdb.get_structure(pdbf, model=1)
    bbmask = filter_backbone(structure)
    structure = structure[bbmask]
    chains = get_chains(structure)
    print(f'Found {len(chains)} chains:', chains, '\n')
    if not list(chains):
        raise ValueError('No chains found in the input file.')
    if chain is None:
        chain = chains[0]
    if chain not in chains:
        raise ValueError(f'Chain {chain} not found in input file')
    structure = structure[structure.chain_id == chain]
    print(f'Loaded chain {chain}\n')
    return structure


def extract_coords_from_structure(structure: biotite.structure.AtomArray):
    """Extract coordinates from structure"""
    coords = get_atom_coords_residuewise(["N", "CA", "C"], structure)
    residue_identities = get_residues(structure)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])
    return coords, seq


def load_coords(fpath, chain):
    """Load coordinates"""
    structure = load_structure(fpath, chain)
    return extract_coords_from_structure(structure)


def get_atom_coords_residuewise(atoms: List[str], struct: biotite.structure.AtomArray):
    """
    Example for atoms argument: ["N", "CA", "C"]
    """

    def filterfn(s, axis=None):
        _ = axis
        filters = np.stack([s.atom_name == name for name in atoms], axis=1)
        filter_sum = filters.sum(0)
        if not np.all(filter_sum <= np.ones(filters.shape[1])):
            raise RuntimeError("structure has multiple atoms with same name")
        index = filters.argmax(0)
        coords = s[index].coord
        coords[filter_sum == 0] = float("nan")
        return coords

    return biotite.structure.apply_residue_wise(struct, struct, filterfn)


def score_sequence(model, alphabet, coords, seq):
    """Score sequences for given structure"""
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(coords, None, seq)]
    coords, confidence, _, tokens, padding_mask = batch_converter(batch)
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    target_padding_mask = (target == alphabet.padding_idx)
    model_input = (coords, padding_mask, confidence, prev_output_tokens)
    logits = model.construct(model_input)
    target = ms.ops.Cast()(target, ms.int32)
    loss = nn.CrossEntropyLoss(reduction='none')(logits, target)
    avgloss = ms_sum(loss * ~target_padding_mask, dim=-1) / ms_sum(ops.Cast()(~target_padding_mask, ms.float32), dim=-1)
    ll_fullseq = -avgloss.asnumpy().item()

    coord_bool = ms.ops.isfinite(coords)
    coord_mask = coord_bool.all(axis=-1).all(axis=-1)
    coord_mask = coord_mask[:, 1:-1]
    avgloss = ms_sum(loss * coord_mask, dim=-1) / ms_sum(ops.Cast()(coord_mask, ms.float32), dim=-1)
    ll_withcoord = -avgloss.asnumpy().item()

    return ll_fullseq, ll_withcoord


def get_encoder_output(model, alphabet, coords):
    """Get encoder output"""
    batch_converter = CoordBatchConverter(alphabet)
    # the batch_converter is essential for forming the correct input format
    batch = [(coords, None, None)]
    coords, confidence, _, _, padding_mask = batch_converter(batch)
    encoder_out = \
        model.encoder.construct(coords, padding_mask, confidence, return_all_hiddens=False)
    # remove beginning and end (bos and eos tokens)
    return encoder_out['encoder_out'][0][1:-1, 0]


def rotate(v, r):
    """Rotate"""
    unsqueeze = ms.ops.ExpandDims()
    r = unsqueeze(r, -3)
    v = unsqueeze(v, -1)
    return ms_sum(v * r, dim=-2)


def get_rotation_frames(coords):
    """Get rotation frames"""
    v1 = coords[:, :, 2] - coords[:, :, 1]
    v2 = coords[:, :, 0] - coords[:, :, 1]
    e1 = normalize(v1, dim=-1)
    u2 = v2 - e1 * ms_sum(e1 * v2, dim=-1, keep_dims=True)
    e2 = normalize(u2, dim=-1)
    e3 = ms.numpy.cross(e1, e2)
    stack = ms.ops.Stack(axis=-2)
    r = stack([e1, e2, e3])
    return r


def nan_to_num(ts, val=0.0):
    val = ms.Tensor(val, dtype=ts.dtype)
    return ms.numpy.where(~ms.ops.IsFinite()(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """Radial basis function"""
    linspace = ms.ops.LinSpace()
    v_min = ms.Tensor(v_min, ms.float32)
    v_max = ms.Tensor(v_max, ms.float32)
    rbf_centers = linspace(v_min, v_max, n_bins)
    rbf_centers = rbf_centers.view(tuple([1] * len(values.shape) + [-1]))
    rbf_std = (v_max - v_min) / n_bins
    expand_dims = ms.ops.ExpandDims()
    v_expand = expand_dims(values, -1)
    z = (v_expand - rbf_centers) / rbf_std
    exp = ms.ops.Exp()
    return exp(-z ** 2)


def norm(tensor, dim, eps=1e-8, keepdim=False):
    sqrt = ms.ops.Sqrt()
    square = ms.ops.Square()
    return sqrt(
        (ops.ReduceSum(keep_dims=keepdim)(square(tensor), axis=dim) + eps))


def normalize(tensor, dim=-1):
    """Normalization"""
    div = ms.ops.Div()
    y = norm(tensor, dim=dim, keepdim=True)
    return nan_to_num(
        div(tensor, y)
    )


class CoordBatchConverter(BatchConverter):
    """Batch conversion of coordinates"""

    def __call__(self, raw_batch: Sequence[Tuple[Sequence, str]], device=None):
        self.alphabet.cls_idx = self.alphabet.get_idx("<cath>")
        batch = []
        for coords, confidence, seq in raw_batch:
            if confidence is None:
                confidence = 1.
            if isinstance(confidence, (float, int)):
                confidence = [float(confidence)] * len(coords)
            if seq is None:
                seq = 'X' * len(coords)
            batch.append(((coords, confidence), seq))

        coords_and_confidence, strs, tokens = super().__call__(batch)

        # pad beginning and end of each protein due to legacy reasons
        coords = [
            ms_padding(ms.Tensor(cd), num=-3, val=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            ms_padding(ms.Tensor(cf), num=-1, val=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)

        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        is_nan = ms.ops.IsNan()
        padding_mask = is_nan(coords[:, :, 0, 0])
        coord_mask = ms.ops.IsFinite()(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        output = [coords, confidence, strs, tokens, padding_mask]

        return output

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """Collate dense tensors"""

        if not samples:
            return ms.Tensor()
        if len(set(x.dim() for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]

        result = ops.Zeros()((len(samples), *max_shape), ms.float32)

        for i, x_sample in enumerate(samples):
            len_sample = x_sample.shape[0]
            shape1 = ops.Zeros()(result[i].shape, ms.int32)
            shape2 = ops.Ones()(x_sample.shape, ms.int32)
            shape1[:len_sample] += shape2
            shape1 = ms.ops.Cast()(shape1, ms.bool_)
            result[i] = ms.ops.masked_fill(result[i], ~shape1, pad_v)
            result[i][:len_sample] = x_sample

        return result

    def from_lists(self, coords_list, confidence_list=None, seq_list=None, device=None):
        batch_size = len(coords_list)
        if confidence_list is None:
            confidence_list = [None] * batch_size
        if seq_list is None:
            seq_list = [None] * batch_size
        raw_batch = zip(coords_list, confidence_list, seq_list)
        return self.__call__(raw_batch, device)


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

        self.cast = ops.Cast()
        self.get_dtype = ops.DType()

    def construct(self, x):
        """Dense construction"""
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
