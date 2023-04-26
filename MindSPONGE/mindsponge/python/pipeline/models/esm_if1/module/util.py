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

import itertools
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
from mindspore import Tensor


proteinseq_toks = {
    'toks': ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N',
             'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-']
}


def ms_transpose(x, index_a, index_b):
    """Transpose"""
    index = list(i for i in range(len(x.shape)))
    index[index_a] = index_b
    index[index_b] = index_a
    input_trans = x.transpose(index)
    return input_trans


def ms_sum(x, dim, keep_dims=False):
    """Sum"""
    op = ms.ops.ReduceSum(keep_dims=keep_dims)
    return op(x, dim)


def ms_padding_without_val(x, padding):
    """Padding"""
    paddings = ()
    num = int(len(padding) / 2)
    zero_pad = len(x.shape) - num
    i = int(0)
    while i < zero_pad:
        i += 1
        paddings = paddings + ((0, 0),)
    for j in range(num):
        paddings = paddings + ((padding[(-2) * j - 2], padding[(-2) * j - 1]),)
    y = ms.nn.Pad(paddings=paddings)(x)
    return y


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
    coords = Tensor(coords)
    padding_mask = Tensor(padding_mask)
    confidence = Tensor(confidence)
    prev_output_tokens = Tensor(prev_output_tokens)
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
    coords = Tensor(coords)
    confidence = Tensor(confidence)
    padding_mask = Tensor(padding_mask)
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


def utils_softmax(x, dim: int, onnx_trace: bool = False):
    """Utils softmax"""
    if onnx_trace:
        return ops.Softmax(axis=dim)(ops.Cast()(x, ms.float32))
    x = x.astype(ms.float32)
    return ops.Softmax(axis=dim)(x)


def tuple_size(tp):
    """Return tuple size"""
    return tuple([0 if a is None else a.size() for a in tp])


def tuple_sum(tp1, tp2):
    """Return the sum of tuple"""
    s1, v1 = tp1
    s2, v2 = tp2
    if v2 is None and v2 is None:
        return (s1 + s2, None)
    return (s1 + s2, v1 + v2)


def tuple_cat(*args, dim=-1):
    """Return the concat of tuple"""
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    concat_op = ops.Concat(axis=dim)
    return concat_op(s_args), concat_op(v_args)


def tuple_index(x, idx):
    """Return the index of tuple"""
    return x[0][idx], x[1][idx]


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    square = ops.Square()
    ops_sum = ops.ReduceSum(keep_dims=keepdims)
    sqrt_1 = ops.Sqrt()
    out = ops_sum(square(x), axis) + eps
    return sqrt_1(out) if sqrt else out


def _split(x, nv):
    """Split"""
    reshape = ops.Reshape()
    v = reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    """Merge"""
    reshape = ops.Reshape()
    v = reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    concat_op = ops.Concat(axis=-1)
    a = concat_op((s, v))
    return a


def nan_to_num(ts, val=0.0):
    val = ms.Tensor(val, dtype=ts.dtype)
    return ms.numpy.where(~ms.ops.IsFinite()(ts), val, ts)


def rbf(values, v_min, v_max, n_bins=16):
    """Radial basis function"""
    rbf_centers = np.linspace(v_min, v_max, n_bins)
    rbf_centers = ms.Tensor(rbf_centers, ms.float32)
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


def ms_flatten(input_tensor, start_dim, end_dim):
    """Flatten"""
    if start_dim == 0:
        shape_list = list(input_tensor.shape[end_dim + 1:])
        dim = 1
        for i in range(start_dim, end_dim + 1):
            dim = input_tensor.shape[i] * dim
        shape_list.insert(0, dim)
        shape_list = tuple(shape_list)
        flatten = ms.ops.Reshape()
        output = flatten(input_tensor, shape_list)
        return output
    if end_dim in (-1, input_tensor.dim() - 1):
        shape_list = list(input_tensor.shape[:start_dim])
        dim = 1
        for i in range(start_dim, end_dim + 1):
            dim = input_tensor.shape[i] * dim
        shape_list.append(dim)
        shape_list = tuple(shape_list)
        flatten = ms.ops.Reshape()
        output = flatten(input_tensor, shape_list)
        return output
    raise ValueError("Unknown dim selected")


def flatten_graph(node_embeddings, edge_embeddings, edge_index):
    """Flatten graph"""
    x_s, x_v = node_embeddings
    e_s, e_v = edge_embeddings
    batch_size, n = x_s.shape[0], x_s.shape[1]
    node_embeddings = (x_s.reshape(((x_s.shape[0] * x_s.shape[1]), x_s.shape[2])),
                       x_v.reshape(((x_v.shape[0] * x_v.shape[1]), x_v.shape[2], x_v.shape[3])))
    edge_embeddings = (e_s.reshape(((e_s.shape[0] * e_s.shape[1]), e_s.shape[2])),
                       e_v.reshape(((e_v.shape[0] * e_v.shape[1]), e_v.shape[2], e_v.shape[3])))
    new_edge_index = ops.Cast()(edge_index != -1, ms.bool_)
    edge_mask = new_edge_index.any(axis=1)

    # Re-number the nodes by adding batch_idx * N to each batch
    unsqueeze = ops.ExpandDims()
    edge_index = edge_index + unsqueeze(unsqueeze((ms.numpy.arange(batch_size) * n), -1), -1)

    permute = ops.Transpose()

    edge_index = permute(edge_index, (1, 0, 2))
    edge_index = edge_index.reshape(edge_index.shape[0], (edge_index.shape[1] * edge_index.shape[2]))

    edge_mask = edge_mask.flatten()
    edge_mask = edge_mask.asnumpy()
    edge_index = edge_index.asnumpy()
    edge_embeddings_0 = edge_embeddings[0].asnumpy()
    edge_embeddings_1 = edge_embeddings[1].asnumpy()

    edge_index = edge_index[:, edge_mask]
    edge_embeddings = (
        ms.Tensor(edge_embeddings_0[edge_mask, :], ms.float32),
        ms.Tensor(edge_embeddings_1[edge_mask, :], ms.float32)
    )

    edge_index = ms.Tensor(edge_index, ms.int32)
    return node_embeddings, edge_embeddings, edge_index


def unflatten_graph(node_embeddings, batch_size):
    """Unflatten graph"""
    x_s, x_v = node_embeddings
    x_s = x_s.reshape((batch_size, -1, x_s.shape[1]))
    x_v = x_v.reshape((batch_size, -1, x_v.shape[1], x_v.shape[2]))
    return (x_s, x_v)


class Alphabet:
    """Create alphabet"""
    def __init__(
            self,
            standard_toks: Sequence[str],
            prepend_toks: Sequence[str] = ("<null_0>", "<pad>", "<eos>", "<unk>"),
            append_toks: Sequence[str] = ("<cls>", "<mask>", "<sep>"),
            prepend_bos: bool = True,
            append_eos: bool = False,
            use_msa: bool = False,
    ):
        self.standard_toks = list(standard_toks)
        self.prepend_toks = list(prepend_toks)
        self.append_toks = list(append_toks)
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.use_msa = use_msa

        self.all_toks = list(self.prepend_toks)
        self.all_toks.extend(self.standard_toks)
        for i in range((8 - (len(self.all_toks) % 8)) % 8):
            self.all_toks.append(f"<null_{i  + 1}>")
        self.all_toks.extend(self.append_toks)

        self.tok_to_idx = {tok: i for i, tok in enumerate(self.all_toks)}

        self.unk_idx = self.tok_to_idx["<unk>"]
        self.padding_idx = self.get_idx("<pad>")
        self.cls_idx = self.get_idx("<cls>")
        self.mask_idx = self.get_idx("<mask>")
        self.eos_idx = self.get_idx("<eos>")
        self.all_special_tokens = ['<eos>', '<unk>', '<pad>', '<cls>', '<mask>']
        self.unique_no_split_tokens = self.all_toks

    def __len__(self):
        return len(self.all_toks)

    @classmethod
    def from_architecture(cls, name: str) -> "Alphabet":
        """Return alphabet"""

        if "invariant_gvp" in name.lower():
            standard_toks = proteinseq_toks.get("toks", "abc")
            prepend_toks = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>", "<cath>", "<af2>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1", "protein_bert_base"):
            standard_toks = proteinseq_toks.get("toks")
            prepend_toks: Tuple[str, ...] = ("<null_0>", "<pad>", "<eos>", "<unk>")
            append_toks: Tuple[str, ...] = ("<cls>", "<mask>", "<sep>")
            prepend_bos = True
            append_eos = False
            use_msa = False
        elif name in ("ESM-1b", "roberta_large"):
            standard_toks = proteinseq_toks.get("toks")
            prepend_toks = ("<cls>", "<pad>", "<eos>", "<unk>")
            append_toks = ("<mask>",)
            prepend_bos = True
            append_eos = True
            use_msa = False
        else:
            raise ValueError("Unknown architecture selected")
        return cls(standard_toks, prepend_toks, append_toks, prepend_bos, append_eos, use_msa)

    @staticmethod
    def _tokenize(text) -> str:
        return text.split()

    def get_idx(self, tok):
        return self.tok_to_idx.get(tok, self.unk_idx)

    def get_tok(self, ind):
        return self.all_toks[ind]

    def get_batch_converter(self):
        return BatchConverter(self)

    def tokenize(self, text) -> List[str]:
        """Tokenization"""

        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()

                if i == 0 and not sub_text:
                    result.append(tok)
                elif i == len(split_text) - 1:
                    if sub_text:
                        result.append(sub_text)
                    else:
                        pass
                else:
                    if sub_text:
                        result.append(sub_text)
                    result.append(tok)
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text.extend(split_on_token(tok, sub_text))
                    else:
                        tokenized_text.append(sub_text)
                text_list = tokenized_text

            return list(
                itertools.chain.from_iterable(
                    (
                        self._tokenize(token)
                        if token not in self.unique_no_split_tokens
                        else [token]
                        for token in tokenized_text
                    )
                )
            )

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

    def to_dict(self):
        return self.tok_to_idx.copy()

    def encode(self, text):
        return [self.tok_to_idx[tok] for tok in self.tokenize(text)]


def np_padding(x, num, val):
    """Padding"""
    num_shape = len(x.shape)
    if num == -3:
        a = np.ones(shape=x.shape[num + 1:]).astype(np.float32)
        a[:] = val
        x_pad = a
    elif num == -1:
        x_pad = val
    else:
        print("wrong with num, it should be -1 or -3")
    pad_tuple = list((0, 0) for i in range(num_shape))
    pad_tuple[num] = (1, 1)
    output = np.pad(x, pad_tuple)
    if num == -3:
        output[..., 0, :, :] = x_pad
        output[..., -1, :, :] = x_pad
    elif num == -1:
        output[..., 0] = x_pad
        output[..., -1] = x_pad
    else:
        output[..., -1] = x_pad
    return output


class BatchConverter:
    """Batch conversion"""

    def __init__(self, alphabet):
        self.alphabet = alphabet

    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encoded_list = [self.alphabet.encode(seq_str) for seq_str in seq_str_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        tokens = np.ones((batch_size, max_len + int(self.alphabet.prepend_bos)
                          + int(self.alphabet.append_eos))).astype(np.float32) * self.alphabet.padding_idx

        labels = []
        strs = []

        for i, (label, seq_str, seq_encoded) in enumerate(
                zip(batch_labels, seq_str_list, seq_encoded_list)
        ):
            labels.append(label)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                tokens[i, 0] = self.alphabet.cls_idx
            seq = np.array(seq_encoded).astype(np.float32)
            tokens[
                i,
                int(self.alphabet.prepend_bos) : len(seq_encoded)
                + int(self.alphabet.prepend_bos),
            ] = seq
            if self.alphabet.append_eos:
                tokens[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

        return labels, strs, tokens


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
            np_padding(np.array(cd), num=-3, val=np.inf)
            for cd, _ in coords_and_confidence
        ]
        confidence = [
            np_padding(np.array(cf), num=-1, val=-1.)
            for _, cf in coords_and_confidence
        ]
        coords = self.collate_dense_tensors(coords, pad_v=np.nan)

        confidence = self.collate_dense_tensors(confidence, pad_v=-1.)
        padding_mask = np.isnan(coords[:, :, 0, 0])
        coord_mask = np.isfinite(coords.sum(-2).sum(-1))
        confidence = confidence * coord_mask + (-1.) * padding_mask
        output = [coords, confidence, strs, tokens, padding_mask]

        return output

    @staticmethod
    def collate_dense_tensors(samples, pad_v):
        """Collate dense tensors"""

        if not samples:
            return None
        if len(set(x.ndim for x in samples)) != 1:
            raise RuntimeError(
                f"Samples has varying dimensions: {[x.dim() for x in samples]}"
            )
        max_shape = [max(lst) for lst in zip(*[x.shape for x in samples])]

        result = np.zeros((len(samples), *max_shape), np.float32)

        for i, x_sample in enumerate(samples):
            len_sample = x_sample.shape[0]
            result[i][len_sample:] = pad_v
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
