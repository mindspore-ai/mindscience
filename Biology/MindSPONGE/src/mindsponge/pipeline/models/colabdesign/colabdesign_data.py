# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
"""colabdesign data"""
import numpy as np
import mindsponge.common.residue_constants as residue_constants
import mindsponge.common.protein as protein

from .module.utils import pdb_to_string, _np_get_cb
from ...dataset import curry1
from mindsponge.common import residue_constants


@curry1
def dict_filter_key(feature, feature_list):
    feature = {k: v for k, v in feature.items() if k in feature_list}
    return feature


@curry1
def dict_replace_key(feature, replaced_key):
    if len(replaced_key) != 2:
        raise ValueError("'replaced_key' should equal to 2")
    origin_key, new_key = replaced_key
    if origin_key in feature:
        feature[new_key] = feature.pop(origin_key)
    return feature


@curry1
def dict_cast(feature, cast_type, filtered_list):
    """dict cast"""
    if len(cast_type) != 2:
        raise ValueError("'cast_type' should equal to 2")
    origin_type = cast_type[0]
    new_type = cast_type[1]
    for k, v in feature.items():
        if k not in filtered_list:
            if v.dtype == origin_type:
                feature[k] = v.astype(new_type)
    return feature


@curry1
def dict_suqeeze(feature=None, filter_list=None, axis=None):
    for k in filter_list:
        if k in feature:
            feat_dim = feature[k].shape[axis]
            if isinstance(feat_dim, int) and feat_dim == 1:
                feature[k] = np.squeeze(feature[k], axis=axis)
    return feature


@curry1
def dict_take(feature, filter_list, axis):
    for k in filter_list:
        if k in feature:
            feature[k] = feature[k][axis]
    return feature


@curry1
def dict_del_key(feature, filter_list):
    for k in filter_list:
        if k in feature:
            del feature[k]
    return feature


@curry1
def one_hot_convert(feature, key, axis):
    if key in feature:
        feature[key] = np.argmax(feature[key], axis=axis)
    return feature


@curry1
def correct_restypes(feature, key):
    new_order_list = residue_constants.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = np.array(new_order_list, dtype=feature[key].dtype)
    feature[key] = new_order[feature[key]]
    return feature


NUM_RES = 'num residues placeholder'
NUM_MSA_SEQ = 'msa placeholder'
NUM_EXTRA_SEQ = 'extra msa placeholder'
NUM_TEMPLATES = 'num templates placeholder'


def prep_pos(pos, residue, chain):
    '''
    given input positions (a string of segment ranges separated by comma,
    for example: "1,3-4,10-15"), return list of indices to constrain.
    '''
    residue_set = []
    chain_set = []
    len_set = []
    for idx in pos.split(","):
        i, j = idx.split("-") if "-" in idx else (idx, None)

        # if chain defined
        if i[0].isalpha():
            c, i = i[0], int(i[1:])
        else:
            c, i = chain[0], int(i)
        if j is None:
            j = i
        else:
            j = int(j[1:] if j[0].isalpha() else j)
        residue_set += list(range(i, j + 1))
        chain_set += [c] * (j - i + 1)
        len_set += [j - i + 1]

    residue = np.asarray(residue)
    chain = np.asarray(chain)
    pos_set = []
    for i, c in zip(residue_set, chain_set):
        idx = np.where((residue == i) & (chain == c))[0]
        if len(idx) != 1:
            raise ValueError(f'ERROR: positions {i} and chain {c} not found')
        pos_set.append(idx[0])

    return {"residue": np.array(residue_set),
            "chain": np.array(chain_set),
            "length": np.array(len_set),
            "pos": np.asarray(pos_set)}


class DesignPrep:
    """DesignPrep"""""

    def __init__(self, cfg, num_seq=1):
        self.cfg_path = cfg
        self.seq_length = self.cfg_path.seq_length
        self.pad_length = self.cfg_path.seq_length
        self.ori_seq_len = self.cfg_path.seq_length
        self.msa_channel = self.cfg_path.model.msa_channel
        self.pair_channel = self.cfg_path.model.pair_channel
        self.extra_msa_num = self.cfg_path.data.max_extra_msa
        self.template_num = self.cfg_path.eval.max_templates
        self.msa_num = self.cfg_path.eval.max_msa_clusters - self.template_num

        self._num = num_seq

    def prep_feature(self, pdb_filename, chain, protocol, lengths=100, nums=1):
        """prep_feature"""""
        if protocol == 'fixbb':
            self._prep_fixbb(pdb_filename=pdb_filename, chain=chain)
            x = 0.01 * np.random.normal(0, 1, size=(self._num, self.ori_seq_len, 20))
            arrays, new_feature, ori_seq_len = self.transfer_input(x)
        elif protocol == 'hallucination':
            self._prep_haillucination(lengths, nums)
            x = 0.01 * np.random.normal(0, 1, size=(nums, lengths, 20))
            arrays, new_feature, ori_seq_len = self.transfer_input(x)
        return arrays, new_feature, ori_seq_len

    def prep_input_features(self, length, num=1, templates=1, enums=1):
        '''
        given [L]ength, [N]umber of sequences and number of Templates
        return dictionary of blank features
        '''
        aatype = np.zeros(length, int)
        setattr(self, "aatype", aatype)
        msa_feat = np.zeros((num, length, 49))
        setattr(self, "msa_feat", msa_feat)
        msa_mask = np.ones((num, length))
        setattr(self, "msa_mask", msa_mask)
        atom37_atom_exists = np.ones((length, 37))
        setattr(self, "atom37_atom_exists", atom37_atom_exists)
        residx_atom37_to_atom14 = np.zeros((length, 37), int)
        setattr(self, "residx_atom37_to_atom14", residx_atom37_to_atom14)
        residue_index = np.arange(length)
        setattr(self, "residue_index", residue_index)
        extra_deletion_value = np.zeros((enums, length))
        setattr(self, "extra_deletion_value", extra_deletion_value)
        extra_has_deletion = np.zeros((enums, length))
        setattr(self, "extra_has_deletion", extra_has_deletion)
        extra_msa = np.zeros((enums, length), int)
        setattr(self, "extra_msa", extra_msa)
        extra_msa_mask = np.zeros((enums, length))
        setattr(self, "extra_msa_mask", extra_msa_mask)

        # for template inputs
        template_aatype = np.zeros((templates, length), int)
        setattr(self, "template_aatype", template_aatype)
        template_all_atom_mask = np.zeros((templates, length, 37))
        setattr(self, "template_all_atom_masks", template_all_atom_mask)
        template_all_atom_positions = np.zeros((templates, length, 37, 3))
        setattr(self, "template_all_atom_positions", template_all_atom_positions)
        template_mask = np.zeros(templates)
        setattr(self, "template_mask", template_mask)
        template_pseudo_beta = np.zeros((templates, length, 3))
        setattr(self, "template_pseudo_beta", template_pseudo_beta)
        template_pseudo_beta_mask = np.zeros((templates, length))
        setattr(self, "template_pseudo_beta_mask", template_pseudo_beta_mask)

    def transfer_input(self, d_params=None):
        """transfer_input"""
        seq_length = self.seq_length
        msa_channel = self.msa_channel
        pair_channel = self.pair_channel

        extra_msa_num = self.extra_msa_num
        template_num = self.template_num
        msa_num = self.msa_num
        ori_seq_len = self.aatype.shape[0]
        pad_length = seq_length - ori_seq_len
        new_feature = {}

        if d_params is not None:
            new_feature['params_seq'] = np.array(d_params).astype(np.float32)
            new_feature['params_seq'] = np.pad(new_feature.get('params_seq'), ((0, 0), (0, pad_length), (0, 0)),
                                               constant_values=(0, 0))
        new_feature['target_feat'] = self.msa_feat[0, :, :21]
        new_feature['target_feat'] = np.pad(new_feature.get('target_feat'), [[0, 0], [1, 0]])
        new_feature['target_feat'] = np.pad(new_feature.get('target_feat'), [[0, pad_length], [0, 0]])
        new_feature['prev_pos'] = np.zeros((seq_length, 37, 3)).astype(np.float32)
        new_feature['prev_msa_first_row'] = np.zeros((seq_length, msa_channel)).astype(np.float32)
        new_feature['prev_pair'] = np.zeros((seq_length, seq_length, pair_channel)).astype(np.float32)
        ori_msa_feat = self.msa_feat.shape[0]
        new_feature['msa_feat'] = np.pad(self.msa_feat, ((0, msa_num - ori_msa_feat), (0, pad_length), (0, 0)),
                                         constant_values=(0, 0))

        new_feature['msa_mask'] = np.pad(self.msa_mask, ((0, msa_num - ori_msa_feat), (0, pad_length)),
                                         constant_values=(0, 0)).astype(np.float32)
        new_feature['seq_mask_batch'] = np.ones((ori_seq_len)).astype(np.float32)
        new_feature['seq_mask_batch'] = np.pad(new_feature.get('seq_mask_batch'), ((0, pad_length)),
                                               constant_values=(0, 0))
        new_feature['aatype_batch'] = np.pad(self.aatype, ((0, pad_length)), constant_values=(0, 0)).astype(np.int32)

        new_feature["template_aatype"] = self.template_aatype
        new_feature["template_all_atom_masks"] = self.template_all_atom_masks
        new_feature["template_all_atom_positions"] = self.template_all_atom_positions
        new_feature["template_mask"] = self.template_mask
        new_feature["template_pseudo_beta_mask"] = self.template_pseudo_beta_mask
        new_feature["template_pseudo_beta"] = self.template_pseudo_beta
        ori_template_num = self.template_aatype.shape[0]
        new_feature['template_aatype'] = np.pad(new_feature.get('template_aatype'),
                                                ((0, template_num - ori_template_num), (0, pad_length)),
                                                constant_values=(0, 0)).astype(np.int32)

        new_feature['template_all_atom_masks'] = np.pad(new_feature.get('template_all_atom_masks'),
                                                        ((0, template_num - ori_template_num), (0, pad_length), (0, 0)),
                                                        constant_values=(0, 0)).astype(np.float32)
        new_feature['template_mask'] = np.pad(new_feature.get('template_mask'), ((0, template_num - ori_template_num)),
                                              constant_values=(0)).astype(np.float32)
        new_feature["template_all_atom_positions"] = np.pad(new_feature.get("template_all_atom_positions"),
                                                            ((0, template_num - ori_template_num), (0, pad_length),
                                                             (0, 0),
                                                             (0, 0)),
                                                            constant_values=(0, 0)).astype(np.float32)
        new_feature['template_pseudo_beta'] = np.pad(new_feature.get("template_pseudo_beta"),
                                                     ((0, template_num - ori_template_num), (0, pad_length), (0, 0)),
                                                     constant_values=(0, 0)).astype(np.float32)

        new_feature['template_pseudo_beta_mask'] = np.pad(new_feature.get("template_pseudo_beta_mask"),
                                                          ((0, template_num - ori_template_num), (0, pad_length)),
                                                          constant_values=(0, 0)).astype(np.float32)
        ori_extra_msa_num = self.extra_msa.shape[0]
        new_feature['extra_msa'] = np.pad(self.extra_msa,
                                          ((0, extra_msa_num - ori_extra_msa_num), (0, pad_length)),
                                          constant_values=(0, 0)).astype(np.int32)
        new_feature['extra_has_deletion'] = np.pad(self.extra_has_deletion,
                                                   ((0, extra_msa_num - ori_extra_msa_num), (0, pad_length)),
                                                   constant_values=(0, 0)).astype(np.float32)
        new_feature['extra_deletion_value'] = np.pad(self.extra_deletion_value,
                                                     ((0, extra_msa_num - ori_extra_msa_num), (0, pad_length)),
                                                     constant_values=(0, 0)).astype(np.float32)
        new_feature['extra_msa_mask'] = np.pad(self.extra_msa_mask,
                                               ((0, extra_msa_num - ori_extra_msa_num), (0, pad_length)),
                                               constant_values=(0, 0)).astype(np.float32)
        new_feature['residx_atom37_to_atom14'] = np.pad(self.residx_atom37_to_atom14,
                                                        ((0, pad_length), (0, 0)),
                                                        constant_values=(0, 0)).astype(np.int32)
        new_feature['atom37_atom_exists_batch'] = np.pad(self.atom37_atom_exists, ((0, pad_length), (0, 0)),
                                                         constant_values=(0, 0)).astype(np.float32)
        new_feature['residue_index_batch'] = np.pad(self.residue_index, ((0, pad_length)),
                                                    constant_values=(0, 0)).astype(np.int32)

        if hasattr(self, 'batch_aatype'):
            new_feature["batch_aatype"] = np.pad(self.batch_aatype, ((0, pad_length)),
                                                 constant_values=(0, 0)).astype(np.float32)
            new_feature["batch_all_atom_mask"] = np.pad(self.batch_all_atom_mask, ((0, pad_length), (0, 0)),
                                                        constant_values=(0, 0)).astype(np.float32)
            new_feature["batch_all_atom_positions"] = np.pad(self.batch_all_atom_positions,
                                                             ((0, pad_length), (0, 0), (0, 0)),
                                                             constant_values=(0, 0)).astype(np.float32)
        else:
            new_feature["batch_aatype"] = np.ones(shape=(seq_length,)).astype(np.float32)
            new_feature["batch_all_atom_mask"] = np.ones(shape=(seq_length, 37)).astype(np.float32)
            new_feature["batch_all_atom_positions"] = np.ones(shape=(seq_length, 37, 3)).astype(np.float32)

        input_keys = ["msa_feat", "msa_mask", "seq_mask_batch", \
                      "template_aatype", "template_all_atom_masks", "template_all_atom_positions", "template_mask", \
                      "template_pseudo_beta_mask", "template_pseudo_beta", \
                      "extra_msa", "extra_has_deletion", "extra_deletion_value", "extra_msa_mask", \
                      "residx_atom37_to_atom14", "atom37_atom_exists_batch", \
                      "residue_index_batch", "batch_aatype", "batch_all_atom_positions", "batch_all_atom_mask"]
        arrays = [new_feature.get(key) for key in input_keys]
        return arrays, new_feature, ori_seq_len

    def _prep_features(self, num_res, num_seq=None, num_templates=1):
        '''process features'''
        if num_seq is None:
            num_seq = self._num
        self.prep_input_features(length=num_res, num=num_seq, templates=num_templates)

    def _prep_fixbb(self, pdb_filename, chain="A",
                    rm_template_seq=True, rm_template_sc=True, ignore_missing=True):
        """_prep_fixbb"""
        o = extract_pdb(pdb_filename, chain=chain, ignore_missing=ignore_missing)
        pdb_residue_index, pdb_idx, pdb_lengths, pdb_batch = \
            o.get("residue_index"), o.get("idx"), o.get("lengths"), o.get("batch")
        self.ori_seq_len = pdb_residue_index.shape[0]
        self.pad_length = self.seq_length - self.ori_seq_len
        # feat dims
        num_seq = self._num
        res_idx = pdb_residue_index

        # configure input features
        self._prep_features(num_res=sum(pdb_lengths), num_seq=num_seq)
        setattr(self, "residue_index", res_idx)
        batch_aatype, batch_all_atom_mask, batch_all_atom_positions = make_fixed_size(pdb_batch,
                                                                                      num_res=sum(pdb_lengths))
        setattr(self, "batch_aatype", batch_aatype)
        setattr(self, "batch_all_atom_mask", batch_all_atom_mask)
        setattr(self, "batch_all_atom_positions", batch_all_atom_positions)

        rm, leng = {}, sum(pdb_lengths)
        for n, x in [["rm_seq", rm_template_seq], ["rm_sc", rm_template_sc]]:
            rm[n] = np.full(leng, False)
            if isinstance(x, str):
                rm.get(n)[prep_pos(x, **pdb_idx).get("pos")] = True
            else:
                rm.get(n)[:] = x

    def _prep_haillucination(self, length=100, num=1):
        """_prep_haillucination"""
        self._prep_features(num_res=length, num_seq=num)
        setattr(self, "residue_index", np.arange(length))


def extract_pdb(pdb_filename, chain=None,
                offsets=None, lengths=None,
                ignore_missing=False):
    """extract_pdb"""

    def add_atom(batch):
        """add missing CB atoms based on N,CA,C"""
        atom_idx = residue_constants.atom_order
        p, m = batch.get("all_atom_positions"), batch.get("all_atom_mask")

        atoms = {k: p[..., atom_idx[k], :] for k in ["N", "CA", "C"]}
        cb = atom_idx["CB"]

        cb_mask = np.prod([m[..., atom_idx[k]] for k in ["N", "CA", "C"]], 0)
        cb_atoms = _np_get_cb(atoms.get("N"), atoms.get("CA"), atoms.get("C"))
        batch["all_atom_positions"][..., cb, :] = np.where(m[:, cb, None], p[:, cb, :], cb_atoms)
        batch["all_atom_mask"][..., cb] = (m[:, cb] + cb_mask) > 0
        return {"atoms": batch["all_atom_positions"][:, cb], "mask": cb_mask}

    # go through each defined chain
    chains = [None] if chain is None else chain.split(",")
    o, last = [], 0
    residue_idx, chain_idx = [], []
    full_lengths = []

    for n, simplechain in enumerate(chains):
        protein_obj = protein.from_pdb_string(pdb_to_string(pdb_filename), chain_id=simplechain)
        batch = {'aatype': protein_obj.aatype,
                 'all_atom_positions': protein_obj.atom_positions,
                 'all_atom_mask': protein_obj.atom_mask,
                 'residue_index': protein_obj.residue_index}

        cb_feat = add_atom(batch)

        im = ignore_missing[n] if isinstance(ignore_missing, list) else ignore_missing
        if im:
            replies = batch.get("all_atom_mask")[:, 0] == 1
            for key in batch:
                batch[key] = batch.get(key)[replies]
            residue_index = batch.get("residue_index") + last

        else:
            offset = 0 if offsets is None else (offsets[n] if isinstance(offsets, list) else offsets)
            replies = offset + (protein_obj.residue_index - protein_obj.residue_index.min())
            lengs = (replies.max() + 1) if lengths is None else (lengths[n] if isinstance(lengths, list) else lengths)

            def scatter(x, value=0, lens=0, re=0):
                shape = (lens,) + x.shape[1:]
                y = np.full(shape, value, dtype=x.dtype)
                y[re] = x
                return y

            batch = {"aatype": scatter(batch.get("aatype"), -1, lens=lengs, re=replies),
                     "all_atom_positions": scatter(batch.get("all_atom_positions"), lens=lengs, re=replies),
                     "all_atom_mask": scatter(batch.get("all_atom_mask"), lens=lengs, re=replies),
                     "residue_index": scatter(batch.get("residue_index"), -1, lens=lengs, re=replies)}

            residue_index = np.arange(lengs) + last

        last = residue_index[-1] + 50
        o.append({"batch": batch,
                  "residue_index": residue_index,
                  "cb_feat": cb_feat})

        residue_idx.append(batch.pop("residue_index"))
        chain_idx.append([chain] * len(residue_idx[-1]))
        full_lengths.append(len(residue_index))

    # concatenate chains
    o_inter = {}
    for i, feature in enumerate(o):
        for key in feature.keys():
            if i == 0:
                o_inter[key] = feature.get(key)
            else:
                o_inter[key] = np.concatenate((o_inter.get(key), feature.get(key)), 0)

    o = o_inter
    o["idx"] = {"residue": np.concatenate(residue_idx), "chain": np.concatenate(chain_idx)}
    o["lengths"] = full_lengths
    return o


def make_fixed_size(feat, num_res):
    """"make_fixed_size"""

    for k, v in feat.items():
        if k == "batch":
            feat[k] = make_fixed_size(v, num_res)
        else:
            continue

    return feat.get("aatype"), feat.get("all_atom_mask"), feat.get("all_atom_positions")


@curry1
def prep(feature=None, cfg=None):
    prev_pos = np.zeros((cfg.seq_length, 37, 3)).astype(np.float32)
    prev_msa_first_row = np.zeros((cfg.seq_length, cfg.model.msa_channel)).astype(np.float32)
    prev_pair = np.zeros((cfg.seq_length, cfg.seq_length, cfg.model.pair_channel)).astype(np.float32)
    feature.append(prev_pos)
    feature.append(prev_msa_first_row)
    feature.append(prev_pair)
    return feature


@curry1
def get_weights(feature=None, cfg=None):
    """get weights"""
    opt_temp = []
    opt_soft = []
    opt_hard = []

    for i in range(cfg.soft_iters):
        opt_temp.append(
            cfg.soft_etemp + (cfg.soft_temp - cfg.soft_etemp) * (1 - (i + 1) / cfg.soft_iters) ** 2)
        opt_soft.append((i + 1) / cfg.soft_iters)
        opt_hard.append(cfg.soft_hard)
    for i in range(cfg.temp_iters):
        opt_temp.append(
            cfg.temp_decay + (cfg.temp_value - cfg.temp_decay) * (1 - (i + 1) / cfg.temp_iters) ** 2)
        opt_soft.append(cfg.temp_esoft + (cfg.temp_soft - cfg.temp_esoft) * ((i + 1) / cfg.temp_iters))
        opt_hard.append(cfg.temp_ehard + (cfg.temp_hard - cfg.temp_ehard) * ((i + 1) / cfg.temp_iters))
    for i in range(cfg.hard_iters):
        opt_temp.append(
            cfg.hard_etemp + (cfg.hard_temp - cfg.hard_etemp) * (1 - (i + 1) / cfg.hard_iters) ** 2)
        opt_soft.append(cfg.hard_esoft + (cfg.hard_soft - cfg.hard_esoft) * ((i + 1) / cfg.hard_iters))
        opt_hard.append(cfg.hard_decay + (cfg.hard_value - cfg.hard_decay) * ((i + 1) / cfg.hard_iters))
    feature.append(opt_temp)
    feature.append(opt_soft)
    feature.append(opt_hard)
    return feature
