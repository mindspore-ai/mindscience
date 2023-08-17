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
"proteinmpnndata"
import random
import numpy as np

import mindspore as ms
import mindspore.ops as ops
from ...dataset import curry1


@curry1
def pre_process(feature=None):
    "pre_process"
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX-'
    alphabet_set = {a for a in alphabet}
    discard_count = {
        'bad_chars': 0,
        'too_long': 0,
        'bad_seq_length': 0
    }

    data = []

    pdb_dict_list = feature
    for _, entry in enumerate(pdb_dict_list):
        seq = entry['seq']

        # Check if in alphabet
        bad_chars = {s for s in seq}.difference(alphabet_set)
        if not bad_chars:
            if len(entry['seq']) <= 1000:
                data.append(entry)
            else:
                discard_count['too_long'] += 1
        else:
            discard_count['bad_chars'] += 1
    return data


@curry1
def tied_featurize(batch=None, chain_dict=None, fixed_position_dict=None, omit_aa_dict=None, tied_positions_dict=None,
                   pssm_dict=None, bias_by_res_dict=None):
    """ Pack and pad batch into tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    b_ = len(batch)
    l_max = max([len(b['seq']) for b in batch])
    x_ = np.zeros([b_, l_max, 4, 3])
    residue_idx = -100 * np.ones([b_, l_max], dtype=np.int32)
    chain_m = np.zeros([b_, l_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    pssm_coef_all = np.zeros([b_, l_max], dtype=np.float32)  # 1.0 for the bits that need to be predicted
    pssm_bias_all = np.zeros([b_, l_max, 21], dtype=np.float32)  # 1.0 for the bits that need to be predicted
    pssm_log_odds_all = 10000.0 * np.ones([b_, l_max, 21],
                                          dtype=np.float32)  # 1.0 for the bits that need to be predicted
    chain_m_pos = np.zeros([b_, l_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    bias_by_res_all = np.zeros([b_, l_max, 21], dtype=np.float32)
    chain_encoding_all = np.zeros([b_, l_max], dtype=np.int32)  # 1.0 for the bits that need to be predicted
    s_ = np.zeros([b_, l_max], dtype=np.int32)
    omit_aa_mask = np.zeros([b_, l_max, len(alphabet)], dtype=np.int32)
    # Build the batch
    letter_list_list = []
    visible_list_list = []
    masked_list_list = []
    masked_chain_length_list_list = []
    tied_pos_list_of_lists_list = []
    # shuffle all chains before the main loop
    for i, b in enumerate(batch):
        if chain_dict is not None:
            masked_chains, visible_chains = chain_dict[
                b['name']]  # masked_chains a list of chain letters to predict [A, D, F]
        else:
            masked_chains = [item[-1:] for item in list(b) if item[:10] == 'seq_chain_']
            visible_chains = []
        all_chains = masked_chains + visible_chains

    for i, b in enumerate(batch):
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        letter_list = []
        global_idx_start_list = [0]
        visible_list = []
        masked_list = []
        masked_chain_length_list = []
        fixed_position_mask_list = []
        omit_aa_mask_list = []
        pssm_coef_list = []
        pssm_bias_list = []
        pssm_log_odds_list = []
        bias_by_res_list = []
        l0 = 0
        l1 = 0
        for _, letter in enumerate(all_chains):
            if letter in visible_chains:
                letter_list.append(letter)
                visible_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in
                                    [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                     f'O_chain_{letter}']], 1)
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                fixed_position_mask_list.append(fixed_position_mask)
                omit_aa_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                omit_aa_mask_list.append(omit_aa_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                bias_by_res_list.append(np.zeros([chain_length, 21]))
            if letter in masked_chains:
                masked_list.append(letter)
                letter_list.append(letter)
                chain_seq = b[f'seq_chain_{letter}']
                chain_seq = ''.join([a if a != '-' else 'X' for a in chain_seq])
                chain_length = len(chain_seq)
                global_idx_start_list.append(global_idx_start_list[-1] + chain_length)
                masked_chain_length_list.append(chain_length)
                chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 1.0 for masked
                x_chain = np.stack([chain_coords[c] for c in
                                    [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                     f'O_chain_{letter}']], 1)
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
                fixed_position_mask = np.ones(chain_length)
                if fixed_position_dict is not None:
                    fixed_pos_list = fixed_position_dict[b['name']][letter]
                    if fixed_pos_list:
                        fixed_position_mask[np.array(fixed_pos_list) - 1] = 0.0
                fixed_position_mask_list.append(fixed_position_mask)
                omit_aa_mask_temp = np.zeros([chain_length, len(alphabet)], np.int32)
                if omit_aa_dict is not None:
                    for item in omit_aa_dict[b['name']][letter]:
                        idx_aa = np.array(item[0]) - 1
                        aa_idx = np.array([np.argwhere(np.array(list(alphabet)) == aa)[0][0] for aa in item[1]]).repeat(
                            idx_aa.shape[0])
                        idx_ = np.array([[a, b] for a in idx_aa for b in aa_idx])
                        omit_aa_mask_temp[idx_[:, 0], idx_[:, 1]] = 1
                omit_aa_mask_list.append(omit_aa_mask_temp)
                pssm_coef = np.zeros(chain_length)
                pssm_bias = np.zeros([chain_length, 21])
                pssm_log_odds = 10000.0 * np.ones([chain_length, 21])
                if pssm_dict:
                    if pssm_dict[b['name']][letter]:
                        pssm_coef = pssm_dict[b['name']][letter]['pssm_coef']
                        pssm_bias = pssm_dict[b['name']][letter]['pssm_bias']
                        pssm_log_odds = pssm_dict[b['name']][letter]['pssm_log_odds']
                pssm_coef_list.append(pssm_coef)
                pssm_bias_list.append(pssm_bias)
                pssm_log_odds_list.append(pssm_log_odds)
                if bias_by_res_dict:
                    bias_by_res_list.append(bias_by_res_dict[b['name']][letter])
                else:
                    bias_by_res_list.append(np.zeros([chain_length, 21]))

        letter_list_np = np.array(letter_list)
        tied_pos_list_of_lists = []
        tied_beta = np.ones(l_max)
        if tied_positions_dict is not None:
            tied_pos_list = tied_positions_dict[b['name']]
            if tied_pos_list:
                for tied_item in tied_pos_list:
                    one_list = []
                    for k, v in tied_item.items():
                        start_idx = global_idx_start_list[np.argwhere(letter_list_np == k)[0][0]]
                        args_if = (v, one_list, start_idx, tied_beta)
                        v, one_list, start_idx, tied_beta = ifreduce(args_if)
                    tied_pos_list_of_lists.append(one_list)
        tied_pos_list_of_lists_list.append(tied_pos_list_of_lists)

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)
        m_pos = np.concatenate(fixed_position_mask_list, 0)  # [L,], 1.0 for places that need to be predicted

        pssm_coef_ = np.concatenate(pssm_coef_list, 0)  # [L,], 1.0 for places that need to be predicted
        pssm_bias_ = np.concatenate(pssm_bias_list, 0)  # [L,], 1.0 for places that need to be predicted
        pssm_log_odds_ = np.concatenate(pssm_log_odds_list, 0)  # [L,], 1.0 for places that need to be predicted

        bias_by_res_ = np.concatenate(bias_by_res_list,
                                      0)  # [L,21], 0.0 for places where aa frequencies don't need to be tweaked

        l = len(all_sequence)
        x_pad = np.pad(x, [[0, l_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
        x_[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, l_max - l]], 'constant', constant_values=(0.0,))
        m_pos_pad = np.pad(m_pos, [[0, l_max - l]], 'constant', constant_values=(0.0,))
        omit_aa_mask_pad = np.pad(np.concatenate(omit_aa_mask_list, 0), [[0, l_max - l]], 'constant',
                                  constant_values=(0.0,))
        chain_m[i, :] = m_pad
        chain_m_pos[i, :] = m_pos_pad
        omit_aa_mask[i,] = omit_aa_mask_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, l_max - l]], 'constant', constant_values=(0.0,))
        chain_encoding_all[i, :] = chain_encoding_pad

        pssm_coef_pad = np.pad(pssm_coef_, [[0, l_max - l]], 'constant', constant_values=(0.0,))
        pssm_bias_pad = np.pad(pssm_bias_, [[0, l_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        pssm_log_odds_pad = np.pad(pssm_log_odds_, [[0, l_max - l], [0, 0]], 'constant', constant_values=(0.0,))

        pssm_coef_all[i, :] = pssm_coef_pad
        pssm_bias_all[i, :] = pssm_bias_pad
        pssm_log_odds_all[i, :] = pssm_log_odds_pad

        bias_by_res_pad = np.pad(bias_by_res_, [[0, l_max - l], [0, 0]], 'constant', constant_values=(0.0,))
        bias_by_res_all[i, :] = bias_by_res_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        s_[i, :l] = indices
        letter_list_list.append(letter_list)
        visible_list_list.append(visible_list)
        masked_list_list.append(masked_list)
        masked_chain_length_list_list.append(masked_chain_length_list)

    isnan = np.isnan(x_)
    mask = np.isfinite(np.sum(x_, (2, 3))).astype(np.float32)
    x_[isnan] = 0.

    # Conversion
    pssm_log_odds_all = ms.Tensor.from_numpy(pssm_log_odds_all).astype(ms.float32)
    bias_by_res_all = ms.Tensor.from_numpy(bias_by_res_all).astype(ms.float32)
    residue_idx = ms.Tensor.from_numpy(residue_idx).astype(ms.int64)
    s = ms.Tensor.from_numpy(s_).astype(ms.int64)
    x = ms.Tensor.from_numpy(x_).astype(ms.float32)
    mask = ms.Tensor.from_numpy(mask).astype(ms.float32)
    chain_m = ms.Tensor.from_numpy(chain_m).astype(ms.float32)
    chain_m_pos = ms.Tensor.from_numpy(chain_m_pos).astype(ms.float32)
    omit_aa_mask = ms.Tensor.from_numpy(omit_aa_mask).astype(ms.float32)
    chain_encoding_all = ms.Tensor.from_numpy(chain_encoding_all).astype(ms.int64)

    pssm_log_odds_mask = (pssm_log_odds_all > 0.0).astype(ms.float32)
    randn_1 = ops.StandardNormal()(chain_m.shape).astype(ms.float32)
    omit_aas_list = 'X'
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'

    omit_aas_np = np.array([aa in omit_aas_list for aa in alphabet]).astype(np.float32)
    bias_aas_np = np.zeros(len(alphabet))

    result = [x, s, mask, chain_m, chain_m_pos, residue_idx, chain_encoding_all, randn_1, omit_aas_np, bias_aas_np,
              omit_aa_mask, pssm_coef, pssm_bias, masked_chain_length_list_list, masked_list_list, bias_by_res_all,
              pssm_log_odds_mask]
    return result


def all_chains_(all_chains, visible_chains, l0, l1, c, residue_idx, masked_chains, b, i):
    """all_chains"""
    x_chain_list = []
    chain_mask_list = []
    chain_seq_list = []
    chain_encoding_list = []
    for _, letter in enumerate(all_chains):
        if letter in visible_chains:
            chain_seq = b[f'seq_chain_{letter}']
            chain_length = len(chain_seq)
            chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
            chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
            x_chain = np.stack([chain_coords[c] for c in
                                [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                 f'O_chain_{letter}']], 1)  # [chain_length,4,3]
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
            l1 += chain_length
            residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c += 1
        elif letter in masked_chains:
            chain_seq = b[f'seq_chain_{letter}']
            chain_length = len(chain_seq)
            chain_coords = b[f'coords_chain_{letter}']  # this is a dictionary
            chain_mask = np.ones(chain_length)  # 0.0 for visible chains
            x_chain = np.stack([chain_coords[c] for c in
                                [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}',
                                 f'O_chain_{letter}']], 1)  # [chain_lenght,4,3]
            x_chain_list.append(x_chain)
            chain_mask_list.append(chain_mask)
            chain_seq_list.append(chain_seq)
            chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
            l1 += chain_length
            residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
            l0 += chain_length
            c += 1
            name__ = x_chain_list, chain_mask_list, chain_seq_list, chain_encoding_list
    return name__


def batch_(batch, l_max, residue_idx, chain_m, chain_encoding_all, x, s, alphabet):
    """batch"""
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for _, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                km = km
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)  # randomly shuffle chain order
        c = 1
        l0 = 0
        l1 = 0
        x_chain_list, chain_mask_list, chain_seq_list, chain_encoding_list = all_chains_(all_chains, \
                                                                                         visible_chains, l0, l1, c,
                                                                                         residue_idx, masked_chains, b,
                                                                                         i)
        x_ = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list, 0)  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)
        x_pad = np.pad(x_, [[0, l_max - l], [0, 0], [0, 0]], 'constant', constant_values=(np.nan,))
        x[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, l_max - l]], 'constant', constant_values=(0.0,))
        chain_m[i, :] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0, l_max - l]], 'constant', constant_values=(0.0,))
        chain_encoding_all[i, :] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        s[i, :l] = indices
        name_ = x, s, chain_m, chain_encoding_all
    return name_


@curry1
def featurize(batch=None):
    """featurize"""
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    b = len(batch)
    l_max = max([len(b['seq']) for b in batch])
    if l_max <= 128:
        l_max = 128
    elif l_max <= 256:
        l_max = 256
    elif l_max <= 512:
        l_max = 512
    else:
        l_max = (int(l_max / 512) + 1) * 512
    x = np.zeros([b, l_max, 4, 3])
    residue_idx = -100 * np.ones([b, l_max], dtype=np.int32)  # residue idx with jumps across chains
    chain_m = np.zeros([b, l_max],
                       dtype=np.int32)  # 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    chain_encoding_all = np.zeros([b, l_max],
                                  dtype=np.int32)  # integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    s = np.zeros([b, l_max], dtype=np.int32)  # sequence AAs integers
    x, s, chain_m, chain_encoding_all = batch_(batch, l_max, residue_idx, chain_m, chain_encoding_all, x, s, alphabet)
    isnan = np.isnan(x)
    mask = np.isfinite(np.sum(x, (2, 3))).astype(np.float32)
    x[isnan] = 0.

    # Conversion
    residue_idx = ms.Tensor.from_numpy(residue_idx).astype(ms.int64)
    s = ms.Tensor.from_numpy(s).astype(ms.int64)
    x = ms.Tensor.from_numpy(x).astype(ms.float32)
    mask = ms.Tensor.from_numpy(mask).astype(ms.float32)
    chain_m = ms.Tensor.from_numpy(chain_m).astype(ms.float32)
    chain_encoding_all = ms.Tensor.from_numpy(chain_encoding_all).astype(ms.int64)
    mask_for_loss = mask * chain_m
    name = [x, s, mask, chain_m, residue_idx, chain_encoding_all, mask_for_loss]
    return name
