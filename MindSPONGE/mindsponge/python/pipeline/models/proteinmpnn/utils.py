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
"""utils"""
import os
import pickle
import random
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class ProcessLinspace(nn.Cell):
    def __init__(self):
        super(ProcessLinspace, self).__init__()
        self.linspace = ops.LinSpace()

    def construct(self, d_min, d_max, d_count):
        output = self.linspace(d_min, d_max, d_count)
        return output


def scores_(s, log_probs, mask):
    """ Negative log probabilities """
    criterion = ops.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.view((-1, log_probs.shape[-1])).astype(ms.float32),
        s.view(-1).astype(ms.int32), ms.Tensor(np.ones((log_probs.shape[-1],)), ms.float32)
    )[0].view(s.shape)
    scores = ops.ReduceSum()(loss * mask, axis=-1) / ops.ReduceSum()(mask, axis=-1)
    return scores


def s_to_seq(s, mask):
    """s_to_seq"""
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    seq = ''.join([alphabet[c] for c, m in zip(s.asnumpy().tolist(), mask.asnumpy().tolist()) if m > 0])
    return seq


def ifreduce(args_if):
    """ifreduce"""
    v, one_list, start_idx, tied_beta = args_if
    if isinstance(v[0], list):
        for v_count in range(len(v[0])):
            one_list.append(start_idx + v[0][v_count] - 1)  # make 0 to be the first
            tied_beta[start_idx + v[0][v_count] - 1] = v[1][v_count]
    else:
        for v_ in v:
            one_list.append(start_idx + v_ - 1)  # make 0 to be the first
    ifresult = (v, one_list, start_idx, tied_beta)
    return ifresult


def tied_featurize(batch, chain_dict, fixed_position_dict=None, omit_aa_dict=None, tied_positions_dict=None,
                   pssm_dict=None, bias_by_res_dict=None):
    """ Pack and pad batch into tensors """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    b_ = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32)  # sum of chain seq lengths
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
    pssm_coef_all = ms.Tensor.from_numpy(pssm_coef_all).astype(ms.float32)
    pssm_bias_all = ms.Tensor.from_numpy(pssm_bias_all).astype(ms.float32)
    pssm_log_odds_all = ms.Tensor.from_numpy(pssm_log_odds_all).astype(ms.float32)

    tied_beta = ms.Tensor.from_numpy(tied_beta).astype(ms.float32)

    jumps = ((residue_idx[:, 1:] - residue_idx[:, :-1]) == 1).astype(np.float32)
    bias_by_res_all = ms.Tensor.from_numpy(bias_by_res_all).astype(ms.float32)
    phi_mask = np.pad(jumps, [[0, 0], [1, 0]])
    psi_mask = np.pad(jumps, [[0, 0], [0, 1]])
    omega_mask = np.pad(jumps, [[0, 0], [0, 1]])
    dihedral_mask = np.concatenate([phi_mask[:, :, None], psi_mask[:, :, None], omega_mask[:, :, None]], -1)  # [B,L,3]
    dihedral_mask = ms.Tensor.from_numpy(dihedral_mask).astype(ms.float32)
    residue_idx = ms.Tensor.from_numpy(residue_idx).astype(ms.int64)
    s_ = ms.Tensor.from_numpy(s_).astype(ms.int64)
    x_ = ms.Tensor.from_numpy(x_).astype(ms.float32)
    mask = ms.Tensor.from_numpy(mask).astype(ms.float32)
    chain_m = ms.Tensor.from_numpy(chain_m).astype(ms.float32)
    chain_m_pos = ms.Tensor.from_numpy(chain_m_pos).astype(ms.float32)
    omit_aa_mask = ms.Tensor.from_numpy(omit_aa_mask).astype(ms.float32)
    chain_encoding_all = ms.Tensor.from_numpy(chain_encoding_all).astype(ms.int64)
    result = (x_, s_, mask, lengths, chain_m, chain_encoding_all, letter_list_list,
              visible_list_list, masked_list_list, masked_chain_length_list_list,
              chain_m_pos, omit_aa_mask, residue_idx, dihedral_mask,
              tied_pos_list_of_lists_list, pssm_coef_all, pssm_bias_all,
              pssm_log_odds_all, bias_by_res_all, tied_beta)
    return result


def loss_nll(s, log_probs, mask):
    """ Negative log probabilities """
    criterion = ops.NLLLoss(reduction='none')
    log_probs_ = log_probs.view(-1, log_probs.shape[-1])
    s_ = ops.Cast()(s.view(-1), ms.int32)
    weight = ms.Tensor(np.ones((log_probs_.shape[1],), dtype='float32'))
    loss, _ = criterion(log_probs_, s_, weight)
    loss = loss.view(s.shape)
    s_argmaxed = ms.ops.Argmax(-1)(log_probs)  # [B, L]
    true_false = ops.Cast()(s == s_argmaxed, ms.float32)
    loss_av = ops.ReduceSum()(loss * mask) / ops.ReduceSum()(mask)
    return loss, loss_av, true_false


class LossSmoothed(nn.Cell):
    """loss_smoothed """

    def __init__(self, weight=0.1):
        """初始化"""
        super(LossSmoothed, self).__init__()
        self.weight = weight

    def construct(self, s, log_probs, mask):
        """ Negative log probabilities """
        s_onehot = ops.Cast()(nn.OneHot(depth=21)(s), ms.float32)

        # Label smoothing
        s_onehot = s_onehot + self.weight / float(s_onehot.shape[-1])
        s_onehot = s_onehot / ops.ReduceSum(keep_dims=True)(s_onehot, -1)

        loss = -(s_onehot * log_probs).sum(-1)
        loss_av = ops.ReduceSum()(loss * mask) / 2000.0
        return loss_av


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer, step):
        """init"""
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
               (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        "zero_grad"
        self.optimizer.zero_grad()


def get_pdbs(data_loader, repeat=1, max_length=10000, num_units=1000000):
    """get_pdbs"""
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet
    c1 = 0
    pdb_dict_list = []
    for _ in range(repeat):
        for t in data_loader:
            t['seq'] = t['seq'].asnumpy().tolist()
            t['label'] = t['label'].asnumpy().tolist()
            t = {k: v[0] for k, v in t.items()}
            c1 += 1
            pdb_dict_list = list_(t, chain_alphabet, max_length, num_units, pdb_dict_list)
    return pdb_dict_list


def list_(t, chain_alphabet, max_length, num_units, pdb_dict_list):
    """list_"""
    if 'label' in list(t):
        my_dict = {}
        concat_seq = ''
        mask_list = []
        visible_list = []
        if len(list(np.unique(t['idx']))) < 352:
            for idx in list(np.unique(t['idx'])):
                letter = chain_alphabet[idx]
                res = ms.Tensor(np.argwhere(t['idx'].asnumpy() == idx.asnumpy()).reshape(1, -1))
                initial_sequence = "".join(list(np.array(list(t['seq']))[res.asnumpy()][0,]))
                if initial_sequence[-6:] == "HHHHHH":
                    res = res[:, :-6]
                if initial_sequence[0:6] == "HHHHHH":
                    res = res[:, 6:]
                if initial_sequence[-7:-1] == "HHHHHH":
                    res = res[:, :-7]
                if initial_sequence[-8:-2] == "HHHHHH":
                    res = res[:, :-8]
                if initial_sequence[-9:-3] == "HHHHHH":
                    res = res[:, :-9]
                if initial_sequence[-10:-4] == "HHHHHH":
                    res = res[:, :-10]
                if initial_sequence[1:7] == "HHHHHH":
                    res = res[:, 7:]
                if initial_sequence[2:8] == "HHHHHH":
                    res = res[:, 8:]
                if initial_sequence[3:9] == "HHHHHH":
                    res = res[:, 9:]
                if initial_sequence[4:10] == "HHHHHH":
                    res = res[:, 10:]
                if res.shape[1] < 4:
                    pass
                else:
                    my_dict['seq_chain_' + letter] = "".join(list(np.array(list(t['seq']))[res.asnumpy()][0,]))
                    try:
                        concat_seq += my_dict['seq_chain_' + letter]
                    except KeyError:
                        pass
                    if idx in t['masked']:
                        mask_list.append(letter)
                    else:
                        visible_list.append(letter)
                    coords_dict_chain = {}
                    all_atoms = np.array(t['xyz'][res,])[0,]  # [L, 14, 3]
                    coords_dict_chain['N_chain_' + letter] = all_atoms[:, 0, :].tolist()
                    coords_dict_chain = coords(coords_dict_chain, letter, all_atoms)
                    my_dict['coords_chain_' + letter] = coords_dict_chain
                    my_dict['name'] = t['label']
                    my_dict['masked_list'] = mask_list
                    my_dict['visible_list'] = visible_list
                    my_dict['num_of_chains'] = len(mask_list) + len(visible_list)
                    my_dict['seq'] = concat_seq
                    if len(concat_seq) <= max_length:
                        pdb_dict_list.append(my_dict)
                    if len(pdb_dict_list) >= num_units:
                        break
    return pdb_dict_list


def coords(coords_dict_chain, letter, all_atoms):
    """coords"""
    try:
        for i, item in enumerate(coords_dict_chain['N_chain_' + letter]):
            for j, m in enumerate(item):
                try:
                    coords_dict_chain['N_chain_' + letter][i][j] = m.asnumpy()
                except KeyError:
                    pass
        coords_dict_chain['CA_chain_' + letter] = all_atoms[:, 1, :].tolist()
        for i, item in enumerate(coords_dict_chain['CA_chain_' + letter]):
            for j, m in enumerate(item):
                try:
                    coords_dict_chain['CA_chain_' + letter][i][j] = m.asnumpy()
                except KeyError:
                    pass
        coords_dict_chain['C_chain_' + letter] = all_atoms[:, 2, :].tolist()
        for i, item in enumerate(coords_dict_chain['C_chain_' + letter]):
            for j, m in enumerate(item):
                try:
                    coords_dict_chain['C_chain_' + letter][i][j] = m.asnumpy()
                except KeyError:
                    pass
        coords_dict_chain['O_chain_' + letter] = all_atoms[:, 3, :].tolist()
        for i, item in enumerate(coords_dict_chain['O_chain_' + letter]):
            for j, m in enumerate(item):
                try:
                    coords_dict_chain['O_chain_' + letter][i][j] = m.asnumpy()
                except KeyError:
                    pass
    except KeyError:
        pass
    return coords_dict_chain


class PdbDateset:
    """PdbDateset"""

    def __init__(self, ids, loader, train_dict, params):
        """init"""
        self.ids = ids
        self.train_dict = train_dict
        self.loader = loader
        self.params = params
        self.data = []
        for i in range(len(self.train_dict)):
            id_ = self.ids[i]
            sel_idx = np.random.randint(0, len(self.train_dict[id_]))
            out = self.loader(self.train_dict[id_][sel_idx], self.params)
            if len(out) > 1:
                self.data.append(out)
            else:
                continue

    def __getitem__(self, index):
        """__getitem__"""
        out = self.data[index]
        output = (out["seq"], out['xyz'], out['idx'], out['masked'], out['label'])
        return output

    def __len__(self):
        """__len__"""
        return len(self.data)


def loader_pdb(item, params):
    """loader_pdb"""
    pdbid, chid = item[0].split('_')
    prefix = "%s/pdb/%s/%s" % (params['DIR'], pdbid[1:3], pdbid)
    prefix_pkl = "%s/pdb_pkl/%s/%s" % (params['DIR'], pdbid[1:3], pdbid)
    # load metadata
    if not os.path.isfile(prefix + ".pt"):
        return {'seq': np.zeros(5)}

    with open(prefix_pkl + '.pkl', 'rb') as f_read:
        meta = pickle.load(f_read)

    for k, v in meta.items():
        if isinstance(meta[k], np.ndarray):
            meta[k] = ms.Tensor(meta[k])
    asmb_ids = meta['asmb_ids']
    asmb_chains = meta['asmb_chains']
    chids = np.array(meta['chains'])

    # find candidate assemblies which contain chid chain
    asmb_candidates = {[a for a, b in zip(asmb_ids, asmb_chains)
                        if chid in b.split(',')]}

    # if the chains is missing is missing from all the assemblies
    # then return this chain alone
    if not asmb_candidates:
        with open("%s_%s.pkl" % (prefix_pkl, chid), 'rb') as f_read:
            chain = pickle.load(f_read)
            for k, v in chain.items():
                try:
                    if isinstance(chain[k], np.ndarray):
                        try:
                            chain[k] = ms.Tensor(chain[k])
                        except KeyError:
                            pass
                except KeyError:
                    pass
        l = len(chain['seq'])
        return {'seq': chain['seq'],
                'xyz': chain['xyz'],
                'idx': ms.ops.zeros(l).int(),
                'masked': ms.Tensor([0]).int(),
                'label': item[0]}

    # randomly pick one assembly from candidates
    asmb_i = random.sample(list(asmb_candidates), 1)

    # indices of selected transforms
    idx = np.where(np.array(asmb_ids) == asmb_i)[0]

    # load relevant chains
    chains = chains_(idx, asmb_chains, meta, prefix_pkl)
    # generate assembly
    asmb = {}
    asmb = asmb_(idx, meta, asmb_chains, asmb, chains)
    # select chains which share considerable similarity to chid
    seqid = meta['tm'][ms.Tensor(np.where(chids == chid))[0]][0, :, 1]
    homo = {ch_j for seqid_j, ch_j in zip(seqid, chids)
            if seqid_j > params['HOMO']}
    # stack all chains in the assembly together
    seq, xyz, idx, masked = "", [], [], []
    seq_list = []
    for counter, (k, v) in enumerate(asmb.items()):
        try:
            seq += chains[k[0]]['seq']
            seq_list.append(chains[k[0]]['seq'])
        except KeyError:
            pass
        xyz.append(v)
        idx.append(ms.numpy.full((v.shape[0],), counter))
        if k[0] in homo:
            masked.append(counter)

    return {'seq': seq,
            'xyz': ms.ops.Concat(axis=0)(xyz).asnumpy(),
            'idx': ms.ops.Concat(axis=0)(idx).asnumpy(),
            'masked': ms.Tensor(masked, ms.int32).asnumpy(),
            'label': item[0]}


def chains_(idx, asmb_chains, meta, prefix_pkl):
    """chains"""
    chains = {}
    for i in idx:
        for c in asmb_chains[i]:
            if c in meta['chains']:
                with open("%s_%s.pkl" % (prefix_pkl, c), 'rb') as f_read:
                    chains[c] = pickle.load(f_read)

                for k, _ in chains.get(c).items():
                    if isinstance(chains.get(c).get(k), np.ndarray):
                        chains.get(c)[k] = ms.Tensor(chains.get(c).get(k))
    return chains


def asmb_(idx, meta, asmb_chains, asmb, chains):
    """asmb"""
    for k in idx:
        # pick k-th xform
        xform = meta['asmb_xform%d' % k]
        u = xform[:, :3, :3]
        r = xform[:, :3, 3]

        # select chains which k-th xform should be applied to
        s1 = set(meta['chains'])
        s2 = set(asmb_chains[k].split(','))
        chains_k = s1 & s2

        # transform selected chains
        for c in chains_k:
            try:
                xyz = chains[c]['xyz']
                xyz_ru = ops.Einsum('bij,raj->brai')((u, xyz)) + r[:, None, None, :]
                asmb.update({(c, k, i): xyz_i for i, xyz_i in enumerate(xyz_ru)})
            except KeyError:
                print('Error!')
                return {'seq': ms.Tensor(np.zeros(5))}
    return asmb


def featurize(batch):
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
    name = x, s, mask, chain_m, residue_idx, chain_encoding_all
    return name


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


class LRLIST:
    """LRLIST"""

    def __init__(self, model_size, factor, warmup):
        """init"""
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def cal_lr(self, total_step):
        """cal_lr"""
        lr = []
        for i in range(total_step):
            step = i
            if i == 0:
                lr.append(0.)
                # continue
            else:
                lr.append(self.factor * (self.model_size ** (-0.5) *
                                         min(step ** (-0.5), step * self.warmup ** (-1.5))))
        lr = np.array(lr).astype(np.float32)
        return lr
