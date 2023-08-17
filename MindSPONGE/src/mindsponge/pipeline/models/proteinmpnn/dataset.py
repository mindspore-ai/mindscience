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
"""Datasets"""
import json
import time
import numpy as np


def parse_pdb_biounits(args):
    '''
    input:  x = PDB filename
            atoms = atoms to extract (optional)
    output: (length, atoms, coords=(x,y,z)), sequence
    '''
    x, atoms, chain = args
    alpha_1 = list("ARNDCQEGHILKMFPSTWYV-")
    alpha_3 = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
               'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'GAP']

    aa_3_n = {a: n for n, a in enumerate(alpha_3)}
    aa_n_1 = {n: a for n, a in enumerate(alpha_1)}

    def n_to_aa(x):
        x = np.array(x)
        if x.ndim == 1:
            x = x[None]
        return ["".join([aa_n_1.get(a, "-") for a in y]) for y in x]

    def ifresn(resn_arg):
        xyz, resn, atoms, xyz_ = resn_arg
        if resn in xyz:
            for k in sorted(xyz.get(resn)):
                for atom in atoms:
                    if atom in xyz.get(resn).get(k):
                        xyz_.append(xyz.get(resn).get(k).get(atom))
                    else:
                        xyz_.append(np.full(3, np.nan))
        else:
            for atom in atoms:
                xyz_.append(np.full(3, np.nan))
        return xyz_

    def ifline(args):
        line, chain, xyz, seq, min_resn, max_resn, resn = args
        if line[:6] == "HETATM" and line[17:17 + 3] == "MSE":
            line = line.replace("HETATM", "ATOM  ")
            line = line.replace("MSE", "MET")

        if line[:4] == "ATOM":
            ch = line[21:22]
            if ch == chain or chain is None:
                atom = line[12:12 + 4].strip()
                resi = line[17:17 + 3]
                resn = line[22:22 + 5].strip()
                x, y, z = [float(line[i:(i + 8)]) for i in [30, 38, 46]]

                if resn[-1].isalpha():
                    resa, resn = resn[-1], int(resn[:-1]) - 1
                else:
                    resa, resn = "", int(resn) - 1

                if resn < min_resn:
                    min_resn = resn
                if resn > max_resn:
                    max_resn = resn
                if resn not in xyz:
                    xyz[resn] = {}
                if resa not in xyz.get(resn):
                    xyz[resn][resa] = {}
                if resn not in seq:
                    seq[resn] = {}
                if resa not in seq.get(resn):
                    seq[resn][resa] = resi
                if atom not in xyz.get(resn).get(resa):
                    xyz[resn][resa][atom] = np.array([x, y, z])
        result = (resn, min_resn, max_resn, seq, xyz)
        return result

    xyz, seq, min_resn, max_resn = {}, {}, 1e6, -1e6
    for line in open(x, "rb"):
        line = line.decode("utf-8", "ignore").rstrip()
        args = (line, chain, xyz, seq, min_resn, max_resn, None)
        resn, min_resn, max_resn, seq, xyz = ifline(args)

    # convert to numpy arrays, fill in missing values
    seq_, xyz_ = [], []
    try:
        for resn in range(min_resn, max_resn + 1):
            if resn in seq:
                for k in sorted(seq.get(resn)):
                    seq_.append(aa_3_n.get(seq.get(resn).get(k), 20))
            else:
                seq_.append(20)
            ifresn_arg = (xyz, resn, atoms, xyz_)
            xyz_ = ifresn(ifresn_arg)
        return np.array(xyz_).reshape((-1, len(atoms), 3)), n_to_aa(np.array(seq_))
    except TypeError:
        return 'no_chain', 'no_chain'


def parse_pdb(path_to_pdb, input_chain_list=None):
    """parse_pdb"""
    c = 0
    pdb_dict_list = []
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                     'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                     'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_alphabet = init_alphabet + extra_alphabet

    if input_chain_list:
        chain_alphabet = input_chain_list

    biounit_names = [path_to_pdb]
    for biounit in biounit_names:
        my_dict = {}
        s = 0
        concat_seq = ''
        for letter in chain_alphabet:
            args = (biounit, ['N', 'CA', 'C', 'O'], letter)
            xyz, seq = parse_pdb_biounits(args)
            if not isinstance(xyz, str):
                concat_seq += seq[0]
                my_dict['seq_chain_' + letter] = seq[0]
                coords_dict_chain = {}
                coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
                my_dict['coords_chain_' + letter] = coords_dict_chain
                s += 1
        fi = biounit.rfind("/")
        my_dict['name'] = biounit[(fi + 1):-4]
        my_dict['num_of_chains'] = s
        my_dict['seq'] = concat_seq
        if s <= len(chain_alphabet):
            pdb_dict_list.append(my_dict)
            c += 1
    return pdb_dict_list


class StructureDataset():
    """StructureDataset"""

    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = {a for a in alphabet}
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        with open(jsonl_file) as f:
            self.data = []

            lines = f.readlines()
            start = time.time()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                name = entry['name']

                # Check if in alphabet
                bad_chars = {s for s in seq}.difference(alphabet_set)
                if not bad_chars:
                    if len(entry['seq']) <= max_length:
                        self.data.append(entry)
                    else:
                        discard_count['too_long'] += 1
                else:
                    print(name, bad_chars, entry['seq'])
                    discard_count['bad_chars'] += 1

                # Truncate early
                if truncate is not None and len(self.data) == truncate:
                    return

                if verbose and (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    print('{} entries ({} loaded) in {:.1f} s'.format(len(self.data), i + 1, elapsed))

            print('discarded', discard_count)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureDatasetPDB():
    """StructureDatasetPDB"""

    def __init__(self, pdb_dict_list, truncate=None, max_length=100,
                 alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = {a for a in alphabet}
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }

        self.data = []

        for _, entry in enumerate(pdb_dict_list):
            seq = entry['seq']

            bad_chars = {s for s in seq}.difference(alphabet_set)
            if not bad_chars:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class StructureLoader():
    """StructureLoader"""

    def __init__(self, dataset, batch_size=100):
        self.dataset = dataset  # 数据
        self.size = len(dataset)  # 有多少个序列
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
            else:
                clusters.append(batch)
                batch, _ = [], 0
        if batch:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def append(sorted_ix, lengths, define_batch):
    """append"""
    clusters, batch_128, batch_256, batch_512, batch_1024, batch_ = [], [], [], [], [], []
    for ix in sorted_ix:
        batch_128, batch_256, batch_512, batch_1024, batch_ = append_batch(ix, lengths, define_batch, batch_128, \
                                                                           batch_256, batch_512, batch_1024, batch_)
        clusters, batch_128, batch_256, batch_512, batch_1024, batch_ = append_cluster(ix, lengths, define_batch, \
                                                                                       clusters, batch_128, batch_256,
                                                                                       batch_512, batch_1024, batch_)
    clusters_ = clusters
    return clusters_


def append_batch(ix, lengths, define_batch, batch_128, batch_256, batch_512, batch_1024, batch_):
    """append_batch"""
    if lengths[ix] <= 128 and len(batch_128) < define_batch[0]:
        batch_128.append(ix)
    elif lengths[ix] <= 256 and lengths[ix] > 128 and len(batch_256) < define_batch[1]:
        batch_256.append(ix)
    elif lengths[ix] > 256 and lengths[ix] <= 512 and len(batch_512) < define_batch[2]:
        batch_512.append(ix)
    elif lengths[ix] > 1024 and len(batch_) < define_batch[4]:
        batch_.append(ix)
    output = (batch_128, batch_256, batch_512, batch_1024, batch_)
    return output


def append_cluster(ix, lengths, define_batch, clusters, batch_128, batch_256, batch_512, batch_1024, batch_):
    """append_cluster"""
    if lengths[ix] <= 128 and len(batch_128) == define_batch[0]:
        clusters.append(batch_128)
        batch_128 = []
    elif lengths[ix] <= 256 and lengths[ix] > 128 and len(batch_256) == define_batch[1]:
        clusters.append(batch_256)
        batch_256 = []
    elif lengths[ix] > 256 and lengths[ix] <= 512 and len(batch_512) == define_batch[2]:
        clusters.append(batch_512)
        batch_512 = []
    elif lengths[ix] > 1024 and len(batch_) == define_batch[4]:
        clusters.append(batch_)
        batch_ = []
    output = (clusters, batch_128, batch_256, batch_512, batch_1024, batch_)
    return output


class Definebatch():
    """StructureLoader"""

    def __init__(self, dataset, num_epoches, batch_size=100):
        self.dataset = dataset
        self.num_epoches = num_epoches
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['seq']) for i in range(self.size)]
        self.batch_size = batch_size
        self.define_batch = [10, 2, 2, 2, 2]
        sorted_ix = np.argsort(self.lengths)
        self.clusters = append(sorted_ix, self.lengths, self.define_batch)

    def __len__(self):
        return self.num_epoches

    def __iter__(self):
        np.random.seed(123)
        np.random.shuffle(self.clusters[:self.num_epoches])
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch
