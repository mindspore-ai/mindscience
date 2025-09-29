# Copyright 2022 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"""train dataset"""
import datetime
import random
import os
import pickle
import time
import numpy as np
from mindspore import dataset as ds
from mindspore.communication import get_rank

from mindsponge1.common.residue_constants import make_atom14_dists_bounds, order_restype_with_x
from mindsponge1.common.protein import from_pdb_string
from mindsponge1.common.utils import make_atom14_positions, get_aligned_seq
from mindsponge1.data.data_transform import pseudo_beta_fn, atom37_to_frames, atom37_to_torsion_angles
from .preprocess import Feature
from .multimer_pipeline import add_assembly_features, pair_and_merge, post_process
from .multimer_process import process_labels


OUTPUT_LABEL_KEYS = ['aatype_per_chain', 'all_atom_positions', 'all_atom_mask', 'atom14_atom_exists', 
                     'atom14_gt_exists', 'atom14_gt_positions', 'residx_atom14_to_atom37', 
                     'atom37_atom_exists_per_chain', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', 
                     'atom14_atom_is_ambiguous', 'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 
                     'rigidgroups_alt_gt_frames', 'backbone_affine_tensor', 'torsion_angles_sin_cos', 
                     'pseudo_beta', 'pseudo_beta_mask', 'chi_mask', 'backbone_affine_mask', 
                     'chain_index']
def create_dataset(pdb_path, pkl_path, paired_pkl_path, all_name_list, data_cfg, resolution_data, shuffle=False,
                   num_parallel_worker=4, hard_rate=0, high=25,
                   is_parallel=False, mixed_precision=False):
    """create train dataset"""

    column_name = ['aatype', 'residue_index', 'template_aatype', 'template_all_atom_masks',
                   'template_all_atom_positions', 'asym_id', 'sym_id', 'entity_id', 'seq_mask', 'msa_mask',
                   'target_feat', 'msa_feat', 'extra_msa', 'extra_msa_deletion_value', 'extra_msa_mask',
                   'residx_atom37_to_atom14', 'atom37_atom_exists',
                   "prev_pos", "prev_msa_first_row", "prev_pair",
                   "num_sym", "bert_mask", "true_msa", ] + \
                   OUTPUT_LABEL_KEYS + \
                   ["atomtype_radius", "restype_atom14_bond_lower_bound", "restype_atom14_bond_upper_bound", \
                   "use_clamped_fape", "filter_by_solution", "prot_name_index"]

    dataset_generator = DatasetGenerator(pdb_path, pkl_path, paired_pkl_path, all_name_list, data_cfg, resolution_data, mixed_precision, hard_rate, high)
    prefetch_size = 1
    print("prefetch_size", prefetch_size)
    ds.config.set_prefetch_size(prefetch_size)

    if is_parallel:
        rank_id = get_rank() % 8
        rank_size = 8
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle,
                                            num_shards=rank_size,
                                            shard_id=rank_id, max_rowsize=16)
    else:
        train_dataset = ds.GeneratorDataset(source=dataset_generator, column_names=column_name,
                                            num_parallel_workers=num_parallel_worker, shuffle=shuffle, max_rowsize=16)
    return train_dataset


class DatasetGenerator:
    """dataset generator"""
    def __init__(self, pdb_path, pkl_path, paired_pkl_path, all_name_list, data_cfg, resolution_data, mixed_precision, hard_rate, high=25):
        self.t1 = time.time()
        self.pdb_path = pdb_path
        self.pkl_path = pkl_path
        self.paired_pkl_path = paired_pkl_path
        self.all_name_list = all_name_list
        self.data_cfg = data_cfg
        self.resolution_info = resolution_data
        self.mixed_precision = mixed_precision
        self.hard_rate = hard_rate
        self.high = high
        print("end dataset init")

    def _random_sample_chains(self, name_list, max_chains=32):
        
        np.random.shuffle(name_list)

        return name_list[:max_chains]

    def __getitem__(self, index):
        # import time
        # tm0 = time.time()
        is_multimer = True 
        try:
            name_list = self.all_name_list[index]
            name_list = self._random_sample_chains(name_list)
            input_arrays, prev_pos, prev_msa_first_row, prev_pair, \
                num_sym, bert_mask, true_msa, labels_arrays \
                    = self._get_train_data(name_list, is_multimer)
        except:
            print('error for name', name_list)
            # raise IOError
            name_list = self.all_name_list[0]
            name_list = self._random_sample_chains(name_list)
            input_arrays, prev_pos, prev_msa_first_row, prev_pair, \
                num_sym, bert_mask, true_msa, labels_arrays \
                    = self._get_train_data(name_list, is_multimer)
            
        prot_name_index = np.array([index]).astype(np.int32)
        atomtype_radius = np.array(
            [1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.55, 1.55,
            1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55,
            1.52, 1.7, 1.7, 1.7, 1.55, 1.52])
        restype_atom14_bond_lower_bound, restype_atom14_bond_upper_bound, _ = \
            make_atom14_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=12.0)
        use_clamped_fape = np.random.binomial(1, 0.9, size=1)
        filter_by_solution = self._get_solution_flag(name_list[0].split("_")[0])
        extra_feats = [atomtype_radius, restype_atom14_bond_lower_bound,
                    restype_atom14_bond_upper_bound, use_clamped_fape, filter_by_solution]
        
        dtype = np.float32
        if self.mixed_precision:
            dtype = np.float16
        extra_feats = [array.astype(dtype) for array in extra_feats] + [prot_name_index]


        all_feats = input_arrays + [prev_pos, prev_msa_first_row, prev_pair, num_sym, bert_mask, true_msa] + labels_arrays + extra_feats

        # print(name_list[0], len(name_list), time.time()-tm0)
        return tuple(all_feats)

    def __len__(self):
        return len(self.all_name_list)

    def _get_solution_flag(self, prot_name):
        """get resolution data"""
        if prot_name not in self.resolution_info:
            return np.array(1.0).astype(np.float32)
        resolution = float(self.resolution_info[prot_name])
        if resolution < 3:
            return np.array(1.0).astype(np.float32)
        return np.array(0.0).astype(np.float32)

    def _get_random_sampled_index(self, total_num, high=25):
        need_num = min(np.random.randint(1, high+1), total_num)
        sampled_index = random.sample(range(total_num), need_num)
        return sampled_index
        
        
        
    
    def _get_train_data(self, name_list, is_multimer=True):
        """get train data"""

        def load_multi_data(name_list):
            
            prot_name = name_list[0].split("_")[0]
            turn_hard = np.random.rand() < self.hard_rate
            
            paired_feature = None
            if len(name_list) > 1 and os.path.exists(f"{self.paired_pkl_path}/{prot_name}.pkl"):
                with open(f"{self.paired_pkl_path}/{prot_name}.pkl", "rb") as f:
                    paired_feature = pickle.load(f)
                if turn_hard and len(paired_feature) > 0:
                    sampled_index = self._get_random_sampled_index(list(paired_feature.values())[0]['msa'].shape[0], self.high)
                    for k, v in paired_feature.items():
                        for k1, v1 in v.items():
                            if k1 in ['msa', 'deletion_matrix']:
                                paired_feature[k][k1] = v1[sampled_index]
            

            all_seq_len = 0
            features_all = []
            sequences = []
            turn_hard_seq_index = {}
            for name in name_list:

                features = {}
                pkl_path_single = os.path.join(self.pkl_path, name + ".pkl")
                
                with open(pkl_path_single, "rb") as f:
                    raw_feature = pickle.load(f)
                    features['aatype']=np.nonzero(raw_feature['aatype'])[1].astype(np.int32)
                    seq_len = raw_feature["msa"].shape[1]
                    features["between_segment_residues"] = raw_feature["between_segment_residues"]
                    features["residue_index"] = raw_feature["residue_index"]
                    seq = raw_feature["sequence"][0].decode()
                    features["sequence"] = np.array(seq)
                    sequences.append(seq)

                    features["msa"] = raw_feature["msa"]
                    features["deletion_matrix"] = raw_feature["deletion_matrix_int"]
                    if turn_hard:
                        if seq not in turn_hard_seq_index:
                            sampled_index = self._get_random_sampled_index(features["msa"].shape[0], self.high)
                            turn_hard_seq_index[seq] = sampled_index
                        else:
                            sampled_index = turn_hard_seq_index[seq]
                        features["msa"] = features["msa"][sampled_index]
                        features["deletion_matrix"] = features["deletion_matrix"][sampled_index]
                    features["num_alignments"] = np.array(features["msa"].shape[0])

                    if (not turn_hard) and (len(raw_feature["template_aatype"].shape) > 1):
                        features["template_aatype"] = np.argmax(raw_feature["template_aatype"], axis=-1)
                        features["template_all_atom_mask"] = raw_feature["template_all_atom_masks"]
                        features["template_all_atom_positions"] = raw_feature["template_all_atom_positions"]
                    else:
                        features["template_aatype"] = np.zeros((1, seq_len)).astype(np.int32)
                        features["template_all_atom_mask"] = np.zeros((1, seq_len, 37)).astype(np.float32)
                        features["template_all_atom_positions"] = np.zeros((1, seq_len, 37, 3)).astype(np.float32)

                    
                    if paired_feature:
                        features["msa_all_seq"] = paired_feature[seq]["msa"]
                        features["deletion_matrix_all_seq"] = paired_feature[seq]["deletion_matrix"]
                        features["num_alignments_all_seq"] = np.array(features["msa_all_seq"].shape[0])
                    all_seq_len += seq_len

                pdb_path_single = os.path.join(self.pdb_path, name + ".pdb")
                with open(pdb_path_single, 'r') as f:
                    prot_pdb = from_pdb_string(f.read())
                    aatype = prot_pdb.aatype
                    seq_len = len(aatype)
                    atom37_positions = prot_pdb.atom_positions.astype(np.float32)
                    atom37_mask = prot_pdb.atom_mask.astype(np.float32)

                    features["seq_length"] = np.array(seq_len)
                    features["aatype_pdb"] = np.array(aatype)
                    features["all_atom_positions"] = atom37_positions
                    features["all_atom_mask"] = atom37_mask
                
                features_all.append(features)
                
            is_homomer = len(set(sequences)) == 1 and len(sequences) > 1
            # is_homomer = len(set(sequences)) == 1 

            if is_homomer and "msa_all_seq" not in features_all[0].keys():
                for features in features_all:
                        features["msa_all_seq"] = features["msa"]
                        features["deletion_matrix_all_seq"] = features["deletion_matrix"]
                        features["num_alignments_all_seq"] = np.array(features["msa_all_seq"].shape[0])

            # print(f"\n\n\n=========================={name_list}")
            # for i, features in enumerate(features_all):
            #     print(f"\n=========================={i}")
            #     for key, value in features.items():
            #         print(key, value.shape, value.dtype)
            
            # print(len(name_list), prot_name, all_seq_len)
            return features_all, all_seq_len


        features, all_seq_len = load_multi_data(name_list)

        # if "msa_all_seq" not in feature and\
        #       np.sum([feature["msa"].shape[0]==features[0]["msa"].shape[0]  for feature in features]) < len(features):
        #     print(f"paired msa num not the same for prot ", name_list[0].split("_")[0])
        #     paired_msa_num = np.min([feature["msa_all_seq"].shape[0] for feature in features])
        #     for feature in features:
        #         feature["msa_all_seq"] = feature["msa_all_seq"][:1]
        #         feature["deletion_matrix_all_seq"] = feature["deletion_matrix_all_seq"][:1]
        #         feature["num_alignments_all_seq"] = np.array(1)                


        features = add_assembly_features(features)
        # for i, feature in enumerate(features):
        #     print("\n\n", i)
        #     for key, value in feature.items():
        #         print(key, value.shape, value.dtype)

        all_labels = [{k: f[k].copy() for k in ["aatype_pdb", "all_atom_positions", "all_atom_mask"]} for f in features]

        asym_len = np.array([c["seq_length"] for c in features], dtype=np.int64)

        features = pair_and_merge(features)
        features = post_process(features)
        features["asym_len"] = asym_len
        processed_feature = Feature(self.data_cfg, features, is_training=True, is_multimer=True)

        seed = global_seed()
        input_arrays, prev_pos, prev_msa_first_row, prev_pair, num_sym, bert_mask, true_msa \
        = processed_feature.pipeline(self.data_cfg, self.mixed_precision, seed=seed)


        all_labels = process_labels(all_labels)
        # print(f"\n\n==========================all_labels")
        # for key, value in all_labels[0].items():
        #     print(key, value.shape, value.dtype, flush=True)
        # keys = list(all_labels[0].keys())
        # print(keys)
        # keys.sort()
        # for i, all_label in enumerate(all_labels):
        #     print("\n\n\n===============", i)
        #     for key in OUTPUT_LABEL_KEYS:
        #         value = all_label[key]
        #         print(key, value.shape, value.dtype, flush=True)

        def merge_label_dicts(all_labels):
            labels_arrays = []
            for key in OUTPUT_LABEL_KEYS:
                values = []
                for all_label in all_labels:
                    values.append(all_label[key])
                value = np.concatenate(values, axis=0)
                if value.dtype == "float64":
                    value = value.astype(np.float16)
                if value.dtype == "float32":
                    value = value.astype(np.float16)
                if value.dtype == "int64":
                    value = value.astype(np.int32)
                labels_arrays.append(value)
            return labels_arrays
        
        labels_arrays = merge_label_dicts(all_labels)
        # for array in labels_arrays:
        #     print(array.shape, array.dtype)

        return input_arrays, prev_pos, prev_msa_first_row, prev_pair, num_sym, bert_mask, true_msa, labels_arrays


class SeedMaker:
    """Return unique seeds."""

    def __init__(self, initial_seed=0):
        self.next_seed = initial_seed

    def __call__(self):
        i = self.next_seed
        self.next_seed += 1
        return i


global_seed = SeedMaker()


def process_pdb(true_aatype, ori_res_length, decoy_pdb_path):
    """get atom information from pdb"""
    with open(decoy_pdb_path, 'r') as f:
        decoy_prot_pdb = from_pdb_string(f.read())
        f.close()
    decoy_aatype = decoy_prot_pdb.aatype
    decoy_atom37_positions = decoy_prot_pdb.atom_positions.astype(np.float32)
    decoy_atom37_mask = decoy_prot_pdb.atom_mask.astype(np.float32)
    padding_val = true_aatype.shape[0] - ori_res_length
    true_aatype = true_aatype[:ori_res_length]
    decoy_aatype, decoy_atom37_positions, decoy_atom37_mask, align_mask = \
        align_with_aatype(true_aatype, decoy_aatype, decoy_atom37_positions, decoy_atom37_mask)
    decoy_atom37_positions = np.pad(decoy_atom37_positions, ((0, padding_val), (0, 0), (0, 0)))
    decoy_atom37_mask = np.pad(decoy_atom37_mask, ((0, padding_val), (0, 0)))
    align_mask = np.pad(align_mask, ((0, padding_val)))

    return decoy_atom37_positions, decoy_atom37_mask, align_mask


def align_with_aatype(true_aatype, aatype, atom37_positions, atom37_mask):
    """align pdb with aatype"""
    if len(true_aatype) == len(aatype):
        out = aatype, atom37_positions, atom37_mask, np.ones((aatype.shape[0])).astype(np.float32)
        return out
    seq1 = [order_restype_with_x.get(x) for x in aatype]
    seq2 = [order_restype_with_x.get(x) for x in true_aatype]
    seq1 = ''.join(seq1)
    seq2 = ''.join(seq2)
    _, align_relationship, _ = get_aligned_seq(seq1, seq2)
    pdb_index = 0
    seq_len = len(true_aatype)
    new_aatype = np.zeros((seq_len,)).astype(np.int32)
    new_atom37_positions = np.zeros((seq_len, 37, 3)).astype(np.float32)
    new_atom37_mask = np.zeros((seq_len, 37)).astype(np.float32)
    align_mask = np.zeros((seq_len,)).astype(np.float32)
    for i in range(len(true_aatype)):
        if align_relationship[i] == "-":
            new_aatype[i] = 20
            new_atom37_positions[i] = np.zeros((37, 3)).astype(np.float32)
            new_atom37_mask[i] = np.zeros((37,)).astype(np.float32)
            align_mask[i] = 0
        else:
            new_aatype[i] = aatype[pdb_index]
            new_atom37_positions[i] = atom37_positions[pdb_index]
            new_atom37_mask[i] = atom37_mask[pdb_index]
            align_mask[i] = 1
            pdb_index += 1
    out = new_aatype, new_atom37_positions, new_atom37_mask, align_mask
    return out
