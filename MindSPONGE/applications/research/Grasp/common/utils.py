# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""utils module"""

import mindspore.numpy as mnp
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindsponge1.common.geometry import vecs_from_tensor
from common.geometry import multimer_rigids_compute_dihedral_angle
from mindspore import  Parameter
import re

def trans_ckpt(param_dict):
    # tmp_dict = {k: v.asnumpy() for k, v in param_dict.items()}
    # import pickle
    # with open('/job/file/step0_numpy.ckpt', 'wb') as f:
    #     pickle.dump(tmp_dict, f)

    # raise IOError('good bye')

    new_param_dict = {}
    for k, v in param_dict.items():
        if re.search('learning_rate|global_step|moment[12]|beta[12]_power|vhat|template_embedding\._flat_.*_slice', k):
            continue
        if re.search('^msa_stack', k):
            new_k = re.sub('^(msa_stack\.)\d+\.', '\\1', k)
            if new_k in new_param_dict:
                new_param_dict[new_k].append(v.asnumpy()[None])
            else:
                new_param_dict[new_k] = [v.asnumpy()[None]]
        else:
            new_param_dict[k] = v
    for k, v in new_param_dict.items():
        if re.search('^msa_stack', k):
            new_param_dict[k] = Parameter(np.concatenate(new_param_dict[k], axis=0))
    for key, value in new_param_dict.items():
        if (('preprocess_1d.weight' in key) or ('left_single.weight' in key) or ('right_single.weight' in key)) and (new_param_dict[key].shape[-1] == 22):
            new_param_dict[key] = Parameter(new_param_dict[key][..., 1:])
    return new_param_dict



# def trans_ckpt(ckpt):
#     # temp_key = []
#     current_path = "/job/file/common/"

#     batch_dict = {}

#     msa_key = []
#     with open(current_path+"old_extra.txt", "r") as f:
#         for line in f.readlines():
#             msa_key.append(line.strip('\n'))
#     msa_keys = []
#     for i in range(4):
#         temp = []
#         for j in range(len(msa_key)):
#             key = msa_key[j].split('0')
#             new_key = key[0] + str(i) + key[1]
#             temp.append(new_key)
#         msa_keys.append(temp)
        
#     msa_new_key = []
#     with open(current_path+"new_extra.txt", "r") as f:
#         for line in f.readlines():
#             msa_new_key.append(line.strip('\n'))
#     msa_new_keys = []
#     for i in range(4):
#         temp = []
#         for j in range(len(msa_new_key)):
#             key = msa_new_key[j].split('0')
#             new_key = key[0] + str(i) + key[1]
#             temp.append(new_key)
#         msa_new_keys.append(temp)

#     envo_key = []
#     with open(current_path+"old_evo.txt", "r") as f:
#         for line in f.readlines():
#             envo_key.append(line.strip('\n'))
#     envo_keys = []
#     for i in range(48):
#         temp = []
#         for j in range(len(envo_key)):
#             key = envo_key[j].split('0')
#             new_key = key[0] + str(i) + key[1]
#             temp.append(new_key)
#         envo_keys.append(temp)

#     envo_new_key = []
#     with open(current_path+"new_evo.txt", "r") as f:
#         for line in f.readlines():
#             envo_new_key.append(line.strip('\n'))
#     envo_new_keys = []
#     for i in range(1):
#         temp = []
#         for j in range(len(envo_new_key)):
#             new_key = envo_new_key[j]
#             temp.append(new_key)
#         envo_new_keys.append(temp)
#     for key in ckpt.keys():
#         flat_msa_keys = sum(msa_keys, [])
#         flat_envo_keys = sum(envo_keys, [])
#         msa_count = len(msa_keys[0])
#         envo_count = len(envo_keys[0])
#         if "learning_rate" in key or "global_step" in key or "moment1" in key or "moment2" in key or "beta1_power" in key or "beta2_power" in key or "vhat" in key:
#            continue
#         if key in flat_msa_keys:
#             row = flat_msa_keys.index(key) // msa_count
#             col = flat_msa_keys.index(key) % msa_count
#             batch_dict[msa_new_keys[row][col]] = ckpt[key]
#         elif key in flat_envo_keys:
#             row = flat_envo_keys.index(key) // envo_count
#             col = flat_envo_keys.index(key) % envo_count
#             if envo_new_keys[0][col] not in batch_dict:
#                 batch_dict[envo_new_keys[0][col]] = np.array(np.expand_dims(ckpt[key].asnumpy(), 0))
#             else:
#                 batch_dict[envo_new_keys[0][col]] = np.array(np.concatenate((batch_dict[envo_new_keys[0][col]], np.expand_dims(ckpt[key].asnumpy(), 0)), axis=0))
#         else:
#             batch_dict[key] = ckpt[key]

#     for k, v in batch_dict.items():
#         # print(k, v.shape, flush=True)
#         if 'template_embedding._flat_query_slice' in k or 'template_embedding._flat_templates_slice' in k:
#             continue
#         batch_dict[k] = Parameter(v)


#     return batch_dict


class CompuyeChiAngles(nn.Cell):
    def __init__(self):
        super(CompuyeChiAngles, self).__init__()
        self.equal = P.Equal()
        self.minimum = P.Minimum().shard(((1,2),()))
        self.reshape = P.Reshape().shard(((1,2,1,1),()))
        self.concat = P.Concat(4).shard(((1, 2, 1, 1, 1), (1, 2, 1, 1, 1),(1, 2, 1, 1, 1)))
        self.gathernd1 = P.GatherNd()#.shard(((1,8,1,1),(1,8,1,1,1)))
        self.gathernd2 = P.GatherNd()#.shard(((1,8,1), (1,8,1,1,1)))
        self.reduceprod =  P.ReduceProd().shard(((1,2,1,1),))
        self.mul = P.Mul().shard(((1,2,1),(1,2,1)))
        self.stack = P.Stack().shard(((2,1),(2,1),(2,1),(2,1)))


    def construct(self, aatype,  # (B, N)
                        all_atom_pos,  # (B, N, 37, 3)
                        all_atom_mask,  # (B, N, 37)
                        chi_atom_indices,
                        chi_angles_mask,
                        indices0,
                        indices1):
        aatype = self.minimum(aatype, 20)
        # Collect the atoms for the chi-angles.
        # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
        # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
        atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)

        # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

        # 4 seq_length 4 4  batch, sequence length, chis, atoms
        seq_length = all_atom_pos.shape[1]
        atom_indices = self.reshape(atom_indices, tuple((4, seq_length, 4, 4, 1))).astype("int32")
        new_indices = self.concat((indices0, indices1, atom_indices))
        chis_atom_pos = self.gathernd1(all_atom_pos, new_indices)
        chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
        chi_angle_atoms_mask = self.gathernd2(all_atom_mask, new_indices)
        # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
        chi_angle_atoms_mask = self.reduceprod(chi_angle_atoms_mask, -1)
        chis_mask = self.mul(chis_mask, (chi_angle_atoms_mask).astype(mnp.float32))
        all_chi_angles = []
        for i in range(aatype.shape[0]):
            template_chi_angles = multimer_rigids_compute_dihedral_angle(vecs_from_tensor(chis_atom_pos[i, :, :, 0, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 1, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 2, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 3, :]))
            all_chi_angles.append(template_chi_angles) 
        chi_angles = self.stack(all_chi_angles)
        return chi_angles, chis_mask


def compute_chi_angles(aatype,  # (B, N)
                       all_atom_pos,  # (B, N, 37, 3)
                       all_atom_mask,  # (B, N, 37)
                       chi_atom_indices,
                       chi_angles_mask,
                       indices0,
                       indices1):
    """compute chi angles"""

    aatype = mnp.minimum(aatype, 20)
    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    # atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)
    atom_indices = chi_atom_indices[aatype,...]
    # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

    # 4 seq_length 4 4  batch, sequence length, chis, atoms
    seq_length = all_atom_pos.shape[1]
    atom_indices = atom_indices.reshape((4, seq_length, 4, 4, 1)).astype("int32")
    new_indices = P.Concat(4)((indices0, indices1, atom_indices))
    chis_atom_pos = P.GatherNd()(all_atom_pos, new_indices)
    # chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
    chis_mask = chi_angles_mask[aatype,:]
    chi_angle_atoms_mask = P.GatherNd()(all_atom_mask, new_indices)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = P.ReduceProd()(chi_angle_atoms_mask, -1)
    chis_mask = chis_mask * (chi_angle_atoms_mask).astype(mnp.float32)
    all_chi_angles = []
    for i in range(aatype.shape[0]):
        template_chi_angles = multimer_rigids_compute_dihedral_angle(vecs_from_tensor(chis_atom_pos[i, :, :, 0, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 1, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 2, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 3, :]))
        all_chi_angles.append(template_chi_angles)
    chi_angles = mnp.stack(all_chi_angles, axis=0)
    return chi_angles, chis_mask


class ComputeChiAngles(nn.Cell):
    def __init__(self, device_num):
        super(ComputeChiAngles, self).__init__()
        self.equal = P.Equal()
        self.minimum = P.Minimum().shard(((1,device_num),()))
        self.reshape = P.Reshape().shard(((1,device_num,1,1),()))
        self.concat = P.Concat(4).shard(((1, device_num, 1, 1, 1), (1, device_num, 1, 1, 1),(1, device_num, 1, 1, 1)))
        self.gathernd1 = P.GatherNd()#.shard(((1,8,1,1),(1,8,1,1,1)))
        self.gathernd2 = P.GatherNd()#.shard(((1,8,1), (1,8,1,1,1)))
        self.reduceprod =  P.ReduceProd().shard(((1,device_num,1,1),))
        self.mul = P.Mul().shard(((1,device_num,1),(1,device_num,1)))
        self.stack = P.Stack().shard(((device_num,1),(device_num,1),(device_num,1),(device_num,1)))
    def construct(self, aatype,  # (B, N)
                        all_atom_pos,  # (B, N, 37, 3)
                        all_atom_mask,  # (B, N, 37)
                        chi_atom_indices,
                        chi_angles_mask,
                        indices0,
                        indices1):
        aatype = self.minimum(aatype, 20)
        # Collect the atoms for the chi-angles.
        # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
        # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
        atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)

        # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

        # 4 seq_length 4 4  batch, sequence length, chis, atoms
        seq_length = all_atom_pos.shape[1]
        atom_indices = self.reshape(atom_indices, tuple((4, seq_length, 4, 4, 1))).astype("int32")
        new_indices = self.concat((indices0, indices1, atom_indices))
        chis_atom_pos = self.gathernd1(all_atom_pos, new_indices)
        chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
        chi_angle_atoms_mask = self.gathernd2(all_atom_mask, new_indices)
        # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
        chi_angle_atoms_mask = self.reduceprod(chi_angle_atoms_mask, -1)
        chis_mask = self.mul(chis_mask, (chi_angle_atoms_mask).astype(mnp.float32))
        all_chi_angles = []
        for i in range(aatype.shape[0]):
            template_chi_angles = multimer_rigids_compute_dihedral_angle(vecs_from_tensor(chis_atom_pos[i, :, :, 0, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 1, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 2, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 3, :]))
            all_chi_angles.append(template_chi_angles) 
        chi_angles = self.stack(all_chi_angles)
        return chi_angles, chis_mask


import numpy as np
from scipy.special import softmax

def compute_confidence(predicted_lddt_logits, return_lddt=False):
    """compute confidence"""

    num_bins = predicted_lddt_logits.shape[-1]
    bin_width = 1 / num_bins
    start_n = bin_width / 2
    plddt = compute_plddt(predicted_lddt_logits, start_n, bin_width)
    confidence = np.mean(plddt)
    if return_lddt:
        return confidence, plddt

    return confidence


def compute_plddt(logits, start_n, bin_width):
    """Computes per-residue pLDDT from logits.

    Args:
      logits: [num_res, num_bins] output from the PredictedLDDTHead.

    Returns:
      plddt: [num_res] per-residue pLDDT.
    """
    bin_centers = np.arange(start=start_n, stop=1.0, step=bin_width)
    probs = softmax(logits, axis=-1)
    predicted_lddt_ca = np.sum(probs * bin_centers[None, :], axis=-1)
    return predicted_lddt_ca * 100