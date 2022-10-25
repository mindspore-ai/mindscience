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
from mindspore.ops import operations as P
from mindsponge.common.geometry import vecs_from_tensor
from common.geometry import multimer_rigids_compute_dihedral_angle


def compute_chi_angles(aatype,  # (B, N)
                       all_atom_pos,  # (B, N, 37, 3)
                       all_atom_mask,  # (B, N, 37)
                       chi_atom_indices,
                       chi_angles_mask,
                       indices0,
                       indices1,
                       batch_size=1):
    """compute chi angles"""

    aatype = mnp.minimum(aatype, 20)
    # Collect the atoms for the chi-angles.
    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    # Select atoms to compute chis. Shape: [batch, num_res, chis=4, atoms=4].
    atom_indices = mnp.take(chi_atom_indices, aatype, axis=0)

    # # Gather atom positions Batch Gather. Shape: [batch, num_res, chis=4, atoms=4, xyz=3].

    # 4 seq_length 4 4  batch, sequence length, chis, atoms
    seq_length = all_atom_pos.shape[1]
    atom_indices = atom_indices.reshape((4, seq_length, 4, 4, 1)).astype("int32")
    new_indices = P.Concat(4)((indices0, indices1, atom_indices))
    chis_atom_pos = P.GatherNd()(all_atom_pos, new_indices)
    chis_mask = mnp.take(chi_angles_mask, aatype, axis=0)
    chi_angle_atoms_mask = P.GatherNd()(all_atom_mask, new_indices)

    # Check if all 4 chi angle atoms were set. Shape: [batch, num_res, chis=4].
    chi_angle_atoms_mask = P.ReduceProd()(chi_angle_atoms_mask, -1)
    chis_mask = chis_mask * (chi_angle_atoms_mask).astype(mnp.float32)
    all_chi_angles = []
    for i in range(batch_size):
        template_chi_angles = multimer_rigids_compute_dihedral_angle(vecs_from_tensor(chis_atom_pos[i, :, :, 0, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 1, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 2, :]),
                                                                     vecs_from_tensor(chis_atom_pos[i, :, :, 3, :]))
        all_chi_angles.append(template_chi_angles)
    chi_angles = mnp.stack(all_chi_angles, axis=0)
    return chi_angles, chis_mask
