# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""violation loss calculation."""

import numpy as np

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P

from ..metrics.structure_violations import find_structural_violations
from ..common import residue_constants
from ..common.utils import get_pdb_info

C_ONE_HOT = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.float32)
N_ONE_HOT = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.float32)
ATOMTYPE_RADIUS = np.array([1.55, 1.7, 1.7, 1.7, 1.52, 1.7, 1.7, 1.7, 1.52, 1.52, 1.8, 1.7, 1.7, 1.7, 1.55, 1.55, 1.52,
                            1.52, 1.8, 1.7, 1.7, 1.7, 1.7, 1.55, 1.55, 1.55, 1.52, 1.52, 1.7, 1.55, 1.55, 1.52, 1.7,
                            1.7, 1.7, 1.55, 1.52])
DISTS_MASK_I = np.eye(14, 14)
LOWER_BOUND, UPPER_BOUND, RESTYPE_ATOM14_BOUND_STDDEV = \
    residue_constants.make_atom14_dists_bounds(overlap_tolerance=1.5, bond_length_tolerance_factor=12.0)

VIOLATION_TOLERANCE_ACTOR = 12.0
CLASH_OVERLAP_TOLERANCE = 1.5
CYS_SG_IDX = 5


def get_violation_loss(pdb_path=None, atom14_atom_exists=None, residue_index=None, residx_atom14_to_atom37=None,
                       atom14_positions=None, aatype=None):
    """calculate violations loss for a given pdb or pdb info
    Args:
        pdb_path: absolute path of pdb
        atom14_atom_exists: weather atom exists in atom14 order from residue constants (ndarray or Tensor in float32)
        residue_index: residue index from pdb (ndarray or Tensor in float32)
        residx_atom14_to_atom37:map atom14 residue_index to atom37 (ndarray or Tensor in int32)
        atom14_positions:atom positions in atom14 order (ndarray or Tensor in float32)
        aatype:amino acid type (ndarray or Tensor in int32)

    Returns:
        pdb's violations loss
    """
    if pdb_path:
        features = get_pdb_info(pdb_path)

        atom14_atom_exists = Tensor(features.get("atom14_gt_exists")).astype(ms.float32)
        residue_index = Tensor(features.get("residue_index")).astype(ms.float32)
        residx_atom14_to_atom37 = Tensor(features.get("residx_atom14_to_atom37")).astype(ms.int32)
        atom14_positions = Tensor(features.get("atom14_gt_positions")).astype(ms.float32)
        aatype = Tensor(features.get("aatype")).astype(ms.int32)
    else:
        atom14_atom_exists = Tensor(atom14_atom_exists).astype(ms.float32) \
            if isinstance(atom14_atom_exists, np.ndarray) else atom14_atom_exists
        residue_index = Tensor(residue_index).astype(ms.float32) \
            if isinstance(residue_index, np.ndarray) else residue_index
        residx_atom14_to_atom37 = Tensor(residx_atom14_to_atom37).astype(ms.int32) \
            if isinstance(residx_atom14_to_atom37, np.ndarray) else residx_atom14_to_atom37
        atom14_positions = Tensor(atom14_positions).astype(ms.float32) \
            if isinstance(atom14_positions, np.ndarray) else atom14_positions
        aatype = Tensor(aatype).astype(ms.int32) if isinstance(aatype, np.ndarray) else aatype

    lower_bound_t = Tensor(LOWER_BOUND, ms.float32)
    upper_bound_t = Tensor(UPPER_BOUND, ms.float32)
    atom_type_radius_t = Tensor(ATOMTYPE_RADIUS, ms.float32)
    c_one_hot_t = Tensor(C_ONE_HOT, ms.int32)
    n_one_hot_t = Tensor(N_ONE_HOT, ms.int32)
    dists_mask_i_t = Tensor(DISTS_MASK_I, ms.int32)
    cys_sg_idx_t = Tensor(CYS_SG_IDX, ms.int32)

    (bonds_c_n_loss_mean, angles_ca_c_n_loss_mean, angles_c_n_ca_loss_mean, _, _, _, clashes_per_atom_loss_sum,
     _, per_atom_loss_sum, _, total_per_residue_violations_mask) = \
        find_structural_violations(atom14_atom_exists, residue_index, aatype, residx_atom14_to_atom37,
                                   atom14_positions, VIOLATION_TOLERANCE_ACTOR,
                                   CLASH_OVERLAP_TOLERANCE, lower_bound_t, upper_bound_t, atom_type_radius_t,
                                   c_one_hot_t, n_one_hot_t, dists_mask_i_t, cys_sg_idx_t)
    num_atoms = P.ReduceSum()(atom14_atom_exists.astype(ms.float32))
    structure_violation_loss = bonds_c_n_loss_mean + angles_ca_c_n_loss_mean + angles_c_n_ca_loss_mean +\
                               P.ReduceSum()(clashes_per_atom_loss_sum + per_atom_loss_sum) / (1e-6 + num_atoms)

    return structure_violation_loss, total_per_residue_violations_mask
