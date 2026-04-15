# Copyright 2023 Huawei Technologies Co., Ltd & CPL YiQin GAO Research Group
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
"run_relax"
import os
import stat
import numpy as np

from mindsponge.common import protein, residue_constants
from .relax import relax

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 1


def make_atom14_masks(feature):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    # create the corresponding mask
    restype_atom37_mask = np.zeros([21, 37], np.float32)
    for restype, restype_letter in enumerate(residue_constants.restypes):
        restype_name = residue_constants.restype_1to3[restype_letter]
        atom_names = residue_constants.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = residue_constants.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[feature.get('aatype')]

    return residx_atom37_mask


def get_amber_input(input_file_path):
    '''get_amber_input'''
    with open(input_file_path, 'r') as f:
        prot_pdb = protein.from_pdb_string(f.read())
    aatype = prot_pdb.aatype
    b_factors = prot_pdb.b_factors
    seq_len = len(aatype)
    atom_positions = prot_pdb.atom_positions.astype(np.float32)
    atom37_mask = prot_pdb.atom_mask.astype(np.float32)
    residue_index = np.array(range(seq_len), dtype=np.int32)
    features = {'aatype': aatype,
                'all_atom_positions': atom_positions,
                'all_atom_mask': atom37_mask}
    atom_mask = make_atom14_masks(features)
    result = (aatype, atom_positions, atom_mask, residue_index, b_factors)

    return result


def run_relax(input_file_path, output_file_path):
    '''run_relax'''

    amber_relaxer = relax.AmberRelaxation(
        max_iterations=RELAX_MAX_ITERATIONS,
        tolerance=RELAX_ENERGY_TOLERANCE,
        stiffness=RELAX_STIFFNESS,
        exclude_residues=RELAX_EXCLUDE_RESIDUES,
        max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS)

    result = get_amber_input(input_file_path)
    aatype, atom_positions, atom_mask, residue_index, b_factors = result
    data = [aatype, residue_index, atom_positions, atom_mask, b_factors]
    unrelaxed_protein = protein.from_prediction_new(data)

    # Relax the prediction.
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=unrelaxed_protein)

    # Save the relaxed PDB.
    os_flags = os.O_RDWR | os.O_CREAT
    os_modes = stat.S_IRWXU
    with os.fdopen(os.open(output_file_path, os_flags, os_modes), "w") as fout:
        fout.write(relaxed_pdb_str)
    print("OpenMM relaxation finished, output pdb file saved at" + \
        output_file_path)
