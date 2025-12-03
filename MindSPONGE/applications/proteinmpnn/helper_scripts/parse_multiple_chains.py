# Modified from ProteinMPNN (https://github.com/dauparas/ProteinMPNN)
# Original license: MIT License
#
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Parse multiple chains in a pdb file.
"""

import argparse
import json
import glob

from proteinmpnn.util_protein_mpnn import parse_PDB_biounits, get_chain_alphabet

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument("--input_path", type=str, help="Path to a folder with pdb files, e.g. /home/my_pdbs/")
argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary of parsed pdbs")
argparser.add_argument("--ca_only", action="store_true",
    default=False, help="parse a backbone-only structure (default: false)")

args = argparser.parse_args()

folder_with_pdbs_path = args.input_path
save_path = args.output_path
ca_only = args.ca_only

pdb_dict_list = []
c = 0

if folder_with_pdbs_path[-1]!='/':
    folder_with_pdbs_path = folder_with_pdbs_path+'/'


chain_alphabet = get_chain_alphabet()

biounit_names = glob.glob(folder_with_pdbs_path+'*.pdb')
for biounit in biounit_names:
    my_dict = {}
    s = 0
    concat_seq = ''
    concat_N = []
    concat_CA = []
    concat_C = []
    concat_O = []
    concat_mask = []
    coords_dict = {}
    for letter in chain_alphabet:
        if ca_only:
            sidechain_atoms = ['CA']
        else:
            sidechain_atoms = ['N', 'CA', 'C', 'O']
        xyz, seq = parse_PDB_biounits(biounit, atoms=sidechain_atoms, chain=letter)
        if not isinstance(xyz, str):
            concat_seq += seq[0]
            my_dict['seq_chain_'+letter]=seq[0]
            coords_dict_chain = {}
            if ca_only:
                coords_dict_chain['CA_chain_'+letter]=xyz.tolist()
            else:
                coords_dict_chain['N_chain_' + letter] = xyz[:, 0, :].tolist()
                coords_dict_chain['CA_chain_' + letter] = xyz[:, 1, :].tolist()
                coords_dict_chain['C_chain_' + letter] = xyz[:, 2, :].tolist()
                coords_dict_chain['O_chain_' + letter] = xyz[:, 3, :].tolist()
            my_dict['coords_chain_'+letter]=coords_dict_chain
            s += 1
    fi = biounit.rfind("/")
    my_dict['name']=biounit[(fi+1):-4]
    my_dict['num_of_chains'] = s
    my_dict['seq'] = concat_seq
    if s < len(chain_alphabet):
        pdb_dict_list.append(my_dict)
        c+=1

with open(save_path, 'w', encoding='utf-8') as f:
    for entry in pdb_dict_list:
        f.write(json.dumps(entry) + '\n')
