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
Make a PSSM input dictionary for each pdb in the parsed PDBs folder.
"""

import argparse
import json
import numpy as np

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

argparser.add_argument("--PSSM_input_path", type=str, help="Path to PSSMs saved as npz files.")
argparser.add_argument("--jsonl_input_path", type=str, help="Path where to load .jsonl dictionary of parsed pdbs.")
argparser.add_argument("--output_path", type=str, help="Path where to save .jsonl dictionary with PSSM bias.")

args = argparser.parse_args()

with open(args.jsonl_input_path, 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

my_dict = {}
for json_str in json_list:
    result = json.loads(json_str)
    all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain']
    path_to_PSSM = args.PSSM_input_path+"/"+result['name'] + ".npz"
    print(path_to_PSSM)
    pssm_input = np.load(path_to_PSSM)
    pssm_dict = {}
    for chain in all_chain_list:
        pssm_dict[chain] = {}
        pssm_dict[chain]['pssm_coef'] = pssm_input[chain+'_coef'].tolist() #[L] per position coefficient to trust PSSM; 0.0 - do not use it; 1.0 - just use PSSM only
        pssm_dict[chain]['pssm_bias'] = pssm_input[chain+'_bias'].tolist() #[L,21] probability (sums up to 1.0 over alphabet of size 21) from PSSM
        pssm_dict[chain]['pssm_log_odds'] = pssm_input[chain+'_odds'].tolist() #[L,21] log_odds ratios coming from PSSM; optional/not needed
    my_dict[result['name']] = pssm_dict

#Write output to:
with open(args.output_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(my_dict) + '\n')
