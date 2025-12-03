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
Make a tied positions dictionary for each pdb in the parsed PDBs folder.
"""

import argparse
import json

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
argparser.add_argument("--chain_list", type=str, default='', help="List of the chains that need to be fixed")
argparser.add_argument("--position_list", type=str, default='',
    help="Position lists, e.g. 11 12 14 18, 1 2 3 4 for first chain and the second chain")
argparser.add_argument("--homooligomer", type=int, default=0, help="If 0 do not use, if 1 then design homooligomer")

args = argparser.parse_args()

with open(args.input_path, 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

homooligomeric_state = args.homooligomer

if homooligomeric_state == 0:
    tied_list = [[int(item) for item in one.split()] for one in args.position_list.split(",")]
    global_designed_chain_list = [str(item) for item in args.chain_list.split()]
    my_dict = {}
    for json_str in json_list:
        result = json.loads(json_str)
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
        tied_positions_list = []
        for i, _ in enumerate(tied_list[0]):
            temp_dict = {}
            for j, chain in enumerate(global_designed_chain_list):
                temp_dict[chain] = [tied_list[j][i]] #needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list
else:
    my_dict = {}
    for json_str in json_list:
        result = json.loads(json_str)
        all_chain_list = sorted([item[-1:] for item in list(result) if item[:9]=='seq_chain']) #A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1,chain_length+1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i] #needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result['name']] = tied_positions_list

with open(args.output_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(my_dict) + '\n')
