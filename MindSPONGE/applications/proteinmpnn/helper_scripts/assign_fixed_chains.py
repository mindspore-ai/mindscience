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
Assign fixed chains for each pdb in the parsed PDBs folder.
"""

import argparse
import json

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--input_path", type=str, help="Path to the parsed PDBs")
argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
argparser.add_argument("--chain_list", type=str, default='', help="List of the chains that need to be designed")

args = argparser.parse_args()

with open(args.input_path, 'r', encoding='utf-8') as json_file:
    json_list = list(json_file)

global_designed_chain_list = []
if args.chain_list != '':
    global_designed_chain_list = [str(item) for item in args.chain_list.split()]
my_dict = {}
for json_str in json_list:
    result = json.loads(json_str)
    all_chain_list = [item[-1:] for item in list(result) if item[:9]=='seq_chain'] #['A','B', 'C',...]
    if len(global_designed_chain_list) > 0:
        designed_chain_list = global_designed_chain_list
    else:
        #manually specify, e.g.
        designed_chain_list = ["A"]
    fixed_chain_list = [letter for letter in all_chain_list if letter not in designed_chain_list] #fix/do not redesign these chains
    my_dict[result['name']]= (designed_chain_list, fixed_chain_list)

with open(args.output_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(my_dict) + '\n')

# Output looks like this:
# {"5TTA": [["A"], ["B"]], "3LIS": [["A"], ["B"]]}
