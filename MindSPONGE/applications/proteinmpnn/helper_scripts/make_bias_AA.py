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
Make a bias dictionary for the AAs to be biased.
"""

import argparse
import json

argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument("--output_path", type=str, help="Path to the output dictionary")
argparser.add_argument("--AA_list", type=str, default='', help="List of AAs to be biased")
argparser.add_argument("--bias_list", type=str, default='', help="AA bias strengths")

args = argparser.parse_args()

bias_list = [float(item) for item in args.bias_list.split()]
AA_list = [str(item) for item in args.AA_list.split()]

my_dict = dict(zip(AA_list, bias_list))

with open(args.output_path, 'w', encoding='utf-8') as f:
    f.write(json.dumps(my_dict) + '\n')

#e.g. output
#{"A": -0.01, "G": 0.02}
