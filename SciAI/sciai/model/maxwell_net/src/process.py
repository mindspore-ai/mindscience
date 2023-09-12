
# Copyright 2023 Huawei Technologies Co., Ltd
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

"""maxwell net process"""
import argparse
import os

import yaml
import numpy as np

from sciai.utils import parse_arg, to_tensor


def prepare(problem=None):
    """prepare"""
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    if problem is not None:
        config_dict["problem"] = problem
    args_ = generate_args(config_dict)
    return (args_,)


def generate_args(config):
    """generate arguments"""
    common_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
    problem_name = find_problem(config)
    problem_config = config.get(problem_name)
    problem_config.update(common_config)
    args = parse_arg(problem_config)
    return args


def find_problem(config):
    """find problem according to config case"""
    parser = argparse.ArgumentParser(description=config.get("problem"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "tm"))
    args = parser.parse_known_args()
    return args[0].problem


def load_data(args, dtype):
    """load data"""
    data = np.load(os.path.join(args.load_data_path, "train.npz"))
    scat_pot_ms, ri_value_ms = to_tensor((data['sample'], data['n'].reshape(1)), dtype)
    return scat_pot_ms, ri_value_ms
