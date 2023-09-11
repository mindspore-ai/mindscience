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

"""utilities"""
import argparse
import os

import yaml

from sciai.utils import parse_arg
from .allen_cahn_20d import AllenCahn20D
from .black_scholes_barenblatt_100_d import BlackScholesBarenblatt100D
from .problem import Problem


def prepare(problem=None):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    if problem is not None:
        config_dict["problem"] = problem
    args_, problem_ = generate_args(config_dict)
    return args_, problem_


def find_problem(config):
    """obtain the target problem"""
    parser = argparse.ArgumentParser(description=config.get("case"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "allen_cahn_20D"))
    args = parser.parse_known_args()
    return args[0].problem


def generate_args(config):
    """generate arguments for main"""
    common_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
    problem_name = find_problem(config)
    problem_config = config.get(problem_name)
    problem_config.update(common_config)
    args = parse_arg(problem_config)
    problem: Problem = {
        "allen_cahn_20D": AllenCahn20D,
        "black_scholes_barenblatt_100D": BlackScholesBarenblatt100D,
    }.get(problem_name, AllenCahn20D)(args)
    return args, problem
