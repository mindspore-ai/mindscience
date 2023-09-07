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
