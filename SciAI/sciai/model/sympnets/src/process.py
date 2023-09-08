"""process"""
import argparse
import os

import yaml

from sciai.utils import parse_arg
from .cases.double_pendulum import DoublePendulum
from .cases.pendulum import Pendulum
from .cases.problem import Problem
from .cases.three_body import ThreeBody


def prepare(problem=None):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    if problem is not None:
        config_dict["problem"] = problem
    args_, problem_ = generate_args(config_dict)
    return args_, problem_


def generate_args(config):
    """generate args"""
    common_config = {k: v for k, v in config.items() if not isinstance(v, dict)}
    problem_name = find_problem(config)
    problem_config = config.get(problem_name)
    problem_config.update(common_config)
    args = parse_arg(problem_config)
    problem: Problem = {
        "pendulum": Pendulum,
        "double_pendulum": DoublePendulum,
        "three_body": ThreeBody
    }.get(problem_name, Pendulum)()
    return args, problem


def find_problem(config):
    parser = argparse.ArgumentParser(description=config.get("case"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "pendulum"))
    args = parser.parse_known_args()
    return args[0].problem
