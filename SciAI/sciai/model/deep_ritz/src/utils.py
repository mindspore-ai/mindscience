"""utilities"""
import argparse
import os

import yaml

from sciai.utils import parse_arg
from .network import Problem
from .poisson_hole import PoissonHole
from .poisson_ls import PoissonLs


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
    problem: Problem = {"poisson_hole": PoissonHole, "poisson_ls": PoissonLs}.get(problem_name, PoissonHole)(args)
    return args, problem


def find_problem(config):
    """find problem"""
    parser = argparse.ArgumentParser(description=config.get("case"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "poisson_hole"))
    args = parser.parse_known_args()
    return args[0].problem
