"""utilities"""
import argparse
import os
import yaml

from sciai.utils import parse_arg


def prepare(problem=None):
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    if problem is not None:
        config_dict["problem"] = problem
    args_ = generate_args(config_dict)
    return args_


def generate_args(config):
    """generate args"""
    problem_name = find_problem(config)
    problem_config = config.get(problem_name)
    problem_config["problem"] = problem_name
    args = parse_arg(problem_config)
    return args


def find_problem(config):
    """find problem"""
    parser = argparse.ArgumentParser(description=config.get("case"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "Schrodinger"))
    args = parser.parse_known_args()
    return args[0].problem
