"""deeponet process"""
import argparse
import os

import yaml
import numpy as np

from sciai.utils import parse_arg


def save_loss_data(data_path, ii, losst, lossv, lossv0):
    """save loss data"""
    np.savetxt(data_path + "/loss_train.txt", losst)
    np.savetxt(data_path + "/loss_test.txt", lossv)
    np.savetxt(data_path + "/loss-test0.txt", lossv0)
    np.savetxt(data_path + "/ii.txt", ii)


def load_data(data_path):
    """load data"""
    data = np.load(data_path + "/train.npz")
    x_u_train, x_y_train, y_train = data["X_u_train"], data["X_y_train"], data["Y_train"]
    data = np.load(data_path + "/test.npz")
    x_u_test, x_y_test, y_test = data["X_u_test"], data["X_y_test"], data["Y_test"]
    data = np.load(data_path + "/test0.npz")
    x_u_test0, x_y_test0, y_test0 = data["X_u_test"], data["X_y_test"], data["Y_test"]
    feed_train = x_u_train, x_y_train, y_train
    feed_test = x_u_test, x_y_test, y_test
    feed_test0 = x_u_test0, x_y_test0, y_test0
    return feed_train, feed_test, feed_test0


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
    """find problem"""
    parser = argparse.ArgumentParser(description=config.get("case"))
    parser.add_argument(f'--problem', type=str, default=config.get("problem", "1d_caputo"))
    args = parser.parse_known_args()
    return args[0].problem
