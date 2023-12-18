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
"""test fbsnns"""
import os
import re
import sys
import shlex
import subprocess
import yaml
import pytest

from mindspore import context
from sciai.context import init_project
from sciai.model.fbsnns.src.process import generate_args
from sciai.model.fbsnns.train import main
from sciai.model.fbsnns.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_1000_allen(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "allen_cahn_20D"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_allen(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "allen_cahn_20D"
    config["allen_cahn_20D"]["load_ckpt"] = True
    config["allen_cahn_20D"]["epochs"] = [1]
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_run_with_full_command_allen():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_allen = config.get("allen_cahn_20D")
    load_ckpt_path = config_allen.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--problem allen_cahn_20D " \
          "--layers 21 256 256 256 256 1 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_fig false " \
          "--save_ckpt_path ./checkpoints/ac " \
          f"--load_ckpt_path {load_ckpt_path} " \
          "--figures_path ./figures " \
          "--lr 1e-3 1e-4 1e-5 1e-6 " \
          "--epochs 1 " \
          "--amp_level O3 " \
          "--batch_size 100 " \
          "--num_snapshots 15 " \
          "--terminal_time 0.3 " \
          "--log_path ./logs"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_1000_double_black_scholes(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "black_scholes_barenblatt_100D"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 50
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_double_black_scholes(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "black_scholes_barenblatt_100D"
    config["black_scholes_barenblatt_100D"]["load_ckpt"] = True
    config["black_scholes_barenblatt_100D"]["epochs"] = [1]
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 50
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val_allen(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    config_dict["problem"] = "allen_cahn_20D"
    args, problem = generate_args(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 1e-5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val_double_black_scholes(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    config_dict["problem"] = "black_scholes_barenblatt_100D"
    args, problem = generate_args(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 0.3
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_error_small_enough_with_full_command_allen():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_allen = config.get("allen_cahn_20D")
    load_ckpt_path = config_allen.get("load_ckpt_path")
    cmd = f"python mock_val.py " \
          "--problem allen_cahn_20D " \
          "--layers 21 256 256 256 256 1 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_fig false " \
          "--save_ckpt_path ./checkpoints/ac " \
          f"--load_ckpt_path {load_ckpt_path} " \
          "--figures_path ./figures " \
          "--lr 1e-3 1e-4 1e-5 1e-6 " \
          "--epochs 1 " \
          "--amp_level O3 " \
          "--batch_size 100 " \
          "--num_snapshots 15 " \
          "--terminal_time 0.3 " \
          "--log_path ./logs"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 1e-5


@pytest.mark.full_epoch
def test_full():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "allen_cahn_20D"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.0001

    config["problem"] = "black_scholes_barenblatt_100D"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.2

    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    # test allen_cahn_20D
    model = AutoModel.from_pretrained("fbsnns")
    model.update_config(mode=mode, epochs=[1001], lr=[1e-3], save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 10
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 0.5

    # test black_scholes_barenblatt_100D
    model = AutoModel.from_pretrained("fbsnns", problem="allen_cahn_20D")
    model.update_config(mode=mode, epochs=[1001], lr=[1e-3], save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 50
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 0.3

    clear_stub(stderr, stdout)
