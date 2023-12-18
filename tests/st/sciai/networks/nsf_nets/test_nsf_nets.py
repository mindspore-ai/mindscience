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
"""test nsf nets"""
import os
import re
import sys
import shlex
import subprocess
import yaml
import pytest

from mindspore import context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.nsf_nets.train import main
from sciai.model.nsf_nets.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_5(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_should_run_with_full_command():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--layers 2 50 50 50 50 3 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--n_train 2601 " \
          "--n_bound 100 " \
          "--lr 1e-3 1e-4 1e-5 1e-6 " \
          "--epochs 1 " \
          "--amp_level O2"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["load_ckpt"] = True
    config["epochs"] = [1]
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    args = parse_arg(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args)
    outputs = sys.stdout.getvalue().strip()
    error_u = float(re.findall(r"Error u: (.*)", outputs)[-1])
    error_v = float(re.findall(r"Error v: (.*)", outputs)[-1])
    error_p = float(re.findall(r"Error p: (.*)", outputs)[-1])
    assert error_u < 0.003
    assert error_v < 0.01
    assert error_p < 8
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_should_error_small_enough_with_full_command_val():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_val.py " \
          "--layers 2 50 50 50 50 3 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--n_train 2601 " \
          "--n_bound 100 " \
          "--lr 1e-3 1e-4 1e-5 1e-6 " \
          "--epochs 1 " \
          "--amp_level O2"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    error_u = float(re.findall(r"Error u: (.*)", outputs)[-1])
    error_v = float(re.findall(r"Error v: (.*)", outputs)[-1])
    error_p = float(re.findall(r"Error p: (.*)", outputs)[-1])
    assert error_u < 0.003
    assert error_v < 0.01
    assert error_p < 8


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
    args = parse_arg(config)
    init_project(device_id=find_card(), args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.00003
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("nsf_nets")
    model.update_config(mode=mode, epochs=[500], lr=[1e-3], save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_u = float(re.findall(r"Error u: (.*)", outputs)[-1])
    error_v = float(re.findall(r"Error v: (.*)", outputs)[-1])
    error_p = float(re.findall(r"Error p: (.*)", outputs)[-1])
    assert error_u < 0.003
    assert error_v < 0.01
    assert error_p < 8

    clear_stub(stderr, stdout)
