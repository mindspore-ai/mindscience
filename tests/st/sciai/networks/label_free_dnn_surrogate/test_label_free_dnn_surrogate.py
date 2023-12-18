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
"""test label free dnn surrogate"""
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
from sciai.model.label_free_dnn_surrogate.train import main
from sciai.model.label_free_dnn_surrogate.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_1(mode):
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
    assert loss_value < 0.05
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    ckpt_path_u = config.get("load_ckpt_path")[0]
    ckpt_path_v = config.get("load_ckpt_path")[1]
    ckpt_path_p = config.get("load_ckpt_path")[2]
    load_data_path = config.get("load_data_path")
    cmd = f"python mock_train.py " \
          "--layers 3 20 20 20 1 " \
          "--save_ckpt false " \
          "--save_fig  false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          "--load_ckpt_path " \
          f"{ckpt_path_u} {ckpt_path_v} {ckpt_path_p} " \
          f"--load_data_path {load_data_path} " \
          f"--save_data_path ./data " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs_train 1 " \
          "--epochs_val 400 " \
          "--batch_size 50 " \
          "--print_interval 100 " \
          "--nu 1e-3 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.05


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
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.05
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    args = parse_arg(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 5e-7
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_error_small_enough_with_full_command_val():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path_u = config.get("load_ckpt_path")[0]
    ckpt_path_v = config.get("load_ckpt_path")[1]
    ckpt_path_p = config.get("load_ckpt_path")[2]
    load_data_path = config.get("load_data_path")
    cmd = f"python mock_val.py " \
          "--layers 3 20 20 20 1 " \
          "--save_ckpt false " \
          "--save_fig  false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          "--load_ckpt_path " \
          f"{ckpt_path_u} {ckpt_path_v} {ckpt_path_p} " \
          f"--load_data_path {load_data_path} " \
          f"--save_data_path ./data " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs_train 1 " \
          "--epochs_val 400 " \
          "--batch_size 50 " \
          "--print_interval 100 " \
          "--nu 1e-3 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 5e-7


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
    assert loss_value < 0.00009
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("label_free_dnn_surrogate")
    model.update_config(mode=mode, epochs_train=1, batch_size=1000, save_fig=False, save_ckpt=False,
                        load_data_path="./data")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.05

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"MSE: (.*)", outputs)[-1])
    assert loss_value < 5e-7

    clear_stub(stderr, stdout)
