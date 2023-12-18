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
"""test pinn_helmholtz"""
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
from sciai.model.pinn_helmholtz.train import main
from sciai.model.pinn_helmholtz.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_100(mode):
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
    final_loss = re.findall(r"loss: (.*), interval: (.*),", outputs)[-1]
    assert float(final_loss[0]) < 30
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
    data_path = config.get("load_data_path")
    cmd = f"python mock_train.py " \
          "--layers 2 40 40 40 40 40 40 40 40 2 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--load_data_path {data_path} " \
          "--save_fig false " \
          "--figures_path ./figures " \
          "--save_results false " \
          "--results_path ./data/results " \
          "--print_interval 20 " \
          "--log_path ./logs " \
          "--lr 0.001 " \
          "--epochs 1 " \
          "--num_batch 1 " \
          "--lbfgs false " \
          "--epochs_lbfgs 50000 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_loss = re.findall(r"loss: (.*), interval: (.*),", outputs)[-1]
    assert float(final_loss[0]) < 30


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
    config["epochs"] = 1
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    final_loss = re.findall(r"loss: (.*), interval: (.*),", outputs)[-1]
    assert float(final_loss[0]) < 30
    clear_stub(stderr, stdout)


@pytest.mark.full_epoch
def test_full():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    print()
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)
    args = parse_arg(config)
    init_project(device_id=find_card(), args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    final_loss = re.findall(r"loss: (.*), interval: (.*),", outputs)[-1]
    assert float(final_loss[0]) < 0.8
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["load_ckpt"] = True
    config["epochs"] = 1
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main_eval(args)
    outputs = sys.stdout.getvalue().strip()
    error_real, error_imag = re.findall(r"Error u_real: (.*), Error u_imag: (.*)", outputs)[-1]
    assert float(error_real) < 0.8
    assert float(error_imag) < 0.8
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("pinn_helmholtz")
    model.update_config(mode=mode, epochs=100, epochs_lbfgs=0, save_fig=False, save_ckpt=False, save_results=False,
                        load_data_path="./data")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_loss = re.findall(r"loss: (.*), interval: (.*),", outputs)[-1]
    assert float(final_loss[0]) < 30

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_real, error_imag = re.findall(r"Error u_real: (.*), Error u_imag: (.*)", outputs)[-1]
    assert float(error_real) < 0.8
    assert float(error_imag) < 0.8

    clear_stub(stderr, stdout)
