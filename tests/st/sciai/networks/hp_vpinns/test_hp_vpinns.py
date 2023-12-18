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
"""test hp_vpinns"""
import os
import re
import sys
import shlex
import subprocess
import yaml
import pytest

import mindspore as ms
from mindspore import context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.hp_vpinns.train import main
from sciai.model.hp_vpinns.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
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
    loss_value = re.findall(r"loss: (.*), lossb: (.*), lossv: (.*), interval", outputs)[-1]
    assert float(loss_value[0]) < 200
    assert float(loss_value[1]) < 1000
    assert float(loss_value[2]) < 200
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
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--layers 1 20 20 20 20 1 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_fig false " \
          "--save_ckpt_path ./checkpoints " \
          "--figures_path ./figures " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs 1 " \
          "--early_stop_loss 2e-32 " \
          "--var_form 1 " \
          "--n_element 4 " \
          "--n_testfcn 60 " \
          "--n_quad 80 " \
          "--n_f 500 " \
          "--lossb_weight 1 " \
          "--amp_level O3 " \
          "--font 24"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = re.findall(r"loss: (.*), lossb: (.*), lossv: (.*), interval", outputs)[-1]
    assert float(loss_value[0]) < 200
    assert float(loss_value[1]) < 1000
    assert float(loss_value[2]) < 200


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
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
    loss_value = re.findall(r"loss: (.*), lossb: (.*), lossv: (.*), interval", outputs)[-1]
    assert float(loss_value[0]) < 200
    assert float(loss_value[1]) < 1000
    assert float(loss_value[2]) < 200
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
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
    error = re.findall(r"MSE: (.*)", outputs)[-1]
    assert float(error) < 5e-6
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
          "--layers 1 20 20 20 20 1 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_fig false " \
          "--save_ckpt_path ./checkpoints " \
          "--figures_path ./figures " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs 1 " \
          "--early_stop_loss 2e-32 " \
          "--var_form 1 " \
          "--n_element 4 " \
          "--n_testfcn 60 " \
          "--n_quad 80 " \
          "--n_f 500 " \
          "--lossb_weight 1 " \
          "--amp_level O3 " \
          "--font 24"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    error = re.findall(r"MSE: (.*)", outputs)[-1]
    assert float(error) < 5e-6


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
    init_project(mode=ms.PYNATIVE_MODE, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"loss: (.*), lossb: (.*), lossv: (.*), interval", outputs)[-1]
    assert float(loss_value[0]) < 0.00009
    assert float(loss_value[1]) < 0.00000003
    assert float(loss_value[2]) < 0.00009
    clear_stub(stderr, stdout)


def test_auto_model():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("hp_vpinns")
    model.update_config(mode=ms.PYNATIVE_MODE, epochs=11, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"loss: (.*), lossb: (.*), lossv: (.*), interval", outputs)[-1]
    assert float(loss_value[0]) < 200
    assert float(loss_value[1]) < 1000
    assert float(loss_value[2]) < 200

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error = re.findall(r"MSE: (.*)", outputs)[-1]
    assert float(error) < 5e-6

    clear_stub(stderr, stdout)
