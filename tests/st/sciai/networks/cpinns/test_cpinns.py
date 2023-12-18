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
"""test cpinns"""
import os
import re
import shlex
import subprocess
import sys

import yaml
import pytest
from mindspore import context
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub

from sciai.context import init_project
from sciai.model import AutoModel
from sciai.model.cpinns.eval import main as main_eval
from sciai.model.cpinns.train import main
from sciai.utils import parse_arg

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_21(mode):
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
    final_losses = re.findall(r"loss1: (.*?), loss2: (.*?), loss3: (.*?), loss4: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 5
    assert float(final_losses[1]) < 5
    assert float(final_losses[2]) < 5
    assert float(final_losses[3]) < 5
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
    load_data_path = config.get("load_data_path")
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--nn_depth 4 6 6 4 " \
          "--nn_width 20 20 20 20 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt True " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--load_data_path {load_data_path} " \
          f"--save_data_path ./data " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--ckpt_interval 1000 " \
          "--lr 8e-4 " \
          "--epochs 1 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"loss1: (.*?), loss2: (.*?), loss3: (.*?), loss4: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 5
    assert float(final_losses[1]) < 5
    assert float(final_losses[2]) < 5
    assert float(final_losses[3]) < 5


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
    final_losses = re.findall(r"loss1: (.*?), loss2: (.*?), loss3: (.*?), loss4: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 5
    assert float(final_losses[1]) < 5
    assert float(final_losses[2]) < 5
    assert float(final_losses[3]) < 5
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
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    args = parse_arg(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args)
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall("error_u: (.*)", outputs)[-1]
    assert float(final_losses) < 0.03
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_should_error_small_with_full_command_val():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    load_data_path = config.get("load_data_path")
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_val.py " \
          "--nn_depth 4 6 6 4 " \
          "--nn_width 20 20 20 20 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt True " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--load_data_path {load_data_path} " \
          f"--save_data_path ./data " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--ckpt_interval 1000 " \
          "--lr 8e-4 " \
          "--epochs 0 " \
          "--amp_level O3"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall("error_u: (.*)", outputs)[-1]
    assert float(final_losses) < 0.03


@pytest.mark.full_epoch
def test_full():
    """
    Feature: ALL TO ALL
    Description: test cases for cpinns
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)
    args = parse_arg(config)
    init_project(device_id=find_card(), args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"loss1: (.*?), loss2: (.*?), loss3: (.*?), loss4: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 0.0008
    assert float(final_losses[1]) < 0.003
    assert float(final_losses[2]) < 0.0005
    assert float(final_losses[3]) < 0.0005
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("cpinns")
    model.update_config(mode=mode, epochs=21, save_fig=False, save_ckpt=False,
                        load_data_path="./data/")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"loss1: (.*?), loss2: (.*?), loss3: (.*?), loss4: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 5
    assert float(final_losses[1]) < 5
    assert float(final_losses[2]) < 5
    assert float(final_losses[3]) < 5

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall("error_u: (.*)", outputs)[-1]
    assert float(final_losses) < 0.03

    clear_stub(stderr, stdout)
