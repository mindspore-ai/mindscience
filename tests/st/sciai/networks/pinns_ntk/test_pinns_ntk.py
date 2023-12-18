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
"""test pinns_ntk"""
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
from sciai.model.pinns_ntk.train import main
from sciai.model.pinns_ntk.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_4001(mode):
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
    final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 50
    assert float(final_losses[1]) < 0.1
    assert float(final_losses[2]) < 50
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
          "--layers 1 512 1 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--num 100 " \
          "--lr 1e-4 " \
          "--epochs 1 " \
          "--amp_level O2"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 400
    assert float(final_losses[1]) < 300
    assert float(final_losses[2]) < 300


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 400
    assert float(final_losses[1]) < 300
    assert float(final_losses[2]) < 300
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_val_should_error_small_enough(mode):
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
    main_eval(args)

    outputs = sys.stdout.getvalue().strip()
    error_u = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    error_r = re.findall(r"Relative L2 error_r: (.*)", outputs)[-1]
    assert float(error_u) < 0.007
    assert float(error_r) < 0.01
    clear_stub(stderr, stdout)


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
    final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 0.002
    assert float(final_losses[1]) < 0.008
    assert float(final_losses[2]) < 0.005
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("pinns_ntk")
    model.update_config(mode=mode, epochs=4001, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 50
    assert float(final_losses[1]) < 0.1
    assert float(final_losses[2]) < 50

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_u = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    error_r = re.findall(r"Relative L2 error_r: (.*)", outputs)[-1]
    assert float(error_u) < 0.007
    assert float(error_r) < 0.01

    clear_stub(stderr, stdout)
