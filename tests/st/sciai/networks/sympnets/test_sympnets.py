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
"""test sympnets"""
import os
import re
import sys
import shlex
import subprocess
import yaml
import pytest

from mindspore import context
from sciai.context import init_project
from sciai.model.sympnets.train import main
from sciai.model.sympnets.eval import main as main_eval
from sciai.model.sympnets.src.process import generate_args
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_5_pendulum(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "pendulum"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_5_double_pendulum(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "double_pendulum"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.3
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_comb_should_run_with_full_command_three_body():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_three_body = config.get("three_body")
    ckpt_path = config_three_body.get("load_ckpt_path")
    data_path = config_three_body.get("save_data_path")
    cmd = f"python mock_train.py " \
          "--problem three_body " \
          "--layers 2 50 50 50 50 3 " \
          "--save_ckpt false " \
          "--save_data false " \
          "--save_fig false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--save_data_path {data_path} " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--print_interval 1 " \
          "--lr 1e-3 " \
          "--batch_size 0 " \
          "--epochs 1 " \
          "--amp_level O3 " \
          "--net_type G " \
          "--la_layers 20 " \
          "--la_sublayers 4 " \
          "--g_layers 20 " \
          "--g_width 50 " \
          "--activation sigmoid " \
          "--h_layers 6 " \
          "--h_width 50 " \
          "--h_activation tanh"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_5_three_body(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "three_body"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.07
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_pendulum(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "pendulum"
    config["pendulum"]["load_ckpt"] = True
    config["pendulum"]["epochs"] = 1
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_double_pendulum(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "double_pendulum"
    config["double_pendulum"]["load_ckpt"] = True
    config["double_pendulum"]["epochs"] = 1
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_three_body(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "three_body"
    config["three_body"]["load_ckpt"] = True
    config["three_body"]["epochs"] = 0
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.1
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_val_should_loss_small_enough(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """

    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    def run_and_test(case, accuracy):
        config["problem"] = case
        args, problem = generate_args(config)
        init_project(mode=mode, args=args)
        main_eval(args, problem)

        outputs = sys.stdout.getvalue().strip()
        loss_val = re.findall(r"validation loss: (.*)", outputs)[-1]
        assert float(loss_val) < accuracy

    run_and_test("pendulum", 3e-7)
    run_and_test("double_pendulum", 2e-4)
    run_and_test("three_body", 5e-4)
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

    config["problem"] = "pendulum"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.0000000006

    config["problem"] = "double_pendulum"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.00003

    config["problem"] = "three_body"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.0003

    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    # test pendulum
    model = AutoModel.from_pretrained("sympnets")
    model.update_config(mode=mode, epochs=5, save_fig=False, save_ckpt=False, save_data=False, print_interval=1)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.01
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_val = re.findall(r"validation loss: (.*)", outputs)[-1]
    assert float(loss_val) < 1e-9

    # test double_pendulum
    model = AutoModel.from_pretrained("sympnets", problem="double_pendulum")
    model.update_config(mode=mode, epochs=5, save_fig=False, save_ckpt=False, save_data=False, print_interval=1)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.3
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_val = re.findall(r"validation loss: (.*)", outputs)[-1]
    assert float(loss_val) < 1e-4

    # test three_body
    model = AutoModel.from_pretrained("sympnets", problem="three_body")
    model.update_config(mode=mode, epochs=5, save_fig=False, save_ckpt=False, save_data=False, print_interval=1)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.07
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_val = re.findall(r"validation loss: (.*)", outputs)[-1]
    assert float(loss_val) < 1e-4

    clear_stub(stderr, stdout)
