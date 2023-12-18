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
"""test fpinns"""
import os
import re
import sys
import shlex
import subprocess
import yaml
import pytest

from mindspore import context
from sciai.context import init_project
from sciai.model.fpinns.src.process import generate_args
from sciai.model.fpinns.train import main
from sciai.model.fpinns.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_iter_300_diffusion(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "diffusion_1d"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_iter_300_fractional_diffusion(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "fractional_diffusion_1d"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_run_with_full_command_fractional_diffusion():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_fractional_diffusion = config.get("fractional_diffusion_1d")
    ckpt_path = config_fractional_diffusion.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--problem fractional_diffusion_1d " \
          "--layers 2 20 20 20 20 1 " \
          "--x_range 0 1 " \
          "--t_range 0 1 " \
          "--num_domain 400 " \
          "--num_boundary 0 " \
          "--num_initial 0 " \
          "--num_test 400 " \
          "--lr 1e-3 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints/fractional_diffusion_1d " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures/fractional_diffusion_1d " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--epochs 1 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.5


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_diffusion(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "diffusion_1d"
    config["diffusion_1d"]["load_ckpt"] = True
    config["diffusion_1d"]["epochs"] = 1
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_fractional_diffusion(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "fractional_diffusion_1d"
    config["fractional_diffusion_1d"]["load_ckpt"] = True
    config["fractional_diffusion_1d"]["epochs"] = 1
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val_fractional_diffusion(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    config_dict["problem"] = "fractional_diffusion_1d"
    args, problem = generate_args(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"MSE:(.*)", outputs)[-1]
    assert float(loss_value) < 1e-9
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val_diffusion(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    config_dict["problem"] = "diffusion_1d"
    args, problem = generate_args(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"MSE:(.*)", outputs)[-1]
    assert float(loss_value) < 0.0007
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_error_small_enough_with_full_command_fractional_diffusion():
    """
        Feature: ALL TO ALL
        Description:  test cases for
        Expectation: pass
        """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_fractional_diffusion = config.get("fractional_diffusion_1d")
    ckpt_path = config_fractional_diffusion.get("load_ckpt_path")
    cmd = f"python mock_val.py " \
          "--problem fractional_diffusion_1d " \
          "--layers 2 20 20 20 20 1 " \
          "--x_range 0 1 " \
          "--t_range 0 1 " \
          "--num_domain 400 " \
          "--num_boundary 0 " \
          "--num_initial 0 " \
          "--num_test 400 " \
          "--lr 1e-3 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints/fractional_diffusion_1d " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures/fractional_diffusion_1d " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--epochs 1 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = re.findall(r"MSE:(.*)", outputs)[-1]
    assert float(loss_value) < 1e-9


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

    config["problem"] = "diffusion_1d"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.0003

    config["problem"] = "fractional_diffusion_1d"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.000006

    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    # test fractional_diffusion_1d
    model = AutoModel.from_pretrained("fpinns")
    model.update_config(mode=mode, epochs=300, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.5
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"MSE:(.*)", outputs)[-1]
    assert float(loss_value) < 1e-9

    # test diffusion_1d
    model = AutoModel.from_pretrained("fpinns", problem="diffusion_1d")
    model.update_config(mode=mode, epochs=300, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.5
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"MSE:(.*)", outputs)[-1]
    assert float(loss_value) < 0.0007

    clear_stub(stderr, stdout)
