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
"""test deep ritz"""
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
from sciai.model.deep_ritz.eval import main as main_eval
from sciai.model.deep_ritz.src.utils import generate_args
from sciai.model.deep_ritz.train import main

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_hole_should_run_with_full_command():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_hole = config.get("poisson_hole")
    data_path = config_hole.get("save_data_path")
    ckpt_path = config_hole.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--layers 2 8 8 8 1 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints/hole " \
          f"--load_ckpt_path {ckpt_path} " \
          "--save_fig false " \
          "--figures_path ./figures " \
          "--save_data false " \
          f"--save_data_path {data_path} " \
          "--log_path ./logs " \
          "--lr 0.01 " \
          "--train_epoch 1 " \
          "--train_epoch_pre 0 " \
          "--body_batch 1024 " \
          "--bdry_batch 1024 " \
          "--write_step 50 " \
          "--sample_step 10 " \
          "--step_size 5000 " \
          "--num_quad 40000 " \
          "--radius 1 " \
          "--penalty 500 " \
          "--diff 0.001 " \
          "--gamma 0.3 " \
          "--decay 0.00001 " \
          "--amp_level O2"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]

    assert float(loss) < 1.5
    assert float(error) < 0.05


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_hole_should_fast_enough_when_epoch_2000(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "poisson_hole"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]

    assert float(loss) < 1.5
    assert float(error) < 0.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_ls_should_fast_enough_when_epoch_2000(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "poisson_ls"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]

    assert float(loss) < 5.0
    assert float(error) < 0.1
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_hole_should_fast_enough_when_load_ckpt(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "poisson_hole"
    config["poisson_hole"]["load_ckpt"] = True
    config["poisson_hole"]["train_epoch"] = 1
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]

    assert float(loss) < 1.0
    assert float(error) < 0.3
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_ls_should_fast_enough_when_load_ckpt(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "poisson_ls"
    config["poisson_ls"]["load_ckpt"] = True
    config["poisson_ls"]["train_epoch"] = 1
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]

    assert float(loss) < 1.5
    assert float(error) < 0.05
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_hole_should_error_small_enough_when_val(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "poisson_hole"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"The test error (.*) is (.*).", outputs)[-1]
    assert float(final_losses[1]) < 0.005
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_ls_should_error_small_enough_when_val(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "poisson_ls"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"The test error (.*) is (.*).", outputs)[-1]
    assert float(final_losses[1]) < 0.005
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_hole_error_small_with_full_command_val():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_hole = config.get("poisson_hole")
    data_path = config_hole.get("save_data_path")
    ckpt_path = config_hole.get("load_ckpt_path")
    cmd = f"python mock_val.py " \
          "--layers 2 8 8 8 1 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints/hole " \
          f"--load_ckpt_path {ckpt_path} " \
          "--save_fig false " \
          "--figures_path ./figures " \
          "--save_data false " \
          f"--save_data_path {data_path} " \
          "--log_path ./logs " \
          "--lr 0.01 " \
          "--train_epoch 0 " \
          "--train_epoch_pre 0 " \
          "--body_batch 1024 " \
          "--bdry_batch 1024 " \
          "--write_step 50 " \
          "--sample_step 10 " \
          "--step_size 5000 " \
          "--num_quad 40000 " \
          "--radius 1 " \
          "--penalty 500 " \
          "--diff 0.001 " \
          "--gamma 0.3 " \
          "--decay 0.00001 " \
          "--amp_level O2"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"The test error (.*) is (.*).", outputs)[-1]
    assert float(final_losses[1]) < 0.005


@pytest.mark.full_epoch
def test_full():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config.yam") as f:
        config = yaml.safe_load(f)

    config["problem"] = "poisson_hole"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 0.4
    assert float(error) < 0.004

    config["problem"] = "poisson_ls"
    args, problem = generate_args(config)
    init_project(device_id=find_card(), args=args)
    main(args, problem)
    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 0.03
    assert float(error) < 0.005

    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    # test poisson_hole
    model = AutoModel.from_pretrained("deep_ritz")
    model.update_config(mode=mode, train_epoch=2000, save_fig=False, save_ckpt=False, save_data=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 1.5
    assert float(error) < 0.5
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"The test error (.*) is (.*).", outputs)[-1]
    assert float(final_losses[1]) < 0.005

    # test poisson_ls
    model = AutoModel.from_pretrained("deep_ritz", problem="poisson_ls")
    model.update_config(mode=mode, train_epoch=2000, save_fig=False, save_ckpt=False, save_data=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 5.0
    assert float(error) < 0.1
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"The test error (.*) is (.*).", outputs)[-1]
    assert float(final_losses[1]) < 0.005

    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model_finetune(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_hole = config.get("poisson_hole")
    ckpt_path = config_hole.get("load_ckpt_path")

    model = AutoModel.from_pretrained("deep_ritz")
    model.update_config(mode=mode, train_epoch=500, save_fig=False, save_ckpt=False, save_data=False)
    model.finetune(load_ckpt_path=ckpt_path)

    outputs = sys.stdout.getvalue().strip()
    loss, error, _, _ = re.findall(r".*loss: (.*), error: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 1.5
    assert float(error) < 0.5

    clear_stub(stderr, stdout)
