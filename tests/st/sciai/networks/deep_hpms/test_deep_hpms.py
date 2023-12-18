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
"""test deep hpms"""
import os
import sys
import re
import subprocess
import shlex
import yaml
import pytest

from mindspore import context
from tests.st.sciai.test_utils.func_utils import copy_dataset
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub

from sciai.context import init_project
from sciai.model import AutoModel
from sciai.model.deep_hpms.eval import main as main_eval
from sciai.model.deep_hpms.src.process import generate_args
from sciai.model.deep_hpms.train import main

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_burgers_should_run_with_full_command(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config_burgers = config.get("burgers_different")
    load_data_idn_path = config_burgers.get("load_data_idn_path")
    load_data_sol_path = config_burgers.get("load_data_sol_path")
    ckpt_path = config_burgers.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--problem burgers_different " \
          "--u_layers 2 50 50 50 50 1 " \
          "--pde_layers 3 100 100 1 " \
          "--layers 2 50 50 50 50 1 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--save_fig false " \
          "--figures_path ./figures " \
          f"--load_data_idn_path {load_data_idn_path} " \
          f"--load_data_sol_path {load_data_sol_path} " \
          "--log_path ./logs " \
          "--lr 0.001 " \
          "--train_epoch 201 " \
          "--train_epoch_lbfgs 0 " \
          "--print_interval 100 " \
          "--amp_level O3 " \
          "--lb_idn 0.0 -8.0 " \
          "--ub_idn 10.0 8.0 " \
          "--lb_sol 0.0 -8.0 " \
          "--ub_sol 10.0 8.0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss = re.findall(r".*loss: (.*), interval", outputs)[-1]
    assert float(loss) < 10


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_burgers_should_fast_enough_when_epoch_500(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "burgers_different"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r".*loss: (.*), interval", outputs)[-1]
    assert float(loss) < 10
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_burgers_should_fast_enough_when_load_ckpt(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "burgers_different"
    config["burgers_different"]["load_ckpt"] = True
    config["burgers_different"]["train_epoch"] = 201
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r".*loss: (.*), interval", outputs)[-1]
    assert float(loss) < 10
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_kdv_should_fast_enough_when_epoch_500(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "kdv_same"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r".*loss: (.*), interval", outputs)[-1]
    assert float(loss) < 12
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_kdv_should_fast_enough_when_load_ckpt(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "kdv_same"
    config["kdv_same"]["load_ckpt"] = True
    config["kdv_same"]["train_epoch"] = 201
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main(args, problem)

    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r".*loss: (.*), interval", outputs)[-1]
    assert float(loss) < 12
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_burgers_should_error_small_enough_when_val(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "burgers_different"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    error_u = re.findall(r"Error u: (.*)", outputs)[-1]
    assert float(error_u) < 0.3
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_kdv_should_error_small_enough_when_val(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    config["problem"] = "kdv_same"
    args, problem = generate_args(config)
    init_project(mode=mode, args=args)
    main_eval(args, problem)
    outputs = sys.stdout.getvalue().strip()
    error_u = re.findall(r"Error u: (.*)", outputs)[-1]
    error_u_idn = re.findall(r"Error u \(idn\): (.*)", outputs)[-1]
    assert float(error_u) < 0.3
    assert float(error_u_idn) < 0.2
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    # test burgers_different
    model = AutoModel.from_pretrained("deep_hpms")
    model.update_config(mode=mode, train_epoch=501, train_epoch_lbfgs=0, save_fig=False, save_ckpt=False,
                        load_data_idn_path=
                        "./data/burgers_sine.mat",
                        load_data_sol_path="./data/burgers.mat")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r".*loss: (.*), interval", outputs)[-1]
    assert float(loss) < 10
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_u = re.findall(r"Error u: (.*)", outputs)[-1]
    assert float(error_u) < 0.3

    # test kdv_same
    model = AutoModel.from_pretrained("deep_hpms", problem="kdv_same")
    model.update_config(mode=mode, train_epoch=501, train_epoch_lbfgs=0, save_fig=False, save_ckpt=False,
                        load_data_idn_path="./data/KdV_sine.mat",
                        load_data_sol_path="./data/KdV_sine.mat")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r".*loss: (.*), interval", outputs)[-1]
    assert float(loss) < 12
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_u = re.findall(r"Error u: (.*)", outputs)[-1]
    error_u_idn = re.findall(r"Error u \(idn\): (.*)", outputs)[-1]
    assert float(error_u) < 0.3
    assert float(error_u_idn) < 0.2

    clear_stub(stderr, stdout)
