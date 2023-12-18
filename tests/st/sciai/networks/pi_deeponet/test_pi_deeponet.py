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
"""test pi_deeponet"""
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
from sciai.model.pi_deeponet.train import main
from sciai.model.pi_deeponet.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_300(mode):
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
    losses = re.findall(r", total loss: (.*), ic_loss: (.*), bc_loss: (.*), res_loss: (.*), interval", outputs)[-1]
    assert float(losses[0]) < 0.123
    assert float(losses[1]) < 0.1
    assert float(losses[2]) < 0.01
    assert float(losses[3]) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_run_with_full_command(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--branch_layers 100 100 100 100 100 100 " \
          "--trunk_layers 2 100 100 100 100 100 " \
          "--save_ckpt false " \
          "--save_data false " \
          "--save_fig false " \
          "--load_ckpt true " \
          f"--save_data_path ./data " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--print_interval 100 " \
          "--lr 8e-4 " \
          "--epochs 1 " \
          "--n_train 1 " \
          "--batch_size 1 " \
          "--amp_level O3"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    losses = re.findall(r", total loss: (.*), ic_loss: (.*), bc_loss: (.*), res_loss: (.*), interval", outputs)[-1]
    assert float(losses[0]) < 0.1
    assert float(losses[1]) < 0.05
    assert float(losses[2]) < 0.01
    assert float(losses[3]) < 0.01


@pytest.mark.level0
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
    losses = re.findall(r", total loss: (.*), ic_loss: (.*), bc_loss: (.*), res_loss: (.*), interval", outputs)[-1]
    assert float(losses[0]) < 0.1
    assert float(losses[1]) < 0.05
    assert float(losses[2]) < 0.01
    assert float(losses[3]) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
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
    l2_error = re.findall(r"Relative l2 error: (.*)", outputs)[-1]
    assert float(l2_error) < 0.02
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("pi_deeponet")
    model.update_config(mode=mode, epochs=301, n_train=1, batch_size=1, save_fig=False, save_ckpt=False,
                        save_data=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    losses = re.findall(r", total loss: (.*), ic_loss: (.*), bc_loss: (.*), res_loss: (.*), interval", outputs)[-1]
    assert float(losses[0]) < 1
    assert float(losses[1]) < 1
    assert float(losses[2]) < 0.1
    assert float(losses[3]) < 0.1

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    losses = re.findall(r", total loss: (.*), ic_loss: (.*), bc_loss: (.*), res_loss: (.*), interval", outputs)[-1]
    assert float(losses[0]) < 1
    assert float(losses[1]) < 1
    assert float(losses[2]) < 0.1
    assert float(losses[3]) < 0.1

    clear_stub(stderr, stdout)
