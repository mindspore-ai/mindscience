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
"""test phylstm"""
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
from sciai.model.phylstm.train import main
from sciai.model.phylstm.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_test_num_1(mode):
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
    losses = re.findall(r"Train_Loss:(.*)", outputs)[-1]
    assert float(losses) < 1e-2
    clear_stub(stderr, stdout)


@pytest.mark.env_onecard
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
    data_path = config.get("load_data_path")
    cmd = f"python mock_train.py " \
          "--print_interval 1 " \
          "--ckpt_interval 1 " \
          "--save_fig true " \
          "--save_ckpt true " \
          "--load_ckpt false " \
          "--save_ckpt_path  ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures " \
          f"--load_data_path {data_path} " \
          "--save_data_path ./data " \
          "--epochs 8000 " \
          "--lr 1e-4 " \
          "--mode train "
    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    losses = re.findall(r"Train_Loss:(.*)", outputs)[-1]
    assert float(losses) < 1e-2


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
    config["epochs"] = 8000
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    losses = re.findall(r"Train_Loss:(.*)", outputs)[-1]
    assert float(losses) < 1e-2
    clear_stub(stderr, stdout)


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
    correlation_score = re.findall(r"Mean correlation coefficient of z_1:(.*)", outputs)[-1]
    assert float(correlation_score) > 0.5
    clear_stub(stderr, stdout)


@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("phylstm")
    model.update_config(mode=mode, epochs=8000)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    losses = re.findall(r"Train_Loss:(.*)", outputs)[-1]
    assert float(losses) < 1e-2
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    correlation_score = re.findall(r"Mean correlation coefficient of z_1:(.*)", outputs)[-1]
    assert float(correlation_score) > 0.5
    clear_stub(stderr, stdout)
