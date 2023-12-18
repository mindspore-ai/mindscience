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
"""test deeponet"""
import os
import re
import sys
import shlex
import subprocess
import yaml
import pytest

from mindspore import context
from sciai.context import init_project
from sciai.model.deeponet.train import main
from sciai.model.deeponet.src.process import generate_args
from sciai.model.deeponet.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.func_utils import copy_dataset
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_1_1d_caputo(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    args = generate_args(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 0.05
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.skip(reason="need to update checkpoints")
def test_comb_should_loss_small_enough_when_load_ckpt_1d_caputo(mode):
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
    args = generate_args(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 1.5e-4
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.skip(reason="need to update checkpoints")
def test_comb_should_error_small_enough_when_val_1d_caputo(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    args = generate_args(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"Validation loss:(.*)", outputs)[-1]
    assert float(loss_value) < 1.5e-4
    clear_stub(stderr, stdout)


@pytest.mark.skip(reason="need to update checkpoints")
def test_comb_should_error_small_enough_with_full_command_val_1d_caputo():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    cmd = "python mock_eval.py " \
          "--layers_u 15 80 80 80 " \
          "--layers_y 2 80 80 80 " \
          "--save_ckpt false " \
          "--load_ckpt false " \
          "--save_ckpt_path ./checkpoints/1d_caputo " \
          "--load_ckpt_path ./checkpoints/1d_caputo/1d_caputo.ckpt " \
          "--save_fig false " \
          "--figures_path ./figures/1d_caputo " \
          "--save_data false " \
          "--load_data_path ./data/1d_caputo " \
          "--save_data_path ./data/1d_caputo " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs 51 " \
          "--batch_size 50 " \
          "--print_interval 10 " \
          "--download_data deeponet " \
          "--force_download false " \
          "--amp_level O3"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = re.findall(r"Validation loss:(.*)", outputs)[-1]
    assert float(loss_value) < 1.5e-4


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    # test 1d_caputo
    model = AutoModel.from_pretrained("deeponet")
    model.update_config(mode=mode, epochs=51, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 0.05

    clear_stub(stderr, stdout)
