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
"""test pinns_swe"""
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
from sciai.model.pinns_swe.train import main
from sciai.model.pinns_swe.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_10(mode):
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
    final_losses = re.findall(r"PDE_loss: (.*?), IC_loss: (.*?), interval", outputs)[-1]
    assert float(final_losses[0]) < 0.003
    assert float(final_losses[1]) < 0.01
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
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--layers 4 20 20 20 20 1 " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--save_fig false " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs 1 " \
          "--amp_level O3 " \
          "--n_pde 100000 " \
          "--n_iv 10000 " \
          "--u 1 " \
          "--h 1000 " \
          "--days 12"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"PDE_loss: (.*?), IC_loss: (.*?), interval", outputs)[-1]
    assert float(final_losses[0]) < 0.1
    assert float(final_losses[1]) < 0.2


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
    final_losses = re.findall(r"PDE_loss: (.*?), IC_loss: (.*?), interval", outputs)[-1]
    assert float(final_losses[0]) < 0.1
    assert float(final_losses[1]) < 0.2
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
    args = parse_arg(config)
    args.amp_level = "O0"
    init_project(mode=mode, args=args)
    main_eval(args)

    outputs = sys.stdout.getvalue().strip()
    loss_val = re.findall(r"validation loss: (.*)", outputs)[-1]
    assert float(loss_val) < 0.01
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
    final_losses = re.findall(r"PDE_loss: (.*?), IC_loss: (.*?), interval", outputs)[-1]
    assert float(final_losses[0]) < 0.003
    assert float(final_losses[1]) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("pinns_swe")
    model.update_config(mode=mode, epochs=10, save_fig=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"PDE_loss: (.*?), IC_loss: (.*?), interval", outputs)[-1]
    assert float(final_losses[0]) < 0.003
    assert float(final_losses[1]) < 0.01

    model.evaluate()
    model.update_config(amp_level="O0")
    outputs = sys.stdout.getvalue().strip()
    loss_val = re.findall(r"validation loss: (.*)", outputs)[-1]
    assert float(loss_val) < 0.01

    clear_stub(stderr, stdout)
