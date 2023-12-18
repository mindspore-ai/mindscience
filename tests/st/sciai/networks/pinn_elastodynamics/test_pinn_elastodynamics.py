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
"""test pinn_elastodynamics"""
import os
import re
import shlex
import subprocess
import sys

import yaml
import pytest
from mindspore import context

from sciai.context import init_project
from sciai.model import AutoModel
from sciai.model.pinn_elastodynamics.eval import main as main_eval
from sciai.model.pinn_elastodynamics.train import main
from sciai.utils import parse_arg
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_2(mode):
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
    final_losses = re.findall(r"loss: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 300
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
    data_path = config.get("load_data_path")
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--uv_layers 3 140 140 140 140 140 140 7 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_data_path {data_path} " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures/output " \
          "--log_path ./logs " \
          "--print_interval 1 " \
          "--ckpt_interval 1000 " \
          "--lr 1e-3 " \
          "--epochs 1 " \
          "--amp_level O0 " \
          "--use_lbfgs false " \
          "--max_iter_lbfgs 1"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"loss: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 300


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
    final_losses = re.findall(r"loss: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 300
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
    final_losses = re.findall(r"loss: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 0.00009
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
    loss = re.findall(r"loss: (.*)", outputs)[-1]
    loss_f_uv = re.findall(r"loss_f_uv: (.*)", outputs)[-1]
    loss_f_s = re.findall(r"loss_f_s: (.*)", outputs)[-1]
    loss_src = re.findall(r"loss_src: (.*)", outputs)[-1]
    loss_ic = re.findall(r"loss_ic: (.*)", outputs)[-1]
    loss_fix = re.findall(r"loss_fix: (.*)", outputs)[-1]
    assert float(loss_f_uv) < 1e-5
    assert float(loss_f_s) < 1e-5
    assert float(loss_src) < 1e-6
    assert float(loss_ic) < 1e-7
    assert float(loss_fix) < 1e-5
    assert float(loss) < 1e-4
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("pinn_elastodynamics")
    model.update_config(mode=mode, epochs=2, print_interval=1, save_fig=False, save_ckpt=False,
                        load_data_path="./data")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"loss: (.*?),", outputs)[-1]
    assert float(final_losses[0]) < 300

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss = re.findall(r"loss: (.*)", outputs)[-1]
    loss_f_uv = re.findall(r"loss_f_uv: (.*)", outputs)[-1]
    loss_f_s = re.findall(r"loss_f_s: (.*)", outputs)[-1]
    loss_src = re.findall(r"loss_src: (.*)", outputs)[-1]
    loss_ic = re.findall(r"loss_ic: (.*)", outputs)[-1]
    loss_fix = re.findall(r"loss_fix: (.*)", outputs)[-1]
    assert float(loss_f_uv) < 1e-5
    assert float(loss_f_s) < 1e-5
    assert float(loss_src) < 1e-6
    assert float(loss_ic) < 1e-7
    assert float(loss_fix) < 1e-5
    assert float(loss) < 1e-4

    clear_stub(stderr, stdout)
