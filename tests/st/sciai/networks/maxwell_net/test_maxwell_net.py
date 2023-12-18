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
from sciai.model.maxwell_net.train import main
from sciai.model.maxwell_net.src.process import generate_args
from sciai.model.maxwell_net.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_2200_tm(mode):
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
    assert loss_value < 1
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_2200_te(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "te"
    args = generate_args(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 1
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_run_with_full_command_tm():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    cmd = f"python train.py " \
          "--in_channels 2 " \
          "--out_channels 4 " \
          "--depth 6 " \
          "--filter 32 " \
          "--norm weight " \
          "--up_mode upconv " \
          "--wavelength 1 " \
          "--dpl 20 " \
          "--nx 160 " \
          "--nz 192 " \
          "--pml_thickness 30 " \
          "--symmetry_x true " \
          "--high_order 4 " \
          "--lr 0.0005 " \
          "--lr_decay 0.5 " \
          "--lr_decay_step 5000 " \
          "--epochs 2200 " \
          "--print_interval 10 " \
          "--ckpt_interval 50000 " \
          "--save_ckpt false " \
          "--load_ckpt false " \
          "--save_ckpt_path ./checkpoints " \
          "--load_ckpt_path ./checkpoints/tm_latest.ckpt " \
          "--load_data_path ./data/spheric_tm " \
          "--save_fig false " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--download_data maxwell_net " \
          "--force_download false " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 1


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_tm(mode):
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
    assert loss_value < 10
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_te(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["problem"] = "te"
    config["load_ckpt"] = True
    config["epochs"] = 1
    args = generate_args(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = float(re.findall(r"loss: (.*?), interval", outputs)[-1])
    assert loss_value < 10
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val_tm(mode):
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
    loss_value = re.findall(r"loss: (.*)", outputs)[-1]
    assert float(loss_value) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_error_small_enough_when_val_te(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    config_dict["problem"] = "te"
    args = generate_args(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args)
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"loss: (.*)", outputs)[-1]
    assert float(loss_value) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_error_small_enough_with_full_command_val_tm():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    cmd = f"python eval.py " \
          "--in_channels 2 " \
          "--out_channels 4 " \
          "--depth 6 " \
          "--filter 32 " \
          "--norm weight " \
          "--up_mode upconv " \
          "--wavelength 1 " \
          "--dpl 20 " \
          "--nx 160 " \
          "--nz 192 " \
          "--pml_thickness 30 " \
          "--symmetry_x true " \
          "--high_order 4 " \
          "--lr 0.0005 " \
          "--lr_decay 0.5 " \
          "--lr_decay_step 5000 " \
          "--epochs 2200 " \
          "--print_interval 10 " \
          "--ckpt_interval 50000 " \
          "--save_ckpt false " \
          "--load_ckpt false " \
          "--save_ckpt_path ./checkpoints " \
          "--load_ckpt_path ./checkpoints/tm_latest.ckpt " \
          "--load_data_path ./data/spheric_tm " \
          "--save_fig false " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--download_data maxwell_net " \
          "--force_download false " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_value = re.findall(r"loss: (.*)", outputs)[-1]
    assert float(loss_value) < 0.01


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    # test tm
    model = AutoModel.from_pretrained("maxwell_net")
    model.update_config(mode=mode, epochs=2200, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 1
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"loss: (.*)", outputs)[-1]
    assert float(loss_value) < 0.01

    # test te
    model = AutoModel.from_pretrained("maxwell_net", problem="te")
    model.update_config(mode=mode, epochs=2200, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r", loss: (.*?),", outputs)[-1]
    assert float(loss_value) < 1
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_value = re.findall(r"loss: (.*)", outputs)[-1]
    assert float(loss_value) < 0.01

    clear_stub(stderr, stdout)
