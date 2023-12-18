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
"""test dgm"""
import os
import sys
import re
import subprocess
import shlex
import yaml
import pytest

from mindspore import context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.dgm.train import main
from sciai.model.dgm.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_fast_enough_when_epoch_800(mode):
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
    loss_domain, loss_ic, _, _ = re.findall(r".*loss_domain: (.*), loss_ic: (.*), interval: (.*), total: (.*)s",
                                            outputs)[-1]

    assert float(loss_domain) < 30
    assert float(loss_ic) < 0.07
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_fast_enough_when_load_ckpt(mode):
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
    loss_domain, loss_ic, _, _ = re.findall(r".*loss_domain: (.*), loss_ic: (.*), interval: (.*), total: (.*)s",
                                            outputs)[-1]

    assert float(loss_domain) < 30
    assert float(loss_ic) < 0.07
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
          "--layers 1 10 10 10 1 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt true " \
          "--save_fig false " \
          "--save_anim false " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--figures_path ./figures " \
          "--lr 0.01 " \
          "--epochs 1 "\
          "--batch_size 256 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_domain, loss_ic, _, _ = re.findall(r".*loss_domain: (.*), loss_ic: (.*), interval: (.*), total: (.*)s",
                                            outputs)[-1]

    assert float(loss_domain) < 30
    assert float(loss_ic) < 0.07


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_should_error_small_enough_when_val(mode):
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
    error = re.findall(r"error: (.*)", outputs)[-1]
    assert float(error) < 0.0008
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_should_error_small_with_full_command_val():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_val.py " \
          "--layers 1 10 10 10 1 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt true " \
          "--save_fig false " \
          "--save_anim false " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--figures_path ./figures " \
          "--lr 0.01 " \
          "--epochs 1 "\
          "--batch_size 256 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    error = re.findall(r"error: (.*)", outputs)[-1]
    assert float(error) < 0.0008


@pytest.mark.full_epoch
def test_full():
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config.yaml") as f:
        config = yaml.safe_load(f)

    args = parse_arg(config)
    init_project(device_id=find_card(), args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_domain, loss_ic, _, _ = re.findall(r".*loss_domain: (.*), loss_ic: (.*), interval: (.*), total: (.*)s",
                                            outputs)[-1]

    assert float(loss_domain) < 0.006
    assert float(loss_ic) < 0.006
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("dgm")
    model.update_config(mode=mode, epochs=800, lr=0.02, save_fig=False, save_ckpt=False, save_anim=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_domain, loss_ic, _, _ = re.findall(r".*loss_domain: (.*), loss_ic: (.*), interval: (.*), total: (.*)s",
                                            outputs)[-1]
    assert float(loss_domain) < 30
    assert float(loss_ic) < 0.07

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error = re.findall(r"error: (.*)", outputs)[-1]
    assert float(error) < 0.0008

    clear_stub(stderr, stdout)
