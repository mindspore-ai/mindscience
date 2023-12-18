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
"""test hfm"""
import os
import sys
import re
import shlex
import subprocess
import yaml
import pytest

from mindspore import context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.hfm.train import main
from sciai.model.hfm.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_fast_enough_when_epoch_100(mode):
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
    loss, _, _ = re.findall(r".*loss: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 0.2
    clear_stub(stderr, stdout)


@pytest.mark.level0
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
    loss, _, _ = re.findall(r".*loss: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 1.5
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
    load_data_path = config.get("load_data_path")
    cmd = f"python mock_train.py "\
          "--layers 3 200 200 200 200 200 200 200 200 200 200 4 "\
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--load_data_path {load_data_path} " \
          f"--save_data_path ./data " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--t 1500 " \
          "--n 1500 " \
          "--total_time 40 " \
          "--epochs 1 " \
          "--batch_size 1000 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss, _, _ = re.findall(r".*loss: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 1.5


@pytest.mark.level0
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
    error_c, error_u, error_v, error_p = \
        re.findall(r"Error c: (.*), Error u: (.*), Error v: (.*), Error p: (.*)", outputs)[-1]
    assert float(error_c) < 1e-1
    assert float(error_u) < 5e-1
    assert float(error_v) < 1e-1
    assert float(error_p) < 1
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_error_small_enough_with_full_command_val():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    load_data_path = config.get("load_data_path")
    cmd = f"python mock_val.py "\
          "--layers 3 200 200 200 200 200 200 200 200 200 200 4 "\
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--load_data_path {load_data_path} " \
          f"--save_data_path ./data " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--t 1500 " \
          "--n 1500 " \
          "--total_time 40 " \
          "--epochs 1 " \
          "--batch_size 1000 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    error_c, error_u, error_v, error_p = \
        re.findall(r"Error c: (.*), Error u: (.*), Error v: (.*), Error p: (.*)", outputs)[-1]
    assert float(error_c) < 1e-1
    assert float(error_u) < 5e-1
    assert float(error_v) < 1e-1
    assert float(error_p) < 1


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
    loss, _, _ = re.findall(r".*loss: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 0.00001
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("hfm")
    model.update_config(mode=mode, epochs=100, save_result=False, save_ckpt=False,
                        load_data_path="./data")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss, _, _ = re.findall(r".*loss: (.*), interval: (.*), total: (.*)s", outputs)[-1]
    assert float(loss) < 0.2

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_c, error_u, error_v, error_p = \
        re.findall(r"Error c: (.*), Error u: (.*), Error v: (.*), Error p: (.*)", outputs)[-1]
    assert float(error_c) < 1e-1
    assert float(error_u) < 5e-1
    assert float(error_v) < 1e-1
    assert float(error_p) < 1

    clear_stub(stderr, stdout)
