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
"""test pfnn"""
import os
import re
import subprocess
import sys
import shlex
import yaml
import pytest

from mindspore import context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.pfnn.train import main
from sciai.model.pfnn.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.func_utils import copy_dataset
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
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
    g_loss = re.findall(r"NETG epoch : (.*), loss : (.*)", outputs)[-1]
    assert float(g_loss[1]) < 1e-4

    error = 0
    num = 5
    for i in range(num):
        f_loss = re.findall(r"NETF epoch : (.*), loss : (.*), error : (.*)", outputs)[-i]
        error += float(f_loss[2])
    error_mean = error / num
    assert error_mean < 0.02
    clear_stub(stderr, stdout)


@pytest.mark.level0
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
    load_ckpt_path_g = config.get("load_ckpt_path")[0]
    load_ckpt_path_f = config.get("load_ckpt_path")[1]
    cmd = f"python mock_train.py " \
          "--problem 1 " \
          "--bound -1 1 -1 1 " \
          "--inset_nx 60 60 " \
          "--bdset_nx 60 60 " \
          "--teset_nx 101 101 " \
          "--g_epochs 6000 " \
          "--f_epochs 6000 " \
          "--g_lr 0.01 " \
          "--f_lr 0.01 " \
          "--tests_num 1 " \
          "--log_path logs " \
          "--load_ckpt false " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {load_ckpt_path_g} {load_ckpt_path_f} " \
          "--amp_level O0 " \
          "--mode 0 "

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    g_loss = re.findall(r"NETG epoch : (.*), loss : (.*)", outputs)[-1]
    assert float(g_loss[1]) < 1e-4

    error = 0
    num = 5
    for i in range(num):
        f_loss = re.findall(r"NETF epoch : (.*), loss : (.*), error : (.*)", outputs)[-i]
        error += float(f_loss[2])
    error_mean = error / num
    assert error_mean < 0.02


@pytest.mark.level0
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
    config["g_epochs"] = 100
    config["f_epochs"] = 100
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    g_loss = re.findall(r"NETG epoch : (.*), loss : (.*)", outputs)[-1]
    f_loss = re.findall(r"NETF epoch : (.*), loss : (.*), error : (.*)", outputs)[-1]
    assert float(g_loss[1]) < 1e-3
    assert float(f_loss[2]) < 0.1
    clear_stub(stderr, stdout)


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
    error = re.findall(r"The Test Error: (.*)", outputs)[-1]
    assert float(error) < 4e-1
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("auq_pinns")
    model.update_config(mode=mode, tests_num=1)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    g_loss = re.findall(r"NETG epoch : (.*), loss : (.*)", outputs)[-1]
    assert float(g_loss[1]) < 1e-4

    error = 0
    num = 5
    for i in range(num):
        f_loss = re.findall(r"NETF epoch : (.*), loss : (.*), error : (.*)", outputs)[-i]
        error += float(f_loss[2])
    error_mean = error / num
    assert error_mean < 0.02

    with pytest.raises(ValueError):
        model.update_config(false_key=0)

    clear_stub(stderr, stdout)
