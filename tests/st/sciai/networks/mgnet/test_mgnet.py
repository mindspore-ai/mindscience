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
"""test mgnet"""
import os
import re
import sys
import shlex
import subprocess
import yaml
import pytest

from mindspore import context, get_context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.mgnet.train import main
from sciai.model.mgnet.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
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
    final_loss = re.findall(r", loss: (.*), interval", outputs)[-1]
    assert float(final_loss) < 4.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
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
    data_path = config.get("load_data_path")
    amp_level = 'O0' if get_context('device_target') == 'GPU' else 'O3'
    cmd = f"python mock_train.py " \
          "--dataset cifar100 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--load_data_path {data_path} " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--ckpt_interval 500 " \
          "--num_ite 2 2 2 2 " \
          "--num_channel_u 256 " \
          "--num_channel_f 256 " \
          "--wise_b true " \
          "--batch_size 128 " \
          "--epochs 1 " \
          "--lr 1e-1 " \
          f"--amp_level {amp_level}"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_loss = re.findall(r", loss: (.*), interval", outputs)[-1]
    assert float(final_loss) < 4.5


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
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    final_loss = re.findall(r", loss: (.*), interval", outputs)[-1]
    assert float(final_loss) < 4.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_accuracy_large_enough_when_val(mode):
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
    final_accuracy = re.findall(r"{'accuracy': (.*)}", outputs)[-1]
    assert float(final_accuracy) > 0.7
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
def test_comb_should_accuracy_large_enough_with_full_command_val():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    data_path = config.get("load_data_path")
    cmd = f"python mock_val.py " \
          "--dataset cifar100 " \
          "--save_ckpt false " \
          "--load_ckpt true " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          f"--load_data_path {data_path} " \
          "--log_path ./logs " \
          "--print_interval 10 " \
          "--ckpt_interval 500 " \
          "--num_ite 2 2 2 2 " \
          "--num_channel_u 256 " \
          "--num_channel_f 256 " \
          "--wise_b true " \
          "--batch_size 128 " \
          "--epochs 2 " \
          "--lr 1e-1 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_accuracy = re.findall(r"{'accuracy': (.*)}", outputs)[-1]
    assert float(final_accuracy) > 0.7


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
    final_loss = re.findall(r", loss: (.*), interval", outputs)[-1]
    assert float(final_loss) < 0.9
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("mgnet")
    amp_train = 'O0' if get_context('device_target') == 'GPU' else 'O3'
    model.update_config(mode=mode, epochs=2, save_ckpt=False, amp_level=amp_train,
                        load_data_path="./data",
                        load_ckpt_path="./checkpoints/model_300.ckpt")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_loss = re.findall(r", loss: (.*), interval", outputs)[-1]
    assert float(final_loss) < 4.5

    model.update_config(amp_level='O0')
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    final_accuracy = re.findall(r"{'accuracy': (.*)}", outputs)[-1]
    assert float(final_accuracy) > 0.7

    clear_stub(stderr, stdout)
