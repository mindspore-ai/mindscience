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
"""test pinns"""
import os
import re
import subprocess
import sys
import shlex
import yaml
import pytest

from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_auto_model():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("pinns")
    model.update_config(epoch=1000, load_data_path="./data/NLS.mat")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    _, _, loss = re.findall(r"epoch: (.*) step: (.*) loss is (.*)", outputs)[-1]
    assert float(loss) < 0.5
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error = re.findall(r"evaluation error is: (.*)", outputs)[-1]
    assert float(error) < 0.02

    model = AutoModel.from_pretrained("pinns", problem="NavierStokes")
    model.update_config(epoch=100,
                        load_data_path="./data/cylinder_nektar_wake.mat")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    _, _, loss = re.findall(r"epoch: (.*) step: (.*) loss is (.*)", outputs)[-1]
    assert float(loss) < 0.5
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_1 = re.findall(r"Error of lambda 1 is (.*)%", outputs)[-1]
    error_2 = re.findall(r"Error of lambda 2 is (.*)%", outputs)[-1]
    assert float(error_1) < 0.3
    assert float(error_2) < 0.5

    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('problem', ['Schrodinger', 'NavierStokes'])
def test_full_command(problem):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f).get(problem)
    load_data_path = config.get("load_data_path")
    epoch = config.get("epoch")
    cmd = f"python mock_train.py " \
          f"--load_data_path {load_data_path} " \
          f"--problem {problem} " \
          f"--epoch {epoch}"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    _, _, loss = re.findall(r"epoch: (.*) step: (.*) loss is (.*)", outputs)[-1]
    assert float(loss) < 0.5


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_eval_full_command_ns():
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f).get('NavierStokes')
    load_data_path = config.get("load_data_path")
    epoch = config.get("epoch")
    cmd = f"python mock_eval.py " \
          f"--load_data_path {load_data_path} " \
          f"--problem NavierStokes " \
          f"--epoch {epoch}"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    error_1 = re.findall(r"Error of lambda 1 is (.*)%", outputs)[-1]
    error_2 = re.findall(r"Error of lambda 2 is (.*)%", outputs)[-1]
    assert float(error_1) < 0.3
    assert float(error_2) < 0.5


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_eval_full_command_sch():
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f).get('Schrodinger')
    load_data_path = config.get("load_data_path")
    epoch = config.get("epoch")
    cmd = f"python mock_eval.py " \
          f"--load_data_path {load_data_path} " \
          f"--problem Schrodinger " \
          f"--epoch {epoch}"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    error = re.findall(r"evaluation error is: (.*)", outputs)[-1]
    assert float(error) < 0.02
