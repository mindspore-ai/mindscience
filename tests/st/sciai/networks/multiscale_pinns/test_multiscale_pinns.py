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
"""test multiscale pinns"""
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
from sciai.model.multiscale_pinns.train import main
from sciai.model.multiscale_pinns.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_5_net_nn(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["net_type"] = "net_nn"
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.03
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_5_net_ff(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["net_type"] = "net_ff"
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_should_run_with_full_command_net_sf_tt():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--layers 2 100 100 100 1 " \
          "--save_ckpt false  " \
          "--save_fig false  " \
          "--load_ckpt true  " \
          "--save_ckpt_path ./checkpoints " \
          "--figures_path ./figures " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--lr 1e-3  " \
          "--epochs 1  " \
          "--batch_size 128  " \
          "--net_type net_st_ff " \
          "--print_interval 100 " \
          "--nnum 1000 " \
          "--amp_level O3"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.01


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_5_net_sf_tt(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["net_type"] = "net_st_ff"
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_load_ckpt_net_sf_tt(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    config["net_type"] = "net_st_ff"
    config["load_ckpt"] = True
    config["epochs"] = 1
    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.01
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
    context.set_context(mode=mode)
    with open("./config_test.yaml") as f:
        config_dict = yaml.safe_load(f)
    args = parse_arg(config_dict)
    init_project(mode=mode, args=args)
    main_eval(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    assert float(loss_values) < 1.5
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_comb_should_error_small_enough_with_full_command_val_net_sf_tt():
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_val.py " \
          "--layers 2 100 100 100 1 " \
          "--save_ckpt false  " \
          "--save_fig false  " \
          "--load_ckpt true  " \
          "--save_ckpt_path ./checkpoints " \
          "--figures_path ./figures " \
          f"--load_ckpt_path {ckpt_path} " \
          "--log_path ./logs " \
          "--lr 1e-3  " \
          "--epochs 1  " \
          "--batch_size 128  " \
          "--net_type net_st_ff " \
          "--print_interval 100 " \
          "--nnum 1000 " \
          "--amp_level O3"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    loss_values = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    assert float(loss_values) < 1.5


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

    config["net_type"] = "net_nn"
    args = parse_arg(config)
    init_project(device_id=find_card(), args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 0.6
    assert float(loss_values[1]) < 0.0000006
    assert float(loss_values[2]) < 0.6
    assert float(loss_values[3]) < 0.000005

    config["net_type"] = "net_ff"
    args = parse_arg(config)
    init_project(device_id=find_card(), args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 0.6
    assert float(loss_values[1]) < 0.0000006
    assert float(loss_values[2]) < 0.6
    assert float(loss_values[3]) < 0.000005

    config["net_type"] = "net_st_ff"
    args = parse_arg(config)
    init_project(device_id=find_card(), args=args)
    main(args)
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 0.6
    assert float(loss_values[1]) < 0.0000006
    assert float(loss_values[2]) < 0.6
    assert float(loss_values[3]) < 0.000005

    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("multiscale_pinns")
    model.update_config(mode=mode, epochs=101, save_fig=False, save_ckpt=False, print_interval=1, nnum=100)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.01

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    assert float(loss_values) < 1.5

    model.update_config(net_type="net_ff")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.01

    model.update_config(net_type="net_nn")
    model.train()
    outputs = sys.stdout.getvalue().strip()
    loss_values = re.findall(r"total_loss: (.*?), bcs_loss: (.*?), ics_loss: (.*?), res_loss: (.*?),", outputs)[-1]
    assert float(loss_values[0]) < 1
    assert float(loss_values[1]) < 0.01
    assert float(loss_values[2]) < 1
    assert float(loss_values[3]) < 0.03

    clear_stub(stderr, stdout)
