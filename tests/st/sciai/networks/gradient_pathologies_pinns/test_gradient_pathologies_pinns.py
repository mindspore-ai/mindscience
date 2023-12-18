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
"""test gradient_pathologies_pinns"""
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
from sciai.model.gradient_pathologies_pinns.eval import main as main_eval
from sciai.model.gradient_pathologies_pinns.train import main
from sciai.utils import parse_arg
from tests.st.sciai.test_utils.func_utils import find_card, copy_dataset
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub

epsilon = 1e-7
copy_dataset(os.path.dirname(os.path.abspath(__file__)))


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
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--method M4 " \
          "--layers 2 50 50 50 1 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt True " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs 1 " \
          "--batch_size 128 " \
          "--amp_level O0"
    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
    error_u = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    error_f = re.findall(r"Relative L2 error_f: (.*)", outputs)[-1]
    assert float(final_losses[0]) < 0.1
    assert float(final_losses[1]) < 0.1
    assert float(final_losses[2]) < 0.1
    assert float(error_u) < 0.3
    assert float(error_f) < 0.3
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_and_error_small_enough_when_load_ckpt(mode):
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
    final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
    error_u = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    error_f = re.findall(r"Relative L2 error_f: (.*)", outputs)[-1]
    assert float(final_losses[0]) < 0.1
    assert float(final_losses[1]) < 0.1
    assert float(final_losses[2]) < 0.1
    assert float(error_u) < 0.3
    assert float(error_f) < 0.3
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_m1_m3_adaptive_not_vary_and_loss_small_enough_when_epoch_21(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    def run_and_test(method):
        config["method"] = method
        args = parse_arg(config)
        init_project(mode=mode, args=args)
        main(args)
        outputs = sys.stdout.getvalue().strip()
        final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
        adap1, adap2 = re.findall(r"adaptive_constant: (.*?),", outputs)[-2:]
        assert float(final_losses[0]) < 10000
        assert float(final_losses[1]) < 20
        assert float(final_losses[2]) < 10000
        assert abs(float(adap1) - 1.0) < epsilon
        assert abs(float(adap2) - 1.0) < epsilon

    run_and_test("M1")
    run_and_test("M3")
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_m2_m4_adaptive_vary_and_loss_small_enough_when_epoch_21(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)

    def run_and_test(method):
        config["method"] = method
        args = parse_arg(config)
        init_project(mode=mode, args=args)
        main(args)
        outputs = sys.stdout.getvalue().strip()
        final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
        adap1, adap2 = re.findall(r"adaptive_constant: (.*?),", outputs)[-2:]
        assert float(final_losses[0]) < 10000
        assert float(final_losses[1]) < 20
        assert float(final_losses[2]) < 10000
        assert abs(float(adap1) - 1.0) > epsilon
        assert abs(float(adap2) - 1.0) > epsilon
        assert abs(float(adap1) - float(adap2)) > epsilon

    run_and_test("M2")
    run_and_test("M4")
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    error_u = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    error_f = re.findall(r"Relative L2 error_f: (.*)", outputs)[-1]
    assert float(error_u) < 0.01
    assert float(error_f) < 0.01
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
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
    cmd = f"python mock_val.py " \
          "--method M4 " \
          "--layers 2 50 50 50 1 " \
          "--save_ckpt false " \
          "--save_fig false " \
          "--load_ckpt True " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path} " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--lr 1e-3 " \
          "--epochs 1 " \
          "--batch_size 128 " \
          "--amp_level O0"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    error_u = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    error_f = re.findall(r"Relative L2 error_f: (.*)", outputs)[-1]
    assert float(error_u) < 0.01
    assert float(error_f) < 0.01


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

    def run_and_test(method):
        config["method"] = method
        args = parse_arg(config)
        init_project(device_id=find_card(), args=args)
        main(args)
        outputs = sys.stdout.getvalue().strip()
        final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
        adap1, adap2 = re.findall(r"adaptive_constant: (.*?),", outputs)[-2:]
        assert float(final_losses[0]) < 0.09
        assert float(final_losses[1]) < 0.03
        assert float(final_losses[2]) < 0.08
        if method in ["M1", "M3"]:
            assert abs(adap1 - 1.0) < epsilon
            assert abs(adap2 - 1.0) < epsilon
        else:
            assert abs(adap1 - 1.0) > epsilon
            assert abs(adap2 - 1.0) > epsilon
            assert abs(adap1 - adap2) > epsilon

    run_and_test("M1")
    run_and_test("M2")
    run_and_test("M3")
    run_and_test("M4")

    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model_m2_m4(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("gradient_pathologies_pinns")
    model.update_config(mode=mode, epochs=21, save_fig=False, save_ckpt=False)

    def run_and_test(method):
        model.update_config(method=method)
        model.train()
        outputs = sys.stdout.getvalue().strip()
        final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
        adap1, adap2 = re.findall(r"adaptive_constant: (.*?),", outputs)[-2:]
        assert float(final_losses[0]) < 10000
        assert float(final_losses[1]) < 20
        assert float(final_losses[2]) < 10000
        assert abs(float(adap1) - 1.0) > epsilon
        assert abs(float(adap2) - 1.0) > epsilon
        assert abs(float(adap1) - float(adap2)) > epsilon

    run_and_test("M2")
    run_and_test("M4")
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model_m1_m3(mode):
    """
    Feature: ALL TO ALL
    Description: test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("gradient_pathologies_pinns")
    model.update_config(mode=mode, epochs=21, save_fig=False, save_ckpt=False)

    def run_and_test(method):
        model.update_config(method=method)
        model.train()
        outputs = sys.stdout.getvalue().strip()
        final_losses = re.findall(r"loss: (.*?), loss_bcs: (.*?), loss_res: (.*?),", outputs)[-1]
        adap1, adap2 = re.findall(r"adaptive_constant: (.*?),", outputs)[-2:]
        assert float(final_losses[0]) < 10000
        assert float(final_losses[1]) < 20
        assert float(final_losses[2]) < 10000
        assert abs(float(adap1) - 1.0) < epsilon
        assert abs(float(adap2) - 1.0) < epsilon

    run_and_test("M1")
    run_and_test("M3")
    clear_stub(stderr, stdout)


@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_auto_model_val(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    model = AutoModel.from_pretrained("gradient_pathologies_pinns")
    model.update_config(mode=mode, epochs=21, save_fig=False, save_ckpt=False)
    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error_u = re.findall(r"Relative L2 error_u: (.*)", outputs)[-1]
    error_f = re.findall(r"Relative L2 error_f: (.*)", outputs)[-1]
    assert float(error_u) < 0.01
    assert float(error_f) < 0.01

    clear_stub(stderr, stdout)
