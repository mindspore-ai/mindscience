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
"""test auq_pinns"""
import os.path
import re
import subprocess
import sys
import shlex
import yaml
import pytest

from mindspore import context
from sciai.utils import parse_arg
from sciai.context import init_project
from sciai.model.auq_pinns.train import main
from sciai.model.auq_pinns.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub

from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_301(mode):
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
    final_losses = re.findall(r"G_loss: (.*), KL_loss: (.*), recon_loss: (.*), pde_loss: (.*), interval:", outputs)[-1]
    assert float(final_losses[0]) < 50
    assert float(final_losses[3]) < 10
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
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
    load_data_path = config.get("load_data_path")
    ckpt_path_t = config.get("load_ckpt_path")[0]
    ckpt_path_kl = config.get("load_ckpt_path")[1]
    cmd = f"python mock_train.py " \
          "--layers_p 2 50 50 50 50 1 " \
          "--layers_q 2 50 50 50 50 1 " \
          "--layers_t 2 50 50 1 " \
          "--print_interval 100 " \
          "--save_fig false " \
          "--save_ckpt false " \
          "--load_ckpt True " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {ckpt_path_t} {ckpt_path_kl} " \
          "--ckpt_interval 400 " \
          "--figures_path ./figures " \
          f"--load_data_path {load_data_path} " \
          "--log_path ./logs " \
          "--lam 1.5 " \
          "--beta 1 " \
          "--n_col 100 " \
          "--n_bound 20 " \
          "--epochs 1 " \
          "--lr 1e-4 " \
          "--term_t 1 " \
          "--term_kl 5 " \
          "--amp_level O2"

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"G_loss: (.*), KL_loss: (.*), recon_loss: (.*), pde_loss: (.*), interval:", outputs)[-1]
    assert float(final_losses[0]) < 1
    assert float(final_losses[3]) < 1


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
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
    final_losses = re.findall(r"G_loss: (.*), KL_loss: (.*), recon_loss: (.*), pde_loss: (.*), interval:", outputs)[-1]
    assert float(final_losses[0]) < 1
    assert float(final_losses[3]) < 1
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
    error = re.findall("Error u: (.*)", outputs)[-1]
    assert float(error) < 1e-2
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
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
    model.update_config(mode=mode, epochs=301, save_fig=False, save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"G_loss: (.*), KL_loss: (.*), recon_loss: (.*), pde_loss: (.*), interval:", outputs)[-1]
    assert float(final_losses[0]) < 50
    assert float(final_losses[3]) < 10

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    error = re.findall("Error u: (.*)", outputs)[-1]
    assert float(error) < 1e-2

    with pytest.raises(ValueError):
        model.update_config(false_key=0)

    clear_stub(stderr, stdout)
