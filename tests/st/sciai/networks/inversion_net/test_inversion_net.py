# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test inversion_net"""
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
from sciai.model.inversion_net.train import main
from sciai.model.inversion_net.eval import main as main_eval
from sciai.model import AutoModel

from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub
from tests.st.sciai.test_utils.func_utils import copy_dataset

copy_dataset(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.level0
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
    load_ckpt_path = config.get("load_ckpt_path")
    cmd = f"python mock_train.py " \
          "--case flatvel_a " \
          "--anno_path ./data " \
          "--train_anno flatvel_a_train_mini.txt " \
          "--val_anno flatvel_a_val_mini.txt " \
          "--device_num 0 " \
          "--dims 32 64 128 256 512 " \
          "--sample_spatial 1.0 " \
          "--sample_temporal 1 " \
          "--lambda_g1v 1.0 " \
          "--lambda_g2v 1.0 " \
          "--batch_size 64 " \
          "--lr 1.0e-4 " \
          "--lr_milestones 100 200 " \
          "--momentum 0.9 " \
          "--weight_decay 1.0e-4 " \
          "--lr_gamma 0.1 " \
          "--lr_warmup_epochs 0 " \
          "--start_epoch 0 " \
          "--epoch_block 10 " \
          "--num_block 1 " \
          "--workers 2 " \
          "--k 1 " \
          "--print_freq 50 " \
          "--save_fig false " \
          "--vis_path ./figures " \
          "--vis_batch 2 " \
          "--vis_sample 3 " \
          "--missing 0 " \
          "--std 0 " \
          "--save_ckpt false " \
          "--load_ckpt false " \
          "--save_ckpt_path ./checkpoints " \
          f"--load_ckpt_path {load_ckpt_path} " \
          "--figures_path ./figures " \
          "--log_path ./logs " \
          "--download_data inversion_net " \
          "--force_download false "

    cmd = shlex.split(cmd)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    outputs = proc.stdout.read().decode()
    final_losses = re.findall(r"Epoch:.*loss: (\d+\.\d+).*loss_g1v: (\d+.\d+).*loss_g2v: (\d+.\d+).*time:", outputs)[-1]
    assert float(final_losses[0]) < 0.2
    assert float(final_losses[1]) < 0.2
    assert float(final_losses[2]) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_loss_small_enough_when_epoch_10(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    with open("./config_test.yaml") as f:
        config = yaml.safe_load(f)
    with open(f"./data/data_config.yaml") as f:
        data_config = yaml.safe_load(f)
    if config["amp_level"] == "O0":
        config["data_type"] = "float32"
    else:
        config["data_type"] = "float16"

    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args, data_config)
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"Epoch:.*loss: (\d+\.\d+).*loss_g1v: (\d+.\d+).*loss_g2v: (\d+.\d+).*time:", outputs)[-1]
    assert float(final_losses[0]) < 0.2
    assert float(final_losses[1]) < 0.2
    assert float(final_losses[2]) < 0.1
    clear_stub(stderr, stdout)


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
    with open(f"./data/data_config.yaml") as f:
        data_config = yaml.safe_load(f)
    if config["amp_level"] == "O0":
        config["data_type"] = "float32"
    else:
        config["data_type"] = "float16"
    config["load_ckpt"] = True
    config["epoch_block"] = 1
    config["num_block"] = 1

    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main(args, data_config)
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"Epoch:.*loss: (\d+\.\d+).*loss_g1v: (\d+.\d+).*loss_g2v: (\d+.\d+).*time:", outputs)[-1]
    assert float(final_losses[0]) < 0.2
    assert float(final_losses[1]) < 0.2
    assert float(final_losses[2]) < 0.1
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
        config = yaml.safe_load(f)
    with open(f"./data/data_config.yaml") as f:
        data_config = yaml.safe_load(f)
    config["load_ckpt"] = True
    if config["amp_level"] == "O0":
        config["data_type"] = "float32"
    else:
        config["data_type"] = "float16"

    args = parse_arg(config)
    init_project(mode=mode, args=args)
    main_eval(args, data_config)
    outputs = sys.stdout.getvalue().strip()
    mse = re.findall("MSE: (.*)", outputs)[-2]
    mae = re.findall("MAE: (.*)", outputs)[-2]
    ssim = re.findall("SSIM: (.*)", outputs)[-1]
    assert float(mse) < 0.001
    assert float(mae) < 0.02
    assert float(ssim) > 0.9
    clear_stub(stderr, stdout)


@pytest.mark.level0
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
    model = AutoModel.from_pretrained("inversion_net")
    model.update_config(
        mode=mode,
        num_block=1,
        epoch_block=10,
        batch_size=64,
        workers=2,
        save_fig=False,
        save_ckpt=False)
    model.train()
    outputs = sys.stdout.getvalue().strip()
    final_losses = re.findall(r"Epoch:.*loss: (\d+\.\d+).*loss_g1v: (\d+.\d+).*loss_g2v: (\d+.\d+).*time:", outputs)[-1]
    assert float(final_losses[0]) < 0.2
    assert float(final_losses[1]) < 0.2
    assert float(final_losses[2]) < 0.1

    model.evaluate()
    outputs = sys.stdout.getvalue().strip()
    mse = re.findall("MSE: (.*)", outputs)[-2]
    mae = re.findall("MAE: (.*)", outputs)[-2]
    ssim = re.findall("SSIM: (.*)", outputs)[-1]
    assert float(mse) < 0.001
    assert float(mae) < 0.02
    assert float(ssim) > 0.9

    with pytest.raises(ValueError):
        model.update_config(false_key=0)

    clear_stub(stderr, stdout)
