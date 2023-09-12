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
"""test sciai common train_cell"""
import os
import re
import shutil
import sys

import mindspore as ms
from mindspore import nn, ops, context
import pytest
from sciai.common import TrainCellWithCallBack, TrainStepCell

from tests.st.sciai.test_utils.basic_nets import Net1In2Out, Net1In1Out, Net1In1OutNumber
from tests.st.sciai.test_utils.test_base import stub_stdout, clear_stub


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_train_cell_should_print_double_loss_with_right_names(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In2Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim, loss_names=("loss_ic", "loss_bc"))
    x = ops.ones((3, 2), ms.float32)
    _, _ = cell(x)
    outputs = sys.stdout.getvalue().strip()
    result = re.search(r"loss_ic: .*, loss_bc: .*", outputs)
    assert result
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_double_loss_with_wrong_names(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In2Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim, loss_names=("loss_ic",))
    x = ops.ones((3, 2), ms.float32)
    _, _ = cell(x)
    outputs = sys.stdout.getvalue().strip()
    result = re.search(r"loss1: .*, loss2: .*", outputs)
    assert result is not None
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_double_loss_with_default_names(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In2Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim)
    x = ops.ones((3, 2), ms.float32)
    _, _ = cell(x)
    outputs = sys.stdout.getvalue().strip()
    result = re.search(r"loss1: .*, loss2: .*", outputs)
    assert result is not None
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_single_loss_with_right_name_str(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In1Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim, loss_names="single_loss")
    x = ops.ones((3, 2), ms.float32)
    _ = cell(x)
    outputs = sys.stdout.getvalue().strip()
    result = re.search(r"single_loss: .*", outputs)
    assert result is not None
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_single_loss_with_wrong_name_tuple(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In1Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim, loss_names=("single_loss",))
    x = ops.ones((3, 2), ms.float32)
    _ = cell(x)
    outputs = sys.stdout.getvalue().strip()
    result = re.search(r"single_loss: .*", outputs)
    assert result is not None
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_single_loss_with_default_name(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In1Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim)
    x = ops.ones((3, 2), ms.float32)
    _ = cell(x)
    outputs = sys.stdout.getvalue().strip()
    result = re.search(r"loss: .*", outputs)
    assert result is not None
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_loss_when_loss_is_number(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In1OutNumber()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim)
    x = ops.ones((3, 2))
    _ = cell(x)
    outputs = sys.stdout.getvalue().strip()
    result = re.search(r"step: 0, loss: 1", outputs)
    assert result is not None
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_ckpt_saved_when_ckpt_interval_gt_0(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In1Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainCellWithCallBack(net, optim, ckpt_interval=1, model_name="pinns")
    x = ops.ones((3, 2), ms.float32)
    try:
        _ = cell(x)
        outputs = sys.stdout.getvalue().strip()
        print(outputs)
        result = re.search(r".*checkpoint saved at: (\./checkpoints/model_pinns_O0_iter_.*\.ckpt)?"
                           r", latest checkpoint re-saved at (.*?checkpoints/Optim_pinns_O0\.ckpt)?.*", outputs)
        assert result is not None
    finally:
        ckpt_path = result.group(1)
        ckpt_dir = os.path.dirname(ckpt_path)
        shutil.rmtree(ckpt_dir)
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_comb_should_print_ckpt_saved_when_batch_num_not_1(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    stderr, stdout = stub_stdout()
    context.set_context(mode=mode)
    net = Net1In1Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    epochs = 3
    ckpt_interval = 2
    batch_num = 5
    cell = TrainCellWithCallBack(net, optim, ckpt_interval=ckpt_interval, batch_num=batch_num)
    x = ops.ones((3, 2))
    results = []
    try:
        for epoch in range(epochs):
            for step in range(batch_num):
                _ = cell(x)
                outputs = sys.stdout.getvalue().strip().split("\n")[-1]
                result = re.search(r".*checkpoint saved at: (\./checkpoints/model_epoch_.*\.ckpt).*", outputs)
                if epoch % ckpt_interval == 0 and step == 0:
                    assert result is not None
                    results.append(result)
                else:
                    assert result is None
    finally:
        ckpt_path = results[-1].group(1)
        ckpt_dir = os.path.dirname(ckpt_path)
        shutil.rmtree(ckpt_dir)
    clear_stub(stderr, stdout)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_train_step_cell_should_return_multi_loss_when_multi_loss(mode):
    """
    Feature: ALL TO ALL
    Description:  test cases for
    Expectation: pass
    """
    context.set_context(mode=mode)
    net = Net1In2Out()
    optim = nn.Adam(net.trainable_params(), 1e-4)
    cell = TrainStepCell(net, optim)
    x = ops.ones((3, 2), ms.float32)
    losses = cell(x)
    assert len(losses) == 2
