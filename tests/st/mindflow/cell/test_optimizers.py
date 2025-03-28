# ============================================================================
# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Optimizers Test Case"""
import os
import random
import sys

import pytest
import numpy as np

import mindspore as ms
from mindspore import ops, set_seed, nn
from mindspore import dtype as mstype
from mindflow import UNet2D, AttentionBlock, AdaHessian
from mindflow.cell.attention import Mlp
from mindflow.cell.unet2d import Down

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

# pylint: disable=wrong-import-position

from common.cell import FP32_RTOL

# pylint: enable=wrong-import-position

set_seed(0)
np.random.seed(0)
random.seed(0)


class TestAdaHessianAccuracy(AdaHessian):
    ''' Child class for testing the accuracy of AdaHessian optimizer '''
    def gen_rand_vecs(self, grads):
        ''' generate certain vector for accuracy test '''
        return [ms.Tensor(np.arange(p.size).reshape(p.shape) - p.size // 2, dtype=ms.float32) for p in grads]


class TestUNet2D(UNet2D):
    ''' Child class for testing optimizing UNet with AdaHessian '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        class TestDown(Down):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                in_channels = args[0]
                kernel_size = kwargs['kernel_size']
                stride = kwargs['stride']
                # replace the `maxpool` layer in the original UNet with `conv` to avoid `vjp` problem
                self.maxpool = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride)

        self.layers_down = nn.CellList()
        for i in range(self.n_layers):
            self.layers_down.append(TestDown(self.base_channels * 2**i, self.base_channels * 2 ** (i+1),
                                             kernel_size=self.kernel_size, stride=self.stride,
                                             activation=self.activation, enable_bn=self.enable_bn))


class TestAttentionBlock(AttentionBlock):
    ''' Child class for testing optimizing Attention with AdaHessian '''
    def __init__(self, in_channels, num_heads, drop_mode="dropout", dropout_rate=0.0, compute_dtype=mstype.float16):
        super().__init__(
            in_channels, num_heads, drop_mode=drop_mode, dropout_rate=dropout_rate, compute_dtype=compute_dtype)

        class TestMlp(Mlp):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.act_fn = nn.ReLU() # replace `gelu` with `relu` to avoid `vjp` problem

        self.ffn = TestMlp(in_channels=in_channels, dropout_rate=dropout_rate, compute_dtype=compute_dtype)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_adahessian_accuracy(mode):
    """
    Feature: AdaHessian forward accuracy test
    Description: Test the accuracy of the AdaHessian optimizer in both GRAPH_MODE and PYNATIVE_MODE
                with input data specified in the code below.
                The expected output is compared to a reference output stored in
                './mindflow/core/optimizers/data/adahessian_output.npy'.
    Expectation: The output should match the target data within the defined relative tolerance,
                ensuring the AdaHessian computation is accurate.
    """
    ms.set_context(mode=mode)

    weight_init = ms.Tensor(np.reshape(range(72), [4, 2, 3, 3]), dtype=ms.float32)
    bias_init = ms.Tensor(np.arange(4), dtype=ms.float32)

    net = nn.Conv2d(
        in_channels=2, out_channels=4, kernel_size=3, has_bias=True, weight_init=weight_init, bias_init=bias_init)

    def forward(a):
        return ops.mean(net(a)**2)**.5

    grad_fn = ms.grad(forward, grad_position=None, weights=net.trainable_params())

    optimizer = TestAdaHessianAccuracy(
        net.trainable_params(),
        learning_rate=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.)

    inputs = ms.Tensor(np.reshape(range(100), [2, 2, 5, 5]), dtype=ms.float32)

    for _ in range(4):
        optimizer(grad_fn, inputs)

    outputs = net(inputs).numpy()
    outputs_ref = np.load('/home/workspace/mindspore_dataset/mindscience/mindflow/optimizers/adahessian_output.npy')
    relative_error = np.max(np.abs(outputs - outputs_ref)) / np.max(np.abs(outputs_ref))
    assert relative_error < FP32_RTOL, "The verification of adahessian accuracy is not successful."

@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('model_option', ['unet'])
def test_adahessian_st(mode, model_option):
    """
    Feature: AdaHessian ST test
    Description: Test the function of the AdaHessian optimizer in both GRAPH_MODE and PYNATIVE_MODE
                on the complex network such as UNet. The input is a Tensor specified in the code
                and the output is the loss after 4 rounds of optimization.
    Expectation: The output should be finite, ensuring the AdaHessian runs successfully on UNet.
    """
    ms.set_context(mode=mode)

    # default test with Attention network
    net = TestAttentionBlock(in_channels=256, num_heads=4)
    inputs = ms.Tensor(np.sin(np.reshape(range(102400), [4, 100, 256])), dtype=ms.float32)

    # test with UNet network
    if model_option.lower() == 'unet':
        net = TestUNet2D(
            in_channels=2,
            out_channels=4,
            base_channels=8,
            n_layers=4,
            kernel_size=2,
            stride=2,
            activation='relu',
            data_format="NCHW",
            enable_bn=True,
        )
        inputs = ms.Tensor(np.sin(np.reshape(range(16384), [2, 2, 64, 64])), dtype=ms.float32)

    def forward(a):
        return ops.mean(net(a)**2)**.5

    grad_fn = ms.grad(forward, grad_position=None, weights=net.trainable_params())

    optimizer = AdaHessian(
        net.trainable_params(),
        learning_rate=0.1, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.)

    for _ in range(4):
        loss = forward(inputs)
        optimizer(grad_fn, inputs)

    assert ops.isfinite(loss)

@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_adahessian_compare():
    """
    Feature: AdaHessian compare with Adam
    Description: Compare the algorithm results of the AdaHessian optimizer with Adam.
                The code runs in PYNATIVE_MODE and the network under comparison is AttentionBlock.
                The optimization runs 100 rounds to demonstrate an essential loss decrease.
    Expectation: The loss of AdaHessian outperforms Adam by 20% under the same configuration on an Attention network.
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)

    def get_loss(optimizer_option):
        ''' compare Adam and  AdaHessian '''
        net = TestAttentionBlock(in_channels=256, num_heads=4)
        inputs = ms.Tensor(np.sin(np.reshape(range(102400), [4, 100, 256])), dtype=ms.float32)

        def forward(a):
            return ops.mean(net(a)**2)**.5

        grad_fn = ms.grad(forward, grad_position=None, weights=net.trainable_params())

        if optimizer_option.lower() == 'adam':
            optimizer = nn.Adam(
                net.trainable_params(),
                learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.)
        else:
            optimizer = AdaHessian(
                net.trainable_params(),
                learning_rate=0.01, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.)

        for _ in range(100):
            loss = forward(inputs)
            if optimizer_option.lower() == 'adam':
                optimizer(grad_fn(inputs))
            else:
                optimizer(grad_fn, inputs)

        return loss

    loss_adam = get_loss('adam')
    loss_adahessian = get_loss('adahessian')

    assert loss_adam * 0.8 > loss_adahessian, (loss_adam, loss_adahessian)
