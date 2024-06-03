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
"""burgers pinns st case"""
import time
import pytest

import numpy as np
from mindspore import context, nn, ops, jit, set_seed, Tensor
import mindspore.common.dtype as mstype

from model import Burgers1D


set_seed(123456)
np.random.seed(123456)


def _calculate_error(label, prediction):
    """calculate l2 error"""
    error = label - prediction
    l2_error = np.sqrt(
        np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))

    return l2_error


def _get_prediction(model, inputs, label_shape, batch_size):
    """get prediction"""
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    return prediction


def calculate_l2_error(model, inputs, label, batch_size):
    """calculate evaluation error"""
    label_shape = label.shape
    prediction = _get_prediction(model, inputs, label_shape, batch_size)
    label = label.reshape((-1, label_shape[1]))
    l2_error = _calculate_error(label, prediction)
    return l2_error


class Net(nn.Cell):
    """MLP"""
    def __init__(self, in_channels=2, hidden_channels=128, out_channels=1):
        super().__init__()
        self.act = nn.Tanh()
        self.layers = nn.SequentialCell(
            nn.Dense(in_channels, hidden_channels, activation=self.act),
            nn.Dense(hidden_channels, hidden_channels, activation=self.act),
            nn.Dense(hidden_channels, hidden_channels, activation=self.act),
            nn.Dense(hidden_channels, hidden_channels, activation=self.act),
            nn.Dense(hidden_channels, out_channels)
        )

    def construct(self, x):
        return self.layers(x)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_mindflow_burgers_pinns():
    """
    Feature: burgers pinns
    Description: test train and eval
    Expectation: success
    """
    context.set_context(mode=context.GRAPH_MODE)
    model = Net()
    optimizer = nn.Adam(model.trainable_params(), 0.0001)
    problem = Burgers1D(model)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None

    pde_data = Tensor([[-0.02541629, 0.12696983],
                       [0.30243418, 0.96671784],
                       [0.6249878, 0.260476],
                       [-0.61371905, 0.8972365],
                       [0.70554, 0.37674972]], mstype.float32)

    ic_data = Tensor([[0.1678119, 0.],
                      [-0.45064327, 0.],
                      [0.01379196, 0.],
                      [0.40799928, 0.],
                      [0.13942307, 0.]], mstype.float32)
    bc_data = Tensor([[1., 0.1909238],
                      [1., 0.70078486],
                      [-1., 0.70864534],
                      [1., 0.7291773],
                      [1., 0.30929238]], mstype.float32)
    inputs = Tensor([[-1., 0.],
                     [-0.99215686, 0.],
                     [-0.9843137, 0.],
                     [-0.9764706, 0.],
                     [-0.96862745, 0.]], mstype.float32)

    label = np.array([[1.22464680e-16],
                      [2.46374492e-02],
                      [4.92599411e-02],
                      [7.38525275e-02],
                      [9.84002783e-02]], np.float32)

    def forward_fn(pde_data, ic_data, bc_data):
        loss = problem.get_loss(pde_data, ic_data, bc_data)
        if use_ascend:
            loss = loss_scaler.scale(loss)

        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, ic_data, bc_data):
        loss, grads = grad_fn(pde_data, ic_data, bc_data)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)

        loss = ops.depend(loss, optimizer(grads))
        return loss

    for epoch in range(1, 1 + 10):
        model.set_train(True)
        time_beg = time.time()
        train_loss = train_step(pde_data, ic_data, bc_data)
        epoch_time = time.time() - time_beg
        print(f"epoch: {epoch} train loss: {train_loss} epoch time: {epoch_time}s")

    model.set_train(False)
    eval_error = calculate_l2_error(model, inputs, label, 5)
    print("eval_error:", eval_error)

    assert epoch_time < 0.01
    assert train_loss < 0.6
    assert eval_error < 0.8
