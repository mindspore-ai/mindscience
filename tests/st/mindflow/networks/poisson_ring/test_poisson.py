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
"""train process"""

import time
import pytest

from mindspore import nn, ops, set_seed, jit, data_sink

from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_test_dataset, calculate_l2_error, Poisson2D

set_seed(123456)


def train():
    '''train and evaluate the network'''
    # load configurations
    config = load_yaml_config(f'./poisson2d_cfg.yaml')

    # create training dataset
    dataset = create_training_dataset(config)
    train_dataset = dataset.batch(batch_size=config["train_batch_size"])

    # create  test dataset
    inputs, label = create_test_dataset(config)

    # define models and optimizers
    model = MultiScaleFCSequential(in_channels=config["model"]["in_channels"],
                                   out_channels=config["model"]["out_channels"],
                                   layers=config["model"]["layers"],
                                   neurons=config["model"]["neurons"],
                                   residual=config["model"]["residual"],
                                   act=config["model"]["activation"],
                                   num_scales=1)

    optimizer = nn.Adam(model.trainable_params(),
                        config["optimizer"]["initial_lr"])
    problem = Poisson2D(model)

    def forward_fn(pde_data, bc_outer_data, bc_inner_data, bc_inner_normal):
        loss = problem.get_loss(pde_data, bc_outer_data,
                                bc_inner_data, bc_inner_normal)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_outer_data, bc_inner_data, bc_inner_normal):
        loss, grads = grad_fn(pde_data, bc_outer_data,
                              bc_inner_data, bc_inner_normal)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    sink_process = data_sink(train_step, train_dataset, sink_size=1)
    model.set_train()
    steps = 100
    for step in range(1, 1+steps):
        local_time_beg = time.time()
        cur_loss = sink_process()
        print(
            f"step: {step} loss: {cur_loss.asnumpy():>7f} epoch time : {time.time() - local_time_beg}s")
    l2_error = calculate_l2_error(
        model, inputs, label, config["train_batch_size"])

    return cur_loss.asnumpy(), l2_error


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_poisson_ring():
    """
    Feature: poisson ring model test
    Description: None.
    Expectation: Success or throw error when l2_error is larger than 0.5 or train_loss is larger than 0.2
    """
    train_loss, l2_error = train()
    assert train_loss < 0.2
    assert l2_error < 0.5
