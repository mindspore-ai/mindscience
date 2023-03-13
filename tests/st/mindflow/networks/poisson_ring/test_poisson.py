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
import os
import pytest

from mindspore import nn, ops, set_seed, jit
import mindspore as ms

from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import load_yaml_config

from tests.st.mindflow.networks.poisson_ring.src import create_training_dataset, \
    create_test_dataset, calculate_l2_error, Poisson2D

set_seed(123456)


def train():
    '''train and evaluate the network'''
    # load configurations
    path = os.path.split(os.path.realpath(__file__))[0]
    config = load_yaml_config(f'{path}/poisson2d_cfg.yaml')

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

    steps = config["train_steps"]
    sink_process = ms.data_sink(train_step, train_dataset, sink_size=1)
    model.set_train()
    l2_error = 0.0
    for step in range(steps):
        local_time_beg = time.time()
        cur_loss = sink_process()
        if step % 100 == 0:
            print(f"loss: {cur_loss.asnumpy():>7f}")
            print("step: {}, time elapsed: {}ms".format(
                step, (time.time() - local_time_beg) * 1000))
            l2_error = calculate_l2_error(
                model, inputs, label, config["train_batch_size"])

    return l2_error


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_poisson_ring_gpu():
    """
    Feature: poisson ring model test in the gpu
    Description: None.
    Expectation: Success or throw error when error is larger than 15
    """
    l2_error = train()
    assert l2_error < 15
