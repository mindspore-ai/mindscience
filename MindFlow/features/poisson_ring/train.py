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
"""train process"""

import os
import time

from mindspore import nn, ops, set_context, set_seed, jit
import mindspore as ms

from mindflow.cell import MultiScaleFCCell
from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_test_dataset, calculate_l2_error, visual_result, Poisson2D

set_seed(123456)
set_context(mode=ms.GRAPH_MODE, device_target="GPU", device_id=3)


def train():
    '''train and evaluate the network'''
    # load configurations
    config = load_yaml_config('poisson2d_cfg.yaml')

    # create training dataset
    dataset = create_training_dataset(config)
    train_dataset = dataset.batch(batch_size=config["train_batch_size"])

    # create  test dataset
    inputs, label = create_test_dataset(config)

    # define models and optimizers
    model = MultiScaleFCCell(in_channels=config["model"]["in_channels"],
                             out_channels=config["model"]["out_channels"],
                             layers=config["model"]["layers"],
                             neurons=config["model"]["neurons"],
                             residual=config["model"]["residual"],
                             act=config["model"]["activation"],
                             num_scales=1)

    optimizer = nn.Adam(model.trainable_params(), config["optimizer"]["initial_lr"])
    problem = Poisson2D(model)

    def forward_fn(pde_data, bc_outer_data, bc_inner_data, bc_inner_normal):
        loss = problem.get_loss(pde_data, bc_outer_data, bc_inner_data, bc_inner_normal)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_outer_data, bc_inner_data, bc_inner_normal):
        loss, grads = grad_fn(pde_data, bc_outer_data, bc_inner_data, bc_inner_normal)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    steps = config["train_steps"]
    sink_process = ms.data_sink(train_step, train_dataset, sink_size=1)
    model.set_train()
    for step in range(steps):
        local_time_beg = time.time()
        cur_loss = sink_process()
        if step % 100 == 0:
            print(f"loss: {cur_loss.asnumpy():>7f}")
            print("step: {}, time elapsed: {}ms".format(step, (time.time() - local_time_beg) * 1000))
            calculate_l2_error(model, inputs, label, config["train_batch_size"])
    visual_result(model, inputs, label, step + 1)


if __name__ == '__main__':
    print("pid:", os.getpid())
    time_beg = time.time()
    train()
    print("End-to-End total time: {} s".format(time.time() - time_beg))
