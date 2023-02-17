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

import numpy as np

import mindspore
from mindspore import context, nn, ops, jit, set_seed
from mindspore import load_checkpoint, load_param_into_net

from mindflow.cell import MultiScaleFCCell
from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_test_dataset, visual_result, calculate_l2_error, Burgers1D


set_seed(123456)
np.random.seed(123456)

context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=5)
use_ascend = context.get_context(attr_key='device_target') == "Ascend"


def train():
    '''Train and evaluate the network'''
    # load configurations
    config = load_yaml_config('burgers_cfg.yaml')

    # create dataset
    burgers_train_dataset = create_training_dataset(config)
    train_dataset = burgers_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                         shuffle=True,
                                                         prebatched_data=True,
                                                         drop_remainder=True)
    # create  test dataset
    inputs, label = create_test_dataset()

    # define models and optimizers
    model = MultiScaleFCCell(in_channels=config["model"]["in_channels"],
                             out_channels=config["model"]["out_channels"],
                             layers=config["model"]["layers"],
                             neurons=config["model"]["neurons"],
                             residual=config["model"]["residual"],
                             act=config["model"]["activation"],
                             num_scales=1)
    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)

    # define optimizer
    optimizer = nn.Adam(model.trainable_params(), config["optimizer"]["initial_lr"])
    problem = Burgers1D(model)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O1')
    else:
        loss_scaler = None

    def forward_fn(pde_data, ic_data, bc_data):
        loss = problem.get_loss(pde_data, ic_data, bc_data)
        if use_ascend:
            loss = loss_scaler.scale(loss)

        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, ic_data, bc_data):
        loss, grads = grad_fn(pde_data, ic_data, bc_data)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)

        loss = ops.depend(loss, optimizer(grads))
        return loss

    steps = config["train_steps"]
    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)
    model.set_train()
    for step in range(steps):
        local_time_beg = time.time()
        cur_loss = sink_process()
        if step % 100 == 0:
            print(f"loss: {cur_loss.asnumpy():>7f}")
            print("step: {}, time elapsed: {}ms".format(step, (time.time() - local_time_beg) * 1000))
            calculate_l2_error(model, inputs, label, config["train_batch_size"])

    visual_result(model, step=step + 1, resolution=config["visual_resolution"])


if __name__ == '__main__':
    print("pid:", os.getpid())
    time_beg = time.time()
    train()
    print("End-to-End total time: {} s".format(time.time() - time_beg))
    