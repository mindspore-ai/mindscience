# ============================================================================
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
import os
import time
import numpy as np

from mindspore import context, nn, ops, jit, set_seed, data_sink

from mindflow.utils import load_yaml_config
from mindflow.cell import FCSequential

from src import create_random_dataset, get_test_data
from src import Darcy2D
from src import visual_result, calculate_l2_error

set_seed(123456)
np.random.seed(123456)

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="GPU",
    save_graphs_path="./graph",
)


def train(config):
    """training process"""
    geom_name = "flow_region"
    # create train dataset
    flow_train_dataset = create_random_dataset(config, geom_name)
    train_data = flow_train_dataset.create_dataset(
        batch_size=config["train_batch_size"], shuffle=True, drop_remainder=True
    )
    test_input, test_label = get_test_data(config)

    # network model
    model = FCSequential(in_channels=config["model"]["input_size"],
                         out_channels=config["model"]["output_size"],
                         neurons=config["model"]["neurons"],
                         layers=config["model"]["layers"],
                         residual=config["model"]["residual"],
                         act=config["model"]["activation"],
                         weight_init=config["model"]["weight_init"])

    # define problem
    problem = Darcy2D(model)

    # optimizer
    params = model.trainable_params()
    optim = nn.Adam(params, learning_rate=config["optimizer"]["lr"])

    def forward_fn(pde_data, bc_data):
        return problem.get_loss(pde_data, bc_data)

    grad_fn = ops.value_and_grad(forward_fn, None, optim.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data):
        loss, grads = grad_fn(pde_data, bc_data)
        loss = ops.depend(loss, optim(grads))
        return loss

    epochs = config["train_epoch"]
    sink_process = data_sink(train_step, train_data, sink_size=1)
    model.set_train()

    for epoch in range(epochs):
        local_time_beg = time.time()
        cur_loss = sink_process()
        if epoch % 200 == 0:
            print(
                "epoch: {}, loss: {}, time: {}ms.".format(
                    epoch, cur_loss, (time.time() - local_time_beg) * 1000
                )
            )
            calculate_l2_error(
                model, test_input, test_label, config["train_batch_size"]
            )

    visual_result(model, config)


if __name__ == "__main__":
    print("pid:", os.getpid())
    configs = load_yaml_config("darcy_cfg.yaml")
    time_beg = time.time()
    train(configs)
    print(f"End-to-End total time: {format(time.time() - time_beg)} s")
