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

import argparse
import os
import time

import numpy as np

import mindspore
from mindspore import context, nn, ops, jit, set_seed
from mindspore import load_checkpoint, load_param_into_net

from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_test_dataset, visual, calculate_l2_error, Burgers1D


set_seed(123456)
np.random.seed(123456)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description="burgers train")
    parser.add_argument("--config_file_path", type=str, default="./burgers_cfg.yaml")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")

    input_args = parser.parse_args()
    return input_args


def train():
    '''Train and evaluate the network'''
    # load configurations
    config = load_yaml_config(args.config_file_path)

    # create dataset
    burgers_train_dataset = create_training_dataset(config)
    train_dataset = burgers_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                         shuffle=True,
                                                         prebatched_data=True,
                                                         drop_remainder=True)
    # create  test dataset
    inputs, label = create_test_dataset(config["test_dataset_path"])

    # define models and optimizers
    model = MultiScaleFCSequential(in_channels=config["model"]["in_channels"],
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
        auto_mixed_precision(model, 'O3')
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

    epochs = config["train_epochs"]
    steps_per_epochs = train_dataset.get_dataset_size()
    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)
    for epoch in range(1, 1 + epochs):
        # train
        time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epochs):
            step_train_loss = sink_process()
        print(f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) * 1000 :.3f}ms")
        model.set_train(False)
        if epoch % config["eval_interval_epochs"] == 0:
            eval_time_start = time.time()
            calculate_l2_error(model, inputs, label, config["train_batch_size"])
            print(f'evaluation time: {time.time() - eval_time_start}s')

    visual(model, epochs=epochs, resolution=config["visual_resolution"])


if __name__ == '__main__':
    print("pid:", os.getpid())
    start_time = time.time()
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    train()
    print("End-to-End total time: {} s".format(time.time() - start_time))
