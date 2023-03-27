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

import argparse
import os
import time

import numpy as np

import mindspore
from mindspore import context, nn, ops, jit, set_seed, load_checkpoint, load_param_into_net

from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import load_yaml_config

from src import create_training_dataset, create_test_dataset, calculate_l2_error, InvNavierStokes
from src import visual, plot_params

set_seed(123456)
np.random.seed(123456)

parser = argparse.ArgumentParser(description="cylinder flow train")
parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                    help="Running in GRAPH_MODE OR PYNATIVE_MODE")
parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                    help="Whether to save intermediate compilation graphs")
parser.add_argument("--save_graphs_path", type=str, default="./graphs")
parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                    help="The target device to run, support 'Ascend', 'GPU'")
parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
parser.add_argument("--config_file_path", type=str, default="./cylinder_flow.yaml")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                    save_graphs=args.save_graphs,
                    save_graphs_path=args.save_graphs_path,
                    device_target=args.device_target,
                    device_id=args.device_id)
print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
use_ascend = context.get_context(attr_key='device_target') == "Ascend"


def train():
    '''Train and evaluate the network'''
    # load configurations
    config = load_yaml_config('inverse_navier_stokes.yaml')

    # create dataset
    inv_ns_train_dataset = create_training_dataset(config)
    train_dataset = inv_ns_train_dataset.create_dataset(batch_size=config["train_batch_size"],
                                                        shuffle=True,
                                                        prebatched_data=True,
                                                        drop_remainder=True)
    # create  test dataset
    inputs, label = create_test_dataset(config)

    coord_min = np.array(config["geometry"]["coord_min"] + [config["geometry"]["time_min"]]).astype(np.float32)
    coord_max = np.array(config["geometry"]["coord_max"] + [config["geometry"]["time_max"]]).astype(np.float32)
    input_center = list(0.5 * (coord_max + coord_min))
    input_scale = list(2.0 / (coord_max - coord_min))

    model = MultiScaleFCSequential(in_channels=config["model"]["in_channels"],
                                   out_channels=config["model"]["out_channels"],
                                   layers=config["model"]["layers"],
                                   neurons=config["model"]["neurons"],
                                   residual=config["model"]["residual"],
                                   act='tanh',
                                   num_scales=1,
                                   input_scale=input_scale,
                                   input_center=input_center)

    if config["load_ckpt"]:
        param_dict = load_checkpoint(config["load_ckpt_path"])
        load_param_into_net(model, param_dict)

    theta = mindspore.Parameter(mindspore.Tensor(np.array([0.0, 0.0]).astype(np.float32)), name="theta")
    params = model.trainable_params()
    params.append(theta)
    optimizer = nn.Adam(params, learning_rate=config["optimizer"]["initial_lr"])
    problem = InvNavierStokes(model, params)
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    def forward_fn(pde_data, train_points, train_label):
        loss = problem.get_loss(pde_data, train_points, train_label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, train_points, train_label):
        loss, grads = grad_fn(pde_data, train_points, train_label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, optimizer(grads))
        else:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    epochs = config["train_epochs"]
    steps_per_epochs = train_dataset.get_dataset_size()
    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)

    param1_hist = []
    param2_hist = []
    for epoch in range(1, 1 + epochs):
        # train
        time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epochs):
            step_train_loss = sink_process()
        print(f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) * 1000 :.3f} ms")
        model.set_train(False)
        if epoch % config["eval_interval_epochs"] == 0:
            print(f"Params are{params[-1].value()}")
            param1_hist.append(params[-1].value()[0])
            param2_hist.append(params[-1].value()[1])
            calculate_l2_error(model, inputs, label, config)
    visual(model, epochs, inputs, label)
    plot_params(param1_hist, param2_hist)


if __name__ == '__main__':
    print("pid:", os.getpid())
    start_time = time.time()
    train()
    print("End-to-End total time: {} s".format(time.time() - start_time))
