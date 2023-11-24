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

from mindspore import context, nn, ops, jit, set_seed
from mindspore import save_checkpoint, data_sink

from mindflow.cell import MultiScaleFCSequential
from mindflow.loss import MTLWeightedLoss
from mindflow.utils import load_yaml_config, log_config, print_log, log_timer

from src import create_training_dataset, create_test_dataset, calculate_l2_error, NavierStokes2D


set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description="cylinder flow train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="././configs/cylinder_flow.yaml")
    input_args = parser.parse_args()
    return input_args


@log_timer
def train(input_args):
    '''Train and evaluate the network'''
    # load configurations
    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]
    geo_params = config["geometry"]
    # create training dataset
    cylinder_flow_train_dataset = create_training_dataset(config)
    cylinder_dataset = cylinder_flow_train_dataset.create_dataset(batch_size=data_params["batch_size"],
                                                                  shuffle=True,
                                                                  prebatched_data=True,
                                                                  drop_remainder=True)

    # create test dataset
    inputs, label = create_test_dataset(data_params['root_dir'])

    coord_min = np.array(geo_params["coord_min"] +
                         [geo_params["time_min"]]).astype(np.float32)
    coord_max = np.array(geo_params["coord_max"] +
                         [geo_params["time_max"]]).astype(np.float32)
    input_center = list(0.5 * (coord_max + coord_min))
    input_scale = list(2.0 / (coord_max - coord_min))
    model = MultiScaleFCSequential(in_channels=model_params["in_channels"],
                                   out_channels=model_params["out_channels"],
                                   layers=model_params["num_layers"],
                                   neurons=model_params["hidden_channels"],
                                   residual=model_params["residual"],
                                   act='tanh',
                                   num_scales=1,
                                   input_scale=input_scale,
                                   input_center=input_center)

    mtl = MTLWeightedLoss(num_losses=cylinder_flow_train_dataset.num_dataset)
    print_log("Use MTLWeightedLoss, num loss: {}".format(mtl.num_losses))

    params = model.trainable_params() + mtl.trainable_params()
    optimizer = nn.Adam(params, optimizer_params["learning_rate"])
    problem = NavierStokes2D(model)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

    def forward_fn(pde_data, bc_data, bc_label, ic_data, ic_label):
        loss = problem.get_loss(pde_data, bc_data, bc_label, ic_data, ic_label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(pde_data, bc_data, bc_label, ic_data, ic_label):
        loss, grads = grad_fn(pde_data, bc_data, bc_label, ic_data, ic_label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            is_finite = all_finite(grads)
            if is_finite:
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, optimizer(grads))
            loss_scaler.adjust(is_finite)
        else:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    epochs = optimizer_params["epochs"]
    steps_per_epochs = cylinder_dataset.get_dataset_size()
    print_log(f"number of steps_per_epochs: {steps_per_epochs}")
    sink_process = data_sink(train_step, cylinder_dataset, sink_size=1)
    os.makedirs(summary_params['ckpt_dir'], exist_ok=True)
    for epoch in range(1, 1 + epochs):
        # train
        local_time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epochs):
            step_train_loss = sink_process()
        local_time_end = time.time()
        epoch_seconds = (local_time_end - local_time_beg) * 1000
        step_seconds = epoch_seconds/steps_per_epochs
        print_log(
            f"epoch: {epoch} train loss: {step_train_loss} "
            f"epoch time: {epoch_seconds:5.3f}ms step time: {step_seconds:5.3f}ms")
        model.set_train(False)
        if epoch % summary_params["test_interval"] == 0:
            # eval
            eval_time_start = time.time()
            calculate_l2_error(model, inputs, label,
                               model_params, data_params["batch_size"])
            print_log(f'evaluation time: {time.time() - eval_time_start}s')

        if epoch % summary_params["save_ckpt_interval"] == 0:
            ckpt_name = f"ns_cylinder_flow-{epoch}.ckpt"
            save_checkpoint(model, os.path.join(
                summary_params['ckpt_dir'], ckpt_name))


if __name__ == '__main__':
    log_config('./logs', 'navier_stokes_2d')
    print_log("pid:", os.getpid())
    args = parse_args()
    print_log(f"device id: {args.device_id}")
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    train(args)
