# Copyright 2024 Huawei Technologies Co., Ltd
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
"""train"""
import os
import time
import datetime
import argparse

import numpy as np
from mindspore import nn, context, ops, Tensor, data_sink, jit, set_seed, save_checkpoint
from mindspore import dtype as mstype
from mindflow.pde import UnsteadyFlowWithLoss
from mindflow import SNO2D, get_poly_transform, RelativeRMSELoss, get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config, log_config, print_log, log_timer

from src import create_training_dataset, load_interp_data, calculate_l2_error, visual

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Navier Stokes problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./configs/sno2d.yaml")
    input_args = parser.parse_args()
    return input_args


@log_timer
def train(input_args):
    '''train and evaluate the network'''
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]

    # prepare dataset
    poly_type = data_params['poly_type']
    load_interp_data(data_params, dataset_type='train')
    train_dataset = create_training_dataset(data_params, shuffle=True)

    test_data = load_interp_data(data_params, dataset_type='test')
    test_input = test_data['test_inputs']
    test_label = test_data['test_labels']

    data_path = data_params['root_dir']

    resolution = data_params['resolution']
    batch_size = data_params['batch_size']
    n_batches = data_params['test_size'] // batch_size
    data_shape = (n_batches, batch_size, resolution, resolution)
    labels_unif = np.load(os.path.join(data_path, "test/label.npy")).reshape(data_shape)

    # prepare model
    n_modes = model_params['modes']

    transform_data = get_poly_transform(resolution, n_modes, poly_type)
    # in this case, resolution and transform type for 'x' and 'y' variables are the same

    transform = Tensor(transform_data["analysis"], mstype.float32)
    inv_transform = Tensor(transform_data["synthesis"], mstype.float32)

    model = SNO2D(
        in_channels=model_params['in_channels'],
        out_channels=model_params['out_channels'],
        hidden_channels=model_params['hidden_channels'],
        num_sno_layers=model_params['sno_layers'],
        kernel_size=model_params['kernel_size'],
        transforms=[[transform, inv_transform]] * 2,
        num_usno_layers=model_params['usno_layers'],
        num_unet_strides=model_params['unet_strides'],
        compute_dtype=mstype.float32)

    total = 0
    for param in model.get_parameters():
        print_log(param.shape)
        total += param.size
    print_log(f"Total Parameters:{total}")

    # prepare optimizer
    steps_per_epoch = train_dataset.get_dataset_size()
    print_log(f"number of steps_per_epochs: {steps_per_epoch}")
    grad_clip_norm = optimizer_params["grad_clip_norm"]

    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params['learning_rate'],
                                        last_epoch=optimizer_params["epochs"],
                                        steps_per_epoch=steps_per_epoch,
                                        warmup_epochs=optimizer_params["warmup_epochs"])

    optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=Tensor(lr),
                                   weight_decay=optimizer_params['weight_decay'])
    problem = UnsteadyFlowWithLoss(model, loss_fn=RelativeRMSELoss(), data_format="NTCHW")

    def forward_fn(train_inputs, train_label):
        loss = problem.get_loss(train_inputs, train_label)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(train_inputs, train_label):
        loss, grads = grad_fn(train_inputs, train_label)
        grads = ops.clip_by_global_norm(grads, grad_clip_norm)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    sink_process = data_sink(train_step, train_dataset, sink_size=1)
    ckpt_dir = os.path.join(model_params["root_dir"], summary_params["ckpt_dir"])
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    for epoch in range(1, 1 + optimizer_params["epochs"]):
        local_time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epoch):
            cur_loss = sink_process()

        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = (epoch_seconds / steps_per_epoch) * 1000
        print_log(f"epoch: {epoch} train loss: {cur_loss} "
                  f"epoch time: {epoch_seconds:.3f}s step time: {step_seconds:5.3f}ms")

        model.set_train(False)
        if epoch % summary_params["save_ckpt_interval"] == 0:
            save_checkpoint(model, os.path.join(ckpt_dir, f"{model_params['name']}_epoch{epoch}"))

        if epoch % summary_params['test_interval'] == 0:
            calculate_l2_error(model, test_input, test_label, labels_unif, data_params)

    visual(model, test_input, labels_unif, data_params)

if __name__ == '__main__':
    log_config('./logs', 'sno2d')
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    train(args)
