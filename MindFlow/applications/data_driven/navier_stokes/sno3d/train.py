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
import argparse
import datetime
from timeit import default_timer

import numpy as np
from mindspore import nn, ops, jit, data_sink, context, Tensor
from mindspore.common import set_seed
from mindspore import dtype as mstype
from mindflow import get_warmup_cosine_annealing_lr, load_yaml_config
from mindflow.utils import print_log, log_config, log_timer

from src import SNO3D, get_poly_transform, calculate_l2_error, UnitGaussianNormalizer, \
    create_training_dataset, load_interp_data

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Navier Stokes 3D problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=3,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/sno3d.yaml")
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

    t1 = default_timer()

    grid_size = data_params["resolution"]
    input_timestep = model_params["input_timestep"]
    output_timestep = model_params["extrapolations"]

    load_interp_data(data_params, 'train')
    train_loader = create_training_dataset(data_params,
                                           shuffle=True)
    test_data = load_interp_data(data_params, 'test')
    test_a = Tensor(test_data['a'], mstype.float32)
    test_u = Tensor(test_data['u'], mstype.float32)

    train_a = Tensor(np.load(os.path.join(
        data_params["root_dir"], "train/train_a_interp.npy")), mstype.float32)
    train_u = Tensor(np.load(os.path.join(
        data_params["root_dir"], "train/train_u_interp.npy")), mstype.float32)

    test_a_unif = np.load(os.path.join(data_params['root_dir'], "test/test_a.npy"))
    test_a_unif = np.transpose(test_a_unif, (0, 3, 1, 2))

    test_u_unif = np.load(os.path.join(data_params['root_dir'], "test/test_u.npy"))
    test_u_unif = np.transpose(test_u_unif, (0, 3, 1, 2))

    t2 = default_timer()

    print_log('preprocessing finished, time used:', t2-t1)

    # prepare model
    n_modes = model_params['modes']
    poly_type = data_params['poly_type']
    transform_data = get_poly_transform(grid_size, n_modes, poly_type)

    transform = Tensor(transform_data["analysis"], mstype.float32)
    inv_transform = Tensor(transform_data["synthesis"], mstype.float32)

    transform_t_axis = get_poly_transform(output_timestep, n_modes, poly_type)
    transform_t = Tensor(transform_t_axis["analysis"], mstype.float32)
    inv_transform_t = Tensor(transform_t_axis["synthesis"], mstype.float32)
    transforms = [[transform, inv_transform]] * 2 + [[transform_t, inv_transform_t]]

    compute_type = mstype.float16 if use_ascend else mstype.float32

    model = SNO3D(
        in_channels=model_params['in_channels'],
        out_channels=model_params['out_channels'],
        hidden_channels=model_params['hidden_channels'],
        num_sno_layers=model_params['sno_layers'],
        transforms=transforms,
        kernel_size=model_params['kernel_size'],
        compute_dtype=compute_type)

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}-{v}")
    model_name = "_".join(model_params_list)

    total = 0
    for param in model.get_parameters():
        print_log(param.shape)
        total += param.size
    print_log(f"Total Parameters:{total}")

    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["learning_rate"],
                                        last_epoch=optimizer_params["epochs"],
                                        steps_per_epoch=train_loader.get_dataset_size(),
                                        warmup_epochs=optimizer_params["warmup_epochs"])

    steps_per_epoch = train_loader.get_dataset_size()
    print_log(f"number of steps_per_epochs: {steps_per_epoch}")

    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=Tensor(lr),
                                   eps=float(optimizer_params['eps']),
                                   weight_decay=optimizer_params['weight_decay'])

    loss_fn = nn.RMSELoss()
    a_normalizer = UnitGaussianNormalizer(train_a)
    u_normalizer = UnitGaussianNormalizer(train_u)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O2')
    else:
        loss_scaler = None

    def forward_fn(data, label):
        bs = data.shape[0]
        data = a_normalizer.encode(data)

        data = data.reshape(bs, input_timestep, grid_size, grid_size, 1).repeat(output_timestep, axis=-1)
        logits = model(data).reshape(bs, output_timestep, grid_size, grid_size)

        logits = u_normalizer.decode(logits)
        loss = loss_fn(logits.reshape(bs, -1), label.reshape(bs, -1))
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(data, label):
        loss, grads = grad_fn(data, label)

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

    sink_process = data_sink(train_step, train_loader, sink_size=steps_per_epoch)
    summary_dir = os.path.join(summary_params["root_dir"], model_name)
    ckpt_dir = os.path.join(summary_dir, summary_params["ckpt_dir"])
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.set_train()
    for epoch in range(1, 1 + optimizer_params["epochs"]):
        local_time_beg = time.time()
        model.set_train(True)
        cur_loss = sink_process()
        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = epoch_seconds / steps_per_epoch
        print_log(
            f"epoch: {epoch} train loss: {cur_loss} epoch time: {epoch_seconds:.3f}s step time: {step_seconds:5.3f}s")
        if epoch % summary_params['test_interval'] == 0:
            model.set_train(False)
            calculate_l2_error(model, test_a, test_u, test_u_unif, data_params, a_normalizer, u_normalizer)


if __name__ == "__main__":
    log_config('./logs', 'sno3d')
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())

    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    train(args)
