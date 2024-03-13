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
"""
train
"""
import os
import time
import argparse
import datetime
from timeit import default_timer
import numpy as np

from mindspore import nn, ops, jit, data_sink, save_checkpoint, context, Tensor
from mindspore.common import set_seed
from mindspore import dtype as mstype

from mindflow import get_warmup_cosine_annealing_lr, load_yaml_config
from mindflow.utils import print_log, log_config
from mindflow.cell.neural_operators.fno import FNO3D

from src import LpLoss, UnitGaussianNormalizer, create_training_dataset

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
                        default="./configs/fno3d.yaml")
    input_args = parser.parse_args()
    return input_args


def train(input_args):
    '''train and evaluate the network'''
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    t1 = default_timer()

    sub = data_params["sub"]
    grid_size = model_params["input_resolution"] // sub
    input_timestep = model_params["input_timestep"]
    output_timestep = model_params["output_timestep"]

    train_a = Tensor(np.load(os.path.join(
        data_params["path"], "train_a.npy")), mstype.float32)
    train_u = Tensor(np.load(os.path.join(
        data_params["path"], "train_u.npy")), mstype.float32)
    test_a = Tensor(np.load(os.path.join(
        data_params["path"], "test_a.npy")), mstype.float32)
    test_u = Tensor(np.load(os.path.join(
        data_params["path"], "test_u.npy")), mstype.float32)

    print_log(train_a.shape, test_a.shape)

    train_loader = create_training_dataset(data_params,
                                           shuffle=True)

    t2 = default_timer()

    print_log('preprocessing finished, time used:', t2-t1)

    if use_ascend:
        compute_type = mstype.float16
    else:
        compute_type = mstype.float32

    model = FNO3D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
                  n_modes=model_params["modes"],
                  resolutions=[model_params["input_resolution"],
                               model_params["input_resolution"], output_timestep],
                  hidden_channels=model_params["width"],
                  n_layers=model_params["depth"],
                  projection_channels=4*model_params["width"],
                  fno_compute_dtype=compute_type
                  )

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}-{v}")
    model_name = "_".join(model_params_list)

    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["initial_lr"],
                                        last_epoch=optimizer_params["train_epochs"],
                                        steps_per_epoch=train_loader.get_dataset_size(),
                                        warmup_epochs=optimizer_params["warmup_epochs"])

    optimizer = nn.optim.Adam(model.trainable_params(),
                              learning_rate=Tensor(lr), weight_decay=optimizer_params['weight_decay'])

    loss_fn = LpLoss()
    a_normalizer = UnitGaussianNormalizer(train_a)
    y_normalizer = UnitGaussianNormalizer(train_u)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O2')
    else:
        loss_scaler = None

    def forward_fn(data, label):
        bs = data.shape[0]
        data = a_normalizer.encode(data)
        label = y_normalizer.encode(label)
        data = data.reshape(bs, grid_size, grid_size, 1, input_timestep).repeat(
            output_timestep, axis=3)
        logits = model(data).reshape(bs, grid_size, grid_size, output_timestep)
        logits = y_normalizer.decode(logits)
        label = y_normalizer.decode(label)
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

    def calculate_l2_error(model, inputs, labels):
        """
        Evaluate the model respect to input data and label.

        Args:
            model (Cell): list of expressions node can by identified by mindspore.
            inputs (Tensor): the input data of network.
            labels (Tensor): the true output value of given inputs.

        """
        print_log("================================Start Evaluation================================")
        time_beg = time.time()
        rms_error = 0.0
        for i in range(labels.shape[0]):
            label = labels[i:i + 1]
            test_batch = inputs[i:i + 1]
            test_batch = a_normalizer.encode(test_batch)
            label = y_normalizer.encode(label)

            test_batch = test_batch.reshape(1, grid_size,
                                            grid_size, 1, input_timestep).repeat(output_timestep, axis=3)
            prediction = model(test_batch).reshape(1, grid_size, grid_size, output_timestep)
            prediction = y_normalizer.decode(prediction)
            label = y_normalizer.decode(label)
            rms_error_step = loss_fn(prediction.reshape(
                1, -1), label.reshape(1, -1))
            rms_error += rms_error_step

        rms_error = rms_error / labels.shape[0]
        print_log("mean rms_error:", rms_error)
        print_log("predict total time: {} s".format(time.time() - time_beg))
        print_log("=================================End Evaluation=================================")

    sink_process = data_sink(train_step, train_loader, sink_size=100)
    summary_dir = os.path.join(config["summary_dir"], model_name)
    ckpt_dir = os.path.join(summary_dir, "ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    model.set_train()
    for step in range(1, 1 + optimizer_params["train_epochs"]):
        local_time_beg = time.time()
        cur_loss = sink_process()
        print_log(
            f"epoch: {step} train loss: {cur_loss} epoch time: {time.time() - local_time_beg:.2f}s")
        if step % 10 == 0:
            print_log(f"loss: {cur_loss.asnumpy():>7f}")
            print_log("step: {}, time elapsed: {}ms".format(
                step, (time.time() - local_time_beg) * 1000))
            calculate_l2_error(model, test_a, test_u)
            save_checkpoint(model, os.path.join(
                ckpt_dir, model_params["name"]))


if __name__ == "__main__":
    log_config('./logs', 'fno3d')
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())

    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    train(args)
