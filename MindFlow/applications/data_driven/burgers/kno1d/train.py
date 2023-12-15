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
# ==============================================================================
"""
train
"""
import os
import time
import argparse
import datetime
import numpy as np

from mindspore import nn, context, ops, Tensor, set_seed, data_sink, save_checkpoint, jit
from mindspore.nn.loss import MSELoss

from mindflow.cell import KNO1D
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config, print_log, log_config, log_timer

from src import create_training_dataset, BurgersWithLoss, visual

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Burgers 1D problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=1,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/kno1d.yaml")
    input_args = parser.parse_args()
    return input_args


@log_timer
def train(input_args):
    '''Train and evaluate the network'''
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]
    # create training dataset
    train_dataset = create_training_dataset(data_params, shuffle=True)

    # create test dataset
    eval_dataset = create_training_dataset(
        data_params, shuffle=False, is_train=False)

    model = KNO1D(in_channels=data_params['in_channels'],
                  channels=model_params['channels'],
                  modes=model_params['modes'],
                  depths=model_params['depths'],
                  resolution=model_params['resolution']
                  )

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}:{v}")
    model_name = "_".join(model_params_list)
    print_log(model_name)
    total = 0
    for param in model.get_parameters():
        print_log(param.shape)
        total += param.size
    print_log(f"Toatal Parameters:{total}")

    train_size = train_dataset.get_dataset_size()
    eval_size = eval_dataset.get_dataset_size()
    print_log(f"number of steps_per_epochs: {train_size}")

    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["learning_rate"],
                                        last_epoch=optimizer_params["epochs"],
                                        steps_per_epoch=train_size,
                                        warmup_epochs=1)
    optimizer = nn.AdamWeightDecay(model.trainable_params(),
                                   learning_rate=Tensor(lr),
                                   weight_decay=optimizer_params["weight_decay"])
    model.set_train()
    loss_fn = MSELoss()
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')
    else:
        loss_scaler = None
    problem = BurgersWithLoss(model, data_params["out_channels"], loss_fn)

    def forward_fn(inputs, labels):
        loss, l_recons, l_pred = problem.get_loss(inputs, labels)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss, l_recons, l_pred

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(inputs, labels):
        (loss, l_recons, l_pred), grads = grad_fn(inputs, labels)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            is_finite = all_finite(grads)
            if is_finite:
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, optimizer(grads))
            loss_scaler.adjust(is_finite)
        else:
            loss = ops.depend(loss, optimizer(grads))
        return loss, l_recons, l_pred

    @jit
    def eval_step(inputs, labels):
        return problem.get_rel_loss(inputs, labels)

    train_sink = data_sink(train_step, train_dataset, sink_size=1)
    eval_sink = data_sink(eval_step, eval_dataset, sink_size=1)

    summary_dir = summary_params["summary_dir"]
    os.makedirs(summary_dir, exist_ok=True)
    print_log(summary_dir)

    for epoch in range(1, optimizer_params["epochs"] + 1):
        local_time_beg = time.time()
        l_recons_train = 0.0
        l_pred_train = 0.0
        l_train = 0.0
        for _ in range(train_size):
            l_full, l_recons, l_pred = train_sink()
            l_recons_train += l_recons.asnumpy()
            l_pred_train += l_pred.asnumpy()
            l_train += l_full.asnumpy()
        l_recons_train = l_recons_train / train_size
        l_pred_train = l_pred_train / train_size
        l_train = l_train / train_size
        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = (epoch_seconds/train_size)*1000
        print_log(f"epoch: {epoch} recons loss: {l_recons_train:>8f} pred loss: {l_pred_train:>8f}"
                  f" train loss: {l_train} epoch time: {epoch_seconds:.3f}s step time: {step_seconds:5.3f}ms")

        if epoch % summary_params['test_interval'] == 0:
            eval_time_start = time.time()
            l_recons_eval = 0.0
            l_pred_eval = 0.0
            print_log(
                "---------------------------start evaluation-------------------------")
            for _ in range(eval_size):
                l_recons, l_pred = eval_sink()
                l_recons_eval += l_recons.asnumpy()
                l_pred_eval += l_pred.asnumpy()
            l_recons_eval = l_recons_eval / eval_size
            l_pred_eval = l_pred_eval / eval_size
            print_log(f'Eval epoch: {epoch}, recons loss: {l_recons_eval},'
                      f' relative pred loss: {l_pred_eval}')
            print_log(
                "---------------------------end evaluation---------------------------")
            print_log(f'evaluation time: {time.time() - eval_time_start}s')
            save_checkpoint(
                model, ckpt_file_name=summary_dir + '/kno1d.ckpt')

    # Infer and plot some data.
    inputs = np.load(os.path.join(
        data_params["root_dir"], "test/inputs.npy"))  # (200,1024,1)
    problem = BurgersWithLoss(model, 10, loss_fn)
    visual(problem, inputs)


if __name__ == '__main__':
    log_config('./logs', 'kno1d')
    print_log("pid:", os.getpid())
    print_log(datetime.datetime.now())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    train(args)
