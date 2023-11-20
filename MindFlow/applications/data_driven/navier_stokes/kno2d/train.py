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
import datetime
import argparse
import numpy as np

from mindspore import nn, context, ops, Tensor, set_seed, dtype, save_checkpoint, data_sink
from mindspore.nn.loss import MSELoss

from mindflow.cell import KNO2D
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.utils import load_yaml_config, print_log, log_config, log_timer

from src import create_training_dataset, NavierStokesWithLoss, visual

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Navier Stokes problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./configs/kno2d.yaml")
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
    test_input = np.load(os.path.join(data_params["root_dir"], "test/inputs.npy"))
    test_label = np.load(os.path.join(data_params["root_dir"], "test/label.npy"))
    print_log('test_input: ', test_input.shape)
    print_log('test_label: ', test_label.shape)

    model = KNO2D(in_channels=data_params['in_channels'],
                  channels=model_params['channels'],
                  modes=model_params['modes'],
                  depths=model_params['depths'],
                  resolution=model_params['resolution'],
                  compute_dtype=dtype.float16 if use_ascend else dtype.float32
                  )

    train_size = train_dataset.get_dataset_size()
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
    problem = NavierStokesWithLoss(model, data_params["out_channels"], loss_fn, data_format="NHWTC")

    def forward_fn(inputs, labels):
        loss, l_recons, l_pred = problem.get_loss(inputs, labels)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss, l_recons, l_pred

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

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

    train_sink = data_sink(train_step, train_dataset, sink_size=1)

    ckpt_dir = os.path.join(summary_params["root_dir"], summary_params["ckpt_dir"])
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, optimizer_params["epochs"] + 1):
        local_time_beg = time.time()
        train_recons_full = 0.0
        train_pred_full = 0.0
        train_full = 0.0
        for _ in range(train_size):
            l_full, l_recons, l_pred = train_sink()
            train_recons_full += l_recons.asnumpy()
            train_pred_full += l_pred.asnumpy()
            train_full += l_full.asnumpy()
        train_recons_full = train_recons_full / train_size
        train_pred_full = train_pred_full / train_size
        train_full = train_full / train_size
        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = (epoch_seconds/train_size)*1000
        print_log(f"epoch: {epoch} recons loss: {train_recons_full:>8f} pred loss: {train_pred_full:>8f}"
                  f" train loss: {train_full:>8f} epoch time: {epoch_seconds:.3f}s step time: {step_seconds:5.3f}ms")

        if epoch % summary_params['test_interval'] == 0:
            eval_time_start = time.time()
            print_log("================================Start Evaluation================================")
            l_recons_all, l_pred_all = problem.test(test_input, test_label)
            print_log(f'Eval epoch: {epoch}, recons loss: {l_recons_all},'
                      f' relative pred loss: {l_pred_all}')
            print_log("=================================End Evaluation=================================")
            print_log(f'evaluation time: {time.time() - eval_time_start}s')
            save_checkpoint(model, ckpt_file_name=os.path.join(ckpt_dir, 'kno2d.ckpt'))

    # Infer and plot some data.
    visual(problem, test_input, test_label, t_out=10)


if __name__ == '__main__':
    log_config('./logs', 'kno2d')
    print_log("pid:", os.getpid())
    print_log(datetime.datetime.now())
    args = parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    train(args)
