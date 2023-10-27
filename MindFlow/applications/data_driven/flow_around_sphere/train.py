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
import argparse

import numpy as np

import mindspore
from mindspore import nn, context, ops, jit, set_seed
from mindspore import Tensor

from mindflow.utils import load_yaml_config
from mindflow.common import get_warmup_cosine_annealing_lr

from src import ResUnet3D, create_dataset, UnsteadyFlow3D, check_file_path, calculate_metric

set_seed(123456)
np.random.seed(123456)


def parse_args():
    """Parse input args"""
    parser = argparse.ArgumentParser(description='model train for 3d unsteady flow')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    parser.add_argument("--norm", type=bool, default=False, choices=[True, False],
                        help="Whether to perform data normalization on original data")
    parser.add_argument("--residual_mode", type=bool, default=True, choices=[True, False],
                        help="Whether to use indirect prediction mode")
    parser.add_argument("--scale", type=float, default=1000.0,
                        help="Whether to use indirect prediction mode")
    input_args = parser.parse_args()
    return input_args


def train():
    """train and evaluate the network"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]

    # prepare dataset
    # data for training
    train_loader = create_dataset(data_params, is_train=True, norm=args.norm, residual=args.residual_mode,
                                  scale=args.scale)
    train_dataset = train_loader.batch(model_params['batch_size'], drop_remainder=True)
    # data for evaluating
    eval_loader = create_dataset(data_params, is_eval=True, residual=args.residual_mode, scale=args.scale)
    eval_dataset = eval_loader.batch(1, drop_remainder=False)

    # prepare model
    model = ResUnet3D(in_channels=model_params['in_dims'], base_channels=model_params['base'],
                      out_channels=model_params['out_dims'])

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O1')
    else:
        loss_scaler = None

    # prepare optimizer and loss function
    steps_per_epoch = train_dataset.get_dataset_size()
    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params['initial_lr'],
                                        last_epoch=optimizer_params['train_epochs'],
                                        steps_per_epoch=steps_per_epoch,
                                        warmup_epochs=optimizer_params['warmup_epochs'])
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=Tensor(lr))

    problem = UnsteadyFlow3D(model, loss_fn=model_params['loss_fn'], metric_fn=model_params['metric_fn'],
                             loss_weight=model_params['loss_weight'], dynamic_flag=model_params['dynamic_flag'],
                             t_in=data_params['t_in'], t_out=data_params['t_out'],
                             residual=args.residual_mode, scale=args.scale)

    def forward_fn(train_inputs, train_label):
        loss = problem.get_loss(train_inputs, train_label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(train_inputs, train_label):
        loss, grads = grad_fn(train_inputs, train_label)
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

    # data sink
    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)

    # The summary_dir path can be adjusted according to the search strategy of task parameters
    summary_dir = os.path.join(summary_params['summary_dir'], f"norm-{args.norm}",
                               f"resi-{args.residual_mode} scale-{args.scale} {model_params['loss_fn']}")
    ckpt_dir = os.path.join(summary_dir, "ckpt")
    check_file_path(ckpt_dir)

    for cur_epoch in range(1, optimizer_params['train_epochs'] + 1):
        local_time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epoch):
            cur_loss = sink_process()

        epoch_time = time.time() - local_time_beg
        print(f"epoch: {cur_epoch:-4d}  loss: {cur_loss.asnumpy():.8f}  epoch time: {epoch_time:.2f}s", flush=True)

        if cur_epoch % summary_params['eval_interval'] == 0:
            model.set_train(False)
            # A uniform metric than total loss is unified as the evaluation standard
            calculate_metric(problem, eval_dataset)
            mindspore.save_checkpoint(model, os.path.join(ckpt_dir, f'ckpt-{cur_epoch}'))


if __name__ == "__main__":
    print("pid:", os.getpid())
    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print(f"Running in {args.mode.upper()} mode within {args.device_target} device, using device id: {args.device_id}.")
    start_time = time.time()
    train()
    print(f"Start-to-End total time: {(time.time() - start_time):.2f}s")
