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
import datetime
import time
import argparse
import numpy as np

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore import dtype as mstype
from mindspore import save_checkpoint, jit, data_sink
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.pde import SteadyFlowWithLoss
from mindflow.loss import WaveletTransformLoss
from mindflow.cell import ViT
from mindflow.utils import load_yaml_config, print_log, log_config, log_timer

from src import AirfoilDataset, plot_u_and_cp, get_ckpt_summary_dir, plot_u_v_p, calculate_test_error

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(
        description='Airfoil 2D_steady Simulation')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Support context mode: 'GRAPH', 'PYNATIVE'")
    parser.add_argument('--train_mode', type=str, default='train', choices=["train", "test", "finetune"],
                        help="Support run mode: 'train', 'test', 'finetune'")
    parser.add_argument('--device_id', type=int, default=4,
                        help="ID of the target device")
    parser.add_argument('--device_target', type=str, default='Ascend', choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/vit.yaml")
    input_args = parser.parse_args()
    return input_args


@log_timer
def train(input_args):
    '''Train and test the network'''
    mode = input_args.train_mode
    print_log(f'running mode: {mode}')
    # read params
    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]
    # prepare dataset
    max_value_list = data_params['max_value_list']
    min_value_list = data_params['min_value_list']
    dataset = AirfoilDataset(max_value_list, min_value_list)
    batch_size = data_params['batch_size']

    train_dataset, test_dataset = dataset.create_dataset(
        dataset_dir=data_params['root_dir'], train_file_name=data_params['train_file_name'],
        test_file_name=data_params['test_file_name'],
        finetune_file_name=data_params['finetune_file_name'],
        batch_size=batch_size,
        shuffle=False,
        mode=mode,
        finetune_size=data_params['finetune_ratio'],
        drop_remainder=True)
    # prepare loss scaler
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        compute_dtype = mstype.float16
    else:
        loss_scaler = None
        compute_dtype = mstype.float32

    # model construction
    model = ViT(in_channels=model_params['in_channels'],
                out_channels=model_params['out_channels'],
                encoder_depths=model_params['encoder_depth'],
                encoder_embed_dim=model_params['encoder_embed_dim'],
                encoder_num_heads=model_params['encoder_num_heads'],
                decoder_depths=model_params['decoder_depth'],
                decoder_embed_dim=model_params['decoder_embed_dim'],
                decoder_num_heads=model_params['decoder_num_heads'],
                compute_dtype=compute_dtype
                )
    grid_path = os.path.join(
        data_params['root_dir'], data_params['grid_file_name'])

    if mode in ('finetune', 'test'):
        # load pretrained model
        param_dict = load_checkpoint(model_params['ckpt_path'])
        load_param_into_net(model, param_dict)
        print_log("Load pre-trained model successfully")
        if mode == 'finetune':
            optimizer_params["epochs"] = 200
            config["save_ckpt_interval"] = 200
        else:
            plot_u_v_p(test_dataset, model,
                       grid_path, summary_params['postprocess_dir'])
            calculate_test_error(test_dataset, model, True,
                                 summary_params['postprocess_dir'])
            return

    model_name = "_".join(
        [model_params['name'], "bs", str(batch_size)])
    # prepare loss
    ckpt_dir, summary_dir = get_ckpt_summary_dir(
        summary_params['summary_dir'], model_name)
    wave_loss = WaveletTransformLoss(wave_level=optimizer_params['wave_level'])
    problem = SteadyFlowWithLoss(model, loss_fn=wave_loss)
    # prepare optimizer
    steps_per_epoch = train_dataset.get_dataset_size()
    print_log(f"number of steps_per_epochs: {steps_per_epoch}")
    epochs = optimizer_params["epochs"]
    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["learning_rate"],
                                        last_epoch=epochs,
                                        steps_per_epoch=steps_per_epoch,
                                        warmup_epochs=1)
    optimizer = nn.Adam(model.trainable_params() +
                        wave_loss.trainable_params(), learning_rate=Tensor(lr))

    def forward_fn(x, y):
        loss = problem.get_loss(x, y)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(x, y):
        loss, grads = grad_fn(x, y)
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

    train_sink_process = data_sink(train_step, train_dataset, sink_size=1)
    test_interval = summary_params['test_interval']
    plot_interval = summary_params['plot_interval']
    save_ckpt_interval = summary_params['save_ckpt_interval']
    # train process
    for epoch in range(1, 1 + epochs):
        # train
        local_time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epoch):
            step_train_loss = train_sink_process()
        local_time_end = time.time()
        epoch_seconds = local_time_end - local_time_beg
        step_seconds = (epoch_seconds/steps_per_epoch)*1000
        print_log(f"epoch: {epoch} train loss: {step_train_loss} "
                  f"epoch time: {epoch_seconds:.3f}s step time: {step_seconds:5.3f}ms")

        model.set_train(False)
        # test
        if epoch % test_interval == 0:
            calculate_test_error(test_dataset, model)
        # plot
        if epoch % plot_interval == 0:
            plot_u_and_cp(test_dataset=test_dataset, model=model,
                          grid_path=grid_path, save_dir=summary_dir)
        # save checkpoint
        if epoch % save_ckpt_interval == 0:
            ckpt_name = f"epoch_{epoch}.ckpt"
            save_checkpoint(model, os.path.join(ckpt_dir, ckpt_name))
            print_log(f'{ckpt_name} save success')


if __name__ == '__main__':
    log_config('./logs', 'vit')
    print_log(f'pid: {os.getpid()}')
    print_log(datetime.datetime.now())

    args = parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.context_mode.upper().startswith("GRAPH")
                        else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print_log(
        f"Running in {args.context_mode.upper()} mode, using device id: {args.device_id}.")
    use_ascend = (args.device_target == "Ascend")
    print_log(f'use_ascend : {use_ascend}')
    train(args)
