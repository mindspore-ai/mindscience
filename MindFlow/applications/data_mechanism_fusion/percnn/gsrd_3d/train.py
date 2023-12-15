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
"""train"""
import argparse
import os
import sys
import time

import numpy as np
from mindspore import context, jit, nn, ops, save_checkpoint, set_seed
import mindspore.common.dtype as mstype
from mindflow.utils import load_yaml_config, print_log, log_config
from src import RecurrentCnn, post_process, Trainer, UpScaler, count_params

set_seed(123456)
np.random.seed(123456)


def train_stage(trainer, stage, config, ckpt_dir, use_ascend):
    """train stage"""
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, all_finite
        loss_scaler = DynamicLossScaler(2**10, 2, 100)

    if 'milestone_num' in config.keys():
        milestone = list([(config['epochs']//config['milestone_num'])*(i + 1)
                          for i in range(config['milestone_num'])])
        learning_rate = config['learning_rate']
        lr = float(config['learning_rate'])*np.array(list([config['gamma']
                                                           ** i for i in range(config['milestone_num'])]))
        learning_rate = nn.piecewise_constant_lr(milestone, list(lr))
    else:
        learning_rate = config['learning_rate']

    if stage == 'pretrain':
        params = trainer.upconv.trainable_params()
    else:
        params = trainer.upconv.trainable_params() + trainer.recurrent_cnn.trainable_params()

    optimizer = nn.Adam(params, learning_rate=learning_rate)

    def forward_fn():
        if stage == 'pretrain':
            loss = trainer.get_ic_loss()
        else:
            loss = trainer.get_loss()
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    if stage == 'pretrain':
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=False)
    else:
        grad_fn = ops.value_and_grad(forward_fn, None, params, has_aux=True)

    @jit
    def train_step():
        loss, grads = grad_fn()
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

    best_loss = sys.maxsize
    for epoch in range(1, 1 + config['epochs']):
        time_beg = time.time()
        trainer.upconv.set_train(True)
        trainer.recurrent_cnn.set_train(True)
        if stage == 'pretrain':
            step_train_loss = train_step()
            print_log(
                f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) :.3f} s")
        else:
            step_train_loss, loss_data, loss_ic, loss_phy, loss_valid = train_step()
            print_log(f"epoch: {epoch} train loss: {step_train_loss} ic_loss: {loss_ic} data_loss: {loss_data} "
                      f"val_loss: {loss_valid} phy_loss: {loss_phy} epoch time: {(time.time() - time_beg): .3f}s")
            if step_train_loss < best_loss:
                best_loss = step_train_loss
                print_log('best loss', best_loss, 'save model')
                save_checkpoint(trainer.upconv, os.path.join(ckpt_dir, "train_upconv.ckpt"))
                save_checkpoint(trainer.recurrent_cnn,
                                os.path.join(ckpt_dir, "train_recurrent_cnn.ckpt"))



def parse_args():
    """parse input args"""
    parser = argparse.ArgumentParser(description="rd train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/percnn_3d_rd.yaml")
    input_args = parser.parse_args()
    return input_args


def train(input_args):
    """train"""
    rd_config = load_yaml_config(input_args.config_file_path)
    data_config = rd_config['data']
    model_config = rd_config['model']
    optim_config = rd_config['optimizer']
    summary_config = rd_config['summary']

    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    if use_ascend:
        compute_dtype = mstype.float16
    else:
        compute_dtype = mstype.float32

    upconv_config = model_config['upconv']
    upconv = UpScaler(in_channels=upconv_config['in_channel'],
                      out_channels=upconv_config['out_channel'],
                      hidden_channels=upconv_config['hidden_channel'],
                      kernel_size=upconv_config['kernel_size'],
                      stride=upconv_config['stride'],
                      has_bias=True)

    if use_ascend:
        from mindspore.amp import auto_mixed_precision
        auto_mixed_precision(upconv, 'O1')

    rcnn_config = model_config['rcnn']
    recurrent_cnn = RecurrentCnn(input_channels=rcnn_config['in_channel'],
                                 hidden_channels=rcnn_config['hidden_channel'],
                                 kernel_size=rcnn_config['kernel_size'],
                                 stride=rcnn_config['stride'],
                                 compute_dtype=compute_dtype)

    percnn_trainer = Trainer(upconv=upconv,
                             recurrent_cnn=recurrent_cnn,
                             timesteps_for_train=data_config['rollout_steps'],
                             dx=data_config['dx'],
                             grid_size=data_config['grid_size'],
                             dt=data_config['dt'],
                             mu=data_config['mu'],
                             data_path=data_config['data_path'],
                             compute_dtype=compute_dtype)

    total_params = int(count_params(upconv.trainable_params()) + count_params(recurrent_cnn.trainable_params()))
    print_log(f"There are {total_params} parameters")

    ckpt_dir = summary_config["ckpt_dir"]
    fig_path = summary_config["fig_save_path"]
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    train_stage(percnn_trainer, 'pretrain', optim_config['pretrain'], ckpt_dir, use_ascend)
    train_stage(percnn_trainer, 'finetune', optim_config['finetune'], ckpt_dir, use_ascend)

    extrapolation_steps = data_config['extrapolations']
    output = percnn_trainer.get_output(extrapolation_steps).asnumpy()
    output = np.transpose(output, (1, 0, 2, 3, 4))[:, :-1:10]

    print_log('output shape is ', output.shape)
    for i in range(0, extrapolation_steps//10, 2):
        post_process(output[0, i], fig_path, is_u=True, num=i)


if __name__ == '__main__':
    args = parse_args()
    log_config('./logs', 'percnn')
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id,
                        max_call_depth=99999999)
    print_log("pid:", os.getpid())
    print_log(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    train(args)
