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
train pde net
"""
import argparse
import os
import time
import numpy as np

import mindspore
from mindspore.common import set_seed
from mindspore import nn, Tensor, context, ops, jit
from mindspore.train.serialization import load_param_into_net

from mindflow.utils import load_yaml_config, print_log, log_config, log_timer
from mindflow.loss import RelativeRMSELoss
from mindflow.pde import UnsteadyFlowWithLoss

from src import init_model, create_dataset, calculate_lp_loss_error
from src import make_dir, scheduler, get_param_dic

set_seed(0)
np.random.seed(0)


def train_single_step(step, config_param, lr, train_dataset, eval_dataset):
    """train PDE-Net with advancing steps"""
    print_log(f"Current step for train loop: {step}")
    model = init_model(config_param["model"])
    data_params = config_param["data"]
    summary_params = config_param["summary"]
    optimizer_params = config_param["optimizer"]

    epoch = optimizer_params["epochs"]
    warm_up_epoch_scale = 10
    if step == 1:
        model.if_fronzen = True
        epoch = warm_up_epoch_scale * epoch
    elif step == 2:
        param_dict = get_param_dic(os.path.join(
            summary_params["root_dir"], summary_params["ckpt_dir"]), step - 1, epoch * 10)
        load_param_into_net(model, param_dict)
        print_log("Load pre-trained model successfully")
    else:
        param_dict = get_param_dic(os.path.join(
            summary_params["root_dir"], summary_params["ckpt_dir"]), step - 1, epoch)
        load_param_into_net(model, param_dict)
        print_log("Load pre-trained model successfully")

    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))
    problem = UnsteadyFlowWithLoss(
        model, t_out=step, loss_fn=RelativeRMSELoss(), data_format="NTCHW")

    def forward_fn(u0, u_t):
        loss = problem.get_loss(u0, u_t)
        return loss

    grad_fn = ops.value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    @jit
    def train_step(u0, u_t):
        loss, grads = grad_fn(u0, u_t)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    steps = train_dataset.get_dataset_size()
    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)

    for cur_epoch in range(1, 1+epoch):
        local_time_beg = time.time()
        model.set_train()
        for _ in range(steps):
            cur_loss = sink_process()
        local_time_end = time.time()
        epoch_seconds = (local_time_end - local_time_beg) * 1000
        step_seconds = epoch_seconds/steps
        print_log(f"epoch: {cur_epoch} train loss: {cur_loss} "
                  f"epoch time: {epoch_seconds:5.3f}ms step time: {step_seconds:5.3f}ms")

        if cur_epoch % summary_params["save_epoch_interval"] == 0:
            ckpt_file_name = f"step_{step}"
            ckpt_dir = os.path.join(
                summary_params["root_dir"], summary_params["ckpt_dir"], ckpt_file_name)
            make_dir(ckpt_dir)
            ckpt_name = f"pdenet-{cur_epoch}.ckpt"
            mindspore.save_checkpoint(model, os.path.join(ckpt_dir, ckpt_name))

        if cur_epoch % summary_params['test_interval'] == 0:
            eval_time_start = time.time()
            calculate_lp_loss_error(
                problem, eval_dataset, data_params["batch_size"])
            print_log(f'evaluation time: {time.time() - eval_time_start}s')


@log_timer
def train(input_args):
    '''Train and evaluate the network'''
    config_param = load_yaml_config(input_args.config_file_path)
    data_params = config_param["data"]
    optimizer_params = config_param["optimizer"]

    db_path = data_params["mindrecord_data_dir"]
    make_dir(db_path)
    lr = optimizer_params["learning_rate"]
    for i in range(1, optimizer_params["multi_step"] + 1):
        db_name = f"train_step{i}.mindrecord"
        dataset = create_dataset(
            config_param, i, db_name, "train", data_size=2 * data_params["batch_size"])
        train_dataset, eval_dataset = dataset.create_train_dataset()
        lr = scheduler(int(
            optimizer_params["multi_step"] / optimizer_params["learning_rate_reduce_times"]), step=i, lr=lr)
        train_single_step(step=i, config_param=config_param, lr=lr, train_dataset=train_dataset,
                          eval_dataset=eval_dataset)


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description="pde net train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/pde_net.yaml")
    input_args = parser.parse_args()
    return input_args


if __name__ == '__main__':
    log_config('./logs', 'pde_net')
    print_log("pid:", os.getpid())

    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH")
                        else context.PYNATIVE_MODE,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print_log(
        f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")

    train(args)
