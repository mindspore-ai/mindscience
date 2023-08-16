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

from mindflow.utils import load_yaml_config
from mindflow.loss import RelativeRMSELoss
from mindflow.pde import UnsteadyFlowWithLoss

from src import init_model, create_dataset, calculate_lp_loss_error
from src import make_dir, scheduler, get_param_dic

set_seed(0)
np.random.seed(0)


def train_single_step(step, config_param, lr, train_dataset, eval_dataset):
    """train PDE-Net with advancing steps"""
    print("Current step for train loop: {}".format(step,))
    model = init_model(config_param)

    epoch = config_param["epochs"]
    warm_up_epoch_scale = 10
    if step == 1:
        model.if_fronzen = True
        epoch = warm_up_epoch_scale * epoch
    elif step == 2:
        param_dict = get_param_dic(config_param["summary_dir"], step - 1, epoch * 10)
        load_param_into_net(model, param_dict)
        print("Load pre-trained model successfully")
    else:
        param_dict = get_param_dic(config_param["summary_dir"], step - 1, epoch)
        load_param_into_net(model, param_dict)
        print("Load pre-trained model successfully")

    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))
    problem = UnsteadyFlowWithLoss(model, t_out=step, loss_fn=RelativeRMSELoss(), data_format="NTCHW")

    def forward_fn(u0, u_t):
        loss = problem.get_loss(u0, u_t)
        return loss

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

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
        print(f"epoch: {cur_epoch} train loss: {cur_loss} epoch time: {epoch_seconds:5.3f}ms", flush=True)

        if cur_epoch % config_param["save_epoch_interval"] == 0:
            ckpt_file_name = "ckpt/step_{}".format(step)
            ckpt_dir = os.path.join(config_param["summary_dir"], ckpt_file_name)
            make_dir(ckpt_dir)
            ckpt_name = "pdenet-{}.ckpt".format(cur_epoch,)
            mindspore.save_checkpoint(model, os.path.join(ckpt_dir, ckpt_name))

        if cur_epoch % config_param['eval_interval'] == 0:
            eval_time_start = time.time()
            calculate_lp_loss_error(problem, eval_dataset, config_param["batch_size"])
            print(f'evaluation total time: {time.time() - eval_time_start}s')



def train(config_param):
    lr = config_param["lr"]
    for i in range(1, config_param["multi_step"] + 1):
        db_name = "train_step{}.mindrecord".format(i)
        dataset = create_dataset(config_param, i, db_name, "train", data_size=2 * config_param["batch_size"])
        train_dataset, eval_dataset = dataset.create_train_dataset()
        lr = scheduler(int(config_param["multi_step"] / config_param["learning_rate_reduce_times"]), step=i, lr=lr)
        train_single_step(step=i, config_param=config_param, lr=lr, train_dataset=train_dataset,
                          eval_dataset=eval_dataset)


if __name__ == '__main__':
    print("pid:", os.getpid())

    parser = argparse.ArgumentParser(description="pde net train")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./pde_net.yaml")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") \
                        else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    print(f"Running in {args.mode.upper()} mode, using device id: {args.device_id}.")
    config = load_yaml_config(args.config_file_path)
    make_dir(config["mindrecord_data_dir"])
    train(config)
