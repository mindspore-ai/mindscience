# Copyright 2022 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore
from mindspore import nn, context, ops, Tensor, jit, set_seed

from mindflow.cell import FNO2D
from mindflow.common import get_warmup_cosine_annealing_lr
from mindflow.loss import RelativeRMSELoss
from mindflow.utils import load_yaml_config
from mindflow.pde import UnsteadyFlowWithLoss
from src import calculate_l2_error, create_training_dataset


set_seed(123456)
np.random.seed(123456)

print("pid:", os.getpid())
print(datetime.datetime.now())

context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=5)
use_ascend = context.get_context(attr_key='device_target') == "Ascend"


def train():
    '''train and evaluate the network'''
    config = load_yaml_config('navier_stokes_2d.yaml')
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    # prepare dataset
    train_dataset = create_training_dataset(data_params,
                                            input_resolution=model_params["input_resolution"],
                                            shuffle=True)
    test_input = np.load(os.path.join(data_params["path"], "test/inputs.npy"))
    test_label = np.load(os.path.join(data_params["path"], "test/label.npy"))

    # prepare model
    model = FNO2D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
                  resolution=model_params["input_resolution"],
                  modes=model_params["modes"],
                  channels=model_params["width"],
                  depths=model_params["depth"]
                  )

    model_params_list = []
    for k, v in model_params.items():
        model_params_list.append(f"{k}-{v}")
    model_name = "_".join(model_params_list)

    # prepare optimizer
    steps_per_epoch = train_dataset.get_dataset_size()
    print("steps_per_epoch: ", steps_per_epoch)
    lr = get_warmup_cosine_annealing_lr(lr_init=optimizer_params["initial_lr"],
                                        last_epoch=optimizer_params["train_epochs"],
                                        steps_per_epoch=steps_per_epoch,
                                        warmup_epochs=optimizer_params["warmup_epochs"])

    optimizer = nn.Adam(model.trainable_params(), learning_rate=Tensor(lr))
    problem = UnsteadyFlowWithLoss(model, loss_fn=RelativeRMSELoss(), data_format="NHWC")

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O3')

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
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
                loss = ops.depend(loss, optimizer(grads))
        else:
            loss = ops.depend(loss, optimizer(grads))
        return loss

    sink_process = mindspore.data_sink(train_step, train_dataset, sink_size=1)
    summary_dir = os.path.join(config["summary_dir"], model_name)

    for cur_epoch in range(optimizer_params["train_epochs"]):
        local_time_beg = time.time()
        model.set_train()

        cur_loss = 0.0
        for _ in range(steps_per_epoch):
            cur_loss = sink_process()

        print("epoch: %s, loss is %s" % (cur_epoch + 1, cur_loss), flush=True)
        local_time_end = time.time()
        epoch_seconds = (local_time_end - local_time_beg) * 1000
        step_seconds = epoch_seconds / steps_per_epoch
        print("Train epoch time: {:5.3f} ms, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds), flush=True)

        if (cur_epoch + 1) % config["save_checkpoint_epoches"] == 0:
            ckpt_dir = os.path.join(summary_dir, "ckpt")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            mindspore.save_checkpoint(model, os.path.join(ckpt_dir, model_params["name"]))

        if (cur_epoch + 1) % config['eval_interval'] == 0:
            calculate_l2_error(model, test_input, test_label, config["test_batch_size"])


if __name__ == '__main__':
    train()
