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
"""train process composed of the following parts.
   1. train neural network
   2. generate metadata and calculate derivatives
   3. finally obtain potential entries
"""
import os
import argparse
import random
import time

from mindspore import nn, context, ops, set_seed
from mindspore import value_and_grad, jit, data_sink, save_checkpoint
from mindspore.amp import DynamicLossScaler, auto_mixed_precision

from mindflow.utils import load_yaml_config
from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import print_log, log_config, log_timer

from src import create_dataset, evaluate, produce_meta_data
from src import gene_algorithm

set_seed(0)
random.seed(0)


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description="train r_glda")
    parser.add_argument("--case", type=str, default="burgers", choices=["burgers", "cylinder_flow", "periodic_hill"],
                        help="choose burgers, cylinder_flow or periodic_hill")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/burgers.yaml")
    result_args = parser.parse_args()
    return result_args


@log_timer
def train(args):
    """train process"""
    # load configuration
    case_name = args.case

    print_log(case_name)

    config = load_yaml_config(args.config_file_path)
    dataset_config = config["dataset"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    epochs = optimizer_config["epochs"]
    summary_config = config["summary"]

    print_log("start creating dataset")
    # create dataset for training and validating
    train_dataset, inputs, label = create_dataset(case_name, dataset_config)
    print_log(train_dataset)
    print_log("start creating model")
    model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                   out_channels=model_config["out_channels"],
                                   layers=model_config["layers"],
                                   neurons=model_config["neurons"],
                                   residual=model_config["residual"],
                                   act=model_config["activation"],
                                   num_scales=1)

    optimizer = nn.Adam(model.trainable_params(),
                        optimizer_config["initial_lr"])

    # set ascend
    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, model_config["amp_level"])
    else:
        loss_scaler = None

    save_ckpt_path = summary_config["save_ckpt_path"]

    # create ckpt dir
    if not os.path.exists(os.path.abspath(save_ckpt_path)):
        os.makedirs(os.path.abspath(save_ckpt_path))

    # define forward function
    def forward_fn(data, label):
        prediction = model(data)
        loss = nn.MSELoss()(prediction, label)
        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    # define gradient function
    grad_fn = value_and_grad(
        forward_fn, None, optimizer.parameters, has_aux=False)

    # define train_step
    @jit
    def train_step(data, label):
        loss, grads = grad_fn(data, label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    # data sink
    sink_process = data_sink(train_step, train_dataset, sink_size=1)
    steps_per_epochs = train_dataset.get_dataset_size()
    print_log(steps_per_epochs)
    print_log(train_dataset)

    print_log("----start training----")
    # train loop for nn
    for epoch in range(1, epochs + 1):
        time_beg = time.time()
        model.set_train(True)
        for _ in range(steps_per_epochs):
            step_train_loss = sink_process()

        # set model to eval mode
        model.set_train(False)

        if epoch % summary_config["validate_interval_epochs"] == 0:
            # current epoch loss
            print_log(
                f"epoch: {epoch} train loss: {step_train_loss} epoch time: {(time.time() - time_beg) * 1000}ms")
            evaluate(model, inputs, label, config)

        # save checkpoint
        if epoch % summary_config["save_checkpoint_epochs"] == 0:
            ckpt_name = f"{case_name}_nn-{epoch + 1}.ckpt"
            save_checkpoint(model, os.path.join(
                save_ckpt_path, ckpt_name))

    # produce meta data
    produce_meta_data(case_name, config)

    # ga
    gene_algorithm(case_name, config)


if __name__ == '__main__':
    input_args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if input_args.mode.upper().startswith("GRAPH")
                        else context.PYNATIVE_MODE,
                        device_target=input_args.device_target,
                        device_id=input_args.device_id)
    log_config('./logs', f'{input_args.case}')
    print_log(
        f"Running in {input_args.mode.upper()} mode, using device id: {input_args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(use_ascend)
    print_log(f"pid: {os.getpid()}")
    start_time = time.time()
    train(input_args)
    print_log(f"End-to-End total time: {time.time() - start_time} s")
