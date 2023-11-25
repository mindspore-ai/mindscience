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
import random
import time
import os
import argparse

from mindspore import context, nn, set_seed, save_checkpoint, data_sink
from mindspore.amp import DynamicLossScaler, auto_mixed_precision
from mindflow.utils import load_yaml_config, print_log, log_config, log_timer

from src import create_dataset, create_model
from src import evaluate, plot
from src import get_train_loss_step

set_seed(0)
random.seed(0)


def parse_args():
    """parse arguments"""
    parser = argparse.ArgumentParser(description="train pinns")
    parser.add_argument("--case", type=str, default="burgers", choices=["burgers", "cylinder_flow", "periodic_hill"],
                        help="choose burgers, cylinder_flow or periodic_hill")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Running in GRAPH_MODE OR PYNATIVE_MODE")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/burgers.yaml")
    result_args = parser.parse_args()
    return result_args


@log_timer
def train(args):
    '''Train and evaluate the network'''
    # load configuration
    case_name = args.case
    config = load_yaml_config(args.config_file_path)
    summary_config = config["summary"]
    optimizer_config = config["optimizer"]
    epochs = optimizer_config["train_epochs"]

    # create dataset for training, calculating loss and testing
    train_dataset, loss_dataset, inputs, label = create_dataset(
        case_name, config)

    model = create_model(case_name, config)
    optimizer = nn.Adam(model.trainable_params(),
                        optimizer_config["initial_lr"])

    if use_ascend:
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, optimizer_config["amp_level"])
    else:
        loss_scaler = None

    # define train_step and loss_step
    train_step, loss_step = get_train_loss_step(use_ascend=use_ascend, case_name=case_name,
                                                config=config, model=model,
                                                optimizer=optimizer, loss_scaler=loss_scaler)
    grad_sink_process = data_sink(train_step, train_dataset, sink_size=1)
    loss_sink_process = data_sink(loss_step, loss_dataset, sink_size=1)

    # create ckpt dir
    save_ckpt_path = summary_config["save_ckpt_path"]
    if not os.path.exists(os.path.abspath(save_ckpt_path)):
        os.makedirs(os.path.abspath(save_ckpt_path))
    steps_per_epochs = train_dataset.get_dataset_size()

    # train loop
    for epoch in range(1, epochs + 1):
        # set begin time
        time_beg = time.time()

        # set model to train mode
        model.set_train(True)

        for _ in range(steps_per_epochs):
            grad_sink_process()

        loss = loss_sink_process()
        print_log(
            f"epoch: {epoch} train loss: {loss} epoch time: {(time.time() - time_beg) * 1000 :.3f}ms")

        # set model to eval mode
        model.set_train(False)

        # evaluate best_params if current epoch reaches setting
        if epoch % summary_config["eval_interval_epochs"] == 0:
            evaluate(case_name, model, inputs, label, config)

        # save checkpoint
        if epoch % summary_config["save_checkpoint_epochs"] == 0:
            ckpt_name = "{}-{}.ckpt".format(case_name, epoch + 1)
            save_checkpoint(model, os.path.join(
                save_ckpt_path, ckpt_name))
            plot(case_name, model, epoch, config, inputs, label)

    # visual best params
    plot(case_name, model, epoch, config, inputs, label)


if __name__ == '__main__':
    input_args = parse_args()
    log_config('./logs', input_args.case)
    context.set_context(mode=context.GRAPH_MODE if input_args.mode.upper().startswith("GRAPH")
                        else context.PYNATIVE_MODE,
                        device_target=input_args.device_target,
                        device_id=input_args.device_id)
    print_log(
        f"Running in {input_args.mode.upper()} mode, using device id: {input_args.device_id}.")
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log("pid:", os.getpid())
    start_time = time.time()
    train(input_args)
    print_log("End-to-End total time: {} s".format(time.time() - start_time))
