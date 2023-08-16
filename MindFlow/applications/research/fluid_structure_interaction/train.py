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

from mindspore import nn, ops, context, save_checkpoint, set_seed, data_sink, jit

from mindflow.utils import load_yaml_config

from src import generate_dataset, AEnet, save_loss_curve

np.random.seed(0)
set_seed(0)


def train():
    """train process"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]

    # prepare file to save the trained model files
    ckpt_dir = optimizer_params["ckpt_dir"]
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # prepare the model to be trained, as well as loss function:MSE and optimizer:Adam
    model = AEnet(in_channels=model_params["in_channels"],
                  num_layers=model_params["num_layers"],
                  kernel_size=model_params["kernel_size"],
                  num_convlstm_layers=model_params["num_convlstm_layers"])

    loss_func = nn.MSELoss()
    optimizer = nn.Adam(params=model.trainable_params(), learning_rate=optimizer_params["lr"])

    # when using Ascend for training, introducing dynamic loss scaler and automatic mixed accuracy training methods
    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision, all_finite
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(model, 'O1')
    else:
        loss_scaler = None

    # define a forward propagation function
    def forward_fn(inputs, velocity, ur, label):
        pred = model(inputs, velocity, ur)
        loss = loss_func(pred, label)

        if use_ascend:
            loss = loss_scaler.scale(loss)
        return loss

    # calculate function forward_ Fn and return the value and gradient of the function
    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    # prepare dataset
    print(f"==================Load data sample ===================")
    dataset_train, dataset_eval = generate_dataset(data_params["data_dir"],
                                                   data_params["time_steps"],
                                                   args.data_list)
    print(f"======================End Load========================\n")

    print(f"====================Start train=======================")

    # define a function decorated with @jit to perform training steps, which calls and saves the function to calculate
    # loss values and gradients. Using decorators can improve the execution efficiency of functions
    @jit
    def train_step(inputs, velocity, ur, label):
        loss, grads = grad_fn(inputs, velocity, ur, label)
        if use_ascend:
            loss = loss_scaler.unscale(loss)
            if all_finite(grads):
                grads = loss_scaler.unscale(grads)
        loss = ops.depend(loss, optimizer(grads))
        return loss

    @jit
    def eval_step(inputs, velocity, ur, label):
        loss = forward_fn(inputs, velocity, ur, label)
        loss = ops.sqrt(loss)
        return loss

    # define train_sink_process and eval_sink_process,obtain data from the dataset, preprocess it and input it into the
    # training steps for model training
    train_sink_process = data_sink(train_step, dataset_train, sink_size=1)
    eval_sink_process = data_sink(eval_step, dataset_eval, sink_size=1)
    train_data_size, eval_data_size = dataset_train.get_dataset_size(), dataset_eval.get_dataset_size()

    # average training loss per epoch
    avg_train_losses = []
    # average validation loss per epoch
    avg_valid_losses = []

    # start epoch training
    for epoch in range(1, optimizer_params["epochs"] + 1):
        train_losses = 0
        valid_losses = 0

        local_time_beg = time.time()
        model.set_train(True)

        for _ in range(train_data_size):
            step_train_loss = ops.squeeze(train_sink_process(), axis=())
            step_train_loss = step_train_loss.asnumpy().item()
            train_losses += step_train_loss

        train_loss = train_losses / train_data_size
        avg_train_losses.append(train_loss)

        print(f"epoch: {epoch}, epoch average train loss: {train_loss :.6f}, "
              f"epoch time: {(time.time() - local_time_beg):.2f}s")

        if epoch % optimizer_params["eval_interval"] == 0:
            print(f"=================Start Evaluation=====================")

            eval_time_beg = time.time()
            model.set_train(False)
            for _ in range(eval_data_size):
                step_eval_loss = ops.squeeze(eval_sink_process(), axis=())
                step_eval_loss = step_eval_loss.asnumpy().item()
                valid_losses += step_eval_loss

            valid_loss = valid_losses / eval_data_size
            avg_valid_losses.append(valid_loss)

            print(f"epoch: {epoch}, epoch average valid loss: {valid_loss :.6f}, "
                  f"epoch time: {(time.time() - eval_time_beg):.2f}s")
            print(f"==================End Evaluation======================")

        # save the ckpt file of the trained model in the folder
        if epoch % optimizer_params["save_ckpt_interval"] == 0:
            save_checkpoint(model, f"{ckpt_dir}/net_{epoch}.ckpt")

    # draw and save curves of training loss and testing loss
    save_loss_curve(avg_train_losses, 'Epoch', 'avg_train_losses', 'Avg_train_losses Curve', 'Avg_train_losses.png')
    save_loss_curve(avg_valid_losses, 'Epoch', 'avg_valid_losses', 'Avg_valid_losses Curve', 'Avg_valid_losses.png')

    print(f"=====================End train========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cylinder around flow ROM")

    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    parser.add_argument("--data_list", type=list, default=['5.0', '5.5', '6.0', '6.5'], help="The type for training")
    parser.add_argument('--batch_size', type=int, default=32, help="batch size")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./summary")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'GPU','Ascend'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")

    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs, save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target, device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print("Process ID:", os.getpid())
    print(f"device id: {args.device_id}")
    start_time = time.time()
    train()
    print(f"End-to-End total time: {(time.time() - start_time):.2f}s")
