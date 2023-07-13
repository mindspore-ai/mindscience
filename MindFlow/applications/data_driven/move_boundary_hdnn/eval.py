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
"""Prediction process"""
import os
import time
import argparse
import numpy as np
from scipy.io import savemat

from mindspore import nn, ops, load_checkpoint, load_param_into_net, set_seed
from mindflow.utils import load_yaml_config

from src import my_test_dataset, AEnet, save_loss_curve

np.random.seed(0)
set_seed(0)


def prediction():
    """Process of prediction with trained net"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    prediction_params = config["prediction"]
    prediction_result_dir = prediction_params["prediction_result_dir"]
    pred_continue_dir = prediction_params["pred_continue_dir"]

    # prepare network
    net = AEnet(in_channels=model_params["in_channels"],
                num_layers=model_params["num_layers"],
                kernel_size=model_params["kernel_size"],
                num_convlstm_layers=model_params["num_convlstm_layers"])
    m_state_dict = load_checkpoint(prediction_params["ckpt_path"])
    load_param_into_net(net, m_state_dict)

    # prepare dataset
    data_set = my_test_dataset(prediction_params["data_dir"], data_params["time_steps"])
    if not os.path.exists(prediction_result_dir):
        os.mkdir(prediction_result_dir)
    if not os.path.exists(pred_continue_dir):
        os.mkdir(pred_continue_dir)

    # prepare loss function: MSE loss function
    loss_func = nn.MSELoss()

    # predicted loss
    test_losses = []

    # predicting next one-step flow field
    if args.infer_mode == "one":
        for i, (input_1, velocity, label, matrix_01) in enumerate(data_set):
            pred = net(input_1, velocity)
            pred = ops.mul(pred, matrix_01)
            loss = ops.sqrt(loss_func(pred, label))
            test_losses.append(loss)
            print(f"test loss: {(loss.asnumpy().item()):.6f}")
            savemat(f"{prediction_result_dir}/prediction_data{i}.mat", {'prediction': pred.asnumpy(),
                                                                        'real': label.asnumpy(),
                                                                        'input': input_1.asnumpy()})

    # predicting a complete periodic flow field
    elif args.infer_mode == "cycle":
        for i, (inputvar, velocityvar, targetvar, matrix_01) in enumerate(data_set):
            if i == 0:
                inputs = inputvar
            label = targetvar
            velocity = velocityvar
            pred = net(inputs, velocity)
            pred = ops.mul(pred, matrix_01)
            loss = ops.sqrt(loss_func(pred, label))
            loss_aver = loss.asnumpy().item()

            # Record training loss
            test_losses.append(loss_aver)
            print(f"test loss: {loss_aver:.6f}")
            savemat(f"{pred_continue_dir}/prediction_data{i}.mat", {'prediction': pred.asnumpy(),
                                                                    'real': label.asnumpy(),
                                                                    'inputs': inputs.asnumpy()})
            # Splicing predicted values as input for the next step
            pred = ops.operations.ExpandDims()(pred, 1)
            cat = ops.concat((inputs, pred), axis=1)
            inputs = cat[:, 1:, :, :, :]

    # draw and save curves of test losses
    save_loss_curve(test_losses, 'Epoch', 'test_losses', 'Test_losses Curve', 'Test_losses.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cylinder around flow ROM")

    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    parser.add_argument("--infer_mode", type=str, default="one", choices=["one", "cycle"],
                        help="The mode to predict next one-step flow field or a complete periodic flow field")

    args = parser.parse_args()

    print("Process ID:", os.getpid())
    print(f"device id: {args.device_id}")
    start_time = time.time()
    prediction()
    print(f"End-to-End total time: {(time.time() - start_time):.2f}s")
