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

from mindspore import nn, ops, Tensor, load_checkpoint, dataset, load_param_into_net, set_seed
from mindflow.utils import load_yaml_config

from src import my_test_dataset, AEnet

np.random.seed(0)
set_seed(0)


def prediction():
    """Process of predict with trained net"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    prediction_params = config["predict"]
    pred_continue_dir = prediction_params["pred_continue_dir"]
    save_prediction_dir = prediction_params["save_prediction_dir"]

    # prepare network
    net = AEnet(in_channels=model_params["in_channels"],
                num_layers=model_params["num_layers"],
                kernel_size=model_params["kernel_size"],
                num_convlstm_layers=model_params["num_convlstm_layers"])
    m_state_dict = load_checkpoint(prediction_params["ckpt_path"])
    load_param_into_net(net, m_state_dict)

    # prepare dataset
    data_set, surf_xy = my_test_dataset(data_params["data_dir"], data_params["time_steps"],
                                        prediction_params["data_list"])

    test_dataset = dataset.GeneratorDataset(data_set, ["input", "velocity", "ur", "label"], shuffle=False)
    test_dataset = test_dataset.batch(batch_size=1, drop_remainder=True)

    if not os.path.exists(pred_continue_dir):
        os.mkdir(pred_continue_dir)
    if not os.path.exists(save_prediction_dir):
        os.mkdir(save_prediction_dir)

    # prepare loss function: MSE loss function
    loss_func = nn.MSELoss()

    # predicted loss
    test_losses = []

    test_v = []
    test_y = []
    test_lift = []
    test_total = []
    real_y = []

    predict = []
    real = []

    for i, (inputvar, velocityvar, urvar, targetvar) in enumerate(test_dataset):
        if i == 0:
            inputs = inputvar
            y = np.max(surf_xy[0, :, 1]) - 1.55
            velocity = velocityvar
        else:
            inputs = ops.operations.ExpandDims()(pred, 1)

        real_y.append(np.max(surf_xy[2 * i, :, 1]) - 1.55)

        pred = net(inputs, velocity, urvar)

        loss = loss_func(inputs, pred)
        loss_aver = loss.asnumpy().item()

        # record training errors
        test_losses.append(loss_aver)
        print(f"test loss: {loss_aver:.6f}")

        surf_x = surf_xy[0, :, 0]

        # output flow field matrix
        real.append(targetvar.numpy())
        predict.append(pred.numpy())

        # Integrate lift based on predicted surface pressure and calculate cylindrical velocity
        m_cylinder = Tensor(0.011775)
        k_spring = Tensor(2.29327)
        d_t = Tensor(0.02)

        surf_p = pred[0, 0, :, 0] * (1.0 * 1.0 * 1.0)
        sum_p = 0.0

        for j in range(127):
            sum_p = sum_p + (surf_p[j] + surf_p[j + 1]) * (surf_x[j] - surf_x[j + 1]) * 0.5
        sum_p = sum_p + (surf_p[127] + surf_p[0]) * (surf_x[127] - surf_x[0]) * 0.5

        y = y + d_t * velocity
        y = Tensor(y.astype(np.float32))

        force_total = sum_p - y * k_spring

        velocity = velocity + d_t * force_total / m_cylinder

        # output velocity, lift force
        test_v.append(velocity.numpy())
        test_y.append(y.numpy())
        test_lift.append(sum_p.numpy())
        test_total.append(force_total.numpy())

    savemat(f"{pred_continue_dir}/prediction_data.mat", {'predict': predict,
                                                         'real': real,
                                                         'surf_x': surf_x})

    savemat(f"{save_prediction_dir}/prediction_v_d.mat", {'test_v': test_v,
                                                          'test_y': test_y,
                                                          'test_lift': test_lift,
                                                          'test_total': test_total,
                                                          'real_y': real_y})

    print(f"mean test loss: {np.mean(test_losses):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cylinder around flow ROM")

    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")


    args = parser.parse_args()

    print("Process ID:", os.getpid())
    print(f"device id: {args.device_id}")
    start_time = time.time()
    prediction()
    print(f"End-to-End total time: {(time.time() - start_time):.2f}s")
