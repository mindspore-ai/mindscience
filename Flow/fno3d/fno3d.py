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
train
"""
import os
import time
import argparse
import datetime
from timeit import default_timer
import numpy as np

from mindspore import nn, ops, jit, data_sink, context, Tensor, load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore import dtype as mstype

from mindflow import get_warmup_cosine_annealing_lr, load_yaml_config
from mindflow.utils import print_log, log_config
from mindflow.cell.neural_operators.fno import FNO3D

from src import LpLoss, UnitGaussianNormalizer, create_training_dataset

set_seed(0)
np.random.seed(0)


def parse_args():
    '''Parse input args'''
    parser = argparse.ArgumentParser(description='Navier Stokes 3D problem')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["GPU", "Ascend", "CPU"],
                        help="The target device to run, support 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=3,
                        help="ID of the target device")
    parser.add_argument("--config_file_path", type=str,
                        default="./configs/fno3d.yaml")
    input_args = parser.parse_args()
    return input_args


def test(input_args):
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"
    print_log(f"use_ascend: {use_ascend}")

    config = load_yaml_config(input_args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    optimizer_params = config["optimizer"]
    summary_params = config["summary"]


    sub = data_params["sub"]
    grid_size = model_params["input_resolution"] // sub
    input_timestep = model_params["input_timestep"]
    output_timestep = model_params["output_timestep"]

    test_a = Tensor(np.load(os.path.join(
        data_params["root_dir"], "test_a.npy")), mstype.float32)
    test_u = Tensor(np.load(os.path.join(
        data_params["root_dir"], "test_u.npy")), mstype.float32)

    if use_ascend:
        compute_type = mstype.float16
    else:
        compute_type = mstype.float32

    model = FNO3D(in_channels=model_params["in_channels"],
                  out_channels=model_params["out_channels"],
                  n_modes=model_params["modes"],
                  resolutions=[model_params["input_resolution"],
                               model_params["input_resolution"], output_timestep],
                  hidden_channels=model_params["width"],
                  n_layers=model_params["depth"],
                  projection_channels=4*model_params["width"],
                  fno_compute_dtype=compute_type
                  )

    param_dict = load_checkpoint(config["summary"]["ckpt_dir"])
    load_param_into_net(model, param_dict)
    loss_fn = LpLoss()
    a_normalizer = UnitGaussianNormalizer(test_a)
    y_normalizer = UnitGaussianNormalizer(test_u)

    def calculate_l2_error(model, inputs, labels):
        """
        Evaluate the model respect to input data and label.

        Args:
            model (Cell): list of expressions node can by identified by mindspore.
            inputs (Tensor): the input data of network.
            labels (Tensor): the true output value of given inputs.

        """
        print_log("================================Start Evaluation================================")
        time_beg = time.time()
        rms_error = 0.0
        for i in range(labels.shape[0]):
            label = labels[i:i + 1]
            test_batch = inputs[i:i + 1]
            test_batch = a_normalizer.encode(test_batch)
            label = y_normalizer.encode(label)

            test_batch = test_batch.reshape(1, grid_size,
                                            grid_size, 1, input_timestep).repeat(output_timestep, axis=3)
            prediction = model(test_batch).reshape(1, grid_size, grid_size, output_timestep)
            prediction = y_normalizer.decode(prediction)
            label = y_normalizer.decode(label)
            rms_error_step = loss_fn(prediction.reshape(
                1, -1), label.reshape(1, -1))
            rms_error += rms_error_step

        rms_error = rms_error / labels.shape[0]
        print_log("mean rms_error:", rms_error)
        print_log("predict total time: {} s".format(time.time() - time_beg))
        print_log("=================================End Evaluation=================================")

    model.set_train(False)
    calculate_l2_error(model, test_a, test_u)


if __name__ == "__main__":
    log_config('./logs', 'fno3d')
    print_log(f"pid: {os.getpid()}")
    print_log(datetime.datetime.now())

    args = parse_args()
    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        device_target=args.device_target, device_id=args.device_id)

    print_log(f"device_id: {context.get_context(attr_key='device_id')}")
    test(args)
