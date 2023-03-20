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
import argparse
import numpy as np
from scipy.io import savemat

from mindspore import ops, Tensor, load_checkpoint, load_param_into_net, set_seed
import mindspore.common.dtype as mstype

from mindflow.utils import load_yaml_config
from src import DataSource, EhdnnNet, PostProcess

np.random.seed(0)
set_seed(0)

parser = argparse.ArgumentParser(description='Prediction')
parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                    help="Context mode, support 'GRAPH', 'PYNATIVE'")
parser.add_argument("--device_target", type=str, default="CPU", choices=["GPU", "CPU", "Ascend"],
                    help="The target device to run, support 'Ascend', 'GPU', 'CPU")
parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
parser.add_argument("--aoa_state", type=list, default=[35], choices=[[33], [34], [35], [36], [37], [375], [38], [39]],
                    help="The data state for prediction, it needs to correspond to the training sample state")
parser.add_argument("--num_memory_layers", type=int, default=2, choices=[2, 4],
                    help="The number of layers of the whole Memory layer， 2 in single_state and 4 in multi state")
parser.add_argument("--config_file_path", type=str, default="./config.yaml")
args = parser.parse_args()


def prediction():
    """Process of prediction with trained net"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]
    prediction_params = config["prediction"]
    postprocess_params = config["postprocess"]

    history_length = prediction_params["history_length"]
    prediction_result_dir = prediction_params["prediction_result_dir"]

    # prepare network
    net = EhdnnNet(model_params["in_channels"],
                   model_params["out_channels"],
                   model_params["num_layers"],
                   args.num_memory_layers,
                   model_params["kernel_size_conv"],
                   model_params["kernel_size_lstm"])
    param_dict = load_checkpoint(prediction_params["ckpt_path"])
    load_param_into_net(net, param_dict)

    # prepare dataset
    data_set = DataSource(data_params["data_dir"], data_params["data_length"], args.aoa_state).prediction_data()

    if not os.path.exists(prediction_result_dir):
        os.mkdir(prediction_result_dir)

    print(f"=================Start Prediction=====================")
    x_set = Tensor(data_set[0:history_length, :, :, :], mstype.float32)
    real = Tensor(data_set[history_length::, :, :, :], mstype.float32)
    for step in range(prediction_params["prediction_length"]):
        input_x = ops.slice(x_set, (step, 0, 0, 0), (history_length, -1, -1, -1))
        out_pred = net(input_x)
        x_set = ops.concat((x_set, out_pred), 0)
    pred = ops.slice(x_set, (history_length, 0, 0, 0), (-1, -1, -1, -1))
    abs_error = ops.abs(pred - real)
    savemat(f"{prediction_result_dir}/prediction_data.mat", {'prediction': pred.asnumpy(),
                                                             'real': real.asnumpy(),
                                                             'abs_error': abs_error.asnumpy()})
    print(f"===================End Prediction====================")
    postpocess = PostProcess(postprocess_params["foil_path"],
                             postprocess_params["size_field"],
                             postprocess_params["x_range"],
                             postprocess_params["y_range"])
    for t in range(3):
        postpocess.plot_flow_field(prediction_result_dir, t)
    print(f"==Output the flow field contours===")


if __name__ == "__main__":
    print(f"pid:{os.getpid()}")
    prediction()
