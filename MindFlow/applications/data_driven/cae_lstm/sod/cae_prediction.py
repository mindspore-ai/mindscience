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
"""prediction process"""
import os
import argparse
import numpy as np

from mindspore import load_checkpoint, load_param_into_net, set_seed, Tensor

from mindflow.utils import load_yaml_config
from src import CaeNet, create_cae_dataset, plot_cae_prediction

np.random.seed(0)
set_seed(0)


def cae_prediction(config_file_path):
    """Process of prediction with cae net"""
    # prepare params
    config = load_yaml_config(config_file_path)
    data_params = config["cae_data"]
    model_params = config["cae_model"]
    prediction_params = config["prediction"]

    # prepare network
    cae = CaeNet(model_params["data_dimension"], model_params["conv_kernel_size"], model_params["maxpool_kernel_size"],
                 model_params["maxpool_stride"], model_params["encoder_channels"], model_params["decoder_channels"])
    cae_param_dict = load_checkpoint(prediction_params["cae_ckpt_path"])
    load_param_into_net(cae, cae_param_dict)

    # prepare dataset
    _, true_data = create_cae_dataset(data_params["data_path"], data_params["batch_size"])
    data_set = np.expand_dims(true_data, 1).astype(np.float32)

    print(f"=================Start cae prediction=====================")
    encoded = cae.encoder(Tensor(data_set))
    cae_predict = np.squeeze(cae(Tensor(data_set)).asnumpy())
    print(f"===================End cae prediction====================")
    plot_cae_prediction(encoded, cae_predict, true_data,
                        prediction_params["prediction_result_dir"], data_params["time_size"])
    return encoded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cae prediction')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "CPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    print(f"pid:{os.getpid()}")
    cae_prediction(args.config_file_path)
