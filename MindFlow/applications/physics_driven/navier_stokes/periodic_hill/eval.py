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
'''evaluation of the rans pinns model over periodic hill dataset'''
import os
import argparse

import matplotlib.pyplot as plt

from mindspore import Tensor, load_checkpoint, load_param_into_net
from mindspore import dtype as mstype
from mindflow.cell import FCSequential
from mindflow.utils import load_yaml_config

from src import create_test_dataset


def predict(model, epochs, input_data, label, path="./prediction_result"):
    """visulization of u/v/p"""
    prediction = model(Tensor(input_data, mstype.float32)).asnumpy()

    x = input_data[:, 0].reshape((300, 700))
    y = input_data[:, 1].reshape((300, 700))

    if not os.path.isdir(os.path.abspath(path)):
        os.makedirs(path)

    _, output_size = label.shape
    label = label.reshape((300, 700, output_size))
    prediction = prediction.reshape((300, 700, output_size))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.pcolor(x.T, y.T, prediction[:, :, 0].T)
    plt.title("U prediction")
    plt.subplot(2, 2, 2)
    plt.pcolor(x.T, y.T, prediction[:, :, 1].T)
    plt.title("V prediction")
    plt.subplot(2, 2, 3)
    plt.pcolormesh(x.T, y.T, label[:, :, 0].T)
    plt.title("U ground truth")
    plt.subplot(2, 2, 4)
    plt.pcolormesh(x.T, y.T, label[:, :, 1].T)
    plt.title("V ground truth")
    plt.tight_layout()
    plt.savefig(os.path.join(path, str(epochs) + ".png"))
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cae-transformer prediction")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()
    print(f"pid:{os.getpid()}")

    config = load_yaml_config(args.config_file_path)
    data_params = config["data"]
    model_params = config["model"]

    rans_model = FCSequential(in_channels=model_params["in_channels"],
                              out_channels=model_params["out_channels"],
                              layers=model_params["layers"],
                              neurons=model_params["neurons"],
                              residual=model_params["residual"],
                              act='tanh')
    inputs, labels = create_test_dataset(data_params["data_path"])

    param_dict = load_checkpoint(config["load_ckpt_path"])
    load_param_into_net(rans_model, param_dict)
    predict(rans_model, config["epochs"], inputs, labels, config["visual_path"])
