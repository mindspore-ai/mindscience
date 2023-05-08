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
from src import CaeNet, Lstm, create_cae_dataset, create_lstm_dataset, plot_cae_lstm_prediction
from cae_prediction import cae_prediction

np.random.seed(0)
set_seed(0)


def cae_lstm_prediction(encoded):
    """Process of prediction with cae-lstm net"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    cae_data_params = config["cae_data"]
    lstm_data_params = config["lstm_data"]
    cae_model_params = config["cae_model"]
    lstm_model_params = config["lstm_model"]
    prediction_params = config["prediction"]

    # prepare network
    lstm = Lstm(lstm_model_params["latent_size"], lstm_model_params["hidden_size"], lstm_model_params["num_layers"])
    lstm_param_dict = load_checkpoint(prediction_params["lstm_ckpt_path"])
    load_param_into_net(lstm, lstm_param_dict)

    cae = CaeNet(cae_model_params["data_dimension"], cae_model_params["conv_kernel_size"],
                 cae_model_params["maxpool_kernel_size"], cae_model_params["maxpool_stride"],
                 cae_model_params["encoder_channels"], cae_model_params["decoder_channels"])
    cae_param_dict = load_checkpoint(prediction_params["cae_ckpt_path"])
    load_param_into_net(cae, cae_param_dict)

    # prepare dataset
    _, input_seq = create_lstm_dataset(encoded, lstm_data_params["batch_size"], lstm_data_params["time_size"],
                                       lstm_data_params["latent_size"], lstm_data_params["time_window"],
                                       lstm_data_params["gaussian_filter_sigma"])

    _, true_data = create_cae_dataset(cae_data_params["data_path"], cae_data_params["batch_size"])

    output_seq_pred = np.zeros(shape=(lstm_data_params["time_size"] - lstm_data_params["time_window"],
                                      lstm_data_params["latent_size"]))

    print(f"=================Start lstm prediction=====================")
    input_seq_pred = input_seq[0].reshape((1, lstm_data_params["time_window"], lstm_data_params["latent_size"]))
    input_seq_pred = input_seq_pred.astype(np.float32)
    for sample in range(0, lstm_data_params["time_size"] - lstm_data_params["time_window"]):
        output_seq_pred[sample, :] = lstm(Tensor(input_seq_pred)).asnumpy()[0, 0, :]
        input_seq_pred[0, : -1, :] = input_seq_pred[0, 1:, :]
        input_seq_pred[0, -1, :] = output_seq_pred[sample, :]
    print(f"===================End lstm prediction====================")
    lstm_latent = np.expand_dims(output_seq_pred, 1)
    lstm_latent = Tensor(lstm_latent.astype(np.float32))
    cae_lstm_predict = np.squeeze(cae.decoder(lstm_latent).asnumpy())
    plot_cae_lstm_prediction(lstm_latent, cae_lstm_predict, true_data, prediction_params["prediction_result_dir"],
                             lstm_data_params["time_size"], lstm_data_params["time_window"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cae-lstm prediction')
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "CPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    print(f"pid:{os.getpid()}")
    cae_latent = cae_prediction()
    cae_lstm_prediction(cae_latent)
