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
"""eval of CaeLstm"""
import os
import argparse

import numpy as np

from mindspore import load_checkpoint, load_param_into_net, set_seed, Tensor, context
from mindflow.utils import load_yaml_config
from src import create_cae_dataset, create_lstm_dataset, CaeNet1D, CaeNet2D, Lstm, plot_cae_lstm_eval
from cae_eval import cae_eval

np.random.seed(0)
set_seed(0)


def cae_lstm_eval(encoded):
    """eval of CaeLstm"""
    # prepare params
    config = load_yaml_config(args.config_file_path)
    if args.case == 'sod' or args.case == 'shu_osher':
        cae_data_params = config["1D_cae_data"]
        lstm_data_params = config["1D_lstm_data"]
        cae_model_params = config["1D_cae_model"]
        lstm_model_params = config["1D_lstm_model"]
        prediction_params = config["1D_prediction"]
    else:
        cae_data_params = config["2D_cae_data"]
        lstm_data_params = config["2D_lstm_data"]
        cae_model_params = config["2D_cae_model"]
        lstm_model_params = config["2D_lstm_model"]
        prediction_params = config["2D_prediction"]

    # prepare network
    lstm = Lstm(lstm_model_params["latent_size"], lstm_model_params["hidden_size"], lstm_model_params["num_layers"])
    lstm_param_dict = load_checkpoint(prediction_params["lstm_ckpt_path"])
    load_param_into_net(lstm, lstm_param_dict)

    if args.case == 'sod' or args.case == 'shu_osher':
        cae = CaeNet1D(cae_model_params["data_dimension"], cae_model_params["conv_kernel_size"],
                       cae_model_params["maxpool_kernel_size"], cae_model_params["maxpool_stride"],
                       cae_model_params["encoder_channels"], cae_model_params["decoder_channels"])
    else:
        cae = CaeNet2D(cae_model_params["data_dimension"], cae_model_params["conv_kernel_size"],
                       cae_model_params["maxpool_kernel_size"], cae_model_params["maxpool_stride"],
                       cae_model_params["encoder_channels"], cae_model_params["decoder_channels"],
                       cae_model_params["channels_dense"])

    cae_param_dict = load_checkpoint(prediction_params["cae_ckpt_path"])
    load_param_into_net(cae, cae_param_dict)

    # prepare dataset
    _, input_seq = create_lstm_dataset(encoded, lstm_data_params["batch_size"], lstm_data_params["time_size"],
                                       lstm_data_params["latent_size"], lstm_data_params["time_window"],
                                       lstm_data_params["gaussian_filter_sigma"])

    _, true_data = create_cae_dataset(cae_data_params["data_path"], cae_data_params["batch_size"],
                                      cae_data_params["multiple"])

    output_seq_pred = np.zeros(shape=(lstm_data_params["time_size"] - lstm_data_params["time_window"],
                                      lstm_data_params["latent_size"]))

    print(f"=================Start Lstm eval=====================")
    input_seq_pred = input_seq[0].reshape((1, lstm_data_params["time_window"], lstm_data_params["latent_size"]))
    input_seq_pred = input_seq_pred.astype(np.float32)
    for sample in range(0, lstm_data_params["time_size"] - lstm_data_params["time_window"]):
        output_seq_pred[sample, :] = lstm(Tensor(input_seq_pred)).asnumpy()[0, 0, :]
        input_seq_pred[0, : -1, :] = input_seq_pred[0, 1:, :]
        input_seq_pred[0, -1, :] = output_seq_pred[sample, :]
    print(f"===================End Lstm eval====================")
    lstm_latent = np.expand_dims(output_seq_pred, 1)
    lstm_latent = Tensor(lstm_latent.astype(np.float32))
    if args.case == 'sod' or args.case == 'shu_osher':
        cae_lstm_predict = np.squeeze(cae.decoder(lstm_latent).asnumpy())
        cae_lstm_predict = cae_lstm_predict / cae_data_params["multiple"]
    else:
        cae_lstm_predict_time = lstm_data_params["time_size"] - lstm_data_params["time_window"]
        cae_lstm_predict = np.zeros((cae_lstm_predict_time, true_data.shape[1], true_data.shape[2]))
        for i in range(prediction_params["decoder_data_split"]):
            time_predict_start, time_predict_end = \
                prediction_params["decoder_time_spilt"][i], prediction_params["decoder_time_spilt"][i + 1]
            cae_lstm_predict[time_predict_start: time_predict_end] = \
                np.squeeze(cae.decoder(lstm_latent[time_predict_start: time_predict_end]).asnumpy())
        cae_lstm_predict = cae_lstm_predict / cae_data_params["multiple"]

    cae_lstm_error_mean = plot_cae_lstm_eval(lstm_latent, cae_lstm_predict, true_data,
                                             prediction_params["prediction_result_dir"], lstm_data_params["time_size"],
                                             lstm_data_params["time_window"])
    print("CaeLstm prediction mean error: " + str(cae_lstm_error_mean))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CaeLstm eval')
    parser.add_argument("--case", type=str, default="sod",
                        choices=["sod", "shu_osher", "riemann", "kh", "cylinder"],
                        help="Which case to run, support 'sod', 'shu_osher', 'riemann', 'kh', 'cylinder")
    parser.add_argument("--mode", type=str, default="GRAPH", choices=["GRAPH", "PYNATIVE"],
                        help="Context mode, support 'GRAPH', 'PYNATIVE'")
    parser.add_argument("--device_target", type=str, default="GPU", choices=["GPU", "CPU", "Ascend"],
                        help="The target device to run, support 'Ascend', 'GPU', 'CPU")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE if args.mode.upper().startswith("GRAPH") else context.PYNATIVE_MODE,
                        save_graphs=args.save_graphs,
                        save_graphs_path=args.save_graphs_path,
                        device_target=args.device_target,
                        device_id=args.device_id)
    use_ascend = context.get_context(attr_key='device_target') == "Ascend"

    print(f"pid:{os.getpid()}")
    cae_latent = cae_eval(args.config_file_path, args.case)
    cae_lstm_eval(cae_latent)
