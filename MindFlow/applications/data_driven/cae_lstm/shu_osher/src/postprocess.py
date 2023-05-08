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
"""Post-processing """
import os

import numpy as np
import matplotlib.pyplot as plt


def plot_train_loss(train_loss, plot_dir, epochs, net_name):
    """Plot change of loss during training"""
    plt.plot(list(range(epochs)), train_loss)
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.savefig(f'{plot_dir}/{net_name}_train_loss.png')
    np.savetxt(f'{plot_dir}/{net_name}_train_loss.txt', train_loss)
    plt.close()


def error(y_true, y_predict):
    relative_error = np.average(np.abs((y_predict - y_true)) / y_true)
    return relative_error


def plot_cae_prediction(cae_encoded, cae_predict, true_data, plot_dir, time_size):
    """Plot cae prediction"""
    # prepare file
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # cae_prediction
    plt.plot(true_data[500, 112:], 'k-', label='true, time=500')
    plt.plot(true_data[1500, 112:], 'b-', label='true, time=1500')
    plt.plot(true_data[2000, 112:], 'r-', label='true, time=2000')
    plt.plot(cae_predict[500, 112:], 'k--', label='cae_prediction, time=500')
    plt.plot(cae_predict[1500, 112:], 'b--', label='cae_prediction, time=1500')
    plt.plot(cae_predict[2000, 112:], 'r--', label='cae_prediction, time=2000')
    plt.ylabel('density')
    plt.xlabel('x')
    plt.legend()
    plt.savefig(f'{plot_dir}/cae_prediction.png')
    plt.close()

    # relative_error
    time_true = np.arange(0, time_size)
    cae_error = np.zeros(time_size)
    for time in np.arange(time_size):
        cae_error[time] = error(true_data[time, :], cae_predict[time, :])

    plt.plot(time_true, cae_error, 'k-')
    plt.title('relative error')
    plt.ylabel('error')
    plt.xlabel('t')
    plt.savefig(f'{plot_dir}/cae_error.png')
    plt.close()

    # save prediction
    np.save(f'{plot_dir}/cae_encoded.npy', np.squeeze(cae_encoded.asnumpy()))
    np.save(f'{plot_dir}/cae_predict.npy', cae_predict)
    np.save(f'{plot_dir}/cae_error.npy', cae_error)


def plot_cae_lstm_prediction(lstm_latent, cae_lstm_predict, true_data, plot_dir, time_size, time_window):
    """Plot prediction"""
    # prepare file
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    # cae_lstm_prediction
    plt.plot(true_data[500, 112:], 'k-', label='true, time=500')
    plt.plot(true_data[1500, 112:], 'b-', label='true, time=1500')
    plt.plot(true_data[2000, 112:], 'r-', label='true, time=2000')
    plt.plot(cae_lstm_predict[500-time_window, 112:], 'k--', label='cae_lstm, time=500')
    plt.plot(cae_lstm_predict[1500-time_window, 112:], 'b--', label='cae_lstm, time=1500')
    plt.plot(cae_lstm_predict[2000-time_window, 112:], 'r--', label='cae_lstm, time=2000')
    plt.ylabel('density')
    plt.xlabel('x')
    plt.legend()
    plt.savefig(f'{plot_dir}/cae_lstm_prediction.png')
    plt.close()

    # relative_error
    time_true = np.arange(0, time_size)
    time_predict = time_true[time_window:]

    cae_lstm_error = np.zeros(time_size - time_window)

    for time in np.arange(time_size - time_window):
        cae_lstm_error[time] = error(true_data[time + time_window, :], cae_lstm_predict[time, :])

    plt.plot(time_predict, cae_lstm_error, 'k-')
    plt.title('relative error')
    plt.ylabel('error')
    plt.xlabel('t')
    plt.savefig(f'{plot_dir}/cae_lstm_error.png')
    plt.close()

    # save prediction
    np.save(f'{plot_dir}/lstm_latent.npy', lstm_latent.asnumpy())
    np.save(f'{plot_dir}/cae_lstm_predict.npy', cae_lstm_predict)
    np.save(f'{plot_dir}/cae_lstm_error.npy', cae_lstm_error)
