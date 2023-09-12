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

"""enso plot"""
import os
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import nn
import numpy as np
import scipy.stats as sts
from sciai.utils import print_log


def plot_corr(predict, obs, figure_dir, title):
    """plot correlation skill"""
    corr_ls = []
    for i in range(17):
        corr, _ = sts.pearsonr(predict[:, i], obs[:, i])
        corr_ls.append(corr)

    plt.plot(np.arange(1, 18), corr_ls, "o-", color="blue")
    plt.xlabel("lead time (month)")
    plt.ylabel("correlation skill")
    plt.axhline(0.5, color="k", linestyle=":")
    plt.savefig(f"{figure_dir}/all_net_corr_skill.png")
    plt.show()
    plt.close()

    time = np.arange(1981, 2019)

    plt.plot(time, obs[11::12, 9], color="red", label="Obs")
    plt.plot(time, predict[11::12, 9], color="blue", label="Prediction")
    plt.savefig(f'{figure_dir}/{title}.png', dpi=300)
    plt.show()
    plt.close()


def plot_loss(history, figure_dir, title):
    """plot train and validation loss"""
    loss = history["train_loss_record"]
    val_loss = history["val_loss_record"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    epochs = range(len(val_loss))
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f'{figure_dir}/train_prog_all_net.png', dpi=300)
    plt.show()
    plt.close()


def evaluate(args, net, ip_var, nino34_var):
    """validation"""
    pre_nino = net(ms.Tensor(ip_var))
    mse = nn.MSE()
    mse_var = mse(pre_nino, ms.Tensor(nino34_var))
    print_log(f"mse_var: {mse_var}")
    pre_nino = pre_nino.asnumpy()
    if args.save_figure:
        plot_corr(pre_nino, nino34_var, args.figures_path, "nino34_validation")

    if args.save_data:
        os.makedirs(f"{args.save_data_path}/htmp_data", exist_ok=True)
        np.save(f"{args.save_data_path}/htmp_data/all_net_ip_data", ip_var)
        np.save(f"{args.save_data_path}/htmp_data/all_net_out_data", nino34_var)
        np.save(f"{args.save_data_path}/htmp_data/all_net_pred_data", pre_nino)
