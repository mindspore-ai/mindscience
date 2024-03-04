# copyright 2024 Huawei Technologies co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the license is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied
# See the License for the specific language governing permissions and
# limitations under the license.
##==========================================================================
"""
    南部非洲大地电磁实测，数据集生成及网络训练
"""
import os
import time
from collections import defaultdict
import json
import math
import logging
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import filters
import matplotlib
import matplotlib.pyplot as plt
import help as hf
import mindspore as ms
from mindspore.dataset import GeneratorDataset
from mindspore import load_checkpoint, save_checkpoint, load_param_into_net
from mindspore import ops, nn
from src import model

ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")  # ,
matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO)

# %% Network & Training parameter
kl_weight = 10e-3  # 1e-2
ssim_w = 6e-2  # 1e-2
N1 = 60000
initial_rate = 8e-4  #
MAX_STEP = 200
model_sel_dim = 32  #

# %% Condition selection
TRAIN = -1  # -1: Generate data set  1: Train the Network;   2: Only test the network

# %% Main parameter
"""Coordinate Parameter"""
ZNUMBER = 64
bar_1 = 0
bar_2 = 4.5
h = np.zeros((1, ZNUMBER))
for ii in range(ZNUMBER):
    h[0, ii] = 2 * math.pow(1.146, ii)  # 64 grids  18 1.045 6000m depth
zedgelocation = np.concatenate(([0], np.cumsum(h)))
zelementlocation = 0.5 * (zedgelocation[0:-1] + zedgelocation[1:])
zlengths = zedgelocation[1:] - zedgelocation[0:-1]
zlengths = ms.Tensor(zlengths, ms.float32)
loss_weight = ops.log10(zlengths[:model_sel_dim])
lw_tens = ms.Tensor(loss_weight, dtype=ms.float32)

zedgelocation1 = np.linspace(0, 80e3, ZNUMBER + 1)
zelementlocation1 = 0.5 * (zedgelocation1[0:-1] + zedgelocation1[1:])

fieldxstart = -30e3
fieldxend = 790e3
XNUMBERMT = 100
xedgelocationmt = np.linspace(fieldxstart, fieldxend, XNUMBERMT + 1)
xelementlocationmt = 0.5 * (xedgelocationmt[0:-1] + xedgelocationmt[1:])
# %%
# """Training set generation"""
if TRAIN == -1:
    #"""ETO-KIM"""
    data_set = np.zeros((ZNUMBER, N1))
    w1 = hf.FspecialGaussian(np.array([1, 8]), 5)
    w1 = np.squeeze(w1)
    for pp in range(N1):
        model_ii = np.zeros(ZNUMBER) + np.random.uniform(3, 4.2)
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        u3 = np.random.uniform(0, 1)
        h4b = np.random.uniform(200, 400, 2)
        h4b = np.sort(h4b)
        if u2 < 1:
            h4 = np.random.uniform(2e3, 20e3)  # 厚度
            h42 = np.random.uniform(1e3, 15e3)  # 顶端位置
            cc = np.where(
                abs(zelementlocation - h42) == np.min(abs(zelementlocation - h42))
            )
            cc5 = cc[0][0]
            cc = np.where(
                abs(zelementlocation - h4 - h42)
                == np.min(abs(zelementlocation - h4 - h42))
            )
            cc6 = cc[0][0]
            if u1 < 0.6:
                model_ii[cc5:cc6] = np.random.uniform(-1.5, 0.4)  # (1,20)
            else:
                model_ii[cc5:cc6] = np.random.uniform(0.4, 1.5)  # (1,20)
            h4b = np.random.uniform(h4 + h42, 80e3, 10)
            h4b = np.sort(h4b)
            h4v = np.random.uniform(
                np.random.uniform(2.6, 2.7), np.random.uniform(4.4, 4.5), 11
            )
            h4v = np.sort(h4v)
            cc = np.where(
                abs(zelementlocation - h4b[0]) == np.min(abs(zelementlocation - h4b[0]))
            )
            cc = cc[0][0]
            model_ii[cc6:cc] = h4v[0]
            for dd in range(9):
                cc = np.where(
                    abs(zelementlocation - h4b[dd])
                    == np.min(abs(zelementlocation - h4b[dd]))
                )
                cc1 = cc[0][0]
                cc = np.where(
                    abs(zelementlocation - h4b[dd + 1])
                    == np.min(abs(zelementlocation - h4b[dd + 1]))
                )
                cc2 = cc[0][0]
                model_ii[cc1:cc2] = h4v[dd + 1]
            cc = np.where(
                abs(zelementlocation - h4b[9]) == np.min(abs(zelementlocation - h4b[9]))
            )
            cc = cc[0][0]
            model_ii[cc:] = h4v[10]

        tmp1 = np.random.randint(9)
        tmp2 = np.zeros(ZNUMBER + 40)
        for _ in range(tmp1):
            mywidth = np.random.randint(6, 20)
            ww = hf.FspecialGaussian(np.array([1, mywidth]), np.random.uniform(1, 4))
            ww = np.reshape(ww, -1)
            try:
                ww = ww * (1 / max(ww))
            except ZeroDivisionError as e:
                logging.error("Error %s", e)
                raise
            pos = np.random.randint(ZNUMBER)
            tmp2[
                20 + pos - int(mywidth / 2) : 20 + pos - int(mywidth / 2) + len(ww)
            ] = (
                tmp2[
                    20 + pos - int(mywidth / 2) : 20 + pos - int(mywidth / 2) + len(ww)
                ]
                + (-1) ** (np.random.randint(0, 2) - 1)
                * (np.random.uniform(0.1, 0.4))
                * ww
            )

        tmp3 = tmp2[20 : 20 + ZNUMBER]
        model_ii = model_ii + tmp3 * 1
        model_ii1 = filters.convolve(model_ii, w1)
        data_set[:, pp] = model_ii1
    os.makedirs(os.path.join("dataset", "VAE_dataset"), exist_ok=True)
    savemat(
        os.path.join("dataset", "VAE_dataset", "dataset.mat"),
        {"Data_set": data_set},
    )  #

    model_training_ave = np.zeros((ZNUMBER, XNUMBERMT))
    try:
        for ii in range(XNUMBERMT):
            model_training_ave[:, ii] = data_set[:, int(ii * N1 / XNUMBERMT)]
    except ZeroDivisionError as e:
        logging.error("Error %s", e)
        raise

    [xw, yw] = np.meshgrid(xedgelocationmt, -zedgelocation)
    plt.ion()
    fig = plt.figure(figsize=(8, 3))
    ax1 = fig.add_subplot(1, 1, 1)
    plt.pcolor(xw, yw, model_training_ave, cmap=plt.get_cmap("jet"))
    plt.xlim(-30e3, 790e3)
    plt.ylim(-80e3, 0)
    plt.xlabel("Distance (m)")
    plt.ylabel("Depth (m)")
    cbar = plt.colorbar()
    plt.clim(bar_1, bar_2)
    cbar.set_label("Resistivity")
    plt.tight_layout()
    plt.savefig(
        os.path.join("dataset", "VAE_dataset", "training_samples.png")
    )
    plt.ioff()
    plt.close()

# %%
#"""Neural Network Training"""
if TRAIN in (1, 2):
    val_rate = 0.2
    data = loadmat(
        os.path.join("dataset", "VAE_dataset", "dataset.mat")
    )
    data_set = data["Data_set"]  # (ZNUMBER, N1)   对数域
    rhotruth1 = data_set.transpose((1, 0))  # (N1, ZNUMBER)

    rhotruth1 = np.expand_dims(rhotruth1, axis=2)  # (N1, ZNUMBER, 1)

    indices = list(range(N1))
    train_ind = indices[: int(N1 * (1 - val_rate))]
    val_ind = indices[int(N1 * (1 - val_rate)) :]

    rho_train = rhotruth1[train_ind, -model_sel_dim:, :]
    rho_test = rhotruth1[val_ind, -model_sel_dim:, :]

    batch_size = int(64)
    checkpoint_path = os.path.join("results", "ckpt", "net", "VAE.ckpt")

    net = model.Model()
    loss_net = model.LossFuncNet(net)
    mse_loss_net = model.MSELoss()
    optimizer = nn.optim.Adam(
        params=loss_net.trainable_params(), learning_rate=initial_rate
    )
    train_cell = nn.TrainOneStepCell(loss_net, optimizer)

if TRAIN == 1:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    class MyIterable:
        """
        MyIterable
        """

        def __init__(self, rho):
            self._index = 0
            self._data = rho.astype(np.float32).transpose((0, 2, 1))

        def __next__(self):
            if self._index >= len(self._data):
                raise StopIteration
            item = (self._data[self._index], self._data[self._index])
            self._index += 1
            return item

        def __iter__(self):
            self._index = 0
            return self

        def __len__(self):
            return len(self._data)

    gd_train = GeneratorDataset(MyIterable(rho_train), column_names=["x", "y"])
    gd_train = gd_train.batch(batch_size)
    gd_val = GeneratorDataset(MyIterable(rho_test), column_names=["x", "y"])
    gd_val = gd_val.batch(batch_size)
    history = defaultdict(list)
    time_last = time.time()
    for epoch in range(MAX_STEP):
        loss_epoch = ms.Tensor(0, ms.float32)
        for inputs, y_true in gd_train:
            loss = train_cell(inputs, y_true, lw_tens, ssim_w, kl_weight)
            outputs, _, _ = net(inputs)
            loss_epoch += loss
            mse_loss = mse_loss_net(outputs, y_true, lw_tens)  ####todo
        history["loss"].append(loss_epoch)
        history["mse_loss"].append(mse_loss)
        loss_epoch_val = ms.Tensor(0, ms.float32)
        time_train = time.time()
        train_spent = time_train - time_last
        for inputs_val, y_true_val in gd_val:
            loss = loss_net(inputs_val, y_true_val, lw_tens, ssim_w, kl_weight)
            outputs_val, _, _ = net(inputs_val)
            loss_epoch_val += loss
            mse_loss_val = mse_loss_net(outputs_val, y_true_val, lw_tens)  ##todo
        history["val_loss"].append(loss_epoch_val)
        history["val_mse_loss"].append(mse_loss_val)
        time_val = time.time()
        val_spent = time_val - time_train
        time_last = time_val
        try:
            logging.info(
                "epoch: %d, loss: %.6f, mse: %.6f, \n "
                "val_loss: %.6f, val_loss_mse: %.6f, \n "
                "train time: %.6f, val time: %.6f",
                epoch, loss_epoch / len(gd_train), mse_loss,
                loss_epoch_val / len(gd_val), mse_loss_val, train_spent, val_spent
            )
        except ZeroDivisionError as e:
            logging.error("Error %s", e)
            raise
    save_checkpoint(net, os.path.join("results", "ckpt", "net", "VAE.ckpt"))
    logging.info("MODEL SAVED")
    import pandas as pd

    hist_df = pd.DataFrame(history)

    # save to json:
    hist_json_file = os.path.join("results", "ckpt", "net", "history.json")
    with open(hist_json_file, mode="w") as f:
        hist_df.to_json(f)

    with open(
            os.path.join("results", "ckpt", "net", "history.json"), "r", encoding="utf8"
    ) as fp:
        json_data = json.load(fp)  #

if TRAIN in (1, 2):
    bar_1 = 10 ** bar_1
    bar_2 = 10 ** bar_2
    checkpoint_dir = os.path.dirname(checkpoint_path)
    params = load_checkpoint(checkpoint_path)
    load_param_into_net(net, params)

    model_dec = np.zeros((model_sel_dim, 100))
    vae_initial = loadmat(
        os.path.join(
            "dataset",
            "conventional_inversion_result", "conventional_inversion_resistivity.mat",
        )
    )
    si_result = vae_initial["model"]
    si_result = np.reshape(si_result, (ZNUMBER, 100), order="f")
    si_result = si_result[-model_sel_dim:, :]

    for jj in range(100):
        # """Test with conventional inversion result"""
        logging.info("Valid %d", jj)
        si_ii = ms.Tensor(
            np.reshape(si_result[:, jj], (1, model_sel_dim, 1), order="f")
            .transpose((0, 2, 1))
            .astype(np.float32),
            ms.float32,
        )
        v_ii, _ = net.mean_model(si_ii)
        si_ii_dec = net.decoder(v_ii)  # decoded model in Logarithm
        si_ii_dec = si_ii_dec.asnumpy()
        model_dec[:, jj] = np.squeeze(si_ii_dec)

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        plt.plot(
            np.squeeze(si_ii_dec),
            zelementlocation[-model_sel_dim:],
            color="blue",
            linewidth=1.5,
        )
        plt.plot(
            np.squeeze(si_ii),
            zelementlocation[-model_sel_dim:],
            color="red",
            linewidth=1.5,
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.xlim(np.log10(bar_1), np.log10(bar_2))
        os.makedirs(
            os.path.join("results", "test_net", "test2"), exist_ok=True
        )
        plt.savefig(
            os.path.join(
                "results",
                "test_net",
                "test2",
                "True_model_No" + str(jj) + "_" + ".png",
            )
        )
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        plt.plot(
            np.squeeze(si_ii_dec),
            zelementlocation1[-model_sel_dim:],
            color="blue",
            linewidth=1.5,
        )
        plt.plot(
            np.squeeze(si_ii),
            zelementlocation1[-model_sel_dim:],
            color="red",
            linewidth=1.5,
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.xlim(np.log10(bar_1), np.log10(bar_2))
        os.makedirs(
            os.path.join("results", "test_net", "test2"), exist_ok=True
        )
        plt.savefig(
            os.path.join(
                "results",
                "test_net",
                "test2",
                "True_model_No" + str(jj) + "_" + "_evengrid.png",
            )
        )
        plt.close()
    coor = {
        "zelementlocation": zelementlocation[-model_sel_dim:],
        "xelementlocation": xelementlocationmt,
        "xx": model_dec,
        "colorbaraxis": [np.log10(bar_1), np.log10(bar_2)],
        "surf_flag": 1,
        "address": os.path.join(
            "results", "test_net", "test2", "True model decode.png"
        ),
    }
    hf.Plot2DImage(
        coor, rangex=[-30, 790], rangez=[-80, 0], iflog=1, use_cmap="jet_r",
    )
    coor = {
        "zelementlocation": zelementlocation1[-model_sel_dim:],
        "xelementlocation": xelementlocationmt,
        "xx": model_dec,
        "colorbaraxis": [np.log10(bar_1), np.log10(bar_2)],
        "surf_flag": 1,
        "address": os.path.join(
            "results",
            "test_net",
            "test2",
            "True model decode_evengrid.png",
        ),
    }
    hf.Plot2DImage(
        coor, rangex=[-30, 790], rangez=[-80, 0], iflog=1, use_cmap="jet_r",
    )

    # """Test with test set"""
    model_dec1 = np.zeros((model_sel_dim, 100))
    for jj in range(99):
        rho_test_no5 = int(jj)
        rho_test_no5 = ms.Tensor(
            rho_test[rho_test_no5 : rho_test_no5 + 1, :, :]
            .astype(np.float32)
            .transpose((0, 2, 1)),
            ms.float32,
        )
        v_true5, _ = net.mean_model(
            rho_test_no5
        )  # input can be tensor or numpy, output is numpy

        test_pred_y = net.decoder(v_true5)  # 解码的是对数域的ss
        test_pred_y = test_pred_y.asnumpy()
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        plt.plot(
            np.squeeze(test_pred_y),
            zelementlocation[-model_sel_dim:],
            color="blue",
            linewidth=1.5,
        )
        plt.plot(
            np.squeeze(rho_test_no5),
            zelementlocation[-model_sel_dim:],
            color="red",
            linewidth=1.5,
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.xlim(np.log10(bar_1), np.log10(bar_2))
        os.makedirs(
            os.path.join("results", "test_net", "test1"), exist_ok=True
        )
        plt.savefig(
            os.path.join(
                "results",
                "test_net",
                "test1",
                "__No" + str(jj) + "_" + ".png",
            )
        )
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)
        plt.plot(
            np.squeeze(test_pred_y),
            zelementlocation1[-model_sel_dim:],
            color="blue",
            linewidth=1.5,
        )
        plt.plot(
            np.squeeze(rho_test_no5),
            zelementlocation1[-model_sel_dim:],
            color="red",
            linewidth=1.5,
        )
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.xlim(np.log10(bar_1), np.log10(bar_2))
        os.makedirs(
            os.path.join("results", "test_net", "test1"), exist_ok=True
        )
        plt.savefig(
            os.path.join(
                "results",
                "test_net",
                "test1",
                "__No" + str(jj) + "_" + "_evengrid.png",
            )
        )
        plt.close()
        model_dec1[:, jj] = np.squeeze(test_pred_y)
    coor = {
        "zelementlocation": zelementlocation[-model_sel_dim:],
        "xelementlocation": xelementlocationmt,
        "xx": model_dec1,
        "colorbaraxis": [np.log10(bar_1), np.log10(bar_2)],
        "surf_flag": 1,
        "address": os.path.join(
            "results", "test_net", "test1", "_ decode.png"
        ),
    }
    hf.Plot2DImage(
        coor, rangex=[-30, 790], rangez=[-80, 0], iflog=1, use_cmap="jet_r",
    )
    coor = {
        "zelementlocation": zelementlocation1[-model_sel_dim:],
        "xelementlocation": xelementlocationmt,
        "xx": model_dec1,
        "colorbaraxis": [np.log10(bar_1), np.log10(bar_2)],
        "surf_flag": 1,
        "address": os.path.join(
            "results", "test_net", "test1", "_ decode_evengrid.png"
        ),
    }
    hf.Plot2DImage(
        coor, rangex=[-30, 790], rangez=[-80, 0], iflog=1, use_cmap="jet_r",
    )
