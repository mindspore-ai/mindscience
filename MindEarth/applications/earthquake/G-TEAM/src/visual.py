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
# ==============================================================================
"visualization"
import os
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics


def generate_true_pred_plot(pred_values, true_values, time, path, suffix=""):
    """
    Generate a plot comparing true values and predicted values, and calculate
    evaluation metrics including MAE, RMSE, R^2, and the standard deviation of residuals.
    Parameters:
    pred_values: List of predicted values
    true_values: List of true values
    time: Time, used for naming the image
    path: Path to save the image
    suffix: Suffix for image naming, default is an empty string
    """
    if suffix:
        suffix += "_"
    fig = plt.figure(figsize=(9, 9))
    plt.plot(true_values, pred_values, "ok", alpha=0.2)
    pred_value = pred_values
    pred_value = np.array([x for x in pred_value])
    r2 = metrics.r2_score(true_values, pred_value)
    rmse = np.sqrt(metrics.mean_squared_error(true_values, pred_value))
    mae = metrics.mean_absolute_error(true_values, pred_value)

    plt.text(
        0.6,
        6,
        f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n$R^{2}$: {r2:.2f}",
        fontsize=30,
        verticalalignment="top",
        horizontalalignment="left",
    )
    plt.plot(np.arange(0, 8), np.arange(0, 8), "-r")
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    ax = plt.gca()
    ax.set_xlabel("True values", fontsize=20)
    ax.set_ylabel("Pred values", fontsize=20)
    ax.set_title(str(time) + " s", fontsize=20)
    fig.savefig(os.path.join(path, f"truepred_{suffix}{time}.png"), bbox_inches="tight")
    plt.close()

    residual = true_values - pred_value
    fig = plt.figure(figsize=(9, 9))
    axs = fig.subplots(1, 1)
    axs.hist(residual)
    axs.set_xlabel("residual", fontsize=25)
    axs.set_ylabel("Event Number", fontsize=25)
    x_lim = axs.get_xlim()
    y_lim = axs.get_ylim()
    plt.text(
        x_lim[1] * 0.95,
        y_lim[1] * 0.95,
        f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\n$R^{{2}}$: {r2:.2f}\nSTD: {np.std(residual):.2f}",
        fontsize=30,
        verticalalignment="top",
        horizontalalignment="right",
    )

    fig.savefig(os.path.join(path, f"Residual_{suffix}{time}.png"), bbox_inches="tight")
    plt.close()
