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
visualization functions
"""
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from mindspore import Tensor, dtype


def visual_static(x, yy, yp):
    """ Plot static figures.

    Args:
        x (Array): Input data with shape e.g. :math:`[H,W,C]`.
        yy (Array): Label data with shape e.g. :math:`[H,W,T,C]`.
        yp (Array): Label data with shape e.g. :math:`[H,W,T,C]`.
    """
    cmap = matplotlib.colormaps['jet']

    plt.figure(figsize=(18, 6))
    plt.subplot(3, 9, 1)
    plt.title("Input")
    plt.imshow(x, cmap=cmap)
    plt.axis('off')

    for i in range(9):
        label = yy[..., i, :]
        vmin, vmax = np.min(label), np.max(label)
        plt.subplot(3, 9, i + 10)
        plt.title(f"Label {i}")
        plt.imshow(yy[..., i, :], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.subplot(3, 9, i + 19)
        plt.title(f"Predict {i}")
        plt.imshow(yp[..., i, :], cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'images/result.jpg')
    plt.show()
    plt.close()


def visual_animate(yy, yp, ye):
    """ Plot animate figures.

    Args:
        yy (Array): Label data with shape e.g. :math:`[H,W,T,C]`.
        yp (Array): Label data with shape e.g. :math:`[H,W,T,C]`.
        ye (Array): Error data with shape e.g. :math:`[H,W,T,C]`.
    """
    cmap = matplotlib.colormaps['jet']
    fig, ax = plt.subplots(1, 3, figsize=[7, 3])

    ax[0].set_title(f'Label')
    im0 = ax[0].imshow(yy[..., 0, :], cmap=cmap)
    ax[1].set_title(f'Prediction')
    im1 = ax[1].imshow(yp[..., 0, :], cmap=cmap)
    ax[2].set_title(f'Error')
    im2 = ax[2].imshow(ye[..., 0, :], cmap=cmap)
    title = fig.suptitle(f't=0')
    fig.tight_layout()
    fig.colorbar(im1, ax=ax)

    def animate(i):
        y, p, e = yy[..., i, :], yp[..., i, :], ye[..., i, :]

        im0.set_data(y)
        im1.set_data(p)
        im2.set_data(e)
        vmin, vmax = np.min(y), np.max(y)
        im0.set_clim(vmin, vmax)
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)
        title.set_text(f't={i}')

    ani = animation.FuncAnimation(fig, animate, interval=200, blit=False, frames=10,
                                  repeat_delay=1000)
    ani.save('images/result.gif')
    plt.show()


def visual(problem, inputs, labels, t_out):
    """ Infer the model and visualize the results.

    Args:
        problem (BurgersWithLoss): A wrapper with step and get_loss method for model.
        inputs (Array): Input data with shape e.g. :math:`[N,T0,H,W,T,C]`.
        labels (Array): Label data with shape e.g. :math:`[N,T0,H,W,T,C]`.
        t_out (int): Number of time steps to predict sequentially.
    """
    t_start = inputs.shape[1] - t_out
    x = inputs[:1, t_start, ...]  # shape [1,H,W,T,C]
    problem.model.set_train(False)
    yp, _ = problem.step(Tensor(x, dtype=dtype.float32), t_out)  # shape [1,H,W,T,C]

    # Get and format input, label, prediction and error.
    x = x[0, :, :, 0, :]
    y_ = labels[0, t_start:, ...]  # shape [T0,H,W,T,C]
    t1, h, w, t2, c = list(y_.shape)
    y_ = y_.transpose((1, 2, 0, 3, 4))  # shape [H,W,T0,T,C]
    yy = y_.reshape((h, w, t1 * t2, c))  # Label, shape [H,W,T,C]
    yp = yp.asnumpy()[0, ...]  # Prediction, shape [H,W,T,C]
    ye = np.abs(yp - yy)  # Error

    # Plot static figures.
    visual_static(x, yy, yp)
    # Plot animate figures.
    visual_animate(yy, yp, ye)
