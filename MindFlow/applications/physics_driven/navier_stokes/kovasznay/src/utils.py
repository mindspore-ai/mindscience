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
"""Utils functions."""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import numpy as np

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import ops

from mindflow import sympy_to_mindspore


def relative_l2(x, y):
    """Calculate the relative L2 error."""
    return np.sqrt(np.mean(np.square(x - y))) / np.sqrt(np.mean(np.square(y)))


def calculate_l2_error(problem, model, ds_test):
    """Calculate the relative L2 error."""
    nodes = sympy_to_mindspore(
        {
            "u": problem.u_func(),
            "v": problem.v_func(),
            "p": problem.p_func(),
        },
        problem.in_vars,
        problem.out_vars,
    )
    for x_domain, x_bc in ds_test:
        y_pred_domain = model(x_domain)
        y_pred_bc = model(x_bc)

        domain_true = problem.parse_node(nodes, x_domain)
        domain_true = ops.Concat(axis=1)(domain_true)
        bc_true = problem.parse_node(nodes, x_bc)
        bc_true = ops.Concat(axis=1)(bc_true)

        metric_domain = relative_l2(y_pred_domain.asnumpy(), domain_true.asnumpy())
        metric_bc = relative_l2(y_pred_bc.asnumpy(), bc_true.asnumpy())
        print(f"Relative L2 error on domain: {metric_domain}")
        print(f"Relative L2 error on boundary: {metric_bc}")


def visual(model, config, resolution=100):
    """visulization of the results."""
    x_flat = np.linspace(0, 1, resolution)
    y_flat = np.linspace(0, 1, resolution)
    y_grid, x_grid = np.meshgrid(x_flat, y_flat)
    x = x_grid.reshape((-1, 1))
    y = y_grid.reshape((-1, 1))
    xy = np.concatenate((x, y), axis=1)
    xy = Tensor(xy, dtype=mstype.float32)
    predict = model(xy)
    u_predict = predict[:, 0]
    u_predict = u_predict.asnumpy()
    v_predict = predict[:, 1]
    v_predict = v_predict.asnumpy()
    p_predict = predict[:, 2]
    p_predict = p_predict.asnumpy()
    u_predict = u_predict.reshape((resolution, resolution))
    v_predict = v_predict.reshape((resolution, resolution))
    p_predict = p_predict.reshape((resolution, resolution))

    fig = plt.figure(figsize=(15, 4))
    gs = GridSpec(1, 3, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("u")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ac = ax1.contourf(x_grid, y_grid, u_predict, cmap=plt.cm.rainbow, levels=100)
    fig.colorbar(ac, ax=ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("v")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ac = ax2.contourf(x_grid, y_grid, v_predict, cmap=plt.cm.rainbow, levels=100)
    fig.colorbar(ac, ax=ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("p")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ac = ax3.contourf(x_grid, y_grid, p_predict, cmap=plt.cm.rainbow, levels=100)
    fig.colorbar(ac, ax=ax3)

    plt.tight_layout()
    plt.savefig(
        "images/kovasznay_epochs_{}_lr_{}.png".format(
            config["epochs"], config["optimizer"]["initial_lr"]
        )
    )
