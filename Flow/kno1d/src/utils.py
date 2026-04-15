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

from mindspore import Tensor, dtype


def visual(problem, inputs):
    """ Infer the model and visualize the results.

    Args:
        problem (BurgersWithLoss): A wrapper with step and get_loss method for model.
        inputs (Array): Input data with shape e.g. :math:`[N,H,T]`.
    """
    x = Tensor(inputs[:6, :, :], dtype=dtype.float32)
    problem.model.set_train(False)
    y, _ = problem.step(x)
    y = y.asnumpy()[:, :, :]

    cmap = matplotlib.colormaps['jet']
    fig, axes = plt.subplots(6, 1)
    im = None
    for i in range(6):
        im = axes[i].imshow(y[i].T, cmap=cmap, interpolation='nearest', aspect='auto')
        axes[i].set_ylabel('t')
    for i in range(5):
        axes[i].set_xticks([])
    axes[-1].set_xlabel('x')
    fig.tight_layout()
    cbar = fig.colorbar(im, ax=axes)
    cbar.set_label('u(t,x)')
    fig.savefig(f'images/result.jpg')
    fig.show()
