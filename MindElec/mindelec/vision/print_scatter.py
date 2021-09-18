# Copyright 2021 Huawei Technologies Co., Ltd
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
"""util functions for tests"""

import os
import numpy as np
import matplotlib.pyplot as plt


def print_graph_1d(name, x, path, clear=True):
    r"""
    Draw 1d scatter image

    Args:
        name (str): name of the graph.
        x (numpy.ndarray): data to draw (shape (dim_print,)).
        path (str): save path of the graph.
        clear (bool): specifies whether clear the current axes. Default: True.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.vision import print_graph_1d
        >>> name = "output.jpg"
        >>> x = np.ones(10)
        >>> path = "./graph_1d"
        >>> clear = True
        >>> print_graph_1d(name, x, path, clear)
    """
    if not isinstance(name, str):
        raise TypeError("The type of name should be str, but get {}".format(type(name)))

    if not isinstance(x, np.ndarray):
        raise TypeError("The type of x should be numpy array, but get {}".format(type(x)))
    shape_x = x.shape
    if len(shape_x) != 1:
        raise ValueError("x shape should be (dim_print,), but get {}".format(shape_x))

    if not isinstance(path, str):
        raise TypeError("The type of path should be str, but get {}".format(type(path)))
    if not os.path.exists(path):
        os.makedirs(path)

    if not isinstance(clear, bool):
        raise TypeError("The type of clear should be bool, but get {}".format(type(clear)))

    if clear:
        plt.cla()
    y = np.zeros(x.shape)
    plt.scatter(x, y, alpha=0.8, s=0.8)
    plt.savefig(os.path.join(path, name), dpi=600)


def print_graph_2d(name, x, y, path, clear=True):
    r"""
    Draw 2d scatter image

    Args:
        name (str): name of the graph.
        x (numpy.ndarray): data x to draw (shape (dim_print,)).
        y (numpy.ndarray): data y to draw (shape (dim_print,)).
        path (str): save path of the graph.
        clear (bool): specifies whether clear the current axes. Default: True.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindelec.vision import print_graph_2d
        >>> name = "output.jpg"
        >>> x = np.ones(10)
        >>> y = np.ones(10)
        >>> path = "./graph_2d"
        >>> clear = True
        >>> print_graph_2d(name, x, y, path, clear)
    """
    if not isinstance(name, str):
        raise TypeError("The type of name should be str, but get {}".format(type(name)))

    if not isinstance(x, np.ndarray):
        raise TypeError("The type of x should be numpy array, but get {}".format(type(x)))
    shape_x = x.shape
    if len(shape_x) != 1:
        raise ValueError("x shape should be (dim_print,), but get {}".format(shape_x))

    if not isinstance(y, np.ndarray):
        raise TypeError("The type of y should be numpy array, but get {}".format(type(y)))
    shape_y = y.shape
    if len(shape_y) != 1:
        raise ValueError("y shape should be (dim_print,), but get {}".format(shape_y))

    if not isinstance(path, str):
        raise TypeError("The type of path should be str, but get {}".format(type(path)))
    if not os.path.exists(path):
        os.makedirs(path)

    if not isinstance(clear, bool):
        raise TypeError("The type of clear should be bool, but get {}".format(type(clear)))

    if clear:
        plt.cla()
    plt.scatter(x, y, alpha=1.0, s=0.8)
    plt.savefig(os.path.join(path, name), dpi=600)
