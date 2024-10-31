# Copyright 2024 Huawei Technologies Co., Ltd
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
"""NSFNet dataset"""
# pylint: disable=C0103
import numpy as np


def ThreeBeltramiflow(x, y, z, t):
    """
    ThreeBeltramiflow
    """
    a = 1.
    d = 1.
    u = -a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(-d * d * t)
    v = -a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(-d * d * t)
    w = -a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(-d * d * t)
    p = -0.5 * a * a * (np.exp(2. * a * x) + np.exp(2. * a * y) + np.exp(2. * a * z) +
                        2. * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                        2. * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                        2. * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) *\
                            np.exp(-2. * d * d * t)

    return u, v, w, p

def read_training_data():
    """prepare data"""
    # prepare the training data
    n_x = 16
    n_y = 16
    n_z = 16
    n_t = 21




    # Geometry
    xmin = -1
    xmax = 1
    ymin = -1
    ymax = 1
    zmin = -1
    zmax = 1
    tmin = 0.
    tmax = 1.

    # Boundary datapoints
    t_ = np.arange(tmin, tmax + 1e-5, (tmax - tmin) / (n_t - 1.)).reshape([-1, 1])
    x_b_sum = np.arange(xmin, xmax + 1e-5, (xmax - xmin) / (n_x - 1.))
    y_b_sum = np.arange(ymin, ymax + 1e-5, (ymax - ymin) / (n_y - 1.))
    z_b_sum = np.arange(zmin, zmax + 1e-5, (zmax - zmin) / (n_z - 1.))

    x_b_top, z_b_top = np.meshgrid(x_b_sum, z_b_sum)
    y_b_top = ymax * np.ones([n_x * n_z, 1])
    x_b_top = x_b_top.reshape([-1, 1])
    z_b_top = z_b_top.reshape([-1, 1])

    x_b_down, z_b_down = np.meshgrid(x_b_sum, z_b_sum)
    y_b_down = ymin * np.ones([n_x * n_z, 1])
    x_b_down = x_b_down.reshape([-1, 1])
    z_b_down = z_b_down.reshape([-1, 1])

    y_b_left, z_b_left = np.meshgrid(y_b_sum, z_b_sum)
    x_b_left = xmin * np.ones([n_y * n_z, 1])
    y_b_left = y_b_left.reshape([-1, 1])
    z_b_left = z_b_left.reshape([-1, 1])

    y_b_right, z_b_right = np.meshgrid(y_b_sum, z_b_sum)
    x_b_right = xmax * np.ones([n_y * n_z, 1])
    y_b_right = y_b_right.reshape([-1, 1])
    z_b_right = z_b_right.reshape([-1, 1])

    x_b_front, y_b_front = np.meshgrid(x_b_sum, y_b_sum)
    z_b_front = zmax * np.ones([n_x * n_y, 1])
    x_b_front = x_b_front.reshape([-1, 1])
    y_b_front = x_b_front.reshape([-1, 1])

    x_b_back, y_b_back = np.meshgrid(x_b_sum, y_b_sum)
    z_b_back = zmin * np.ones([n_x * n_y, 1])
    x_b_back = x_b_back.reshape([-1, 1])
    y_b_back = x_b_back.reshape([-1, 1])

    x_ = np.concatenate([x_b_top, x_b_down, x_b_left, x_b_right, x_b_front, x_b_back], axis=0).reshape([-1, 1])

    y_ = np.concatenate([y_b_top, y_b_down, y_b_left, y_b_right, y_b_front, y_b_back], axis=0).reshape([-1, 1])

    z_ = np.concatenate([z_b_top, z_b_down, z_b_left, z_b_right, z_b_front, z_b_back], axis=0).reshape([-1, 1])

    T_ = np.tile(t_, (1, x_.shape[0])).T  # N x T
    X_ = np.tile(x_, (1, t_.shape[0]))  # N x T
    Y_ = np.tile(y_, (1, t_.shape[0]))  # N x T
    Z_ = np.tile(z_, (1, t_.shape[0]))  # N x T

    t_b = T_.flatten()[:, None]  # NT x 1
    x_b = X_.flatten()[:, None]  # NT x 1
    y_b = Y_.flatten()[:, None]  # NT x 1
    z_b = Z_.flatten()[:, None]  # NT x 1

    # data on the initial condition
    x_ = np.arange(xmin, xmax + 1e-5, (xmax - xmin) / (n_x - 1.))
    y_ = np.arange(ymin, ymax + 1e-5, (ymax - ymin) / (n_y - 1.))
    z_ = np.arange(zmin, zmax + 1e-5, (zmax - zmin) / (n_z - 1.))
    xx, yy, zz = np.meshgrid(x_, y_, z_)
    x_i = xx.flatten()[:, None]
    y_i = yy.flatten()[:, None]
    z_i = zz.flatten()[:, None]
    t_i = np.ones([x_i.shape[0], 1]) * tmin

    t = np.concatenate([t_b, t_i], axis=0)
    x = np.concatenate([x_b, x_i], axis=0)
    y = np.concatenate([y_b, y_i], axis=0)
    z = np.concatenate([z_b, z_i], axis=0)


    u_b, v_b, w_b, _ = ThreeBeltramiflow(x_b, y_b, z_b, t_b)
    u_i, v_i, w_i, _ = ThreeBeltramiflow(x_i, y_i, z_i, t_i)




    X = np.concatenate([x, y, z, t], 1)
    X_min = X.min(0)
    X_max = X.max(0)

    # data on the velocity (inside the domain)
    xx, yy, zz, tt = np.meshgrid(x_, y_, z_, t_)

    t_f = tt.flatten()[:, None]
    x_f = xx.flatten()[:, None]
    y_f = yy.flatten()[:, None]
    z_f = zz.flatten()[:, None]


    return x_b, y_b, z_b, t_b, x_i, y_i, z_i, t_i, u_b, v_b, w_b, u_i, v_i, w_i, x_f, y_f, z_f, t_f, X_min, X_max
