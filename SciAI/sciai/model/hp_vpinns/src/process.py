
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

"""process"""
import os

import yaml
import numpy as np
import mindspore as ms
from pyDOE import lhs

from sciai.utils import to_tensor, parse_arg
from .network import jacobi_poly, gauss_lobatto_jacobi_weights


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def get_data(args, dtype, np_dtype):
    """get data"""

    def test_fcn(n, x):
        test = jacobi_poly(n + 1, 0, 0, x) - jacobi_poly(n - 1, 0, 0, x)
        return test

    omega, amp, r1 = 8 * np.pi, 1, 80

    def u_ext(x):
        utemp = 0.1 * np.sin(omega * x) + np.tanh(r1 * x)
        return amp * utemp

    def f_ext(x):
        gtemp = -0.1 * (omega ** 2) * np.sin(omega * x) - (2 * r1 ** 2) * (np.tanh(r1 * x)) / ((np.cosh(r1 * x)) ** 2)
        return -amp * gtemp

    x_quad, w_quad = gauss_lobatto_jacobi_weights(args.n_quad, 0, 0)
    ne = args.n_element
    x_l, x_r = -1, 1
    delta_x = (x_r - x_l) / ne
    grid = np.asarray([x_l + i * delta_x for i in range(ne + 1)], dtype=np_dtype)
    n_testfcn_total = np.array((len(grid) - 1) * [args.n_testfcn])
    if args.n_element == 3:
        grid = np.array([-1, -0.1, 0.1, 1], dtype=np_dtype)
        ne = len(grid) - 1
        n_testfcn_total = np.array([args.n_testfcn, args.n_testfcn, args.n_testfcn])
    u_ext_total = []
    f_ext_total = []
    for e in range(ne):
        x_quad_element = grid[e] + (grid[e + 1] - grid[e]) / 2 * (x_quad + 1)
        jacobian = (grid[e + 1] - grid[e]) / 2
        n_testfcn_temp = n_testfcn_total[e]
        testfcn_element = np.asarray([test_fcn(n, x_quad) for n in range(1, n_testfcn_temp + 1)], dtype=np_dtype)

        u_quad_element = u_ext(x_quad_element)
        u_ext_element = jacobian * np.asarray(
            [sum(w_quad * u_quad_element * testfcn_element[i]) for i in range(n_testfcn_temp)], dtype=np_dtype)
        u_ext_element = u_ext_element[:, None]
        u_ext_total.append(u_ext_element)

        f_quad_element = f_ext(x_quad_element)
        f_ext_element = jacobian * np.asarray(
            [sum(w_quad * f_quad_element * testfcn_element[i]) for i in range(n_testfcn_temp)], dtype=np_dtype)
        f_ext_element = f_ext_element[:, None]
        f_ext_total.append(f_ext_element)
    f_ext_total = ms.Tensor(np.asarray(f_ext_total), dtype=dtype)
    # Training points
    x_u_train = np.asarray([-1.0, 1.0], dtype=np_dtype)[:, None]
    u_train = u_ext(x_u_train)
    x_f_train = (2 * lhs(1, args.n_f) - 1)
    f_train = f_ext(x_f_train)
    # Quadrature points
    x_quad, w_quad = gauss_lobatto_jacobi_weights(args.n_quad, 0, 0)
    x_quad_train = x_quad[:, None]
    w_quad_train = w_quad[:, None]
    # Test point
    delta_test = 0.001
    xtest = np.arange(-1, 1 + delta_test, delta_test)
    data_temp = np.asarray([[xtest[i], u_ext(xtest[i])] for i in range(len(xtest))], dtype=np_dtype)
    x_test = data_temp.flatten()[0::2]
    u_test = data_temp.flatten()[1::2]
    x_test = x_test[:, None]
    u_test = u_test[:, None]
    x_u_train, u_train, f_train, x_test = to_tensor((x_u_train, u_train, f_train, x_test), dtype)
    return f_ext_total, w_quad_train, x_quad_train, x_test, x_u_train, grid, u_test, u_train
