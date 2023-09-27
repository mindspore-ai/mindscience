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
import scipy.io
from pyDOE import lhs

from sciai.utils import to_tensor, flatten_add_dim, parse_arg
from .plot import plot_uvmag, plot_sigma


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def plot_res(*args):
    """plot result"""
    max_t, n_t, model, save_figures_path, data_path, srcs, dtype = args
    x_star, y_star = xy_stars(srcs)
    for i in range(n_t):
        t_star = np.ones((x_star.size, 1)) * (i * max_t / (n_t - 1))
        x_star_, y_star_, t_star_ = to_tensor((x_star, y_star, t_star), dtype)
        u_pred, v_pred, s11_pred, s22_pred, s12_pred, _, _, _ = model.predict(x_star_, y_star_, t_star_)
        field = x_star, y_star, t_star, u_pred, v_pred, s11_pred, s22_pred, s12_pred
        post_process(field=field, figures_path=save_figures_path, data_path=data_path, num=i)


def xy_stars(srcs):
    """xy stars"""
    xc_src, yc_src, r_src = srcs
    x_star, y_star = np.linspace(-15, 15, 201), np.linspace(-15, 15, 201)
    x_star, y_star = flatten_add_dim(*np.meshgrid(x_star, y_star))
    dst = ((x_star - xc_src) ** 2 + (y_star - yc_src) ** 2) ** 0.5
    x_star, y_star = x_star[dst >= r_src], y_star[dst >= r_src]
    return flatten_add_dim(x_star, y_star)


def generate_data(dtype):
    """generate data"""
    max_t = 14.0  # Pretraining will help convergence (train 7s -> 14s)
    lb, ub = np.array([-15.0, -15.0, 0.0]), np.array([15.0, 15.0, max_t])  # Domain bounds
    xc_src, yc_src, r_src = 0.0, 0.0, 2.0  # Properties of source
    srcs = xc_src, yc_src, r_src
    n_f, n_t = 120000, int(max_t * 4 + 1)  # Num of collocation point in x, y, t.  4 frames per second
    fixed, ic = init_condition(lb, max_t)
    xyt_c = collocation_point(lb, ub, srcs, n_f, max_t)  # Collocation point
    src = wave_src(max_t, srcs)  # Wave source, Gauss pulse
    x_c, y_c, t_c = xyt_c[:, :1], xyt_c[:, 1:2], xyt_c[:, 2:3]  # Collocation point
    x_src, y_src, t_src, u_src, v_src = src[:, :1], src[:, 1:2], src[:, 2:3], src[:, 3:4], src[:, 4:5]  # Source wave
    x_ic, y_ic, t_ic = ic[:, :1], ic[:, 1:2], ic[:, 2:3]  # Initial condition point, t: 0
    x_fix, y_fix, t_fix = fixed[:, :1], fixed[:, 1:2], fixed[:, 2:3]
    input_data = to_tensor((x_c, y_c, t_c, x_src, y_src, t_src, u_src, v_src, x_ic, y_ic, t_ic, x_fix, y_fix, t_fix),
                           dtype=dtype)
    return max_t, n_t, input_data, srcs


def wave_src(max_t, srcs):
    """wave src"""
    xc_src, yc_src, r_src = srcs
    xx, yy = gen_circle_pt(*srcs, n_pt=200)  # n_pt=500
    tt = np.concatenate((np.linspace(0, 4, 141), np.linspace(4, max_t, 141)), 0)[1:]
    x_src, t_src = np.meshgrid(xx, tt)
    y_src, _ = np.meshgrid(yy, tt)
    x_src, y_src, t_src = flatten_add_dim(x_src, y_src, t_src)
    amplitude = 0.5 * np.exp(-((t_src - 2.0) / 0.5) ** 2)
    u_src = amplitude * (x_src - xc_src) / r_src
    v_src = amplitude * (y_src - yc_src) / r_src
    return np.concatenate((x_src, y_src, t_src, u_src, v_src), 1)


def collocation_point(lb, ub, src, n_f, max_t):
    """collocation point"""
    xc_src, yc_src, r_src = src
    xyt_c = lb + (ub - lb) * lhs(3, n_f)
    xyt_c_ext = np.array([xc_src - r_src - 1, yc_src - r_src - 1, 0.0]) + np.array(
        [2 * (r_src + 1), 2 * (r_src + 1), max_t]) * lhs(3, 15000)  # Refinement around source
    xyt_c_ext2 = lb + (ub - lb) * lhs(3, 50000)
    flag = (np.abs(xyt_c_ext2[:, 0]) > 12) | (np.abs(xyt_c_ext2[:, 1]) > 12)
    xyt_c_ext2 = xyt_c_ext2[flag, :]
    xyt_c = np.concatenate((xyt_c, xyt_c_ext, xyt_c_ext2), axis=0)
    return del_src_pt(xyt_c, *src)


def init_condition(lb, max_t):
    """Initial condition point for u, v, ut, vt"""
    ic = lb + np.array([30.0, 30.0, 0.0]) * lhs(3, 6000)
    ic = del_src_pt(ic, xc=0, yc=0, r=2.0)
    low = np.array([-15.0, -15.0, 0.0]) + np.array([30.0, 0.0, max_t]) * lhs(3, 7000)
    up = np.array([-15.0, 15.0, 0.0]) + np.array([30.0, 0.0, max_t]) * lhs(3, 7000)
    left = np.array([-15.0, -15.0, 0.0]) + np.array([0.0, 30.0, max_t]) * lhs(3, 7000)
    right = np.array([15.0, -15.0, 0.0]) + np.array([0.0, 30.0, max_t]) * lhs(3, 7000)
    fixed = np.concatenate((left, right, low, up), 0)
    return fixed, ic


def gen_dist_pt(**kwargs):
    """generate dist pt"""
    # num: number per edge. num_t: number time step
    xmin, xmax, ymin, ymax, tmin, tmax, xc, yc, r, num_surf_pt, num, num_t = kwargs.values()
    x, y = np.meshgrid(np.linspace(xmin, xmax, num=num), np.linspace(ymin, ymax, num=num))
    dst = ((x - xc) ** 2 + (y - yc) ** 2) ** 0.5  # Delete point in hole
    x, y = flatten_add_dim(x[dst >= r], y[dst >= r])
    theta = np.linspace(0.0, np.pi * 2, num_surf_pt)  # Add point on hole surface
    x_surf, y_surf = flatten_add_dim(np.multiply(r, np.cos(theta)) + xc, np.multiply(r, np.sin(theta)) + yc)
    x, y = np.concatenate((x, x_surf), 0), np.concatenate((y, y_surf), 0)
    t = np.linspace(tmin, tmax, num=num_t)
    xxx, ttt = np.meshgrid(x, t)
    yyy, _ = np.meshgrid(y, t)
    return flatten_add_dim(xxx, yyy, ttt)


def cart_grid(*args):
    """cart grid"""
    xmin, xmax, ymin, ymax, tmin, tmax, num, num_t = args
    # num: number per edge. num_t: number time step
    x = np.linspace(xmin, xmax, num=num)
    y = np.linspace(ymin, ymax, num=num)
    t = np.linspace(tmin, tmax, num=num_t)
    return flatten_add_dim(np.meshgrid(x, y, t))


def post_process(field, figures_path, data_path, num):
    """
    post process.
    num: Number of time step
    """
    data = scipy.io.loadmat(f'{data_path}/FEM_result/30x30_gauss_fine/ProbeData-{num}.mat')
    x_star, y_star, u_star, v_star, a_star, s11_star, s22_star, s12_star, _ \
        = flatten_add_dim(data['x'], data['y'], data['u'], data['v'], data['amp'],
                          data['s11'], data['s22'], data['s12'], data['Mises'])
    x_pred, y_pred, _, u_pred, v_pred, s11_pred, s22_pred, s12_pred = field
    amp_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5

    os.makedirs(figures_path, exist_ok=True)

    plot_uvmag(x_star, y_star, u_star, v_star, a_star, x_pred, y_pred, u_pred, v_pred,
               figures_path, num, amp_pred)
    plot_sigma(x_star, y_star, s11_star, s22_star, s12_star, x_pred, y_pred, s11_pred, s22_pred, s12_pred,
               figures_path, num)


def gen_circle_pt(xc, yc, r, n_pt):
    """generate circle pt"""
    theta = np.linspace(0.0, np.pi * 2, n_pt)
    xx = np.multiply(r, np.cos(theta)) + xc
    yy = np.multiply(r, np.sin(theta)) + yc
    return flatten_add_dim(xx, yy)


def del_src_pt(xyt_c, xc, yc, r):
    """del src pt"""
    dst = np.array([((xyt[0] - xc) ** 2 + (xyt[1] - yc) ** 2) ** 0.5 for xyt in xyt_c])
    return xyt_c[dst > r, :]


def shuffle(xyt_c, src, ic, up):
    """Shuffle along the first dimension"""
    np.random.shuffle(xyt_c)
    np.random.shuffle(src)
    np.random.shuffle(ic)
    np.random.shuffle(up)
