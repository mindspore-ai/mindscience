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

"""Plotting results"""
import math
import os

import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from mindspore import ops
from sciai.utils import print_log


def visualize(args):
    """Results visualize"""
    filename = args.save_data_path + f"/data.txt"
    with open(filename) as file:
        lines = file.readlines()
        split_lines = list(map(lambda line: line.strip().split(" "), lines))
        num_list = []
        for line in split_lines:
            line_float = list(map(float, line))
            num_list.append(line_float)

    num_r = int(len(num_list) / 3)
    num_theta = len(num_list[0])
    x = np.zeros((num_r, num_theta))
    y = np.zeros((num_r, num_theta))
    z = np.zeros((num_r, num_theta))
    for idx in range(num_r):
        x[idx, :] = np.array(num_list[3 * idx])
        y[idx, :] = np.array(num_list[3 * idx + 1])
        z[idx, :] = np.array(num_list[3 * idx + 2])

    idx = np.where(z < np.inf)
    z_valid = z[idx]
    levels = np.linspace(np.amin(z_valid), np.amax(z_valid), 100)

    fig, ax = plt.subplots()
    cs = ax.contourf(x, y, z, levels=levels)
    fig.colorbar(cs)

    plt.title('2d possion')
    plt.savefig(f"{args.figures_path}/{args.case}.png")


def write_row(list_, file):
    for i in list_:
        file.write("%s " % i)
    file.write("\n")


def write(x, y, z, n_sampling, file):
    for k1 in range(n_sampling):
        write_row(x[k1], file)
        write_row(y[k1], file)
        write_row(z[k1], file)


def write_boundary(edge_ist, path, edge_list2=None):
    """Write boundary"""
    with open(path + "/boundary_coord.txt", mode="w+") as file:
        for i in edge_ist:
            write_row(i, file)
        if edge_list2 is not None:
            for i in edge_list2:
                write_row(i, file)

    with open(path + "/boundary_number.txt", mode="w+") as file:
        length = [len(edge_ist)] if edge_list2 is None else [len(edge_ist), len(edge_list2)]
        for i in length:
            file.write("%s\n" % i)


def write_result(args, model, n_sample, dtype):
    """Write results"""
    if args.case == 'ls':
        write_result_ls(args, model, n_sample, dtype)
    elif args.case == 'hole':
        write_result_hole(args, model, n_sample, dtype)
    else:
        print_log("Unsupported case")


def write_result_ls(args, model, n_sample, dtype):
    """Write results for poisson_ls"""
    radius = args.radius
    expand_dim = ops.ExpandDims()
    r_list = np.linspace(0, radius, n_sample)
    theta_list = np.linspace(0, math.pi * 2, n_sample)

    xx = np.zeros([n_sample, n_sample])
    yy = np.zeros([n_sample, n_sample])
    zz = np.zeros([n_sample, n_sample])
    for i in range(n_sample):
        for j in range(n_sample):
            xx[i, j] = r_list[i] * math.cos(theta_list[j])
            yy[i, j] = r_list[i] * math.sin(theta_list[j])
            coord = np.array([xx[i, j], yy[i, j]])
            x = expand_dim(ms.Tensor.from_numpy(coord).astype(dtype), 0)
            res = model(x)
            zz[i, j] = res.asnumpy()[0][0]

    if not os.path.exists(args.save_data_path):
        os.makedirs(args.save_data_path)

    with open(f"{args.save_data_path}/n_sample.txt", mode="w+") as file:
        file.write(str(n_sample))

    with open(f"{args.save_data_path}/data.txt", mode="w+") as file:
        write(xx, yy, zz, n_sample, file)

    edge_list = [[radius * math.cos(i), radius * math.sin(i)] for i in theta_list]
    write_boundary(edge_list, path=args.save_data_path)


def write_result_hole(args, model, n_sample, dtype):
    """Write results for poisson_hole"""
    x_list = np.linspace(-1, 1, n_sample)
    y_list = np.linspace(-1, 1, n_sample)
    theta_list = np.linspace(0, 2 * math.pi, 50)

    expand_dim = ops.ExpandDims()

    xx = np.zeros([n_sample, n_sample])
    yy = np.zeros([n_sample, n_sample])
    zz = np.zeros([n_sample, n_sample])
    for i in range(n_sample):
        for j in range(n_sample):
            xx[i, j] = x_list[i]
            yy[i, j] = y_list[j]
            coord = np.array([xx[i, j], yy[i, j]])
            x = expand_dim(ms.Tensor.from_numpy(coord).astype(dtype), 0)
            res = model(x)
            zz[i, j] = res.asnumpy()[0][0]
            if np.linalg.norm(coord - np.array([0.3, 0.0])) < 0.3:
                zz[i, j] = np.inf

    if not os.path.exists(args.save_data_path):
        os.makedirs(args.save_data_path)

    with open(f"{args.save_data_path}/n_sample.txt", mode="w+") as file:
        file.write(str(n_sample))

    with open(f"{args.save_data_path}/data.txt", mode="w+") as file:
        write(xx, yy, zz, n_sample, file)

    edge_list1 = [[0.3 * math.cos(i) + 0.3, 0.3 * math.sin(i)] for i in theta_list]
    edge_list2 = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]
    write_boundary(edge_list1 + edge_list2, path=args.save_data_path)
