# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# without warranties or conditions of any kind, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""produce meta data"""
import os
import numpy as np

from mindspore import load_checkpoint, load_param_into_net, ops

from mindflow.cell import MultiScaleFCSequential

from .grads import UNet, VNet, PNet, Grad


def burgers_meta_data(config):
    """prepare data for producing meta data"""
    meta_data_config = config["meta_data"]
    model_config = config["model"]
    summary_config = config["summary"]
    nx = meta_data_config["nx"]
    nt = meta_data_config["nt"]
    x = ops.linspace(meta_data_config["x_min"], meta_data_config["x_max"], nx)
    t = ops.linspace(meta_data_config["t_min"], meta_data_config["t_max"], nt)

    total = nx*nt
    num = 0
    data = ops.zeros(2)
    dataset = ops.zeros([total, 2])
    for j in range(nx):
        for i in range(nt):
            data[0] = x[j]
            data[1] = t[i]
            dataset[num] = data
            num += 1

    # prepare model for load parameters
    model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                   out_channels=model_config["out_channels"],
                                   layers=model_config["layers"],
                                   neurons=model_config["neurons"],
                                   residual=model_config["residual"],
                                   act=model_config["activation"],
                                   num_scales=1)

    ckpt_name = "burgers_nn-{}.ckpt".format(config["optimizer"]["epochs"] + 1)
    model_dict = load_checkpoint(os.path.join(
        summary_config["save_ckpt_path"], ckpt_name))
    load_param_into_net(model, model_dict)

    pinnstatic = model(dataset)
    h_n = pinnstatic.asnumpy()

    grad1 = Grad(model)(dataset)
    grad2 = Grad(Grad(model))(dataset)
    grad3 = Grad(Grad(Grad(model)))(dataset)

    h_x = grad1[:, 0].reshape(total, 1)
    h_x_n = h_x.asnumpy()
    h_t = grad1[:, 1].reshape(total, 1)
    h_t_n = h_t.asnumpy()
    h_xx = grad2[:, 0].reshape(total, 1)
    h_xx_n = h_xx.asnumpy()
    h_xxx = grad3[:, 0].reshape(total, 1)
    h_xxx_n = h_xxx.asnumpy()
    h_tt = grad2[:, 1].reshape(total, 1)
    h_tt_n = h_tt.asnumpy()

    theta = h_n
    theta = np.hstack((theta, h_x_n))
    theta = np.hstack((theta, h_xx_n))
    theta = np.hstack((theta, h_xxx_n))
    theta = np.hstack((theta, h_t_n))
    theta = np.hstack((theta, h_tt_n))
    return theta


def get_grads(grad1, grad2, grad3, total, include_time=True):
    """get grads"""
    results = []

    if include_time:
        h_t = grad1[:, 2].reshape(total, 1)
        results.append(h_t.asnumpy())

    h_x = grad1[:, 0].reshape(total, 1)
    results.append(h_x.asnumpy())
    h_y = grad1[:, 1].reshape(total, 1)
    results.append(h_y.asnumpy())

    h_xx = grad2[:, 0].reshape(total, 1)
    results.append(h_xx.asnumpy())
    h_yy = grad2[:, 1].reshape(total, 1)
    results.append(h_yy.asnumpy())

    h_xxx = grad3[:, 0].reshape(total, 1)
    results.append(h_xxx.asnumpy())
    h_yyy = grad3[:, 1].reshape(total, 1)
    results.append(h_yyy.asnumpy())

    return results


def cylinder_flow_meta_data(config):
    """prepare data for producing meta data"""
    meta_data_config = config["meta_data"]
    model_config = config["model"]
    summary_config = config["summary"]

    nx = meta_data_config["nx"]
    ny = meta_data_config["ny"]
    nt = meta_data_config["nt"]

    x = ops.linspace(meta_data_config["x_min"], meta_data_config["x_max"], nx)
    y = ops.linspace(meta_data_config["y_min"], meta_data_config["y_max"], ny)
    t = ops.linspace(meta_data_config["t_min"], meta_data_config["t_max"], nt)

    total = nx*ny*nt
    num = 0
    data = ops.zeros(3)
    dataset = ops.zeros([total, 3])
    for k in range(nx):
        for j in range(ny):
            for i in range(nt):
                data[0] = x[k]
                data[1] = y[j]
                data[2] = t[i]
                dataset[num] = data
                num += 1

    model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                   out_channels=model_config["out_channels"],
                                   layers=model_config["layers"],
                                   neurons=model_config["neurons"],
                                   residual=model_config["residual"],
                                   act=model_config["activation"],
                                   num_scales=1)
    ckpt_name = "cylinder_flow_nn-{}.ckpt".format(config["optimizer"]["epochs"] + 1)
    model_dict = load_checkpoint(os.path.join(
        summary_config["save_ckpt_path"], ckpt_name))
    load_param_into_net(model, model_dict)

    u = UNet(model)
    v = VNet(model)
    p = PNet(model)

    u_n = u(dataset).reshape(total, 1).asnumpy()
    v_n = v(dataset).reshape(total, 1).asnumpy()
    p_n = p(dataset).reshape(total, 1).asnumpy()

    u_grad1 = Grad(u)(dataset)
    u_grad2 = Grad(Grad(u))(dataset)
    u_grad3 = Grad(Grad(Grad(u)))(dataset)

    v_grad1 = Grad(v)(dataset)
    v_grad2 = Grad(Grad(v))(dataset)
    v_grad3 = Grad(Grad(Grad(v)))(dataset)

    p_grad1 = Grad(p)(dataset)
    p_grad2 = Grad(Grad(p))(dataset)
    p_grad3 = Grad(Grad(Grad(p)))(dataset)

    u_grads = get_grads(u_grad1, u_grad2, u_grad3, total)
    v_grads = get_grads(v_grad1, v_grad2, v_grad3, total)
    p_grads = get_grads(p_grad1, p_grad2, p_grad3, total)

    theta = u_n
    for u_grad in u_grads:
        theta = np.hstack((theta, u_grad))

    theta = np.hstack((theta, v_n))
    for v_grad in v_grads:
        theta = np.hstack((theta, v_grad))

    theta = np.hstack((theta, p_n))
    for p_grad in p_grads:
        theta = np.hstack((theta, p_grad))

    return theta


def periodic_hill_meta_data(config):
    """prepare data for producing meta data"""
    meta_data_config = config["meta_data"]
    model_config = config["model"]
    summary_config = config["summary"]

    nx = meta_data_config["nx"]
    ny = meta_data_config["ny"]

    x = ops.linspace(meta_data_config["x_min"], meta_data_config["x_max"], nx)
    y = ops.linspace(meta_data_config["y_min"], meta_data_config["y_max"], ny)

    total = nx * ny
    num = 0
    data = ops.zeros(2)
    dataset = ops.zeros([total, 2])
    for j in range(ny):
        for i in range(nx):
            data[0] = x[i]
            data[1] = y[j]
            dataset[num] = data
            num += 1

    model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                   out_channels=model_config["out_channels"],
                                   layers=model_config["layers"],
                                   neurons=model_config["neurons"],
                                   residual=model_config["residual"],
                                   act=model_config["activation"],
                                   num_scales=1)
    ckpt_name = "periodic_hill_nn-{}.ckpt".format(config["optimizer"]["epochs"] + 1)
    model_dict = load_checkpoint(os.path.join(
        summary_config["save_ckpt_path"], ckpt_name))
    load_param_into_net(model, model_dict)

    u = UNet(model)
    v = VNet(model)
    p = PNet(model)

    u_n = u(dataset).reshape(total, 1).asnumpy()
    v_n = v(dataset).reshape(total, 1).asnumpy()
    p_n = p(dataset).reshape(total, 1).asnumpy()

    u_grad1 = Grad(u)(dataset)
    u_grad2 = Grad(Grad(u))(dataset)
    u_grad3 = Grad(Grad(Grad(u)))(dataset)

    v_grad1 = Grad(v)(dataset)
    v_grad2 = Grad(Grad(v))(dataset)
    v_grad3 = Grad(Grad(Grad(v)))(dataset)

    p_grad1 = Grad(p)(dataset)
    p_grad2 = Grad(Grad(p))(dataset)
    p_grad3 = Grad(Grad(Grad(p)))(dataset)

    u_grads = get_grads(u_grad1, u_grad2, u_grad3, total, False)
    v_grads = get_grads(v_grad1, v_grad2, v_grad3, total, False)
    p_grads = get_grads(p_grad1, p_grad2, p_grad3, total, False)

    theta = u_n
    for u_grad in u_grads:
        theta = np.hstack((theta, u_grad))

    theta = np.hstack((theta, v_n))
    for v_grad in v_grads:
        theta = np.hstack((theta, v_grad))

    theta = np.hstack((theta, p_n))
    for p_grad in p_grads:
        theta = np.hstack((theta, p_grad))

    return theta


def produce_meta_data(case_name, config):
    """produce_meta_data"""
    meta_data_config = config["meta_data"]
    if case_name == "burgers":
        theta = burgers_meta_data(config)
    elif case_name == "cylinder_flow":
        # cylinder_flow
        theta = cylinder_flow_meta_data(config)
    else:
        #  periodic_hill
        theta = periodic_hill_meta_data(config)

    # create meta-data dir
    meta_data_save_path = meta_data_config["meta_data_save_path"]
    if not os.path.exists(os.path.abspath(meta_data_save_path)):
        os.makedirs(os.path.abspath(meta_data_save_path))

    np.save(os.path.join(
        meta_data_save_path, f"{case_name}_theta-ga"), theta)
