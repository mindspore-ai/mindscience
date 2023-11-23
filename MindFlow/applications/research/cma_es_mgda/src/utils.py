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
"""utils for training"""
import collections
import os
import time

import io
import cv2
import PIL
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

from mindspore import Tensor
from mindspore import dtype as mstype

from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import print_log

from .model import Burgers1D, NavierStokes2D, NavierStokesRANS
from .dataset import create_burgers_train_dataset, create_burgers_test_dataset
from .dataset import create_cylinder_flow_train_dataset, create_cylinder_flow_test_dataset
from .dataset import create_periodic_hill_train_dataset, create_periodic_hill_test_dataset
from .trainer import BurgersTrainer, CylinderflowTrainer, PeriodichillTrainer


def visual_burgers(model, epochs=1, resolution=100):
    r"""visulization of ex/ey/hz"""
    t_flat = np.linspace(0, 1, resolution)
    x_flat = np.linspace(-1, 1, resolution)
    t_grid, x_grid = np.meshgrid(t_flat, x_flat)
    x = x_grid.reshape((-1, 1))
    t = t_grid.reshape((-1, 1))
    xt = Tensor(np.concatenate((x, t), axis=1), dtype=mstype.float32)
    u_predict = model(xt)
    u_predict = u_predict.asnumpy()
    gs = GridSpec(2, 3)
    plt.subplot(gs[0, :])
    plt.scatter(t, x, c=u_predict, cmap=plt.cm.rainbow)
    plt.xlabel('t')
    plt.ylabel('x')
    cbar = plt.colorbar(pad=0.05, aspect=10)
    cbar.set_label('u(t,x)')
    cbar.mappable.set_clim(-1, 1)
    t_cross_sections = [0.25, 0.5, 0.75]
    for i, t_cs in enumerate(t_cross_sections):
        plt.subplot(gs[1, i])
        xt = Tensor(
            np.stack([x_flat, np.full(x_flat.shape, t_cs)], axis=-1), dtype=mstype.float32)
        u = model(xt).asnumpy()
        plt.plot(x_flat, u)
        plt.title('t={}'.format(t_cs))
        plt.xlabel('x')
        plt.ylabel('u(t,x)')
    plt.tight_layout()
    plt.savefig(f'images/{epochs + 1}-result.jpg')


def calculate_burgers_error(label, prediction):
    r'''calculate l2-error to evaluate accuracy'''
    error = label - prediction
    l2_error = np.sqrt(
        np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))

    return l2_error


def get_burgers_prediction(model, inputs, label_shape):
    r'''calculate the prediction respect to the given inputs'''
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    inputs = inputs.reshape((-1, inputs.shape[1]))

    time_beg = time.time()
    batch_size = inputs.shape[0]
    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print_log("predict total time: {} ms".format(
        (time.time() - time_beg)*1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[1]))
    return prediction


def calculate_burgers_l2_error(model, inputs, label):
    r"""
    Evaluate the model respect to input data and label.

    Args:
         model (Cell): list of expressions node can by identified by mindspore.
         inputs (Tensor): the input data of network.
         label (Tensor): the true output value of given inputs.

    """
    label_shape = label.shape
    prediction = get_burgers_prediction(model, inputs, label_shape)
    label = label.reshape((-1, label_shape[1]))
    l2_error = calculate_burgers_error(label, prediction)
    print_log("    l2_error: ", l2_error)
    print_log(
        "==================================================================================================")


plt.rcParams['figure.dpi'] = 300


def cal_abs_error(u_predict_2d, u_label_2d, v_predict_2d,
                  v_label_2d, p_predict_2d, p_label_2d):
    r"""Evaluate abs error"""
    u_error_2d = np.abs(u_predict_2d - u_label_2d)
    v_error_2d = np.abs(v_predict_2d - v_label_2d)
    p_error_2d = np.abs(p_predict_2d - p_label_2d)
    return u_error_2d, v_error_2d, p_error_2d


def cal_l2_error(error, label):
    r"""Evaluate l2 error"""
    l2_error_u = np.sqrt(np.sum(np.square(
        error[:, :, 0]), axis=1)) / np.sqrt(np.sum(np.square(label[:, :, 0]), axis=1))
    l2_error_v = np.sqrt(np.sum(np.square(
        error[:, :, 1]), axis=1)) / np.sqrt(np.sum(np.square(label[:, :, 1]), axis=1))
    l2_error_p = np.sqrt(np.sum(np.square(
        error[:, :, 2]), axis=1)) / np.sqrt(np.sum(np.square(label[:, :, 2]), axis=1))
    return l2_error_u, l2_error_v, l2_error_p


def visual_cylinder_flow(model, epochs, input_data, label, path="./videos"):
    r"""visulization of u/v/p"""
    predict = model(Tensor(input_data, mstype.float32)).asnumpy()
    [sample_t, sample_x, sample_y, _] = np.shape(input_data)

    u_vmin, u_vmax = np.percentile(label[:, :, :, 0], [0.5, 99.5])
    v_vmin, v_vmax = np.percentile(label[:, :, :, 1], [0.5, 99.5])
    p_vmin, p_vmax = np.percentile(label[:, :, :, 2], [0.5, 99.5])

    vmin_list = [u_vmin, v_vmin, p_vmin]
    vmax_list = [u_vmax, v_vmax, p_vmax]

    output_names = ["U", "V", "P"]

    if not os.path.isdir(os.path.abspath(path)):
        os.makedirs(path)

    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    fps = 10
    size = (1920, 1440)

    # writFe value video
    video = cv2.VideoWriter(os.path.join(
        path, "FlowField_" + str(epochs + 1) + ".avi"), fourcc, fps, size)

    t_set = []
    if sample_t < 100:
        t_set = np.arange(sample_t, dtype=np.int32)
    else:
        for t in range(sample_t):
            if t % int(sample_t / 50) == 0 or t == sample_t - 1:
                t_set.append(t)

    for t in t_set:
        # get 1d true value
        u_label = label[t, :, :, 0]
        v_label = label[t, :, :, 1]
        p_label = label[t, :, :, 2]

        # get 2d predict value
        u_predict = predict[t, :, :, 0]
        v_predict = predict[t, :, :, 1]
        p_predict = predict[t, :, :, 2]

        # get 2d true value
        u_label_2d = np.reshape(np.array(u_label), (sample_x, sample_y))
        v_label_2d = np.reshape(np.array(v_label), (sample_x, sample_y))
        p_label_2d = np.reshape(np.array(p_label), (sample_x, sample_y))

        # get 2d predict value
        u_predict_2d = np.reshape(np.array(u_predict), (sample_x, sample_y))
        v_predict_2d = np.reshape(np.array(v_predict), (sample_x, sample_y))
        p_predict_2d = np.reshape(np.array(p_predict), (sample_x, sample_y))

        # calculate error
        u_error_2d, v_error_2d, p_error_2d = cal_abs_error(u_predict_2d, u_label_2d, v_predict_2d,
                                                           v_label_2d, p_predict_2d, p_label_2d)

        label_2d = [u_label_2d, v_label_2d, p_label_2d]
        predict_2d = [u_predict_2d, v_predict_2d, p_predict_2d]
        error_2d = [u_error_2d, v_error_2d, p_error_2d]

        lpe_2d = [label_2d, predict_2d, error_2d]
        lpe_names = ["label", "predict", "error"]

        fig = plt.figure()

        gs = GridSpec(3, 3)

        title = "t={:d}".format(t)
        plt.suptitle(title, fontsize=14)

        gs_idx = int(0)

        for i, data_2d in enumerate(lpe_2d):
            for j, data in enumerate(data_2d):
                ax = fig.add_subplot(gs[gs_idx])
                gs_idx += 1

                if lpe_names[i] == "error":
                    img = ax.imshow(data.T, vmin=0, vmax=1,
                                    cmap=plt.get_cmap("jet"), origin='lower')
                else:
                    img = ax.imshow(data.T, vmin=vmin_list[j], vmax=vmax_list[j],
                                    cmap=plt.get_cmap("jet"), origin='lower')

                ax.set_title(output_names[j] + " " + lpe_names[i], fontsize=4)
                plt.xticks(size=4)
                plt.yticks(size=4)

                aspect = 20
                pad_fraction = 0.5
                divider = make_axes_locatable(ax)
                width = axes_size.AxesY(ax, aspect=1 / aspect)
                pad = axes_size.Fraction(pad_fraction, width)
                cax = divider.append_axes("right", size=width, pad=pad)
                cb = plt.colorbar(img, cax=cax)
                cb.ax.tick_params(labelsize=4)

        gs.tight_layout(fig, pad=0.4, w_pad=0.4, h_pad=0.4)

        buffer_ = io.BytesIO()
        fig.savefig(buffer_, format="jpg")
        buffer_.seek(0)
        image = PIL.Image.open(buffer_)

        video.write(np.asarray(image))

        buffer_.close()

        plt.close()

    video.release()

    numt, _, _, output_size = label.shape
    label = label.reshape((numt, -1, output_size))
    predict = predict.reshape((numt, -1, output_size))
    error = label - predict
    l2_error_u, l2_error_v, l2_error_p = cal_l2_error(
        error, label)
    l2_error_total = np.sqrt(np.sum(np.square(error[:, :, :]), axis=(1, 2))) / \
        np.sqrt(np.sum(np.square(label[:, :, :]), axis=(1, 2)))

    plt.figure()
    plt.plot(input_data[:, 0, 0, 2], l2_error_u, 'b--', label="l2_error of U")
    plt.plot(input_data[:, 0, 0, 2], l2_error_v, 'g-.', label="l2_error of V")
    plt.plot(input_data[:, 0, 0, 2], l2_error_p, 'k:', label="l2_error of P")
    plt.plot(input_data[:, 0, 0, 2], l2_error_total,
             'r-', label="l2_error of All")
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('l2_error')
    plt.xticks(np.arange(0, 7.0, 1.0))
    plt.savefig(os.path.join(path, "TimeError_" + str(epochs) + ".png"))


def calculate_cylinder_flow_error(label, prediction):
    r'''calculate l2-error to evaluate accuracy'''
    error = label - prediction
    l2_error_u = np.sqrt(
        np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error_v = np.sqrt(
        np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
    l2_error_p = np.sqrt(
        np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
    l2_error = np.sqrt(np.sum(np.square(error))) / \
        np.sqrt(np.sum(np.square(label)))
    CylinderFlowError = collections.namedtuple(
        "CylinderFlowError", ["l2_error", "l2_error_u", "l2_error_v", "l2_error_p"])
    errors = CylinderFlowError(l2_error, l2_error_u, l2_error_v, l2_error_p)
    return errors


def get_cylinder_flow_prediction(model, inputs, label_shape, config):
    r'''calculate the prediction respect to the given inputs'''
    output_size = config.get("output_size", 3)
    input_size = config.get("input_size", 3)
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, output_size))
    inputs = inputs.reshape((-1, input_size))

    time_beg = time.time()

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + config["data"]
                        ["test_batch_size"], inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print_log("    predict total time: {} ms".format(
        (time.time() - time_beg)*1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, output_size))
    return prediction


def calculate_cylinder_flow_l2_error(model, inputs, label, config):
    r"""
    Evaluate the model respect to input data and label.

    Args:
         model (mindspore.nn.Cell): list of expressions node can by identified by mindspore.
         inputs (Tensor): the input data of network.
         label (Tensor): the true output value of given inputs.
         config (dict): the configuration of dataset.

    """
    label_shape = label.shape
    prediction = get_cylinder_flow_prediction(
        model, inputs, label_shape, config)
    output_size = config.get("output_size", 3)
    label = label.reshape((-1, output_size))
    l2_errors = calculate_cylinder_flow_error(label, prediction)
    print_log("l2_error, U: ", l2_errors.l2_error_u, ", V: ", l2_errors.l2_error_v, ", P: ", l2_errors.l2_error_p,
              ", Total: ", l2_errors.l2_error)
    print_log(
        "==================================================================================================")


def calculate_periodic_hill_error(label, prediction):
    '''calculate l2-error to evaluate accuracy'''
    PeriodicHillError = collections.namedtuple("PeriodicHillError",
                                               ["l2_error", "l2_error_u", "l2_error_v",
                                                "l2_error_p", "l2_error_uu",
                                                "l2_error_uv", "l2_error_vv"])
    error = label - prediction
    # x, y, u, v, p, uu, uv, vv, rho, nu
    l2_error_u = np.sqrt(
        np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error_v = np.sqrt(
        np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
    l2_error_p = np.sqrt(
        np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
    l2_error_uu = np.sqrt(
        np.sum(np.square(error[..., 3]))) / np.sqrt(np.sum(np.square(label[..., 3])))
    l2_error_uv = np.sqrt(
        np.sum(np.square(error[..., 4]))) / np.sqrt(np.sum(np.square(label[..., 4])))
    l2_error_vv = np.sqrt(
        np.sum(np.square(error[..., 5]))) / np.sqrt(np.sum(np.square(label[..., 5])))

    l2_error = np.sqrt(np.sum(np.square(error))) / \
        np.sqrt(np.sum(np.square(label)))
    errors = PeriodicHillError(l2_error, l2_error_u, l2_error_v, l2_error_p,
                               l2_error_uu, l2_error_uv, l2_error_vv)
    return errors


def get_periodic_hill_prediction(model, inputs, label_shape, config):
    '''calculate the prediction respect to the given inputs'''
    output_size = config['model']['out_channels']
    input_size = config['model']['in_channels']

    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, output_size))
    inputs = inputs.reshape((-1, input_size))

    time_beg = time.time()

    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + config["data"]
                        ['test_batch_size'], inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    print_log("    predict total time: {} ms".format(
        (time.time() - time_beg)*1000))
    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, output_size))
    return prediction


def calculate_periodic_hill_l2_error(model, inputs, label, config):
    """
    Evaluate the model respect to input data and label.

    Args:
         model (mindspore.nn.Cell): list of expressions node can by identified by mindspore.
         inputs (Tensor): the input data of network.
         label (Tensor): the true output value of given inputs.
         config (dict): the configuration of dataset.

    """
    label_shape = label.shape
    prediction = get_periodic_hill_prediction(
        model, inputs, label_shape, config)
    output_size = config["model"]["out_channels"]
    label = label.reshape((-1, output_size))
    l2_errors = calculate_periodic_hill_error(label, prediction)
    print_log("    l2_error, U: ", l2_errors.l2_error_u, ", V: ",
              l2_errors.l2_error_v, ", P: ", l2_errors.l2_error_p)
    print_log("    l2_error, uu: ", l2_errors.l2_error_uu, ", \
              uv: ", l2_errors.l2_error_uv, ", vv: ", l2_errors.l2_error_vv,
              ", Total: ", l2_errors.l2_error)
    print_log(
        "==================================================================================================")


def visual_periodic_hill(model, epochs, input_data, label, path="./images"):
    """visulization of u/v/p"""
    prediction = (model(Tensor(input_data, mstype.float32))).asnumpy()

    x = input_data[:, 0].reshape((300, 700))
    y = input_data[:, 1].reshape((300, 700))

    if not os.path.isdir(os.path.abspath(path)):
        os.makedirs(path)

    _, output_size = label.shape
    label = label.reshape((300, 700, output_size))
    prediction = prediction.reshape((300, 700, output_size))

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.pcolor(x.T, y.T, prediction[:, :, 0].T)
    plt.title("U prediction")
    plt.subplot(2, 2, 2)
    plt.pcolor(x.T, y.T, prediction[:, :, 1].T)
    plt.title("V prediction")
    plt.subplot(2, 2, 3)
    plt.pcolormesh(x.T, y.T, label[:, :, 0].T)
    plt.title("U ground truth")
    plt.subplot(2, 2, 4)
    plt.pcolormesh(x.T, y.T, label[:, :, 1].T)
    plt.title("V ground truth")
    plt.tight_layout()
    plt.savefig(os.path.join(path, str(epochs) + ".png"))
    plt.show()


create_dataset_hooks = {'burgers': create_burgers_train_dataset,
                        'cylinder_flow': create_cylinder_flow_train_dataset,
                        'periodic_hill': create_periodic_hill_train_dataset}
create_test_dataset_hooks = {'burgers': create_burgers_test_dataset,
                             'cylinder_flow': create_cylinder_flow_test_dataset,
                             'periodic_hill': create_periodic_hill_test_dataset}


def create_dataset(case_name, config):
    r"""create dataset for training loss and calculating loss"""
    data_config = config["data"]
    pre_train_dataset = create_dataset_hooks[case_name](config)
    if case_name != "periodic_hill":
        train_dataset = pre_train_dataset.create_dataset(batch_size=data_config["train_batch_size"],
                                                         shuffle=True,
                                                         prebatched_data=True,
                                                         drop_remainder=True)
    else:
        train_dataset = pre_train_dataset
    # create dataset for calculating loss
    pre_loss_dataset = create_dataset_hooks[case_name](config)
    if case_name != "periodic_hill":
        loss_dataset = pre_loss_dataset.create_dataset(batch_size=data_config["test_batch_size"],
                                                       shuffle=True,
                                                       prebatched_data=True,
                                                       drop_remainder=True)
    else:
        loss_dataset = pre_loss_dataset
    # create dataset for test
    inputs, label = create_test_dataset_hooks[case_name](
        data_config["test_data_path"])
    return train_dataset, loss_dataset, inputs, label


def create_model(case_name, config):
    r"""create model"""
    if case_name == "cylinder_flow":
        geometry_config = config["geometry"]
        coord_min = np.array(geometry_config["coord_min"] +
                             [geometry_config["time_min"]]).astype(np.float32)
        coord_max = np.array(geometry_config["coord_max"] +
                             [geometry_config["time_max"]]).astype(np.float32)
        input_center = list(0.5 * (coord_max + coord_min))
        input_scale = list(2.0 / (coord_max - coord_min))
    else:
        input_scale = None
        input_center = None
    model_config = config["model"]
    model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                   out_channels=model_config["out_channels"],
                                   layers=model_config["layers"],
                                   neurons=model_config["neurons"],
                                   residual=model_config["residual"],
                                   act=model_config["activation"],
                                   num_scales=1,
                                   input_scale=input_scale,
                                   input_center=input_center)
    return model


def create_problem(case_name, model):
    r"""define problem"""
    # define problem for burgers
    if case_name == "burgers":
        problem = Burgers1D(model)
    # define problem for NavierStokes2D
    elif case_name == "cylinder_flow":
        problem = NavierStokes2D(model)
    else:
        problem = NavierStokesRANS(model)
    return problem


def visual(case_name, model, epoch, config, inputs, label):
    r"""visualize the result"""
    if case_name == "burgers":
        visual_burgers(model=model, epochs=epoch,
                       resolution=config["summary"]["visual_resolution"])
    elif case_name == "cylinder_flow":
        visual_cylinder_flow(
            model=model, epochs=epoch, input_data=inputs, label=label)
    else:
        visual_periodic_hill(
            model=model, epochs=epoch, input_data=inputs, label=label)


def evaluate(case_name, model, inputs, label, config):
    if case_name == "burgers":
        calculate_burgers_l2_error(model, inputs, label)
    elif case_name == "cylinder_flow":
        calculate_cylinder_flow_l2_error(model, inputs, label, config)
    else:
        calculate_periodic_hill_l2_error(model, inputs, label, config)


def get_train_loss_step(use_ascend, case_name, model, optimizer, loss_scaler, config):
    """get train_step and loss_step"""
    problem = create_problem(case_name, model)
    if case_name == "burgers":
        trainer = BurgersTrainer(
            case_name, model, optimizer, problem, use_ascend, loss_scaler, config)
        return trainer.train_step, trainer.loss_step
    if case_name == "cylinder_flow":
        trainer = CylinderflowTrainer(
            case_name, model, optimizer, problem, use_ascend, loss_scaler, config)
        return trainer.train_step, trainer.loss_step
    trainer = PeriodichillTrainer(
        case_name, model, optimizer, problem, use_ascend, loss_scaler, config)
    return trainer.train_step, trainer.loss_step


def get_losses(loss_sink_process, solutions, model):
    r"""given solutions, return losses"""
    losses = []
    for solution in solutions:
        model.trainable_parameters = solution
        loss = loss_sink_process()
        losses.append(float(loss))
    return losses
