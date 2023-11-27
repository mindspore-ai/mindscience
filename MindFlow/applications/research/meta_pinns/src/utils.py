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
"""utils"""
import time
import os

import matplotlib.pyplot as plt
import numpy as np

from mindspore import set_seed, get_seed, nn, Parameter, data_sink, Tensor, save_checkpoint
from mindspore import dtype as mstype
from mindspore.common.initializer import initializer

from mindflow.cell import MultiScaleFCSequential
from mindflow.utils import print_log

from .dataset import sample_dataset, create_test_dataset, create_lburgers_train_dataset, create_periodic_hill_train_dataset
from .dataset import create_cd_dataset, create_cylinder_flow_train_dataset
from .model import Burgers, LBurgers, ConvectionDiffusion, NavierStokes2D, NavierStokesRANS
from .trainer import Trainer
from .divide import divide_with_error


class WorkspaceConfig:
    """work space config"""

    def __init__(self, cur_epoch, case_name, config, params,
                 iterations, cal_l2_interval,
                 use_ascend, loss_scaler,
                 if_constr_sigmoid, if_weighted, if_saveckpt, ckpt_name):
        self.cur_epoch = cur_epoch
        self.case_name = case_name
        self.config = config
        self.params = params
        self.iterations = iterations
        self.cal_l2_interval = cal_l2_interval
        self.use_ascend = use_ascend
        self.loss_scaler = loss_scaler
        self.if_constr_sigmoid = if_constr_sigmoid
        self.if_weighted = if_weighted
        self.if_saveckpt = if_saveckpt
        self.ckpt_name = ckpt_name


def create_train_dataset(case_name, config, seed):
    """create train dataset"""
    train_batch_size = config["data"]["train_batch_size"]
    if case_name == "l_burgers":
        train_dataset = create_lburgers_train_dataset(config)
    elif case_name == "convection_diffusion":
        train_dataset = create_cd_dataset(config)
    elif case_name == "cylinder_flow":
        train_dataset = create_cylinder_flow_train_dataset(config).create_dataset(batch_size=train_batch_size,
                                                                                  shuffle=True,
                                                                                  prebatched_data=True,
                                                                                  drop_remainder=True)
    elif case_name == "periodic_hill":
        train_dataset = create_periodic_hill_train_dataset(config)
    else:
        cur_seed = get_seed()
        set_seed(seed)
        sample = sample_dataset(config)
        train_dataset = sample.create_dataset(batch_size=train_batch_size,
                                              shuffle=True,
                                              prebatched_data=True,
                                              drop_remainder=True)
        set_seed(cur_seed)
    return train_dataset


def create_problem(lamda, case_name, model, config):
    """create problem"""
    if case_name == "burgers":
        problem = Burgers(lamda, model, config)
    elif case_name == "l_burgers":
        problem = LBurgers(lamda, model, config)
    elif case_name == "convection_diffusion":
        problem = ConvectionDiffusion(lamda, model, config)
    elif case_name == "cylinder_flow":
        problem = NavierStokes2D(lamda, model, config)
    else:
        problem = NavierStokesRANS(lamda, model, config)
    return problem


def create_trainer(case_name, model, optimizer, problem):
    """create trainer"""
    trainer = Trainer(case_name, model, optimizer, problem)
    return trainer


def re_initialize_model(model, epoch):
    """method for reinitializing"""
    origin_seed = get_seed()

    set_seed(epoch + 1)

    for param in model.trainable_params():
        if isinstance(param, Parameter):
            param_shape = param.data.shape
            param.set_data(Parameter(initializer('normal', param_shape)))

    set_seed(origin_seed)


def get_prediction(model, inputs, label_shape):
    r'''calculate the prediction respect to the given inputs'''
    prediction = np.zeros(label_shape)
    prediction = prediction.reshape((-1, label_shape[-1]))
    inputs = inputs.reshape((-1, inputs.shape[-1]))

    batch_size = inputs.shape[0]
    index = 0
    while index < inputs.shape[0]:
        index_end = min(index + batch_size, inputs.shape[0])
        test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
        prediction[index: index_end, :] = model(test_batch).asnumpy()
        index = index_end

    prediction = prediction.reshape(label_shape)
    prediction = prediction.reshape((-1, label_shape[-1]))
    return prediction


def cal_l2_error(case_name, eva_model, inputs, label):
    r"""
    Evaluate the model respect to input data and label.

    Args:
         model (Cell): list of expressions node can by identified by mindspore.
         inputs (Tensor): the input data of network.
         label (Tensor): the true output value of given inputs.

    """
    label_shape = label.shape
    prediction = get_prediction(eva_model, inputs, label_shape)
    label = label.reshape((-1, label_shape[-1]))
    error = label - prediction
    if case_name in ('cylinder_flow', 'periodic_hill'):
        l2_error = divide_with_error(np.sqrt(np.sum(np.square(error))),
                                     np.sqrt(np.sum(np.square(label))))
    else:
        l2_error = divide_with_error(np.sqrt(
            np.sum(np.square(error[..., 0]))), np.sqrt(np.sum(np.square(label[..., 0]))))
    return l2_error


def create_normal_params(case_name):
    """create plain params"""
    w1 = Parameter(Tensor([1], mstype.float32), name="w1")
    w2 = Parameter(Tensor([1], mstype.float32), name="w2")
    normal_params = [w1, w2]
    if case_name in ("burgers", "cylinder_flow"):
        w3 = Parameter(Tensor([1], mstype.float32), name="w3")
        normal_params.append(w3)
    return normal_params


def evaluate(work_space_config):
    """evaluate"""
    cur_epoch = work_space_config.cur_epoch
    case_name = work_space_config.case_name
    config = work_space_config.config
    params = work_space_config.params
    iterations = work_space_config.iterations
    cal_l2_interval = work_space_config.cal_l2_interval
    use_ascend = work_space_config.use_ascend
    loss_scaler = work_space_config.loss_scaler
    if_constr_sigmoid = work_space_config.if_constr_sigmoid
    if_weighted = work_space_config.if_weighted
    if_saveckpt = work_space_config.if_saveckpt
    ckpt_name = work_space_config.ckpt_name

    l2_error_list = []
    model_config = config["model"]
    summary_config = config["summary"]
    data_config = config["data"]
    initial_lr = config["optimizer"]["initial_lr"]
    eva_model = MultiScaleFCSequential(in_channels=model_config["in_channels"],
                                       out_channels=model_config["out_channels"],
                                       layers=model_config["layers"],
                                       neurons=model_config["neurons"],
                                       residual=model_config["residual"],
                                       act=model_config["activation"],
                                       num_scales=1)

    eva_problem = create_problem(
        config["lamda"]["eva_lamda"], case_name, eva_model, config)
    eva_problem.set_hp_params(params)
    if config["meta_test"]["if_adam"]:
        eva_optimizer = nn.Adam(eva_model.trainable_params(),
                                initial_lr)
    else:
        eva_optimizer = nn.SGD(eva_model.trainable_params(),
                               initial_lr)

    if use_ascend:
        from mindspore.amp import DynamicLossScaler, auto_mixed_precision
        loss_scaler = DynamicLossScaler(1024, 2, 100)
        auto_mixed_precision(eva_model, model_config["amp_level"])
    else:
        loss_scaler = None

    eva_train_dataset = create_train_dataset(case_name, config, get_seed() + 3)
    inputs, label = create_test_dataset(
        case_name, data_config["test_data_path"], config)
    steps_per_epochs = eva_train_dataset.get_dataset_size()

    eva_trainer = create_trainer(
        case_name, eva_model, eva_optimizer, eva_problem)

    eva_trainer.set_params(use_ascend, loss_scaler,
                           if_constr_sigmoid, if_weighted)

    eva_sink_process = data_sink(
        eva_trainer.train_step, eva_train_dataset, sink_size=1)

    if if_saveckpt:
        os.makedirs(summary_config["save_ckpt_path"], exist_ok=True)

    for eva_iter in range(iterations):
        time_beg = time.time()
        eva_model.set_train(True)
        for _ in range(steps_per_epochs):
            eva_loss = eva_sink_process()
        eva_model.set_train(False)
        if eva_iter % cal_l2_interval == 0:
            l2_error = cal_l2_error(case_name, eva_model, inputs, label)
            l2_error_list.append(l2_error)
            # validate between training
            if iterations == cal_l2_interval:
                print_log(
                    f"epoch: {cur_epoch} l2_error: {l2_error} epoch time: {(time.time() - time_beg) * 1000}ms")
            # pure training
            else:
                if if_saveckpt and eva_iter % summary_config["save_checkpoint_epochs"] == 0:
                    save_checkpoint(eva_model, os.path.join(
                        summary_config["save_ckpt_path"], f"{ckpt_name}_{eva_iter}"))
                print_log(f"epoch: {eva_iter} loss: {eva_loss}")
                print_log(
                    f"epoch: {eva_iter} l2_error: {l2_error} epoch time: {(time.time() - time_beg) * 1000}ms")
            print_log(
                "==================================================================================================")
    return l2_error_list


def plot_l2_error(case_name, visual_path, interval, l2_errors):
    """function for drawing l2 error"""
    # Generate the x-axis array
    x = [i * interval for i in range(len(l2_errors))]
    # Draw the line chart
    plt.clf()
    plt.plot(x, l2_errors, label='L2 Error')

    # Add legend, title, and axis labels
    plt.legend()
    plt.title('L2 Error Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')

    # Create the 'images' folder if it doesn't exist
    os.makedirs(visual_path, exist_ok=True)

    # Save the image to the 'images' folder in the current directory
    plt.savefig(f'{visual_path}/{case_name}_meta_training_l2_error.png')

    # Show the image
    plt.show()


def plot_l2_comparison_error(case_name, visual_path, interval, meta_l2_errors, normal_l2_errors):
    """plot comparison error"""
    x = [i * interval for i in range(len(meta_l2_errors))]

    plt.clf()
    plt.plot(x, meta_l2_errors, label='Meta L2 Error')
    plt.plot(x, normal_l2_errors, label='Normal L2 Error')

    plt.legend()
    plt.title('Comparison of L2 Errors')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Error')

    os.makedirs(visual_path, exist_ok=True)

    plt.savefig(f'{visual_path}/{case_name}_l2_error_comparison.png')

    plt.show()
