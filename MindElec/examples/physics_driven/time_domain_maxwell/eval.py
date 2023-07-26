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
"""evaluation process"""
import copy
import json
import os
import time

import numpy as np
import mindspore as ms
from mindspore import context, Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindelec.architecture import MultiScaleFCCell
from src import get_test_data, visual_result


def load_config():
    with open("./config.json") as f:
        config = json.load(f)
    return config


def evaluation(config):
    """evaluation"""
    # define network
    model = MultiScaleFCCell(config["input_size"],
                             config["output_size"],
                             layers=config["layers"],
                             neurons=config["neurons"],
                             input_scale=config["input_scale"],
                             residual=config["residual"],
                             act="sin",
                             num_scales=config["num_scales"],
                             amp_factor=config["amp_factor"],
                             scale_factor=config["scale_factor"]
                             )

    model.to_float(ms.float16)
    model.input_scale.to_float(ms.float32)

    # load parameters
    param_dict = load_checkpoint(config["load_ckpt_path"])
    convert_ckpt_dict = {}
    for _, param in model.parameters_and_names():
        convert_name1 = "model.cell_list." + param.name
        convert_name2 = "model.cell_list." + ".".join(param.name.split(".")[2:])
        for key in [convert_name1, convert_name2]:
            if key in param_dict:
                convert_ckpt_dict[param.name] = param_dict[key]
    load_param_into_net(model, convert_ckpt_dict)

    # load test dataset
    inputs, label = get_test_data(config["test_data_path"])
    outputs_size = config.get("outputs_size", 3)
    inputs_size = config.get("inputs_size", 3)
    outputs_scale = np.array(config["output_scale"], dtype=np.float32)
    batch_size = config.get("test_batch_size", 8192 * 4)

    dx = inputs[0, 1, 0, 0] - inputs[0, 0, 0, 0]
    dy = inputs[0, 0, 1, 1] - inputs[0, 0, 0, 1]
    dt = inputs[1, 0, 0, 2] - inputs[0, 0, 0, 2]

    ex_inputs = copy.deepcopy(inputs)
    ey_inputs = copy.deepcopy(inputs)
    hz_inputs = copy.deepcopy(inputs)
    ex_inputs = ex_inputs.reshape(-1, inputs_size)
    ey_inputs = ey_inputs.reshape(-1, inputs_size)
    hz_inputs = hz_inputs.reshape(-1, inputs_size)
    ex_inputs[:, 1] += dy / 2.0
    ex_inputs[:, 2] += dt / 2.0
    ey_inputs[:, 0] += dx / 2.0
    ey_inputs[:, 2] += dt / 2.0
    inputs_each = [ex_inputs, ey_inputs, hz_inputs]

    index = 0
    prediction_each = np.zeros(label.shape)
    prediction_each = prediction_each.reshape((-1, outputs_size))
    time_beg = time.time()
    while index < len(inputs_each[0]):
        index_end = min(index + batch_size, len(inputs_each[0]))
        # predict each physical quantity respectively in order to keep consistent with fdtd on staggered mesh
        for i in range(outputs_size):
            test_batch = Tensor(inputs_each[i][index: index_end, :], ms.float32)
            predict = model(test_batch)
            predict = predict.asnumpy()
            prediction_each[index: index_end, i] = predict[:, i] * outputs_scale[i]
        index = index_end
    print("==================================================================================================")
    print("predict total time: {} s".format(time.time() - time_beg))
    prediction = prediction_each.reshape(label.shape)
    vision_path = config.get("vision_path", "./vision")
    visual_result(inputs, label, prediction, path=vision_path, name="predict")

    # get accuracy
    error = label - prediction
    l2_error_ex = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
    l2_error_ey = np.sqrt(np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
    l2_error_hz = np.sqrt(np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
    print("l2_error, Ex: ", l2_error_ex, ", Ey: ", l2_error_ey, ", Hz: ", l2_error_hz)


if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", save_graphs_path="./graph")
    print("pid:", os.getpid())
    configs = load_config()
    print("check config: {}".format(configs))
    time0 = time.time()
    evaluation(configs)
    print("End-to-End total time: {} s".format(time.time() - time0))
