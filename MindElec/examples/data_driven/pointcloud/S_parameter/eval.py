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
# ==============================================================================
"""
eval
"""
import argparse
import numpy as np

import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore import context
from mindspore import load_checkpoint

from src.model import S11Predictor
from src.metric import EvalMetric
from src.config import config
from src.dataset import create_dataset

from mindelec.solver import Solver

set_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--label_path', type=str)
parser.add_argument('--data_config_path', default='./src/data_config.npz', type=str)
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--model_path', help='checkpoint directory')
parser.add_argument('--output_path', default="./eval_result")

opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=opt.device_num)

def evaluation():
    """evaluation"""

    model_net = S11Predictor(input_dim=config["input_channels"])
    load_checkpoint(opt.model_path, model_net)
    print("model loaded")

    data_config = np.load(opt.data_config_path)
    scale_s11 = data_config["scale_s11"]

    eval_dataset = create_dataset(input_path=opt.input_path,
                                  label_path=opt.label_path,
                                  batch_size=config["batch_size"],
                                  shuffle=False)

    model_net.set_train(False)

    eval_ds_size = eval_dataset.get_dataset_size() * config["batch_size"]

    eval_error_mrc = EvalMetric(scale_s11=scale_s11,
                                length=eval_ds_size,
                                frequency=np.linspace(0, 4*10**8, 1001),
                                show_pic_number=4,
                                file_path=opt.output_path)

    solver = Solver(network=model_net,
                    mode="Data",
                    optimizer=nn.Adam(model_net.trainable_params(), 0.001),
                    metrics={'eval_mrc': eval_error_mrc},
                    loss_fn=nn.MSELoss())

    res_eval = solver.model.eval(valid_dataset=eval_dataset, dataset_sink_mode=True)

    loss_mse, l2_s11 = res_eval["eval_mrc"]["loss_error"], res_eval["eval_mrc"]["l2_error"]
    print(f'Loss_mse: {loss_mse:.10f}  ',
          f'L2_S11: {l2_s11:.10f}')


if __name__ == '__main__':
    evaluation()
