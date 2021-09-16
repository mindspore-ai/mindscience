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
train
"""
import os
import argparse
import datetime
import numpy as np

import mindspore.nn as nn
from mindspore.common import set_seed
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindelec.solver import Solver
from src.dataset import create_dataset
from src.loss import MyMSELoss, EvaLMetric
from src.maxwell_model import Maxwell3D
from src.config import config

set_seed(0)
np.random.seed(0)

print("pid:", os.getpid())
print(datetime.datetime.now())
parser = argparse.ArgumentParser(description='Electromagnetic Simulation')
parser.add_argument('--device_id', type=int, default=2)
parser.add_argument('--checkpoint_path', default='', help='checkpoint path')
parser.add_argument('--data_path', default='', help='data path')

opt = parser.parse_args()
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="Ascend", device_id=opt.device_id)


def evaluation():
    """eval"""
    dataset, config_scale = create_dataset(opt.data_path, batch_size=config.batch_size,
                                           shuffle=False, drop_remainder=False, is_train=False)
    step_size = dataset.get_dataset_size()
    batch_size = dataset.get_batch_size()
    data_length = step_size * batch_size // config.t_solution
    evl_error_mrc = EvaLMetric(data_length, config_scale, batch_size)

    model_net = Maxwell3D(6)
    param_dict = load_checkpoint(opt.checkpoint_path)
    load_param_into_net(model_net, param_dict)
    model_net.set_train(False)
    loss_net = MyMSELoss()
    optimizer = nn.Adam(model_net.trainable_params())
    solver = Solver(model_net, optimizer=optimizer, loss_fn=loss_net, metrics={"evl_mrc": evl_error_mrc})
    res = solver.model.eval(dataset, dataset_sink_mode=False)
    l2_s11 = res['evl_mrc']['l2_error']
    print('test_res:', f'l2_error: {l2_s11:.10f} ')


if __name__ == '__main__':
    evaluation()
