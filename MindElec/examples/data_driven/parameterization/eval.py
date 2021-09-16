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

import os
import argparse
import numpy as np
import mindspore.nn as nn
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from mindelec.solver import Solver

from src.dataset import create_dataset
from src.maxwell_model import S11Predictor
from src.loss import EvalMetric

set_seed(123456)
np.random.seed(123456)

print("pid:", os.getpid())
parser = argparse.ArgumentParser(description='Parametrization S11 Simulation')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--input_dim', type=int, default=3)
parser.add_argument('--device_num', type=int, default=1)
parser.add_argument('--device_target', type=str, default="Ascend")
parser.add_argument('--checkpoint_dir', default='./ckpt/', help='checkpoint directory')
parser.add_argument('--input_path', default='./dataset/Butterfly_antenna/data_input.npy')
parser.add_argument('--label_path', default='./dataset/Butterfly_antenna/data_label.npy')
opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target=opt.device_target,
                    device_id=opt.device_num)


def eval_s11():
    """evaluate s11"""
    data, config_data = create_dataset(opt)

    model_net = S11Predictor(opt.input_dim)
    model_net.to_float(mstype.float16)

    param_dict = load_checkpoint(os.path.join(opt.checkpoint_dir, 'model.ckpt'))
    load_param_into_net(model_net, param_dict)

    eval_error_mrc = EvalMetric(scale_s11=config_data["scale_S11"],
                                length=data["eval_data_length"],
                                frequency=data["frequency"],
                                show_pic_number=4,
                                file_path='./eval_result')

    solver = Solver(network=model_net,
                    mode="Data",
                    optimizer=nn.Adam(model_net.trainable_params(), 0.001),
                    metrics={'eval_mrc': eval_error_mrc},
                    loss_fn=nn.MSELoss())

    res_eval = solver.model.eval(valid_dataset=data["eval_loader"], dataset_sink_mode=True)

    loss_mse, l2_s11 = res_eval["eval_mrc"]["loss_error"], res_eval["eval_mrc"]["l2_error"]
    print(f'Loss_mse: {loss_mse:.10f}  ',
          f'L2_S11: {l2_s11:.10f}')


if __name__ == '__main__':
    eval_s11()
