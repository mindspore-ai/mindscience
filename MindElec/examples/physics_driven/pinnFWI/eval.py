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
"""eval process"""

import os
import yaml
import mindspore as ms
from src.model import Net, Net0
from src.data_gen import xx, zz, X_S, u_ini1x, u_ini1z, u_ini2x, u_ini2z, u_specx, u_specz
from src.data_gen import dataset01, dataset02, dataset2, dataset_seism
from src.customloss import CustomWithEval2Cell, alpha_true_func
from utils.plot import plot_alpha, plot_wave_pnential


os.environ['CUDA_VISIBLE_DEVICES'] = '2'


with open('src/dafault_config.yaml', 'r') as y:
    cfg = yaml.full_load(y)


layers = [3]+[100]*8+[1]
neural_net = Net(layers=layers)

layers0 = [2]+[20]*5+[1]
neural_net0 = Net0(layers=layers0)

param_dict = ms.load_checkpoint("MyNet.ckpt")
ms.load_param_into_net(neural_net, param_dict)

param_dict = ms.load_checkpoint("MyNet0.ckpt")
ms.load_param_into_net(neural_net0, param_dict)

alpha_true0 = alpha_true_func(dataset01)
alpha_true0 = alpha_true0.reshape((xx.shape))


eval_net = CustomWithEval2Cell(
    neural_net=neural_net, neural_net0=neural_net0)
eval_net.set_train(False)
_, _, alpha_plot = eval_net(dataset01)


alpha_plot = alpha_plot.reshape(xx.shape)

ux01, uz01, alpha0 = eval_net(dataset01)
ux02, uz02, _ = eval_net(dataset02)
uxt, uzt, _ = eval_net(dataset2)
uz_seism_pred, ux_seism_pred, _ = eval_net(dataset_seism)


plot_wave_pnential(xx, zz, ux01, uz01, u_ini1x, u_ini1z, ux02,
                   uz02, u_ini2x, u_ini2z, uxt, uzt, u_specx, u_specz)
plot_alpha(xx, zz, X_S, alpha_true0, alpha0)
