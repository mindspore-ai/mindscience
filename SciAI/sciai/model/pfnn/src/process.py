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
"""process for pfnn"""
import os
import yaml
import numpy as np

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.train.model import Model

from sciai.utils import parse_arg, print_log
from .callback import SaveCallbackNETG, SaveCallbackNETLoss


def prepare():
    abs_dir = os.path.abspath(os.path.dirname(__file__))
    with open(f"{abs_dir}/../config.yaml") as f:
        config_dict = yaml.safe_load(f)
    args_ = parse_arg(config_dict)
    return (args_,)


def calerror(netg_calerror, netf_calerror, lenfac_calerror, teset_calerror):
    """The eval function"""
    x = Tensor(teset_calerror.x, ms.float32)
    test_set_u = (netg_calerror(x) + lenfac_calerror(Tensor(x[:, 0])).reshape(-1, 1) * netf_calerror(x)).asnumpy()
    test_error = (((test_set_u - teset_calerror.ua) ** 2).sum() / teset_calerror.ua.sum()) ** 0.5
    return test_error


def train_netg(args_netg, net_netg, optim_netg, dataset_netg):
    """The process of preprocess and process to train NetG"""
    print_log("START TRAIN NEURAL NETWORK G")
    model = Model(network=net_netg, loss_fn=None, optimizer=optim_netg)
    model.train(args_netg.g_epochs, dataset_netg, callbacks=[
        SaveCallbackNETG(net_netg, args_netg.load_ckpt_path[0])])


def train_netloss(*inputs):
    """The process of preprocess and process to train NetF/NetLoss"""
    (args_netloss, netg_netloss, netf_netloss, netloss_netloss,
     lenfac_netloss, optim_netloss, inset_netloss, bdset_netloss, dataset_netloss, dtype) = inputs

    grad_ = ops.composite.GradOperation(get_all=True)

    inset_l = lenfac_netloss(Tensor(inset_netloss.x[:, 0], dtype))
    inset_l = inset_l.reshape((len(inset_l), 1))
    inset_lx = grad_(lenfac_netloss)(Tensor(inset_netloss.x[:, 0], dtype))[0].asnumpy()[:, np.newaxis]
    inset_lx = np.hstack((inset_lx, np.zeros(inset_lx.shape)))
    bdset_nl = lenfac_netloss(Tensor(bdset_netloss.n_x[:, 0], dtype)).asnumpy()[:, np.newaxis]

    ms.load_param_into_net(netg_netloss, ms.load_checkpoint(
        args_netloss.load_ckpt_path[0]), strict_load=True)
    inset_g = netg_netloss(Tensor(inset_netloss.x, dtype))
    inset_gx = grad_(netg_netloss)(Tensor(inset_netloss.x, dtype))[0]
    bdset_ng = netg_netloss(Tensor(bdset_netloss.n_x, dtype))

    netloss_netloss.get_variable(inset_g, inset_l, inset_gx, inset_lx, inset_netloss.a,
                                 inset_netloss.size, inset_netloss.dim, inset_netloss.area, inset_netloss.c,
                                 bdset_netloss.n_length, bdset_netloss.n_r, bdset_nl, bdset_ng)
    print_log("START TRAIN NEURAL NETWORK F")
    model = Model(network=netloss_netloss, loss_fn=None, optimizer=optim_netloss)
    model.train(args_netloss.f_epochs, dataset_netloss, callbacks=[
        SaveCallbackNETLoss(netf_netloss, args_netloss.load_ckpt_path[1], inset_netloss.x, inset_l, inset_g,
                            inset_netloss.ua)],
                dataset_sink_mode=True)
    if not os.path.exists(args_netloss.load_ckpt_path[1]):
        ms.save_checkpoint(netf_netloss, args_netloss.load_ckpt_path[1])
