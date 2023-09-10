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
"""Run PFNN"""
import time
import numpy as np

from mindspore import Tensor
from mindspore import nn
from mindspore import load_param_into_net, load_checkpoint

from sciai.context import init_project
from sciai.utils import print_time, print_log, amp2datatype
from src import pfnnmodel
from src.process import prepare, calerror, train_netg, train_netloss
from data import gendata, dataset


def trainer(*inputs):
    """
    The traing process that's includes traning network G and network F/Loss
    """
    (args, net_g, net_f, net_loss, lenfac, optim_g, optim_f,
     inset, bdset, dataset_g, dataset_loss, dtype) = inputs

    print_log("START TRAINING")

    start_gnet_time = time.time()
    train_netg(args, net_g, optim_g, dataset_g)
    elapsed_gnet = time.time() - start_gnet_time

    start_fnet_time = time.time()
    train_netloss(args, net_g, net_f, net_loss, lenfac, optim_f, inset, bdset, dataset_loss, dtype)
    elapsed_fnet = time.time() - start_fnet_time

    return elapsed_gnet, elapsed_fnet


@print_time("train")
def main(args):
    dtype = amp2datatype(args.amp_level)

    errors = np.zeros(args.tests_num)

    for ii in range(args.tests_num):
        inset, bdset, teset = gendata.generate_set(args)
        dsg, dsloss = dataset.GenerateDataSet(inset, bdset)

        lenfac = pfnnmodel.LenFac(Tensor(args.bound, dtype).reshape(2, 2), 1)
        netg = pfnnmodel.NetG()
        netf = pfnnmodel.NetF()
        if ii == 0 and args.load_ckpt:
            load_param_into_net(netg, load_checkpoint(args.load_ckpt_path[0]))
            load_param_into_net(netf, load_checkpoint(args.load_ckpt_path[1]))

        netloss = pfnnmodel.Loss(netf)
        netg.to_float(dtype)
        netloss.to_float(dtype)

        optimg = nn.Adam(netg.trainable_params(), learning_rate=args.g_lr)
        optimf = nn.Adam(netf.trainable_params(), learning_rate=args.f_lr)

        net_g_ime, net_f_ime = trainer(
            args, netg, netf, netloss, lenfac, optimg, optimf, inset, bdset, dsg, dsloss, dtype)
        print_log("Train NetG total time: %.2f, train NetG one step time: %.5f" %
                  (net_g_ime, net_g_ime / args.g_epochs))
        print_log("Train NetF total time: %.2f, train NetF one step time: %.5f" %
                  (net_f_ime, net_f_ime / args.f_epochs))

        load_param_into_net(netg, load_checkpoint(args.load_ckpt_path[0]))
        load_param_into_net(netf, load_checkpoint(args.load_ckpt_path[1]))

        errors[ii] = calerror(netg, netf, lenfac, teset)
        print_log("test_error = %.3e\n" % (errors[ii].item()))

    print_log(errors)
    errors_mean = errors.mean()
    errors_std = errors.std()

    print_log("test_error_mean = %.3e, test_error_std = %.3e" % (errors_mean.item(), errors_std.item()))


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
