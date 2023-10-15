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

"""phylstm_2 train"""
import os
from random import shuffle

from mindspore import nn, context, save_checkpoint
import numpy as np
from tqdm.autonotebook import trange

from sciai.context import init_project
from sciai.utils import print_log, calc_ckpt_name
from src.network import prepare_network
from src.process import prepare_dataset, prepare
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class MyWithLossCell(nn.Cell):
    """custom train_network with loss"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, ag_train, phi, ag_c_train, phi_c, u_train, ut_train, lift_train):
        a, b = self._backbone(ag_train, phi, ag_c_train, phi_c)
        return self._loss_fn(a[0], u_train, a[2], ut_train, b[0], b[1], b[2], ag_c_train, lift_train)

class CustomWithEvalCell(nn.Cell):
    """Custom eval_network with loss"""
    def __init__(self, network, loss_fn):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn

    def construct(self, ag_train, phi, ag_c_train, phi_c, u_train, ut_train, lift_train):
        a, b = self.network(ag_train, phi, ag_c_train, phi_c)
        return self._loss_fn(a[0], u_train, a[2], ut_train, b[0], b[1], b[2], ag_c_train, lift_train)

def train(args, train_dataset, addition_dataset, network, loss):
    """Training and validating logic"""
    epochs = args.epochs
    train_loss_list = []
    eval_loss_list = []
    best_loss = 100
    best_epoch = 0

    lr = args.lr
    opt = nn.Adam(params=network.trainable_params(), learning_rate=lr)

    # train_net
    net_with_loss = MyWithLossCell(network, loss)
    train_net = nn.TrainOneStepCell(net_with_loss, opt)
    # eval_net
    eval_net = CustomWithEvalCell(network, loss)

    for e in trange(0, epochs, desc="Train", leave=False):
        for _, (ag_train, u_train, ut_train, _, g_train, phi) in enumerate(train_dataset.create_tuple_iterator()):
            # Shuffle data, ensuring that the training and validation set data are 8:2
            ind = list(range(ag_train.shape[0]))
            shuffle(ind)
            ratio_split = 0.8
            ind_tr = ind[0:round(ratio_split * ag_train.shape[0])]
            ind_val = ind[round(ratio_split * ag_train.shape[0]):]
            # Training set data
            ag_tr, u_tr, ut_tr = ag_train[ind_tr], u_train[ind_tr], ut_train[ind_tr]
            _, phi_tr = g_train[ind_tr], phi[ind_tr]
            # Validation set data
            ag_val, u_val, ut_val = ag_train[ind_val], u_train[ind_val], ut_train[ind_val]
            _, phi_val = g_train[ind_val], phi[ind_val]
        for _, (ag_c_train, lift_train, phi_c) in enumerate(addition_dataset.create_tuple_iterator()):
            # Training
            train_net.set_train()
            train_loss = train_net(ag_tr, phi_tr, ag_c_train, phi_c, u_tr, ut_tr, lift_train)
            train_loss_list.append(train_loss.asnumpy())

            if train_loss.asnumpy() < best_loss:
                best_loss = train_loss.asnumpy()
                best_epoch = e

            # Validating
            train_net.set_train(False)
            eval_loss = eval_net(ag_val, phi_val, ag_c_train, phi_c, u_val, ut_val, lift_train)
            eval_loss_list.append(eval_loss.asnumpy())
        if e % args.print_interval == 0:
            print_log("[Adam]Epoch:{},Train_Loss:{},Eval_Loss:{},bestLoss:{},bestEpoch:{}"
                      .format(e, train_loss, eval_loss, best_loss, best_epoch))
        if args.save_ckpt and e % args.ckpt_interval == 0:
            save_checkpoint(network, os.path.join(args.save_ckpt_path, calc_ckpt_name(args)))
    np.save(args.save_data_path + '/train_loss.npy', np.array(train_loss_list))
    np.save(args.save_data_path + '/eval_loss.npy', np.array(eval_loss_list))

def main(args):
    train_dataset, addition_dataset = prepare_dataset(args)
    network, loss = prepare_network(args)
    train(args, train_dataset, addition_dataset, network, loss)


if __name__ == "__main__":
    args_ = prepare()
    init_project(args=args_[0])
    main(*args_)
