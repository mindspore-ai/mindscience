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
"""brain"""
import mindspore as ms
from mindspore import nn
from mindspore.nn import optim
import numpy as np
from sciai.common import TrainCellWithCallBack
from sciai.utils import print_log

from .nn.module import LossNN
from .utils import cross_entropy_loss


class Brain:
    """Brain"""
    def __init__(self, args, data, net, criterion):
        self.case = args.case
        self.data = data
        self.net = net
        self.criterion = criterion
        self.lr = args.lr
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.save_ckpt = args.save_ckpt
        self.save_data = args.save_data
        self.save_ckpt_path = args.save_ckpt_path
        self.save_data_path = args.save_data_path
        self.print_interval = args.print_interval
        self.amp_level = args.amp_level

        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None

        self._optimizer = None
        self._criterion = None
        self.__init_brain()

    def train(self):
        """train"""
        print_log('Training...')
        loss_history = []
        for i in range(self.epochs + 1):
            if self.batch_size:
                mask = np.random.choice(self.data.x_train.size(0), self.batch_size, replace=False)
                loss = self.train_net(self.data.x_train[mask], self.data.y_train[mask])
            else:
                loss = self.train_net(self.data.x_train, self.data.y_train)
            if i % self.print_interval == 0 or i == self.epochs:
                loss_test = self.with_loss_cell(self.data.x_test, self.data.y_test)
                loss_history.append([i, loss, loss_test])
                if self.save_ckpt:
                    ms.save_checkpoint(self.net, f'{self.save_ckpt_path}'
                                                 f'/model_{self.case}_{self.net.__class__.__name__}_iter{i}.ckpt')
        self.loss_history = np.array(loss_history)
        print_log('training finished.')

        if self.loss_history is not None:
            if self.save_ckpt:
                best_loss_index = np.argmin(self.loss_history[:, 1])
                epoch = int(self.loss_history[best_loss_index, 0])
                loss_train = self.loss_history[best_loss_index, 1]
                loss_test = self.loss_history[best_loss_index, 2]
                print_log('Best model at epoch {}:'.format(epoch))
                print_log('Train loss:', loss_train, 'Test loss:', loss_test)
                best_model_params = ms.load_checkpoint(
                    f'{self.save_ckpt_path}/model_{self.case}_{self.net.__class__.__name__}_iter{epoch}.ckpt')
                self.best_model = type(self.net)(self.data.dim, self.net.layers, self.net.third_parameter,
                                                 self.net.activation)
                ms.load_param_into_net(self.best_model, best_model_params)
            else:
                self.best_model = self.net
                print_log("warning: loading last model as the best model!")
        else:
            raise RuntimeError('restore before running or without saved models')
        return self.best_model

    def evaluate(self):
        val_loss = self.with_loss_cell(self.data.x_test, self.data.y_test)
        print_log('validation loss:', val_loss)

    def save_txt(self):
        """save txt"""
        np.savetxt(f'{self.save_data_path}/{self.case}_X_train.txt', self.data.x_train_np)
        np.savetxt(f'{self.save_data_path}/{self.case}_y_train.txt', self.data.y_train_np)
        np.savetxt(f'{self.save_data_path}/{self.case}_X_test.txt', self.data.x_test_np)
        np.savetxt(f'{self.save_data_path}/{self.case}_y_test.txt', self.data.y_test_np)
        np.savetxt(f'{self.save_data_path}/{self.case}_loss.txt', self.loss_history)

    def __init_brain(self):
        """init brain"""
        self.loss_history = None
        self.encounter_nan = False
        self.best_model = None
        self.data.dtype = self.amp_level
        self.__init_criterion()
        self._optimizer = optim.Adam(self.net.get_parameters(), learning_rate=self.lr)
        self.with_loss_cell = nn.WithLossCell(self.net, self._criterion)
        self.train_net = TrainCellWithCallBack(self.with_loss_cell, self._optimizer, loss_interval=self.print_interval,
                                               time_interval=self.print_interval,
                                               amp_level=self.amp_level)

    def __init_criterion(self):
        """init criterion"""
        if isinstance(self.net, LossNN):
            self._criterion = self.net.criterion
            if self.criterion is not None:
                raise Warning('loss-oriented neural network has already implemented its loss function')
        elif self.criterion == 'MSE':
            self._criterion = nn.MSELoss()
        elif self.criterion == 'CrossEntropy':
            self._criterion = cross_entropy_loss
        else:
            raise NotImplementedError()
