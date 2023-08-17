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
"""mgbert"""
import numpy as np
from sklearn.metrics import roc_auc_score, r2_score

from mindspore import jit, context
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ..model import Model
from .nn_arch import MGBertModel, CustomWithLossCell
from .utils import SampleLoss


class MGBert(Model):
    '''MGBert'''

    def __init__(self, config):
        self.name = "MGBert"
        context.set_context(memory_optimize_level="O1", max_call_depth=6000)
        if context.get_context("device_target") == "GPU":
            self.mixed_precision = False
        else:
            self.mixed_precision = True

        self.config = config
        self.use_jit = self.config.use_jit
        self.checkpoint_url = \
            'https://download.mindspore.cn/mindscience/mindsponge/MGBert/checkpoint/bert_weightsMedium_100.ckpt'
        self.checkpoint_path = "./bert_weightsMedium_100.ckpt"
        self.network = MGBertModel(self.config)
        opt = nn.Adam(params=self.network.trainable_params(), learning_rate=1e-4)
        self.task_name = config.task_name
        if self.task_name == 'classification':
            loss_fn = nn.BCEWithLogitsLoss()
            self.pretraining = self.config.pretraining
            self.trained_epoch = config.trained_epoch
            loss_net = nn.WithLossCell(self.network, loss_fn)
            self.train_net = nn.TrainOneStepCell(loss_net, opt)
        elif self.task_name == 'regression':
            self.pretraining = self.config.pretraining
            loss_fn = nn.MSELoss()
            self.trained_epoch = config.trained_epoch
            loss_net = nn.WithLossCell(self.network, loss_fn)
            self.train_net = nn.TrainOneStepCell(loss_net, opt)
        else:
            loss_fn = SampleLoss()
            self.trained_epoch = config.trained_epoch
            loss_net = CustomWithLossCell(self.network, loss_fn)
            self.train_net = nn.TrainOneStepCell(loss_net, opt)
        super().__init__(self.checkpoint_url, self.checkpoint_path, self.network, self.name,
                         mixed_precision=self.mixed_precision)

    # pylint: disable=invalid-name
    def forward(self, data):
        """forward"""

    # pylint: disable=arguments-differ
    @jit
    def backward(self, feat):
        """backward"""
        loss = self.train_net(*feat)
        return loss

    # pylint: disable=arguments-differ
    def predict(self, data):
        """predict"""
        param_dict_model = ms.load_checkpoint(self.checkpoint_path)
        ms.load_param_into_net(model, param_dict_model)
        if self.task_name == 'classification':
            sigmoid = ops.Sigmoid()
            y_true = []
            y_preds = []
            y_preds_label = []
            self.network.set_train(False)
            for x, adjoin_matrix, y in data:
                preds = self.network([x, adjoin_matrix])
                y_true.append(y.asnumpy())
                y_preds.append(preds.asnumpy())
            y_true = np.concatenate(y_true, axis=0).reshape(-1)
            y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
            y_preds = sigmoid(ms.Tensor(y_preds, ms.float32)).asnumpy()
            test_auc = roc_auc_score(y_true, y_preds)

            for _, yp_item in enumerate(y_preds):
                if yp_item - 0.5 > 0:
                    y_preds_label.append(1)
                else:
                    y_preds_label.append(0)

            test_acc = np.count_nonzero(np.equal(np.array(y_preds_label), y_true)) / len(y_preds)
            print('test auc:{:.4f}'.format(test_auc), 'test accuracy:{:.4f}'.format(test_acc))
        else:
            y_true = []
            y_preds = []
            self.network.set_train(False)
            val_dataset, value_range = data
            for x, adjoin_matrix, y in val_dataset:
                preds = self.network([x, adjoin_matrix])
                y_true.append(y.asnumpy())
                y_preds.append(preds.asnumpy())
            y_true = np.concatenate(y_true, axis=0).reshape(-1)
            y_preds = np.concatenate(y_preds, axis=0).reshape(-1)
            r2_new = r2_score(y_true, y_preds)

            val_mse = ms.nn.metrics.MSE()
            val_mse.clear()
            val_mse.update(y_true, y_preds)
            val_mse = val_mse.eval() * (value_range ** 2)
            print('val r2: {:.4f}'.format(r2_new), 'val mse:{:.4f}'.format(val_mse))

    def loss(self, data):
        """loss"""

    def grad_operations(self, gradient):
        """grad_operations"""

    def train_step(self, data):
        """train_step"""
        if self.task_name == 'pretrain':
            x = data[0]
            adjoin_matrix = data[1]
            y = data[2]
            char_weight = data[3]
            loss = self.train_net(x, adjoin_matrix, y, char_weight)
        else:
            x = data[0]
            adjoin_matrix = data[1]
            y = data[2]
            if self.pretraining:
                param_dict = ms.load_checkpoint(self.checkpoint_path)
                ms.load_param_into_net(self.network.encoder, param_dict)
                print('load_weights')
            loss = self.train_net([x, adjoin_matrix], y)
        return loss

    @jit
    def _jit_forward(self, x, adjoin_matrix, mask):
        result = self.network(x, adjoin_matrix, mask)
        return result

    def _pynative_forward(self, x, adjoin_matrix, mask):
        result = self.network(x, adjoin_matrix, mask)
        return result
