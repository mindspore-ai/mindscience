# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Gaussian process modules"""
import os
import stat
import pickle
import logging
import numpy as np
from mindspore import Parameter, Tensor, value_and_grad
import mindspore as ms
from src.dataset import Normalize, RBFKernel, MaternKernel, LaplacianKernel, shuffle
from mindchemistry.cell import ElasticNet


class GaussianProcessTensors:
    """Training and prediction module for Gaussian ProcessTensors"""

    def __init__(self, optimizer_type='adam', verbose=0, run_id=1, mae=False, workspace_dir=None, kernel=None,
                 kernel_params=None):
        """
        init
        """
        self.optimizer = optimizer_type
        self.verbose = verbose
        self.id = run_id
        self.mae = mae
        self.errors = None
        self.workspace_dir = workspace_dir
        self.kernel_type = kernel
        kernel_params = {} if kernel_params is None else kernel_params
        if self.kernel_type is None:
            self.kernel = RBFKernel(**kernel_params)
        elif self.kernel_type == 'rbf':
            self.kernel = RBFKernel(**kernel_params)
        elif self.kernel_type == 'mat':
            self.kernel = MaternKernel(**kernel_params)
        elif self.kernel_type == 'lap':
            self.kernel = LaplacianKernel(**kernel_params)
        else:
            raise RuntimeError('Not Implemented Kernel')

    def _error_score(self, prediction, targets, iw, test_index):
        """calculation error"""
        if self.mae:
            err = ms.ops.mean((prediction - targets.astype(ms.float32)).abs() * iw[ms.Tensor(test_index), None].pow(2),
                              0)
        else:
            err = ms.ops.mean(((prediction - targets).pow(2)) * iw[ms.Tensor(test_index), None].pow(2), 0)
        return err

    def fit(self, x: ms.Tensor, y: ms.Tensor, importance_weights=None, max_iter=None, save_type=None, mean_std=None):
        """fit"""
        if y.ndim < 2:
            y = y.reshape(-1, 1)
        train_indices, test_indices = shuffle(x)

        if importance_weights is None:
            importance_weights = ms.ops.ones((x.shape[0],))
        importance_weights = importance_weights.pow(0.5)
        iw = importance_weights.pow(0)
        y = Normalize(data_type=save_type, mean_std_params=mean_std).apply_to_data(y)
        kernel_train = x[train_indices]
        train_indices = [i for i in range(y.shape[0])] if train_indices is None else train_indices
        train_samples = [i for i in range(y.shape[1])]
        self.model = model = ElasticNet(n_inputs=kernel_train.shape[1], n_outputs=y.shape[1], save_type=save_type)
        self.loss_fn = ms.nn.MSELoss()
        optimizer = ms.nn.AdamWeightDecay(self.model.trainable_params(), learning_rate=0.05, eps=1e-08,
                                          weight_decay=0.01)  # AdamW优化器
        net_backward = value_and_grad(self.net_forward, None, optimizer.parameters, has_aux=False)
        loss, loss_p = 0, 0
        x_t = Parameter(kernel_train).astype(ms.float32)
        x_t.requires_grad = False
        y_t = Tensor(ms.ops.matmul(ms.ops.diag(iw[Tensor(train_indices)]),
                                   y[Tensor(train_indices), :][:, Tensor(train_samples)].astype(ms.float32)))
        model_record = model.parameters_dict()
        model.set_train()
        for ii in range(max_iter):
            loss, grad = net_backward(x_t, y_t)
            optimizer(grad)
            if self.verbose == 0 and ii % 10 == 0:
                logging.info("第 %d 步: \t 训练集误差: %s", ii + 1, np.round(loss.asnumpy(), 4))
            if abs(loss - loss_p) < 0.0000001:
                break
            elif loss < loss_p or loss_p == 0:
                loss_p = loss
                model_record = model.parameters_dict()
        ms.load_param_into_net(model, model_record)
        model.set_train(False)
        pred_mean = model.construct(x[test_indices])
        errors = self._error_score(prediction=pred_mean, targets=y[Tensor(test_indices)],
                                   iw=importance_weights,
                                   test_index=test_indices).asnumpy().astype(float)
        ms.save_checkpoint(model, f"{self.workspace_dir}/elasticNet_{save_type}.ckpt")
        self.errors = errors

    def predict(self, x: ms.Tensor, save_type=None, mean_std=None):
        """predict"""
        input_data = x.shape[1]
        if save_type == "density":
            n_outputs = 125000
        else:
            n_outputs = 1
        model = ElasticNet(save_type=save_type, n_inputs=input_data, n_outputs=n_outputs)
        param_dict = ms.load_checkpoint(f"{self.workspace_dir}/elasticNet_{save_type}.ckpt")
        param_not_load, _ = ms.load_param_into_net(model, param_dict)
        if not bool(param_not_load):
            logging.info("预测模型加载成功")
        model.set_train(False)
        y = model.construct(x)
        y = Normalize(data_type=save_type, mean_std_params=mean_std).recover(y)
        return y

    def _load_model(self, save_type, n_training=None):
        """load model"""
        if save_type == "density":
            n_inputs = n_training
            n_outputs = 125000
        else:
            n_inputs = n_training
            n_outputs = 1
        model = ElasticNet(save_type=save_type, n_inputs=n_inputs, n_outputs=n_outputs)
        param_dict = ms.load_checkpoint(f"{self.workspace_dir}/elasticNet_{save_type}.ckpt")
        param_not_load, _ = ms.load_param_into_net(model, param_dict)
        if not bool(param_not_load):
            logging.info("模型加载成功")
        model.set_train(False)
        return model

    def save_file(self, filename):
        """save as npy file"""
        np.save(filename + '_errors', self.errors)
        with os.fdopen(os.open(filename + '_kernel.pkl', os.O_WRONLY | os.O_CREAT, stat.S_IWUSR),
                       'wb') as kernel_output:
            kernel_output.write(pickle.dumps(self.kernel))

    def load_file(self, filename):
        """load errors file"""
        self.errors = np.load(filename + '_errors.npy', allow_pickle=True)
        with open(filename + '_kernel.pkl', 'rb') as kernel_input_file:
            self.kernel = pickle.loads(kernel_input_file.read())

    def net_forward(self, x_train: ms.Tensor, y_train: ms.Tensor):
        """net forward"""
        y_pred = self.model(x_train)
        loss = self.loss_fn(y_pred, y_train)
        return loss
