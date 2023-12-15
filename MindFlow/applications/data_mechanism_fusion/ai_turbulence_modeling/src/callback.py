# ============================================================================
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
"""
callback
"""
import time

import numpy as np

import mindspore
from mindspore import Tensor
from mindspore.train.callback import Callback


def loss_data_2d(loss):
    """2d定义callback函数"""
    if isinstance(loss, (tuple, list)):
        if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
            loss = loss[0]
    if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = np.mean(loss.asnumpy())
    return loss


def eval_loss_fn_2d(network, data):
    """定义测试函数"""
    batch_num_eval = 0
    out_eval_loss = 0.0
    out_rs_loss = 0.0

    time_eval_start = time.time()
    for value in data:
        batch_num_eval += 1
        # network返回值是[loss, predict, label]
        output = network(value[0], value[1], value[2], value[3])
        loss_eval, loss_rs = output[0], output[1]
        out_eval_loss += loss_eval.asnumpy()
        out_rs_loss += loss_rs.asnumpy()

    out_eval_loss = out_eval_loss / batch_num_eval
    out_eval = [out_eval_loss]

    out_rs_loss = out_rs_loss / batch_num_eval
    out_rs = [out_rs_loss]

    time_eval_end = time.time()
    epoch_seconds_eval = time_eval_end - time_eval_start
    return epoch_seconds_eval, out_eval, out_rs


def eval_fn_2d(network, data):
    """eval_fn_2d"""
    predict = []
    label = []
    for value in data:
        output = network(value[0])
        predict.extend([x[0] for x in output.asnumpy().tolist()])
        label.extend([x[0] for x in value[1].asnumpy().tolist()])
    return predict, label


class Callback2D(Callback):
    """自定义Callback2D"""
    def __init__(self, model_path, network, eval_network, eval_1, eval_2):
        super(Callback2D, self).__init__()
        self.out_train = 0.0
        self.time_train_start = time.time()
        self.eval_net = eval_network
        self.network = network
        self.eval_1 = eval_1
        self.eval_2 = eval_2
        self.best_loss = 1e7

        self.ckpt_path = model_path

        self.train_loss_log = []
        self.val_loss_log = []

        self.predict_list = []
        self.label_list = []
        self.train_predict_list = []
        self.trian_label_list = []
        self.epoch_num = 0
        self.best_epoch = 0
        self.last_step_loss = 0

    def epoch_begin(self, run_context):
        """epoch_begin"""
        self.out_train = 0.0
        cb_params = run_context.original_args()
        print(f"----EPOCH = {cb_params.cur_epoch_num}------")
        self.time_train_start = time.time()

    def step_end(self, run_context):
        """step_end"""
        cb_params = run_context.original_args()
        loss_train = cb_params.net_outputs  # 损失值
        loss_train = loss_data_2d(loss_train)
        self.out_train += loss_train
        self.last_step_loss = loss_train

    def epoch_end(self, run_context):
        """epoch_end"""
        cb_params = run_context.original_args()
        batch_num_train = cb_params.batch_num
        output_train = self.out_train / batch_num_train
        time_train_end = time.time()
        epoch_seconds_train = time_train_end - self.time_train_start

        print(f"Epoch: {cb_params.cur_epoch_num}, time: {epoch_seconds_train:5.3f}s, \
              train_loss:{output_train:.3e}, last_step_loss:{self.last_step_loss:.3e}", flush=True)

        self.train_loss_log.append(output_train)

        if self.eval_2 is not None:
            epoch_seconds_2, out_eval_2, out_rs_2 = eval_loss_fn_2d(self.eval_net, self.eval_2)
            print(f"val set: time: {epoch_seconds_2:5.3f}s, loss:{out_eval_2[0]:.3e}", flush=True)
            print(f"val set: rs loss:{out_rs_2[0]:.3e}", flush=True)
            self.val_loss_log.append(out_eval_2[0])

            # 保存验证集损失最小的模型
            if self.val_loss_log[-1] < self.best_loss:
                self.best_loss = self.val_loss_log[-1]
                mindspore.save_checkpoint(cb_params.train_network, self.ckpt_path)
                print("update best checkpoint ...")

                self.predict_list, self.label_list = eval_fn_2d(self.network, self.eval_2)
                self.train_predict_list, _ = eval_fn_2d(self.network, self.eval_1)
                self.best_epoch = cb_params.cur_epoch_num

        elif self.eval_2 is None and self.train_loss_log[-1] < self.best_loss:
            mindspore.save_checkpoint(cb_params.train_network, self.ckpt_path)
            self.train_predict_list, _ = eval_fn_2d(self.network, self.eval_1)
            self.best_epoch = cb_params.cur_epoch_nu


def loss_data_3d(loss):
    """loss_data_3d"""
    if isinstance(loss, (tuple, list)):
        if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
            loss = loss[0]
    if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
        loss = np.mean(loss.asnumpy())
    return loss


def eval_fn_3d(network, data):
    """eval_fn_3d"""
    batch_num_eval = 0
    out_eval_loss = 0.0
    out_rs_loss = 0.0
    out_r2_loss = 0.0
    loss12_1 = 0.0
    loss12_2 = 0.0
    time_eval_start = time.time()
    for value in data:
        batch_num_eval += 1
        output = network(value[0], value[1], value[2], value[3], value[4])
        loss_eval, loss_rs, loss_r2, loss12 = output[0], output[1], output[2], output[3]
        out_eval_loss += loss_eval.asnumpy()
        out_rs_loss += loss_rs.asnumpy()
        out_r2_loss += loss_r2.asnumpy()
        loss12_1 += loss12[0].asnumpy()
        loss12_2 += loss12[1].asnumpy()
    out_eval_loss = out_eval_loss / batch_num_eval
    out_eval = [out_eval_loss]

    out_rs_loss = out_rs_loss / batch_num_eval
    out_rs = [out_rs_loss]

    out_r2_loss = out_r2_loss / batch_num_eval
    out_r2 = [out_r2_loss]

    out1 = [loss12_1 / batch_num_eval]
    out2 = [loss12_2 / batch_num_eval]

    time_eval_end = time.time()
    epoch_seconds_eval = time_eval_end - time_eval_start
    return epoch_seconds_eval, out_eval, out_rs, out_r2, out1, out2


class Callback3D(Callback):
    """Callback3D"""
    def __init__(self, ckpt_path, eval_network, eval_1, eval_2):
        super(Callback3D, self).__init__()
        self.out_train = 0.0
        self.time_train_start = time.time()
        self.eval_net = eval_network
        self.eval_1 = eval_1
        self.eval_2 = eval_2
        self.best_loss = 1e7
        self.ckpt_path = ckpt_path

        self.train_loss_log = []
        self.val_loss_log = []
        self.rs_loss_log = []

    def epoch_begin(self, run_context):
        """epoch_begin"""
        self.out_train = 0.0
        cb_params = run_context.original_args()
        print(f"----EPOCH = {cb_params.cur_epoch_num}------")
        self.time_train_start = time.time()

    def step_end(self, run_context):
        """step_end"""
        cb_params = run_context.original_args()
        loss_train = cb_params.net_outputs  # 损失值
        loss_train = loss_data_3d(loss_train)
        self.out_train += loss_train

    def epoch_end(self, run_context):
        """epoch_end"""
        cb_params = run_context.original_args()
        batch_num_train = cb_params.batch_num
        output_train = self.out_train / batch_num_train
        time_train_end = time.time()
        epoch_seconds_train = time_train_end - self.time_train_start
        print(f"Epoch: {cb_params.cur_epoch_num}, time: {epoch_seconds_train:5.3f}s,\
               train_loss:{output_train:.3e}", flush=True)

        self.train_loss_log.append(output_train)

        epoch_seconds_2, out_eval_2, out_rs_2, \
            out_r2_2, lx0, lx2 = eval_fn_3d(self.eval_net, self.eval_2)
        r2_score = 1.0 - out_r2_2[0]
        print(f"val set: time: {epoch_seconds_2:5.3f}s, loss:{out_eval_2[0]:.3e}", flush=True)
        print(f"val set: rs loss:{out_rs_2[0]:.3e}", flush=True)
        print(f"val set: r2 :{r2_score:.5f}", flush=True)
        print(f"val set: Lx0 :{lx0[0]:.5e}", flush=True)
        print(f"val set: data loss :{lx2[0]:.5e}", flush=True)
        self.val_loss_log.append(out_eval_2[0])
        self.rs_loss_log.append(out_rs_2[0])

        # 保存验证集损失最小的模型
        cb_params = run_context.original_args()
        if self.val_loss_log[-1] < self.best_loss:
            self.best_loss = self.val_loss_log[-1]
            mindspore.save_checkpoint(cb_params.train_network, self.ckpt_path)
            print("update best checkpoint ...")

    def end(self, run_context):
        """end"""
        # 保存训练过程
        cb_params = run_context.original_args()
        epochs = cb_params.epoch_num
        temp_loss = np.zeros((epochs, 4))
        temp_loss[:, 0] = np.arange(epochs)
        temp_loss[:, 1] = self.train_loss_log
        temp_loss[:, 2] = self.val_loss_log
        temp_loss[:, 3] = self.rs_loss_log
        txt_name = './train_loss_log.dat'
        np.savetxt(txt_name, temp_loss)
