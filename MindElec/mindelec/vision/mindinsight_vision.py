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
"""Callback functions for model.train and model.eval"""

import os
import numpy as np
from mindspore.dataset.engine.datasets import BatchDataset as ds
from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord
from ..solver import Solver


class MonitorTrain(Callback):
    r"""
    Loss monitor for train.

    Args:
        per_print_times (int): print loss interval. Default: 1.
        summary_dir (str): summary save path. Default: './summary_train'.

    Returns:
        Callback monitor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> from mindelec.vision import MonitorTrain
        >>> per_print_times = 1
        >>> summary_dir = './summary_train'
        >>> MonitorTrain(per_print_times, summary_dir)
    """

    def __init__(self, per_print_times=1, summary_dir='./summary_train'):
        super(MonitorTrain, self).__init__()
        if not isinstance(per_print_times, int):
            raise TypeError("per_print_times must be int, but get {}".format(type(per_print_times)))
        if isinstance(per_print_times, bool):
            raise TypeError("per_print_times must be int, but get {}".format(type(per_print_times)))
        if per_print_times <= 0:
            raise ValueError("per_print_times must be > 0, but get {}".format(per_print_times))

        if not isinstance(summary_dir, str):
            raise TypeError("summary_dir must be str, but get {}".format(type(summary_dir)))
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        self._per_print_times = per_print_times
        self._summary_dir = summary_dir
        self._step_counter = 0
        self.final_loss = 0

    def __enter__(self):
        self.summary_record = SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def step_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        self._step_counter += 1
        params = run_context.original_args()

        loss = params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0].asnumpy(), np.ndarray) and isinstance(loss[0], Tensor):
                loss = loss[0]

        if isinstance(loss.asnumpy(), np.ndarray) and isinstance(loss, Tensor):
            loss = np.mean(loss.asnumpy())

        cur_step = (params.cur_step_num - 1) % params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, training end.".format(
                params.cur_epoch_num, cur_step))

        if self._per_print_times != 0 and params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (params.cur_epoch_num, cur_step, loss), flush=True)
            self.summary_record.add_value('scalar', 'train_loss', Tensor(loss))
            self.summary_record.record(self._step_counter)
            self.final_loss = loss


class MonitorEval(Callback):
    r"""
    LossMonitor for eval.

    Args:
        summary_dir (str): summary save path. Default: './summary_eval'.
        model (Solver): Model object for eval. Default: None.
        eval_ds (Dataset): eval dataset. Default: None.
        eval_interval (int): eval interval. Default: 10.
        draw_flag (bool): specifies if save summary_record. Default: True.

    Returns:
        Callback monitor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore.nn as nn
        >>> from mindelec.solver import Solver
        >>> from mindelec.vision import MonitorEval
        >>> class S11Predictor(nn.Cell):
        ...     def __init__(self, input_dimension):
        ...         super(S11Predictor, self).__init__()
        ...         self.fc1 = nn.Dense(input_dimension, 128)
        ...         self.fc2 = nn.Dense(128, 128)
        ...         self.fc3 = nn.Dense(128, 1001)
        ...         self.relu = nn.ReLU()
        ...
        ...     def construct(self, x):
        ...         x0 = x
        ...         x1 = self.relu(self.fc1(x0))
        ...         x2 = self.relu(self.fc2(x1))
        ...         x = self.fc3(x1 + x2)
        ...         return x
        >>> model_net = S11Predictor(3)
        >>> model = Solver(network=model_net, mode="Data",
        >>>                optimizer=nn.Adam(model_net.trainable_params(), 0.001), loss_fn=nn.MSELoss())
        >>> # For details about how to build the dataset, please refer to the tutorial
        >>> # document on the official website.
        >>> eval_ds = Dataset()
        >>> summary_dir = './summary_eval_path'
        >>> eval_interval = 10
        >>> draw_flag = True
        >>> MonitorEval(summary_dir, model, eval_ds, eval_interval, draw_flag)
    """

    def __init__(self,
                 summary_dir='./summary_eval',
                 model=None,
                 eval_ds=None,
                 eval_interval=10,
                 draw_flag=True):
        super(MonitorEval, self).__init__()
        if not isinstance(summary_dir, str):
            raise TypeError("summary_dir must be str, but get {}".format(type(summary_dir)))

        if not isinstance(model, Solver):
            raise TypeError("model must be mindelec solver, but get {}".format(type(model)))

        if not isinstance(eval_ds, ds):
            raise TypeError("eval_ds must be mindelec dataset, but get {}".format(type(eval_ds)))

        if not isinstance(eval_interval, int):
            raise TypeError("eval_interval must be int, but get {}".format(type(eval_interval)))
        if isinstance(eval_interval, bool):
            raise TypeError("eval_interval must be int, but get {}".format(type(eval_interval)))
        if eval_interval <= 0:
            raise ValueError("eval_interval must be > 0, but get {}".format(eval_interval))

        if not isinstance(draw_flag, bool):
            raise TypeError("draw_flag must be bool, but get {}".format(type(draw_flag)))

        self._summary_dir = summary_dir
        self._model = model
        self._eval_ds = eval_ds
        self._eval_interval = eval_interval
        self._draw_flag = draw_flag

        self._eval_count = 0
        self.temp = None
        self.loss_final = 0.0
        self.l2_s11_final = 0.0

    def __enter__(self):
        self.summary_record = SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        self.temp = run_context
        self._eval_count += 1
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self._eval_interval == 0:
            res_eval = self._model.model.eval(valid_dataset=self._eval_ds, dataset_sink_mode=True)
            loss_eval_print, l2_s11_print = res_eval['eval_mrc']['loss_error'], res_eval['eval_mrc']['l2_error']

            self.loss_final = loss_eval_print
            self.l2_s11_final = l2_s11_print
            print('Eval   current epoch:', cur_epoch, ' loss:', loss_eval_print, ' l2_s11:', l2_s11_print)

            self.summary_record.add_value('scalar', 'eval_loss', Tensor(loss_eval_print))
            self.summary_record.record(self._eval_count * self._eval_interval)

            self.summary_record.add_value('scalar', 'l2_s11', Tensor(l2_s11_print))
            self.summary_record.record(self._eval_count * self._eval_interval)

            if self._draw_flag:
                pic_res = res_eval['eval_mrc']['pic_res']
                for i in range(len(pic_res)):
                    self.summary_record.add_value('image', 'l2_s11_image_' + str(i),
                                                  Tensor(np.expand_dims(pic_res[i], 0).transpose((0, 3, 1, 2))))
                    self.summary_record.record(self._eval_count * self._eval_interval)
