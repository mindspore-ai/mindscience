# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
Callback functions
"""

import os
from shutil import copyfile
from collections import deque
import numpy as np
from numpy import ndarray
import mindspore as ms
from mindspore import Model
from mindspore import Tensor
from mindspore.nn import Cell
from mindspore.nn import Optimizer
from mindspore.nn import TrainOneStepCell
from mindspore.ops import functional as F
from mindspore.dataset import Dataset
from mindspore.train import save_checkpoint
from mindspore.train.callback import Callback, RunContext
from mindspore.train.callback._callback import InternalCallbackParam
from mindspore.train._utils import _make_directory
from .. import utils

_cur_dir = os.getcwd()


__all__ = [
    "TrainMonitor",
]


class TrainMonitor(Callback):
    r"""A callback to show and record the information during training process.

    Args:
        model (Model):              Mindspore model.

        file_name (str):            Name of the file to record the training information.

        directory (str):            Name of output directory. Default: ``None``.

        per_epoch (int):            The epoch interval for outputting training information. Default: 1

        per_step (int):             The step interval for outputting training information. Default: 0

        avg_steps (int):            Number of step for the moving average of loss function.
                                    If 0 is given, the loss will be averaged over all previous steps.
                                    Default: 0

        eval_dataset (Dataset):     Evaluate dataset. Default: ``None``.

        best_ckpt_metrics (str):    Reference metric to record the best parameters. Default: ``None``.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """

    def __init__(self,
                 model: Model,
                 file_name: str,
                 directory: str = None,
                 per_epoch: int = 1,
                 per_step: int = 0,
                 avg_steps: int = 0,
                 eval_dataset: Dataset = None,
                 best_ckpt_metrics: str = None,
                 ):

        super().__init__()
        if not isinstance(per_epoch, int) or per_epoch < 0:
            raise ValueError("per_epoch must be int and >= 0.")
        if not isinstance(per_step, int) or per_step < 0:
            raise ValueError("per_step must be int and >= 0.")

        self.avg_steps = avg_steps
        self.loss_record = 0
        self.train_num = 0
        if avg_steps > 0:
            self.train_num = deque(maxlen=avg_steps)
            self.loss_record = deque(maxlen=avg_steps)

        if per_epoch * per_step != 0:
            if per_epoch == 1:
                per_epoch = 0
            else:
                raise ValueError(
                    "per_epoch and per_step cannot larger than 0 at same time.")
        self.model = model
        self._per_epoch = per_epoch
        self._per_step = per_step
        self.eval_dataset = eval_dataset

        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir

        self._filename = file_name + '-info.data'
        self._ckptfile = file_name + '-best'
        self._ckptdata = file_name + '-ckpt.data'

        self.num_ckpt = 1
        self.best_value = 5e4
        self.best_ckpt_metrics = best_ckpt_metrics

        self.last_loss = 0
        self._last_print_time = 0
        self.record = []

        self.output_title = True
        filename = os.path.join(self._directory, self._filename)
        if os.path.exists(filename):
            with open(filename, "r") as f:
                lines = f.readlines()
                if len(lines) > 1:
                    os.remove(filename)

    def begin(self, run_context: RunContext):
        """
        Called once before the network executing.

        Args:
            run_context (RunContext): Include some information of the model.
        """

    def step_end(self, run_context: RunContext):
        """step end"""
        cb_params: InternalCallbackParam = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        nbatch = len(cb_params.train_dataset_element[0])
        batch_loss = loss * nbatch

        self.last_loss = loss
        if self.avg_steps > 0:
            self.loss_record.append(batch_loss)
            self.train_num.append(nbatch)
        else:
            self.loss_record += batch_loss
            self.train_num += nbatch

        # In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.
        if self._per_step != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -=\
                    max(self._per_step, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_step != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_step:
            self._last_print_time = cb_params.cur_step_num
            self._output_data(cb_params)

    def epoch_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num

        if self._per_epoch > 0 and cur_epoch % self._per_epoch == 0:
            self._output_data(cb_params)

    def _write_ckpt_file(self, filename: str, info: str, network: TrainOneStepCell):
        """write checkpoint (.ckpt) file"""
        ckptfile = os.path.join(self._directory, filename + '.ckpt')
        ckptbck = os.path.join(self._directory, filename + '.bck.ckpt')
        ckptdata = os.path.join(self._directory, self._ckptdata)

        if os.path.exists(ckptfile):
            os.rename(ckptfile, ckptbck)

        save_checkpoint(network, ckptfile)
        with utils.fdopen(ckptdata, "a") as f:
            f.write(info + os.linesep)

    def _output_data(self, cb_params: InternalCallbackParam):
        """output data"""
        cur_epoch = cb_params.get("cur_epoch_num", 1)

        opt: Optimizer = cb_params.get('optimizer')
        if opt is None:
            opt = cb_params.train_network.optimizer

        if opt.dynamic_lr:
            step = opt.global_step
            if not isinstance(step, int):
                step = step.asnumpy()[0]
        else:
            step = cb_params.cur_step_num

        if self.avg_steps > 0:
            mov_avg = sum(self.loss_record) / sum(self.train_num)
        else:
            mov_avg = self.loss_record / self.train_num

        title = "#! FIELDS step"
        info = 'Epoch: ' + str(cur_epoch) + ', Step: ' + str(step)
        outdata = '{:>10d}'.format(step)

        lr = opt.learning_rate
        if opt.dynamic_lr:
            step = F.cast(step, ms.int32)
            if opt.is_group_lr:
                lr = ()
                for learning_rate in opt.learning_rate:
                    current_dynamic_lr = learning_rate(step-1)
                    lr += (current_dynamic_lr,)
            else:
                lr = opt.learning_rate(step-1)
        lr = lr.asnumpy()

        title += ' learning_rate'
        info += ', Learning_rate: ' + str(lr)
        outdata += '{:>15e}'.format(lr)

        title += " last_loss avg_loss"
        info += ', Last_Loss: ' + \
            str(self.last_loss) + ', Avg_loss: ' + str(mov_avg)
        outdata += '{:>15e}'.format(self.last_loss) + '{:>15e}'.format(mov_avg)

        _make_directory(self._directory)

        if self.eval_dataset is not None:
            eval_metrics = self.model.eval(
                self.eval_dataset, dataset_sink_mode=False)
            for k, v in eval_metrics.items():
                info += ', '
                info += k
                info += ': '
                info += str(v)

                if isinstance(v, ndarray) and v.size > 1:
                    for i in range(v.size):
                        title += (' ' + k + str(i))
                        outdata += '{:>15e}'.format(v[i])
                else:
                    title += (' ' + k)
                    outdata += '{:>15e}'.format(v)

            if self.best_ckpt_metrics in eval_metrics.keys():
                best_value = eval_metrics[self.best_ckpt_metrics]
                self._write_best_ckpt(best_value, info, cb_params.train_network)

        print(info, flush=True)
        filename = os.path.join(self._directory, self._filename)
        if self.output_title:
            with utils.fdopen(filename, "a") as f:
                f.write(title + os.linesep)
            self.output_title = False
        with utils.fdopen(filename, "a") as f:
            f.write(outdata + os.linesep)

    def _write_best_ckpt(self, best_value: ndarray, info: str, network: Cell):
        """write the best parameter of checkpoint file"""
        if isinstance(best_value, ndarray) and len(best_value) > 1:
            output_ckpt = best_value < self.best_value
            num_best = np.count_nonzero(output_ckpt)
            if num_best > 0:
                self._write_ckpt_file(
                    self._ckptfile, info, network)
                source_ckpt = os.path.join(
                    self._directory, self._ckptfile + '.ckpt')
                for i in range(len(best_value)):
                    if output_ckpt[i]:
                        dest_ckpt = os.path.join(
                            self._directory, self._ckptfile + '-' + str(i) + '.ckpt')
                        bck_ckpt = os.path.join(
                            self._directory, self._ckptfile + '-' + str(i) + '.ckpt.bck')
                        if os.path.exists(dest_ckpt):
                            os.rename(dest_ckpt, bck_ckpt)
                        copyfile(source_ckpt, dest_ckpt)
                self.best_value = np.minimum(best_value, self.best_value)
        else:
            if best_value < self.best_value:
                self._write_ckpt_file(
                    self._ckptfile, info, network)
                self.best_value = best_value
        return self
