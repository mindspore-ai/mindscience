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
# ==============================================================================
"""
call back functions
"""
import time
import os

import mindspore as ms
from mindspore.train.callback import Callback


class TimeMonitor(Callback):
    """
    Monitor the time in training.
    """

    def __init__(self, data_size=None):
        """
        Args:
            data_size (int): Iteration steps to run one epoch of the whole dataset.
        """
        super(TimeMonitor, self).__init__()
        self.data_size = data_size
        self.epoch_time = time.time()
        self.per_step_time = 0

    def epoch_begin(self, run_context):
        """
        Set begin time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        run_context.original_args()
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Print process cost time at the end of epoch.
        """
        epoch_seconds = (time.time() - self.epoch_time)
        step_size = self.data_size
        cb_params = run_context.original_args()
        if hasattr(cb_params, "batch_num"):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        self.per_step_time = epoch_seconds / step_size
        print("epoch time: {:5.1f} s, per step time: {:5.3f} ms".format(epoch_seconds,
                                                                        1000*self.per_step_time), flush=True)

    def get_step_time(self):
        return self.per_step_time


class SaveCkptMonitor(Callback):
    """
    Save (.ckpt) model with less loss than latest one.
    """

    def __init__(self, loss=1.0, save_dir="./checkpoints/", comment=""):
        super(SaveCkptMonitor, self).__init__()
        self.loss = loss
        self.save_dir = save_dir
        self.comment = comment

    def epoch_end(self, run_context):
        """
        Save model if current loss is less than minimum loss.
        """
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        cur_loss = cb_params.net_outputs.asnumpy()

        if cur_loss < self.loss:
            self.loss = cur_loss
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            file_name = f"{self.comment}_epoch_{epoch}_loss_{cur_loss:.6f}.ckpt"
            file_name = os.path.join(self.save_dir, file_name)
            ms.save_checkpoint(
                save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print(
                f"Current epoch: {epoch}, loss: {cur_loss}, saved checkpoint.")
