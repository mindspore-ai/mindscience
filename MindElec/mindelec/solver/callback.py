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
"""
call back functions
"""
import time
import numpy as np

from mindspore.train.callback import Callback
from mindspore.train import Model
from mindspore import dataset as ds
from mindspore import Tensor


class EvalCallback(Callback):
    """
    Evaluate the model during training.

    Args:
        model (Model): A testing network.
        eval_ds (Dataset): Dataset to evaluate the model.
        eval_interval (int): Specifies how many epochs to train before evaluating.

    Supported Platforms:
        ``Ascend``
    """
    def __init__(self, model, eval_ds, eval_interval):
        super(EvalCallback, self).__init__()
        if not isinstance(model, Model):
            raise TypeError("the type of model should be instance of Cell but got {}".format(type(model)))
        if not isinstance(eval_interval, int):
            raise TypeError("the type of eval_interval should be int, but got {}".format(type(eval_interval)))
        if not isinstance(eval_ds, ds.Dataset):
            raise TypeError("the type of eval_ds should be be instance of Dataset, but got {}"
                            .format(type(eval_interval)))
        self.model = model
        self.eval_ds = eval_ds
        self.eval_interval = eval_interval
        self.eval_count = 0
        self.num_samples = 1

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.eval_interval == 0:
            time_beg = time.time()
            res = self.model.eval(self.eval_ds, dataset_sink_mode=True)
            print("===========================")
            text = " ".join(i + ": " + "%.4f" % res[i] for i in res.keys())
            print(text, " cost time: {} s".format(time.time() - time_beg))
            print("===========================")
            self.eval_count += 1


class LossAndTimeMonitor(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF, it will terminate training.

    Note:
        If per_print_times is 0, do not print loss.

    Args:
        data_size (int): number of batches of each epoch dataset
        per_print_times (int): Print the loss each every seconds. Default: 1.

    Raises:
        ValueError: If data_size is not an integer or less than zero.
        ValueError: If per_print_times is not an integer or less than zero.

    Supported Platforms:
        ``Ascend``
    """

    def __init__(self, data_size, per_print_times=1):
        super(LossAndTimeMonitor, self).__init__()
        if not isinstance(data_size, int) or data_size < 0:
            raise ValueError("data_size must be int and >= 0, but got: {}".format(data_size))
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0, gut got: {}".format(per_print_times))
        self._per_print_times = per_print_times
        self.data_size = data_size
        self.epoch_time = time.time()
        self.time_per_iter = 0
        self.loss = np.inf

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
        Get loss at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())
        self.loss = loss

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss), flush=True)
            epoch_seconds = time.time() - self.epoch_time
            step_size = self.data_size
            step_seconds = epoch_seconds / step_size * 1000.0
            self.time_per_iter = step_seconds
            print("epoch time: {:5.3f} s, per step time: {:5.3f} ms".format(epoch_seconds, step_seconds), flush=True)

    def get_step_time(self):
        return self.time_per_iter

    def get_loss(self):
        return self.loss
