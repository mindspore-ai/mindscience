# Copyright 2022 Huawei Technologies Co., Ltd
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
from mindspore.train.summary import SummaryRecord
from mindspore import Tensor


class PredictCallback(Callback):
    """
    Monitor the prediction accuracy in training.
    """

    def __init__(self, model, eval_dataset, logger, step, config):
        super(PredictCallback, self).__init__()
        self.model = model
        self.eval_dataset = eval_dataset
        self.length = eval_dataset.get_dataset_size()
        self.summary_dir = config['summary_dir']
        self.predict_interval = config['eval_interval']
        self.batch_size = config['batch_size']
        self.logger = logger
        self.step = step
        self.config = config
        self.lploss_error = 0
        self.summary_record = None

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()
        self.logger.info("train step-{} {}".format(self.step, self.lploss_error))

    def epoch_end(self, run_context):
        """Evaluate the model at the end of epoch."""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            print("================================Start Evaluation================================")
            time_beg = time.time()
            lploss_error = 0.0
            max_error = 0.0
            for data in self.eval_dataset.create_dict_iterator():
                label = data["uT"].asnumpy()
                test_batch = data["u0"]
                prediction = self.model(test_batch)
                prediction = prediction.asnumpy()

                lploss_error_step = _calculate_error(label, prediction, self.batch_size, size_average=False,
                                                     reduction=True)
                lploss_error += lploss_error_step

                if lploss_error_step >= max_error:
                    max_error = lploss_error_step

            self.lploss_error = lploss_error / self.length
            print("LpLoss_error:", self.lploss_error)
            self.summary_record.add_value('scalar', 'LpLoss_error', Tensor(self.lploss_error))
            print("=================================End Evaluation=================================")
            print("predict total time: {} s".format(time.time() - time_beg))
            self.summary_record.record(cb_params.cur_step_num)

    def get_lploss_error(self):
        return self.lploss_error


def _calculate_error(label, prediction, batch_size, size_average=False, reduction=True):
    """calculate in a batch"""
    num_examples = label.shape[0]

    diff_norms = np.linalg.norm(label.reshape(num_examples, -1) - prediction.reshape(num_examples, -1), axis=1)
    y_norms = np.linalg.norm(label.reshape(num_examples, -1), axis=1)

    res = 0
    if reduction:
        if size_average:
            res = (diff_norms / y_norms).mean() / batch_size
        else:
            res = (diff_norms / y_norms).sum() / batch_size
    else:
        res = diff_norms / y_norms / batch_size

    return res
