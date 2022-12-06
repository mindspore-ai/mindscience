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
import mindspore.common.dtype as mstype
from mindspore import Tensor
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord


class PredictCallback(Callback):
    """
    Monitor the prediction accuracy in training.

    Args:
        model (Cell): Prediction network cell.
        inputs (Array): Input data of prediction.
        label (Array): Label data of prediction.
        config (dict): config info of prediction.
        visual_fn (dict): Visualization function. Default: None.
    """

    def __init__(self, model, inputs, label, config, summary_dir):
        super(PredictCallback, self).__init__()
        self.model = model
        self.inputs = inputs
        self.label = label
        self.length = label.shape[0]
        self.summary_dir = summary_dir
        self.predict_interval = config.get("eval_interval", 10)
        self.batch_size = config.get("test_batch_size", 1)
        self.rms_error = 1.0
        self.summary_record = None
        print("check test dataset shape: {}, {}".format(self.inputs.shape, self.label.shape))

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            print("================================Start Evaluation================================")
            time_beg = time.time()
            rms_error = 0.0
            max_error = 0.0
            for i in range(self.length):
                label = self.label[i:i + 1]
                test_batch = Tensor(self.inputs[i:i + 1], dtype=mstype.float32)
                prediction = self.model(test_batch)
                prediction = prediction.asnumpy()
                rms_error_step = self._calculate_error(label, prediction)
                rms_error += rms_error_step

                if rms_error_step >= max_error:
                    max_error = rms_error_step

            self.rms_error = rms_error / self.length
            print("mean rms_error:", self.rms_error)
            self.summary_record.add_value('scalar', 'rms_error', Tensor(self.rms_error))
            print("=================================End Evaluation=================================")
            print("predict total time: {} s".format(time.time() - time_beg))
            self.summary_record.record(cb_params.cur_step_num)

    def get_rms_error(self):
        return self.rms_error

    def _calculate_error(self, label, prediction):
        """calculate l2-error to evaluate accuracy"""
        rel_error = np.sqrt(np.sum(np.square(label.reshape(self.batch_size, -1) -
                                             prediction.reshape(self.batch_size, -1)))) / \
                    np.sqrt(np.sum(np.square(label.reshape(self.batch_size, -1))))
        return rel_error
