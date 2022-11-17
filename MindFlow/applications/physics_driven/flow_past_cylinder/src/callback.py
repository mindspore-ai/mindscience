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
"""call back functions"""
import time
import numpy as np
from mindspore.train.callback import Callback
from mindspore import Tensor
import mindspore.common.dtype as mstype


class PredictCallback(Callback):
    """Monitor the prediction accuracy in training."""

    def __init__(self, model, inputs, label, config, visual_fn=None):
        super(PredictCallback, self).__init__()
        self.model = model
        self.inputs = inputs
        self.label = label
        self.label_shape = label.shape
        self.visual_fn = visual_fn
        self.vision_path = config.get("vision_path", "./vision")
        self.summary_dir = config.get("summary_path", "./summary")

        self.output_size = config.get("output_size", 3)
        self.input_size = config.get("input_size", 3)
        self.eval_interval_epochs = config.get("eval_interval_epochs", 10)
        self.batch_size = config.get("test_batch_size", 8192 * 4)

        self._step_counter = 0
        self.l2_error = (1.0, 1.0, 1.0)

    def epoch_end(self, run_context):
        """Evaluate the model at the end of epoch."""
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.eval_interval_epochs == 0:
            # predict each quantity
            index = 0
            prediction = np.zeros(self.label_shape)
            prediction = prediction.reshape((-1, self.output_size))
            time_beg = time.time()
            inputs = self.inputs.reshape((-1, self.input_size))
            while index < inputs.shape[0]:
                index_end = min(index + self.batch_size, inputs.shape[0])
                test_batch = Tensor(inputs[index: index_end, :], mstype.float32)
                prediction[index: index_end, :] = self.model(test_batch).asnumpy()
                index = index_end
            print("==================================================================================================")
            print("predict total time: {} s".format(time.time() - time_beg))
            prediction = prediction.reshape(self.label_shape)
            if self.visual_fn is not None:
                self.visual_fn(self.inputs, self.label, prediction, path=self.vision_path,
                               name="epoch" + str(cb_params.cur_epoch_num))

            label = self.label.reshape((-1, self.output_size))
            prediction = prediction.reshape((-1, self.output_size))
            self.l2_error = self._calculate_error(label, prediction)
            print("==================================================================================================")

    def get_l2_error(self):
        return self.l2_error

    def _calculate_error(self, label, prediction):
        """calculate l2-error to evaluate accuracy"""
        self._step_counter += 1
        error = label - prediction
        l2_error_u = np.sqrt(np.sum(np.square(error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
        l2_error_v = np.sqrt(np.sum(np.square(error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
        l2_error_p = np.sqrt(np.sum(np.square(error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
        l2_error = np.sqrt(np.sum(np.square(error))) / np.sqrt(np.sum(np.square(label)))
        print("l2_error, U: ", l2_error_u, ", V: ", l2_error_v, ", P: ", l2_error_p, ", Total: ", l2_error)
        return l2_error_u, l2_error_v, l2_error_p, l2_error
