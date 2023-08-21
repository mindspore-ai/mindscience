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
import copy

import numpy as np
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord
from mindspore import Tensor
import mindspore.common.dtype as mstype

class PredictCallback(Callback):
    """
    Monitor the prediction accuracy in training.

    Args:
        net (Cell): Prediction network cell.
        input_data (Array): Input data of prediction.
        label (Array): Label data of prediction.
        config (dict): config info of prediction.
        visual_fn (dict): Visualization function. Default: ``None``.
    """
    def __init__(self, net, input_data, label, config, visual_fn=None):
        super(PredictCallback, self).__init__()
        self.net = net
        self.input_data = input_data
        self.label = label
        self.latent_vector = None
        for params in net.trainable_params():
            if "net.latent_vector" in params.name:
                self.latent_vector = params
                break
        self.num_scenarios = config.get("num_scenarios", 1)

        self.output_shape = label.shape
        self.visual_fn = visual_fn
        self.video_path = config.get("vision_path", "./vision")
        self.summary_path = config.get("summary_path", "./summary")

        self.output_size = config.get("output_size", 3)
        self.input_size = config.get("input_size", 3)
        self.output_scale = np.array(config["output_scale"], dtype=np.float32)
        self.predict_interval = config.get("predict_interval", 10)
        self.batch_size = config.get("batch_data_size", 8192*4)

        self.delta_x = input_data[0, 1, 0, 0] - input_data[0, 0, 0, 0]
        self.delta_x = input_data[0, 0, 1, 1] - input_data[0, 0, 0, 1]
        self.delta_t = input_data[1, 0, 0, 2] - input_data[0, 0, 0, 2]
        print("check yee delta: {}, {}, {}".format(self.delta_x, self.delta_x, self.delta_t))

        self.ex_data = copy.deepcopy(input_data)
        self.ey_data = copy.deepcopy(input_data)
        self.ey_data = copy.deepcopy(input_data)
        self.ex_data = self.ex_data.reshape(-1, self.input_size)
        self.ey_data = self.ey_data.reshape(-1, self.input_size)
        self.ey_data = self.ey_data.reshape(-1, self.input_size)
        self.ex_data[:, 1] += self.delta_x / 2.0
        self.ex_data[:, 2] += self.delta_t / 2.0
        self.ey_data[:, 0] += self.delta_x / 2.0
        self.ey_data[:, 2] += self.delta_t / 2.0
        self.each_data = [self.ex_data, self.ey_data, self.ey_data]
        self._step_counter = 0
        self.l2_error = (1.0, 1.0, 1.0)

    def __enter__(self):
        self.summary_record = SummaryRecord(self.summary_path)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def epoch_end(self, run_context):
        """
        Evaluate the net at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        if self.latent_vector is not None:
            print("latent vector norm: ", np.linalg.norm(self.latent_vector.data.asnumpy(), axis=1))
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            # predict each quantity separately
            index = 0
            prediction = np.zeros(self.output_shape)
            prediction = prediction.reshape((-1, self.output_size))
            time_beg = time.time()
            while index < len(self.each_data[0]):
                index_end = min(index + self.batch_size, len(self.each_data[0]))
                for i in range(self.output_size):
                    batch_data = Tensor(self.each_data[i][index: index_end, :], mstype.float32)
                    batch_data = batch_data.view(1, -1, 3).repeat(self.num_scenarios, axis=0).view(-1, 3)
                    net_out = self.net(batch_data)
                    net_out_size = len(net_out)
                    net_out = net_out.asnumpy()
                    net_out = net_out[:int(net_out_size // self.num_scenarios), :]
                    prediction[index: index_end, i] = net_out[:, i] * self.output_scale[i]
                index = index_end
            print("==================================================================================================")
            print("Prediction total time: {} s".format(time.time() - time_beg))
            prediction = prediction.reshape(self.output_shape)
            if self.visual_fn is not None:
                self.visual_fn(self.input_data, self.label, prediction, save_path=self.video_path,
                               name="epoch" + str(cb_params.cur_epoch_num))

            label = self.label.reshape((-1, self.output_size))
            prediction = prediction.reshape((-1, self.output_size))
            self.l2_error = self._get_l2_error(label, prediction)
            print("l2_error, Ex: ", self.l2_error[0], ", Ey: ", self.l2_error[1], ", Hz: ", self.l2_error[2])
            print("==================================================================================================")

    def _get_l2_error(self, label, prediction):
        """calculate l2-error to evaluate accuracy"""
        self._step_counter += 1
        abs_error = label - prediction
        error_ex = np.sqrt(np.sum(np.square(abs_error[..., 0]))) / np.sqrt(np.sum(np.square(label[..., 0])))
        error_ey = np.sqrt(np.sum(np.square(abs_error[..., 1]))) / np.sqrt(np.sum(np.square(label[..., 1])))
        error_hz = np.sqrt(np.sum(np.square(abs_error[..., 2]))) / np.sqrt(np.sum(np.square(label[..., 2])))
        self.summary_record.add_value('scalar', 'l2_ex', Tensor(error_ex))
        self.summary_record.add_value('scalar', 'l2_ey', Tensor(error_ey))
        self.summary_record.add_value('scalar', 'l2_hz', Tensor(error_hz))
        return error_ex, error_ey, error_hz
