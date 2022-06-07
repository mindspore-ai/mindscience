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
"""monitor eval"""
from mindspore.common.tensor import Tensor
from mindspore.train.callback import Callback
from mindspore.train.summary import SummaryRecord


class MonitorEval(Callback):
    """
    Monitor the prediction accuracy in training
    """
    def __init__(self,
                 summary_dir='./summary_eval',
                 model=None,
                 eval_ds=None,
                 eval_interval=10,
                 draw_flag=True):
        super(MonitorEval, self).__init__()

        self._summary_dir = summary_dir
        self._model = model
        self._eval_ds = eval_ds
        self._eval_interval = eval_interval
        self._draw_flag = draw_flag

        self._eval_count = 0
        self.temp = None
        self.loss_final = 0.0
        self.l2_final = 0.0
        self.summary_record = None

    def __enter__(self):
        self.summary_record = SummaryRecord(self._summary_dir)
        return self

    def __exit__(self, *exc_args):
        self.summary_record.close()

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch
        """
        self.temp = run_context
        self._eval_count += 1
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self._eval_interval == 0:
            res_eval = self._model.model.eval(valid_dataset=self._eval_ds, dataset_sink_mode=True)
            loss_eval_print, l2_print = res_eval['eval_mrc']['loss_error'], res_eval['eval_mrc']['l2_error']

            self.loss_final = loss_eval_print
            self.l2_final = l2_print
            print('Eval   current epoch:', cur_epoch, ' loss:', loss_eval_print, ' l2:', l2_print)

            self.summary_record.add_value('scalar', 'eval_loss', Tensor(loss_eval_print))
            self.summary_record.record(self._eval_count * self._eval_interval)

            self.summary_record.add_value('scalar', 'l2', Tensor(l2_print))
            self.summary_record.record(self._eval_count * self._eval_interval)
