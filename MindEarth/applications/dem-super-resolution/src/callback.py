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
# ==============================================================================
"""InferenceModule and EvaluateCallBack"""
from mindspore import nn
from mindspore.train.callback import Callback


class InferenceModule:
    r"""Perform the model inference in Dem Super-resolution."""
    def __init__(self, model, config, logger):
        self.model = model
        self.logger = logger
        self.config = config
        self.loss_mse = nn.MSELoss(reduction='mean')

    def eval(self, test_dataset):
        r"""Evaluate the test dataset"""
        self.logger.info("================================Start Evaluation================================")
        mean_rmse_all = 0
        eval_data_length = 0
        for data in test_dataset.create_dict_iterator():
            inputs = data['inputs']
            batch_size = inputs.shape[0]
            labels = data['labels']
            pred = self.forecast(inputs)
            mean_mse_step = self.loss_mse(pred, labels)
            mean_rmse_all += mean_mse_step.asnumpy()
            eval_data_length += batch_size
            self.logger.info(f"{eval_data_length}, mean mse per step: {mean_mse_step}")
        self.logger.info(f'test dataset size: {eval_data_length}')
        self.logger.info(f"mean mse: {mean_rmse_all / eval_data_length}")
        self.logger.info("================================End Evaluation================================")

    def forecast(self, inputs):
        pred = self.model(inputs)
        return pred


class EvaluateCallBack(Callback):
    r"""Monitor the prediction accuracy in training."""
    def __init__(self,
                 model,
                 test_dataset,
                 config,
                 logger
                 ):
        super(EvaluateCallBack, self).__init__()
        self.model = model
        self.test_dataset = test_dataset
        self.predict_interval = config['summary']["valid_frequency"]
        self.logger = logger
        self.eval_net = InferenceModule(model,
                                        config,
                                        logger)

    def epoch_end(self, run_context):
        """
        Evaluate the model at the end of epoch.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        if cb_params.cur_epoch_num % self.predict_interval == 0:
            self.eval_net.eval(self.test_dataset)
