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
"""dem solver"""
from mindspore import nn, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor

from mindearth.module import Trainer

from .callback import EvaluateCallBack


class DemSrTrainer(Trainer):
    r"""
    Self-define forecast model inherited from `Trainer`.

    Args:
        config (dict): parameters for training.
        model (Cell): network for training.
        loss_fn (str): user-defined loss function.
        logger (logging.RootLogger): tools for logging.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, config, model, loss_fn, logger):
        super(DemSrTrainer, self).__init__(config, model, loss_fn, logger, weather_data_source="DemSR")
        self.optimizer_params = config["optimizer"]
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.optimizer = self.get_optimizer()
        self.solver = self.get_solver()

    def get_optimizer(self):
        r"""define the optimizer of the model, abstract method."""
        self.steps_per_epoch = self.train_dataset.get_dataset_size()
        if self.logger:
            self.logger.info(f'steps_per_epoch: {self.steps_per_epoch}')
        if self.optimizer_params['name']:
            optimizer = nn.Adam(self.model.trainable_params(),
                                learning_rate=Tensor(self.optimizer_params['learning_rate']))
        else:
            raise NotImplementedError(
                "self.optimizer_params['name'] not implemented, please overwrite get_optimizer()")
        return optimizer

    def get_callback(self):
        r"""define the callback of the model, abstract method."""
        pred_cb = EvaluateCallBack(self.model, self.valid_dataset, self.config, self.logger)
        return pred_cb

    def train(self):
        r"""Train."""
        callback_lst = [LossMonitor(), TimeMonitor(), self.ckpt_cb]
        if self.pred_cb:
            callback_lst.append(self.pred_cb)
        self.solver.train(epoch=self.optimizer_params["epochs"],
                          train_dataset=self.train_dataset,
                          callbacks=callback_lst,
                          dataset_sink_mode=True)
