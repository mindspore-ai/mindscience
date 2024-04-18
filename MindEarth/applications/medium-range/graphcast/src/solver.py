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
# ============================================================================
"""GraphCastTrainer"""
from mindspore import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import LossMonitor, TimeMonitor

from mindearth.module import Trainer
from mindearth.data import Dataset
from .callback import EvaluateCallBack
from .precip_dataset import Era5DataTp
from .net_with_clip import TrainOneStepCell


class GraphCastTrainer(Trainer):
    """
    Self-defined forecast model inherited from `Trainer`.

    Args:
        config (dict): parameters for training.
        model (Cell): network for training.
        loss_fn (str): user-defined loss function.
        logger (logging.RootLogger): tools for logging.

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, config, model, loss_fn, logger):
        super().__init__(config, model, loss_fn, logger)
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.pred_cb = self.get_callback()
        self.solver = self.get_solver()

    def get_solver(self):
        """
        define the solver of the model, abstract method.
        """
        loss_scale = DynamicLossScaleManager()
        solver = Model(network=self.loss_fn,
                       optimizer=self.optimizer,
                       loss_scale_manager=loss_scale,
                       amp_level=self.train_params.get('amp_level', 'O2'),
                       )
        return solver

    def get_callback(self):
        """
        define the callback of the model, abstract method.
        """
        pred_cb = EvaluateCallBack(self.model, self.valid_dataset, self.config, self.logger)
        return pred_cb


class GraphCastTrainerTp(Trainer):
    """
    Self-defined forecast precipitation model inherited from `Trainer`.

    Args:
        config (dict): parameters for training.
        model (Cell): network for training.
        loss_fn (str): user-defined loss function.
        logger (logging.RootLogger): tools for logging.
        loss_scale (LossScaler): loss manager

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, config, model, loss_fn, logger, loss_scale):
        super().__init__(config, model, loss_fn, logger, loss_scale=loss_scale)
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.pred_cb = self.get_callback()
        self.solver = self.get_solver()
        self.loss_scale = loss_scale
        self.train_params = config.get("train")

    def get_solver(self):
        """
        define the solver of the model, abstract method.
        """
        loss_with_clip = TrainOneStepCell(self.loss_fn, self.optimizer, self.loss_scale, enable_clip_grad=True,
                                          gradient_clip_value=self.train_params.get("gradient_clip_value", 16))
        solver = Model(network=loss_with_clip)
        return solver

    def get_callback(self):
        """
        define the callback of the model, abstract method.
        """
        pred_cb = EvaluateCallBack(self.model, self.valid_dataset, self.config, self.logger)
        return pred_cb

    def get_dataset(self):
        """
        Get medium precipitation train and valid dataset.

        Returns:
            Dataset, train dataset.
            Dataset, valid dataset.
        """
        if self.weather_data_source == 'ERA5':
            train_dataset_generator = Era5DataTp(data_params=self.data_params, run_mode='train')
            valid_dataset_generator = Era5DataTp(data_params=self.data_params, run_mode='valid')
        else:
            raise NotImplementedError(
                f"{self.weather_data_source} not implemented")
        train_dataset = Dataset(train_dataset_generator,
                                distribute=self.train_params.get('distribute'),
                                num_workers=self.data_params.get('num_workers'))
        valid_dataset = Dataset(valid_dataset_generator,
                                distribute=False,
                                num_workers=self.data_params.get('num_workers'),
                                shuffle=False)
        train_dataset = train_dataset.create_dataset(self.data_params.get('batch_size'))
        valid_dataset = valid_dataset.create_dataset(self.data_params.get('batch_size'))
        return train_dataset, valid_dataset

    def train(self):
        """Train."""
        callback_lst = [LossMonitor(), TimeMonitor(), self.ckpt_cb]
        if self.pred_cb:
            callback_lst.append(self.pred_cb)
        if self.step == 1:
            self.solver.train(epoch=self.optimizer_params.get("epochs"),
                              train_dataset=self.train_dataset,
                              callbacks=callback_lst,
                              dataset_sink_mode=self.data_params.get('data_sink'))
        else:
            self.solver.train(epoch=self.optimizer_params.get("finetune_epochs"),
                              train_dataset=self.train_dataset,
                              callbacks=callback_lst,
                              dataset_sink_mode=self.data_params.get('data_sink'))
