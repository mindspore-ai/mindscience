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
"""ViTKNOTrainer"""

from mindspore import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager

from mindearth.module import Trainer
from mindearth.data import Era5Data, Dataset

from src.callback import EvaluateCallBack



class ViTKNOEra5Data(Era5Data):
    def _patch(self, *args):
        """ Partition the data into patches. """
        x = args[0]
        if self.run_mode == 'valid' or self.run_mode == 'test':
            x = x.transpose(1, 0, 2, 3)
        return x

class ViTKNOTrainer(Trainer):
    r"""
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
        super(ViTKNOTrainer, self).__init__(config, model, loss_fn, logger)
        self.pred_cb = self.get_callback()

    def get_dataset(self):
        """
        Get train and valid dataset.

        Returns:
            Dataset, train dataset.
            Dataset, valid dataset.
        """
        train_dataset_generator = ViTKNOEra5Data(data_params=self.data_params, run_mode='train')
        valid_dataset_generator = ViTKNOEra5Data(data_params=self.data_params, run_mode='valid')

        train_dataset = Dataset(train_dataset_generator, distribute=self.train_params['distribute'],
                                num_workers=self.data_params['num_workers'])
        valid_dataset = Dataset(valid_dataset_generator, distribute=False, num_workers=self.data_params['num_workers'],
                                shuffle=False)
        train_dataset = train_dataset.create_dataset(self.data_params['batch_size'])
        valid_dataset = valid_dataset.create_dataset(self.data_params['batch_size'])
        return train_dataset, valid_dataset

    def get_callback(self):
        pred_cb = EvaluateCallBack(self.model, self.valid_dataset, self.config, self.logger)
        return pred_cb

    def get_solver(self):
        loss_scale = DynamicLossScaleManager()
        solver = Model(self.loss_fn,
                       optimizer=self.optimizer,
                       loss_scale_manager=loss_scale,
                       amp_level=self.train_params['amp_level']
                       )
        return solver
