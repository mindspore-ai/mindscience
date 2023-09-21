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
"""dgmr solver"""
import mindspore as ms
from mindspore import nn, ops

from mindearth.data import RadarData, Dataset

from .callback import EvaluateCallBack


class DgmrTrainer:
    r"""Self-define forecast model for dgmr."""
    def __init__(self, config, g_model, d_model, g_loss_fn, d_loss_fn, logger):
        self.config = config
        self.model_params = config.get("model")
        self.data_params = config.get("data")
        self.train_params = config.get("train")
        self.optimizer_params = config.get("optimizer")
        self.callback_params = config.get("summary")
        self.logger = logger

        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.dataset_size = self.train_dataset.get_dataset_size()
        self.g_model = g_model
        self.d_model = d_model
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        self.g_optimizer, self.d_optimizer = self.get_optimizer()
        self.g_solver, self.d_solver = self.get_solver()

    def get_dataset(self):
        """
        Get train and valid dataset.

        Args:
            process (bool, optional): Whether to process the dataset.

        Returns:
            Dataset, train dataset.
            Dataset, valid dataset.
        """
        train_dataset_generator = RadarData(data_params=self.data_params, run_mode='train')
        valid_dataset_generator = RadarData(data_params=self.data_params, run_mode='valid')

        train_dataset = Dataset(train_dataset_generator, distribute=self.train_params.get('distribute'),
                                num_workers=self.data_params.get('num_workers'))
        valid_dataset = Dataset(valid_dataset_generator, distribute=self.train_params.get('distribute'),
                                num_workers=self.data_params.get('num_workers'),
                                shuffle=False)
        train_dataset = train_dataset.create_dataset(self.data_params.get('batch_size'))
        valid_dataset = valid_dataset.create_dataset(self.data_params.get('batch_size'))
        return train_dataset, valid_dataset

    def get_optimizer(self):
        """
        Get the training optimizer.

        Returns:
            Optimizer, Optimizer of the model.
        """
        self.steps_per_epoch = self.train_dataset.get_dataset_size()
        if self.logger:
            self.logger.info(f'steps_per_epoch: {self.steps_per_epoch}')

        if self.optimizer_params.get('name'):
            beta1 = self.config.get("optimizer").get("beta1")
            beta2 = self.config.get("optimizer").get("beta2")
            g_optimizer = nn.Adam(self.g_model.trainable_params(),
                                  self.config.get("optimizer").get("gen_lr"),
                                  beta1=beta1,
                                  beta2=beta2)
            d_optimizer = nn.Adam(self.d_model.trainable_params(),
                                  self.config.get("optimizer").get("disc_lr"),
                                  beta1=beta1,
                                  beta2=beta2)
        else:
            return NotImplemented
        return g_optimizer, d_optimizer

    def get_solver(self):
        loss_scale = nn.FixedLossScaleUpdateCell(loss_scale_value=self.config.get("optimizer").get("loss_scale"))

        g_solver = nn.TrainOneStepWithLossScaleCell(self.g_loss_fn, self.g_optimizer, scale_sense=loss_scale)
        d_solver = nn.TrainOneStepWithLossScaleCell(self.d_loss_fn, self.d_optimizer, scale_sense=loss_scale)

        return g_solver, d_solver

    def train(self):
        """dgmr train function"""
        evaluator = EvaluateCallBack(config=self.config, dataset_size=self.dataset_size, logger=self.logger)
        for epoch in range(self.config.get("train").get("epochs")):
            evaluator.epoch_start()
            for data in self.train_dataset.create_dict_iterator():
                images = ops.cast(data.get("inputs"), ms.float32)
                future_images = ops.cast(data.get("labels"), ms.float32)
                for _ in range(2):
                    d_res = self.d_solver(images, future_images)
                g_res = self.g_solver(images, future_images)
            evaluator.print_loss(g_res, d_res)
            if epoch % self.callback_params.get("save_checkpoint_steps") == 0:
                evaluator.save_ckpt(self.g_solver)
        evaluator.summary()
