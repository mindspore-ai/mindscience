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
"""NowcastNet Trainer"""
import math

import numpy as np
import mindspore as ms
from mindspore import nn, Model
from mindspore.train.callback import LossMonitor, TimeMonitor

from .dataset import RadarData, NowcastDataset
from .callback import NowcastCallBack, EvolutionCallBack
from .forecast import GenerationPredictor


class GenerationTrainer:
    """
    Trainer class of generation module

    Args:
        config (dict): parameters for training.
        g_model (Cell): network of generator.
        d_model (Cell): network of discriminator.
        g_loss_fn (Cell): user-defined loss function of generator.
        d_loss_fn (Cell): user-defined loss function of discriminator.
        logger (logging.RootLogger): tools for logging.
        loss_scale (LossScaler): loss manager

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, config, g_model, d_model, g_loss_fn, d_loss_fn, logger, loss_scale):
        self.data_params = config.get("data")
        self.train_params = config.get("train")
        self.optimizer_params = config.get("optimizer-gen")
        self.summary_params = config.get("summary")
        self.model_params = config.get("model")
        self.noise_scale = self.data_params.get("noise_scale", 32)
        self.epochs = self.optimizer_params.get("epochs", 20)
        self.save_ckpt_epochs = self.summary_params.get("save_checkpoint_epochs", 2)
        self.w_size = self.data_params.get("w_size", 512)
        self.h_size = self.data_params.get("h_size", 512)
        self.ngf = self.model_params.get("ngf", 32)
        self.batch_size = self.data_params.get("batch_size", 1)
        self.pool_ensemble_num = self.data_params.get("pool_ensemble_num", 4)
        self.eval_interval = self.summary_params.get("eval_interval", 10)
        self.loss_scale = loss_scale
        self.g_model = g_model
        self.d_model = d_model
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn
        self.logger = logger
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.dataset_size = self.train_dataset.get_dataset_size()
        self.g_optimizer, self.d_optimizer = self.get_optimizer()
        self.g_solver, self.d_solver = self.get_solver()
        self.callback = NowcastCallBack(config,
                                        dataset_size=self.dataset_size,
                                        logger=self.logger)
        self.predictor = GenerationPredictor(config, self.g_model, logger)

    @staticmethod
    def _get_cosine_annealing_lr(lr_init, steps_per_epoch, epochs, eta_min=1e-6):
        """cosine annealing lr"""
        total_steps = epochs * steps_per_epoch
        delta = 0.5 * (lr_init - eta_min)
        lr = []
        try:
            for i in range(total_steps):
                tmp_epoch = min(math.floor(i / steps_per_epoch), epochs)
                lr.append(eta_min + delta * (1 + math.cos(math.pi * tmp_epoch / epochs)))
        except ZeroDivisionError:
            return lr
        return lr

    def get_dataset(self):
        """
        Get NowcastNet train and valid dataset.

        Returns:
            Dataset, train dataset.
            Dataset, valid dataset.
        """
        train_dataset_generator = RadarData(self.data_params, run_mode='train', module_name='generation')
        valid_dataset_generator = RadarData(self.data_params, run_mode='valid', module_name='generation')

        train_dataset = NowcastDataset(train_dataset_generator,
                                       module_name='generation',
                                       distribute=self.train_params.get('distribute'),
                                       num_workers=self.data_params.get('num_workers'))
        valid_dataset = NowcastDataset(valid_dataset_generator,
                                       module_name='generation',
                                       distribute=self.train_params.get('distribute'),
                                       num_workers=self.data_params.get('num_workers'),
                                       shuffle=False)
        train_dataset = train_dataset.create_dataset(self.data_params.get('batch_size'))
        valid_dataset = valid_dataset.create_dataset(self.data_params.get('batch_size'))
        return train_dataset, valid_dataset

    def get_optimizer(self):
        """
        Get the training optimizer.

        Returns:
            g_optimizer, optimizer of the generation module.
            d_optimizer, optimizer of the evolution module.
        """
        g_init_lr = float(self.optimizer_params.get("g_lr", 1.5e-5))
        d_init_lr = float(self.optimizer_params.get("d_lr", 6e-5))
        g_lr = self._get_cosine_annealing_lr(g_init_lr, self.dataset_size, self.optimizer_params.get("epochs"))
        d_lr = self._get_cosine_annealing_lr(d_init_lr, self.dataset_size, self.optimizer_params.get("epochs"))
        g_optimizer = nn.Adam(self.g_model.trainable_params(),
                              learning_rate=g_lr,
                              beta1=self.optimizer_params.get("beta1"),
                              beta2=self.optimizer_params.get("beta2"))
        d_optimizer = nn.Adam(self.d_model.trainable_params(),
                              learning_rate=d_lr,
                              beta1=self.optimizer_params.get("beta1"),
                              beta2=self.optimizer_params.get("beta2"))
        return g_optimizer, d_optimizer

    def get_solver(self):
        g_solver = nn.TrainOneStepWithLossScaleCell(self.g_loss_fn, self.g_optimizer, scale_sense=self.loss_scale)
        d_solver = nn.TrainOneStepWithLossScaleCell(self.d_loss_fn, self.d_optimizer, scale_sense=self.loss_scale)
        return g_solver, d_solver

    def train(self):
        """Train."""
        for epoch in range(self.epochs):
            self.g_solver.set_train(True)
            self.d_solver.set_train(True)
            epoch_g_loss, epoch_d_loss = 0.0, 0.0
            self.callback.epoch_start()
            for data in self.train_dataset.create_dict_iterator():
                self.callback.step_start()
                inp, evo_result, labels = data.get("inputs"), data.get("evo"), data.get("labels")
                g_noise = ms.Tensor(np.random.randn(self.batch_size,
                                                    self.ngf,
                                                    self.h_size // self.noise_scale,
                                                    self.w_size // self.noise_scale,
                                                    self.pool_ensemble_num + 1), inp.dtype)
                g_res = self.g_solver(inp, evo_result, g_noise, labels)
                d_noise = ms.Tensor(np.random.randn(self.batch_size,
                                                    self.ngf,
                                                    self.h_size // self.noise_scale,
                                                    self.w_size // self.noise_scale), inp.dtype)
                d_res = self.d_solver(inp, evo_result, d_noise, labels)
                self.callback.print_loss(g_res[0].asnumpy(), d_res[0].asnumpy(), step=True)
                epoch_g_loss += g_res[0].asnumpy()
                epoch_d_loss += d_res[0].asnumpy()
            try:
                epoch_g_loss = epoch_g_loss / self.dataset_size
                epoch_d_loss = epoch_d_loss / self.dataset_size
            except ZeroDivisionError:
                self.logger.info('dataset size is 0')
            self.callback.print_loss(epoch_g_loss, epoch_d_loss)
            if epoch % self.eval_interval == 0 or epoch == self.epochs - 1:
                self.predictor.eval(self.valid_dataset)
            if epoch % self.save_ckpt_epochs == 0 or epoch == self.epochs - 1:
                self.logger.info(f"saving the model at the end of epoch {epoch}")
                self.callback.save_generation_ckpt(self.g_solver)
            self.callback.epoch_end()
        self.callback.summary()


class EvolutionTrainer:
    """
    Trainer class of evolution module

    Args:
        config (dict): parameters for training.
        model (Cell): network of evolution.
        loss_fn (Cell): user-defined loss function of evolution.
        logger (logging.RootLogger): tools for logging.
        loss_scale (LossScaler): loss manager

    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, config, model, loss_fn, logger, loss_scale):
        self.config = config
        self.logger = logger
        self.model_params = config.get("model")
        self.data_params = config.get("data")
        self.train_params = config.get("train")
        self.optimizer_params = config.get("optimizer-evo")
        self.callback_params = config.get("summary")
        self.loss_scale = loss_scale
        self.model = model
        self.loss_fn = loss_fn
        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.dataset_size = self.train_dataset.get_dataset_size()
        self.optimizer = self.get_optimizer()
        self.solver = self.get_solver()
        self.pred_cb = self.get_callback()
        self.ckpt_cb = self.pred_cb.save_evolution_ckpt()

    def get_dataset(self):
        """
        Get train and valid dataset.

        Args:
            process (bool, optional): Whether to process the dataset.

        Returns:
            Dataset, train dataset.
            Dataset, valid dataset.
        """
        train_dataset_generator = RadarData(self.data_params, run_mode='train', module_name='evolution')
        valid_dataset_generator = RadarData(self.data_params, run_mode='valid', module_name='evolution')

        train_dataset = NowcastDataset(train_dataset_generator,
                                       module_name='evolution',
                                       distribute=self.train_params.get('distribute', False),
                                       num_workers=self.data_params.get('num_workers', 1))
        valid_dataset = NowcastDataset(valid_dataset_generator,
                                       module_name='evolution',
                                       distribute=self.train_params.get('distribute', False),
                                       num_workers=self.data_params.get('num_workers', 1),
                                       shuffle=False)
        train_dataset = train_dataset.create_dataset(self.data_params.get('batch_size', 8))
        valid_dataset = valid_dataset.create_dataset(self.data_params.get('batch_size', 8))
        return train_dataset, valid_dataset

    def get_optimizer(self):
        """
        Get the training optimizer.

        Returns:
            Optimizer, Optimizer of the model.
        """
        lr = self.optimizer_params.get('lr')
        optimizer = nn.Adam(self.model.trainable_params(),
                            learning_rate=lr,
                            weight_decay=self.optimizer_params.get('weight_decay', 0.1)
                            )
        return optimizer

    def get_solver(self):
        """
        define the solver of the model, abstract method.
        """
        solver = Model(network=self.loss_fn,
                       optimizer=self.optimizer,
                       loss_scale_manager=self.loss_scale,
                       )
        return solver

    def get_callback(self):
        """
        define the callback of the model, abstract method.
        """
        pred_cb = EvolutionCallBack(self.model, self.valid_dataset, self.config, self.logger)
        return pred_cb

    def train(self):
        """Train."""
        callback_lst = [LossMonitor(), TimeMonitor(), self.pred_cb, self.ckpt_cb]
        self.solver.train(epoch=self.optimizer_params.get("epochs", 200),
                          train_dataset=self.train_dataset,
                          callbacks=callback_lst,
                          dataset_sink_mode=self.data_params.get('data_sink'))
