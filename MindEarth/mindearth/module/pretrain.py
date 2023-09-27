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
# ==============================================================================
"""Trainer"""

import os

from mindspore import nn, Model, Tensor
from mindspore.train.callback import LossMonitor, TimeMonitor, CheckpointConfig, ModelCheckpoint
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
import mindspore.communication.management as D

from ..utils import make_dir
from ..data import Era5Data, DemData, Dataset
from ..core import get_warmup_cosine_annealing_lr

# MindSpore 2.0 has changed the APIs of _checkparam, the following try except is for compatibility
try:
    from mindspore._checkparam import Validator as validator
except ImportError:
    import mindspore._checkparam as validator

_optimizer_metric = {
    'adam': nn.Adam,
    'adamw': nn.AdamWeightDecay,
    'sgd': nn.SGD
}


class Trainer:
    """
    Base class of Weather Forecast model training.
    All user-define forecast model should be inherited from this class during training.
    This class generates datasets, optimizer, callbacks, and solver components based on the input model, loss function,
    and related configurations. For example, if you want to train your model, you can rewrite the get_dataset(),
    get_optimizer(), or other member functions to suit your needs, or instantiate the class directly.
    Then you can use the Trainer.train() function to start model training.

    Args:
        config (dict): configurations of model, dataset, train details, etc.
        model (mindspore.nn.Cell): network for training.
        loss_fn (mindspore.nn.Cell): loss function.
        logger (logging.RootLogger, optional): logger of the training process. Default: None.
        weatherdata_type (str, optional): the dataset type. Default: 'Era5Data'.
        loss_scale (mindspore.amp.LossScaleManager, optional): the class of loss scale manager when using mixed
            precision. Default: mindspore.amp.DynamicLossScaleManager().

    Raises:
        TypeError: If `model` or `loss_fn` is not mindspore.nn.Cell.
        NotImplementedError: If the member function `get_callback` is not implemented.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, nn
        >>> from mindearth.module import Trainer
        >>> from mindearth.core import RelativeRMSELoss
        ...
        >>> class Net(nn.Cell):
        >>>     def __init__(self, input_dim, output_dim):
        >>>         super(Net, self).__init__()
        >>>         self.fc1 = nn.Dense(input_dim, 128, weight_init='ones')
        >>>         self.fc2 = nn.Dense(128, output_dim, weight_init='ones')
        ...
        >>>     def construct(self, x):
        >>>         x = x.transpose(0, 2, 3, 1)
        >>>         x = self.fc1(x)
        >>>         x = self.fc2(x)
        >>>         x = x.transpose(0, 3, 1, 2)
        >>>         return x
        ...
        >>> loss_fn = RelativeRMSELoss()
        >>> config={
        ...     "model": {
        ...         'name': 'Net'
        ...     },
        ...     "data": {
        ...         'name': 'era5',
        ...         'root_dir': './dataset',
        ...         'feature_dims': 69,
        ...         't_in': 1,
        ...         't_out_train': 1,
        ...         't_out_valid': 20,
        ...         't_out_test': 20,
        ...         'train_interval': 1,
        ...         'valid_interval': 1,
        ...         'test_interval': 1,
        ...         'pred_lead_time': 6,
        ...         'data_frequency': 6,
        ...         'train_period': [2015, 2015],
        ...         'valid_period': [2016, 2016],
        ...         'test_period': [2017, 2017],
        ...         'patch': True,
        ...         'patch_size': 8,
        ...         'batch_size': 8,
        ...         'num_workers': 1,
        ...         'grid_resolution': 1.4,
        ...         'h_size': 128,
        ...         'w_size': 256
        ...     },
        ...     "optimizer": {
        ...         'name': 'adam',
        ...         'weight_decay': 0.0,
        ...         'epochs': 200,
        ...         'finetune_epochs': 1,
        ...         'warmup_epochs': 1,
        ...         'initial_lr': 0.0005
        ...     },
        ...     "summary": {
        ...         'save_checkpoint_steps': 1,
        ...         'keep_checkpoint_max': 10,
        ...         'valid_frequency': 10,
        ...         'summary_dir': '/path/to/summary',
        ...         'ckpt_path': '.'
        ...     },
        ...     "train": {
        ...         'name': 'oop',
        ...         'distribute': False,
        ...         'device_id': 2,
        ...         'amp_level': 'O2',
        ...         'run_mode': 'test',
        ...         'load_ckpt': False
        ...     }
        ... }
        ...
        >>> model = Net(input_dim = config['data']['feature_dims'], output_dim = config['data']['feature_dims'])
        >>> trainer = Trainer(config, model, loss_fn)
        >>> trainer.train()
    """

    def __init__(self,
                 config,
                 model,
                 loss_fn,
                 logger=None,
                 weather_data_source="ERA5",
                 loss_scale=DynamicLossScaleManager()
                 ):

        validator.check_value_type("model", model, nn.Cell)
        validator.check_value_type("loss_fn", loss_fn, nn.Cell)

        self.config = config
        self.model_params = config.get("model")
        self.data_params = config.get("data")
        self.train_params = config.get("train")
        self.optimizer_params = config.get("optimizer")
        self.callback_params = config.get("summary")
        self.step = self.data_params.get("t_out_train")

        self.logger = logger
        self.model = model
        self.loss_fn = loss_fn
        self.weather_data_source = weather_data_source
        self.loss_scale = loss_scale

        self.train_dataset, self.valid_dataset = self.get_dataset()
        self.optimizer = self.get_optimizer()
        self.ckpt_cb = self.get_checkpoint()
        self.pred_cb = self.get_callback()
        self.solver = self.get_solver()

    def get_dataset(self):
        """
        Get train and valid dataset.

        Returns:
            Dataset, train dataset.
            Dataset, valid dataset.
        """
        if self.weather_data_source == 'ERA5':
            train_dataset_generator = Era5Data(data_params=self.data_params, run_mode='train')
            valid_dataset_generator = Era5Data(data_params=self.data_params, run_mode='valid')
        elif self.weather_data_source == 'DemSR':
            train_dataset_generator = DemData(data_params=self.data_params, run_mode='train')
            valid_dataset_generator = DemData(data_params=self.data_params, run_mode='valid')
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

    def get_optimizer(self):
        """
        Get the training optimizer.

        Returns:
            Optimizer, Optimizer of the model.
        """
        self.steps_per_epoch = self.train_dataset.get_dataset_size()
        if self.logger:
            self.logger.info(f'steps_per_epoch: {self.steps_per_epoch}')
        if self.step == 1:
            lr = get_warmup_cosine_annealing_lr(self.optimizer_params.get('initial_lr'), self.steps_per_epoch,
                                                self.optimizer_params.get("epochs"),
                                                warmup_epochs=self.optimizer_params.get("warmup_epochs"))
        else:
            lr = self.optimizer_params.get('finetune_lr')

        if self.optimizer_params.get('name'):
            optimizer = _optimizer_metric.get(self.optimizer_params.get('name'))(self.model.trainable_params(),
                                                                                 learning_rate=Tensor(lr),
                                                                                 weight_decay=self.optimizer_params.get(
                                                                                     'weight_decay'))
        else:
            raise NotImplementedError(f"{self.optimizer_params.get('name')} not implemented")
        return optimizer

    def get_checkpoint(self):
        """
        Get the checkpoint callback of the model.

        Returns:
            Callback, The checkpoint callback of the model.
        """
        ckpt_file_name = "ckpt/step_{}".format(self.step)
        ckpt_dir = os.path.join(self.callback_params.get('summary_dir'), ckpt_file_name)
        make_dir(ckpt_dir)
        model_name = self.model_params.get('name')
        if self.train_params.get('distribute'):
            rank_id = D.get_rank()
            ckpt_name = "{}-device{}".format(model_name, rank_id)
        else:
            ckpt_name = model_name
        ckpt_config = CheckpointConfig(
            save_checkpoint_steps=self.callback_params.get("save_checkpoint_steps") * self.steps_per_epoch,
            keep_checkpoint_max=self.callback_params.get("keep_checkpoint_max"))
        ckpt_cb = ModelCheckpoint(prefix=ckpt_name, directory=ckpt_dir, config=ckpt_config)
        return ckpt_cb

    def get_callback(self):
        """
        Used to build a Callback class. You can use this mechanism to do some custom operations.
        """
        raise NotImplementedError("get_callback not implemented")

    def get_solver(self):
        """
        Get the model solver for training.

        Returns:
            Model, the model solver for training.
        """
        solver = Model(self.model,
                       optimizer=self.optimizer,
                       loss_scale_manager=self.loss_scale,
                       loss_fn=self.loss_fn,
                       amp_level=self.train_params.get('amp_level'),
                       )
        return solver

    def train(self):
        """ train """
        callback_lst = [LossMonitor(), TimeMonitor()]
        if self.pred_cb:
            callback_lst.append(self.pred_cb)
        if not self.train_params.get('distribute') or D.get_rank() == 0:
            callback_lst.append(self.ckpt_cb)
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
