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
"""FuXiTrainer"""
from mindspore import Model
from mindspore import amp
from mindearth.module import Trainer

from .callback import EvaluateCallBack

class FuXiTrainer(Trainer):
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
        super().__init__(config, model, loss_fn, logger)
        self.pred_cb = self.get_callback()
        self.solver = self.get_solver()

    def get_solver(self):
        """
        define the solver of the model, abstract method.
        """
        solver = Model(network=self.loss_fn,
                       optimizer=self.optimizer,
                       loss_scale_manager=amp.DynamicLossScaleManager(2 ** 24, 2, 2000),
                       amp_level='O2'
                       )
        return solver

    def get_callback(self):
        """
        define the callback of the model, abstract method.
        """
        pred_cb = EvaluateCallBack(self.model, self.valid_dataset_generator, self.config, self.logger)
        return pred_cb
