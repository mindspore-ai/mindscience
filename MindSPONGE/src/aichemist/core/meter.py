# Copyright 2021-2023 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of AIchemist package.
#
# The AIchemist is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""meter"""

import time
from collections import defaultdict
import mindspore as ms
import numpy as np
from ..utils import pretty


class Meter:
    """
    Meter for recording metrics and training progress.

    Args:
        log_interval (int, optional): log every n updates
        silent (int, optional): suppress all outputs or not
        logger (core.LoggerBase, optional): log handler
    """

    def __init__(self, log_interval=100, silent=False, logger=None):
        self.records = defaultdict(list)
        self.log_interval = log_interval
        self.epoch2batch = [0]
        self.time = [time.time()]
        self.epoch_id = 0
        self.batch_id = 0
        self.silent = silent
        self.logger = logger
        self.start_epoch = 0
        self.end_epoch = 0

    def __call__(self, num_epoch):
        self.start_epoch = self.epoch_id
        self.end_epoch = self.start_epoch + num_epoch

        for epoch in range(self.start_epoch, self.end_epoch):
            if not self.silent:
                self.logger.warning(pretty.SEP)
                self.logger.warning(f"Epoch {epoch} begin")
            yield epoch
            if not self.silent:
                self.logger.warning(pretty.SEP)
                self.logger.warning(f"Epoch {epoch} end")
            self.step()

    def log(self, record, category="train/batch"):
        """
        Log a record.

       Args:
            record (dict): dict of any metric
            category (str, optional): log category.
                Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``.
        """
        if self.silent:
            return

        if category.endswith("batch"):
            self.logger.warning('>' * 30)
        elif category.endswith("epoch"):
            self.logger.warning('-' * 30)
        if category == "train/epoch":
            for k in sorted(record.keys()):
                self.logger.warning(f"average {k}: {record[k]}")
        else:
            for k in sorted(record.keys()):
                self.logger.warning(f"{k}: {record[k]}")

    def log_config(self, config):
        """
        Log a hyperparameter config.

        Args:
            config (dict): hyperparameter config
        """
        if self.silent:
            return

        self.logger.log_config(config)

    def update(self, record):
        """
        Update the meter with a record.

        Args:
            record (dict): dict of any metric
        """
        if self.batch_id % self.log_interval == 0:
            self.log(record, category="train/batch")
        self.batch_id += 1

        for k, v in record.items():
            if isinstance(v, ms.Tensor):
                v = v.asnumpy()
            self.records[k].append(v)

    def step(self):
        """
        Step an epoch for this meter.

        Instead of manually invoking :meth:`step()`, it is suggested to use the following line
            >>> for epoch in meter(num_epoch):
            >>>     # do something
        """
        self.epoch_id += 1
        self.epoch2batch.append(self.batch_id)
        self.time.append(time.time())
        index = slice(self.epoch2batch[-2], self.epoch2batch[-1])
        duration = self.time[-1] - self.time[-2]
        speed = (self.epoch2batch[-1] - self.epoch2batch[-2]) / duration
        if self.silent:
            return

        self.logger.warning(f"duration: {pretty.time(duration)}")
        self.logger.warning(f"speed: {speed:.2f} batch / sec")

        eta = (self.time[-1] - self.time[self.start_epoch]) \
            / (self.epoch_id - self.start_epoch) * (self.end_epoch - self.epoch_id)
        self.logger.warning(f"ETA: {pretty.time(eta)}")

        record = {}
        for k, v in self.records.items():
            record[k] = np.mean(v[index])
        self.log(record, category="train/epoch")
