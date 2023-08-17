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
"""Logger"""

import logging
from abc import ABC, abstractmethod
from ..configs import Registry as R
from ..utils import pretty


class LoggerBase(ABC):
    """
    Base class for loggers.
    Any custom logger should be derived from this class.
    """

    @abstractmethod
    def log(self, record, category="train/batch"):
        """
        Log a record.

        Args:
            record (dict): dict of any metric
            step_id (int): index of this log step
            category (str, optional): log category.
                Available types are ``train/batch``, ``train/epoch``, ``valid/epoch`` and ``test/epoch``.
        """
        raise NotImplementedError

    @abstractmethod
    def log_config(self, config):
        """
        Log a hyperparameter config.

        Args:
            config (dict): hyperparameter config
        """
        raise NotImplementedError


@R.register("core.LoggingLogger")
class LoggingLogger(LoggerBase):
    """
    Log outputs with the builtin logging module of Python.
    By default, the logs will be printed to the console. To additionally log outputs to a file,
    add the following lines in the beginning of your code.

    Examples:
        >>> import logging
        >>> format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")
        >>> handler = logging.FileHandler("log.txt")
        >>> handler.setFormatter(format)
        >>> logger = logging.getLogger("")
        >>> logger.addHandler(handler)
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def log(self, record: dict, category: str = "train/batch"):
        """log output

        Args:
            record (dict): The content of the logs.
            category (str, optional): The category of the logs. Defaults to "train/batch".
        """
        if category.endswith("batch"):
            self.logger.warning('>' * 30)
        elif category.endswith("epoch"):
            self.logger.warning('-' * 30)
        if category == "train/epoch":
            for k in sorted(record.keys()):
                self.logger.warning("average %d: %d", k, record[k])
        else:
            for k in sorted(record.keys()):
                self.logger.warning("%d: %d", k, record[k])

    def log_config(self, config):
        self.logger.warning(pretty.format(config, compact=True))
