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
"""logger"""
import logging


def create_logger(path="./log.log", level=logging.INFO):
    """
    Create logger

    Args:
        path (str): Directory to save log. Default: ``"./log.log"`` .
        level (int): Log level. Default: ``logging.INFO`` .

    Returns:
        logging.RootLogger. Logger for log information.

    Supported Platforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> from mindearth.utils import create_logger
        >>> logger = create_logger("/path/to/log.log")
    """
    logger = logging.getLogger()
    logger.setLevel(level)

    logfile = path
    fh = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    fh.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
