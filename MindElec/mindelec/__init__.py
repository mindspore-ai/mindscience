# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
init
"""
from distutils.version import LooseVersion
from .solver import *
from .common import *

__all__ = []
__all__.extend(solver.__all__)
__all__.extend(common.__all__)


def _mindspore_version_check():
    """
    Do the MindSpore version check for MindSpore Elec. If the
    MindSpore can not be imported, it will raise ImportError. If its
    version is not compatibale with current MindSpore Elec verision,
    it will print a warning.

    Raise:
        ImportError: If the MindSpore can not be imported.
    """

    try:
        import mindspore as ms
        from mindspore import log as logger
        import time
    except (ImportError, ModuleNotFoundError):
        print("Can not find MindSpore in current environment. Please install "
              "MindSpore before using MindSpore Elec, by following "
              "the instruction at https://www.mindspore.cn/install")
        raise

    ms_version = ms.__version__
    ms_requires = '2.0.0'
    logger.info("Current Mindspore version is {} ".format(ms_version))

    if LooseVersion(ms_version) < LooseVersion(ms_requires):
        logger.warning("Current version of MindSpore is not compatible with MindSpore Elec. "
                       "Some functions might not work or even raise error. Please install MindSpore "
                       f"version >= {ms_requires}. For more details about dependency setting, please check "
                       "the instructions at MindSpore official website https://www.mindspore.cn/install "
                       "or check the README.md at https://gitee.com/mindspore/mindscience/tree/master/MindElec")
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(
                f"Please pay attention to the above warning, countdown: {i}")
            time.sleep(1)


_mindspore_version_check()
