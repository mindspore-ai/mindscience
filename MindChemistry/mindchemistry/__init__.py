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
# ============================================================================
"""initialization for mindchemistry APIs"""
import time
from .cell import *
from .utils import *
from .e3 import *


__all__ = []
__all__.extend(cell.__all__)
__all__.extend(utils.__all__)
__all__.extend(e3.__all__)


def _mindspore_version_check():
    """
       Do the MindSpore version check for MindChemistry. If the
       MindSpore can not be imported, it will raise ImportError. If its
       version is not compatibale with current MindChemistry verision,
       it will print a warning.

       Raise:
           ImportError: If the MindSpore can not be imported.
       """

    try:
        import mindspore as ms
        from mindspore import log as logger
    except ImportError:
        raise ImportError("Can not find MindSpore in current environment. Please install "
                          "MindSpore before using MindChemistry, by following "
                          "the instruction at https://www.mindspore.cn/install")

    ms_version = ms.__version__[:5]
    required_mindspore_verision = '1.8.1'

    if ms_version < required_mindspore_verision:
        logger.warning("Current version of MindSpore is not compatible with MindChemistry. "
                       "Some functions might not work or even raise error. Please install MindSpore "
                       "version >= {} For more details about dependency setting, please check "
                       "the instructions at MindSpore official website https://www.mindspore.cn/install "
                       "or check the README.md at https://gitee.com/mindspore/mindscience"
                       .format(required_mindspore_verision))
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(
                f"Please pay attention to the above warning, countdown: {i}")
            time.sleep(1)

_mindspore_version_check()
