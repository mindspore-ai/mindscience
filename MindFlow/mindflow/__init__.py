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
import re
import time


def _mindspore_version_check():
    """
       Do the MindSpore version check for MindFlow. If the
       MindSpore can not be imported, it will raise ImportError. If its
       version is not compatibale with current MindFlow verision,
       it will print a warning.

       Raise:
           ImportError: If the MindSpore can not be imported.
       """

    try:
        import mindspore as ms
        from mindspore import log as logger
    except ImportError:
        raise ImportError("Can not find MindSpore in current environment. Please install "
                          "MindSpore before using MindFlow, by following "
                          "the instruction at https://www.mindspore.cn/install")

    pattern = r'\d+\.\d+\.\d+'
    ms_version = re.match(pattern, ms.__version__)
    required_mindspore_version = '2.0.0'

    logger.info("Current Mindspore version is {}.".format(ms_version))

    if not ms_version or ms_version.group() < required_mindspore_version:
        logger.warning("Current version of MindSpore is not compatible with MindFlow. "
                       "Some functions might not work or even raise error. Please install MindSpore "
                       "version >= {} For more details about dependency setting, please check "
                       "the instructions at MindSpore official website https://www.mindspore.cn/install "
                       "or check the README.md at https://gitee.com/mindspore/mindscience"
                       .format(required_mindspore_version))
        warning_countdown = 3
        for i in range(warning_countdown, 0, -1):
            logger.warning(
                f"Please pay attention to the above warning, countdown: {i}")
            time.sleep(1)


_mindspore_version_check()

from .data import *
from .geometry import *
from .common import *
from .operators import *
from .pde import *
from .loss import *
from .cell import *
from .cfd import *
from .utils import *

__all__ = []
__all__.extend(data.__all__)
__all__.extend(geometry.__all__)
__all__.extend(common.__all__)
__all__.extend(operators.__all__)
__all__.extend(pde.__all__)
__all__.extend(loss.__all__)
__all__.extend(cell.__all__)
__all__.extend(cfd.__all__)
__all__.extend(utils.__all__)
