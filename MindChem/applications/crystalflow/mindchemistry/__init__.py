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
"""Initialization for MindChemistry APIs."""

import time
import mindspore as ms
from mindspore import log as logger
from mindscience.e3nn import *
from .graph import *

def _mindspore_version_check():
    """
    Check MindSpore version for MindChemistry.

    Raises:
        ImportError: If MindSpore cannot be imported.
    """
    try:
        _ = ms.__version__
    except ImportError as exc:
        raise ImportError(
            "Cannot find MindSpore in the current environment. Please install "
            "MindSpore before using MindChemistry, by following the instruction at "
            "https://www.mindspore.cn/install"
        ) from exc

    ms_version = ms.__version__[:5]
    required_mindspore_version = "1.8.1"

    if ms_version < required_mindspore_version:
        logger.warning(
            f"Current version of MindSpore ({ms_version}) is not compatible with MindChemistry. "
            f"Some functions might not work or even raise errors. Please install MindSpore "
            f"version >= {required_mindspore_version}. For more details about dependency settings, "
            f"please check the instructions at the MindSpore official website "
            f"https://www.mindspore.cn/install or check the README.md at "
            f"https://gitee.com/mindspore/mindscience"
        )

        for i in range(3, 0, -1):
            logger.warning(f"Please pay attention to the above warning, countdown: {i}")
            time.sleep(1)


_mindspore_version_check()
