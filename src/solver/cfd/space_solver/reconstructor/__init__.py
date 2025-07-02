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
"""init of reconstructor."""
from .weno3 import WENO3
from .weno5 import WENO5
from .weno7 import WENO7

_reconstructor_dict = {
    'WENO3': WENO3,
    'WENO5': WENO5,
    'WENO7': WENO7,
}


def define_reconstructor(name):
    """Define interpolator according to interpolator configuration"""
    ret = _reconstructor_dict.get(name)
    if ret is None:
        err = "reconstructor {} has not been implied".format(name)
        raise NameError(err)
    return ret
