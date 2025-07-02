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
"""init of derivative computer"""
from .forth_order_face_derivative_computer import FourthOrderFaceDerivativeComputer
from .forth_order_central_derivative_computer import FourthOrderCentralDerivativeComputer

_derivative_dict = {
    'FourthOrderFaceDerivativeComputer': FourthOrderFaceDerivativeComputer,
    'FourthOrderCentralDerivativeComputer': FourthOrderCentralDerivativeComputer,
}


def define_derivative_computer(name):
    """Define derivative computer according to derivative computer configuration"""
    ret = _derivative_dict.get(name)
    if ret is None:
        err = "derivative {} has not been implied".format(name)
        raise NameError(err)
    return ret
