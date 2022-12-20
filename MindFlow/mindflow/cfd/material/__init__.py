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
"""init of material"""
from .ideal_gas import IdealGas

_material_dict = {
    'IdealGas': IdealGas,
}


def define_material(config):
    """Define boundary according to boundary configuration"""
    ret = _material_dict.get(config["type"])
    if ret is None:
        err = "material {} has not been implied".format(config["type"])
        raise NameError(err)
    return ret(config)
