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
"""Register for network model functions"""
from sciai.utils import print_log


class FunctionType:
    """function types for each network model"""
    def __init__(self):
        pass

    TRAINER = 'trainer'
    VAL = 'validation'
    TRAIN_PREPARE = 'train_prepare'
    EVAL_PREPARE = 'eval_prepare'


class Register:
    """Register class"""
    def __init__(self):
        pass

    dict_functions = {FunctionType.TRAINER: {}, FunctionType.VAL: {}, FunctionType.TRAIN_PREPARE: {}}

    @classmethod
    def register(cls, target, function_type=FunctionType.TRAINER):
        """wrapper fot registration"""
        def register_item(func_type, key, value):
            cls.dict_functions[func_type][key] = value
            return value

        if function_type not in cls.dict_functions.keys():
            raise ValueError(f"unsupported function type: {function_type}")

        if callable(target):
            return register_item(function_type, target.__name__, target)
        if isinstance(target, str):
            return lambda x: register_item(function_type, target, x)
        raise ValueError("target should be function or string")

    @classmethod
    def get_functions(cls, model_name: str, function_type=FunctionType.TRAINER):
        """return the function according to model name and function type"""
        if function_type not in cls.dict_functions.keys():
            print_log("The available function types:")
            for key in cls.dict_functions:
                print_log(key)
            raise ValueError("Invalid function type, check the available model names above")

        if model_name not in cls.dict_functions.get(function_type).keys():
            print_log("The available networks:")
            for key in cls.dict_functions.get(function_type):
                print_log(key)
            raise ValueError("Invalid model name, check the available model names above")

        return cls.dict_functions.get(function_type).get(model_name)
