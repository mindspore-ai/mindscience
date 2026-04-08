# Copyright 2026 Huawei Technologies Co., Ltd
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
from abc import ABCMeta, abstractmethod


def config_to_str(cls, gap=2 * " "):
    """Return class attribute str for print."""
    attributes = vars(cls)
    print_str = "\n" + cls.__class__.__name__ + "\n"
    for name, val in attributes.items():
        new_str = str(val)
        new_str = new_str.replace("\n", "\n" + gap)
        print_str += f"{gap}{name}: {new_str}\n"

    return print_str


class BaseConfig(metaclass=ABCMeta):
    _validation_func_dict = {}

    @abstractmethod
    def __init__(self, config_name):
        self._config_name = config_name

    def __setattr__(self, name, value):
        validator_func = self._validation_func_dict.get(name)
        if validator_func is not None:
            value = validator_func(self, value)
        super().__setattr__(name, value)

    @classmethod
    def validator(cls, name):
        def decorator(func):
            cls._validation_func_dict[name] = func
            return func

        return decorator

    def update_attrs(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        return config_to_str(self)

    def get_config_name(self):
        return self._config_name
