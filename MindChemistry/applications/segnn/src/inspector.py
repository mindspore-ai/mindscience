# Copyright 2024 Huawei Technologies Co., Ltd
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
Get func signature
"""
import re
import inspect
from collections import OrderedDict


class Inspector:
    """func signature get and handle"""
    def __init__(self, base_class):
        self.base_clas = base_class
        self.params = {}

    def inspect(self, func, pop_first=False):
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names=None):
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params[func].keys()
        return set(keys)

    def __implements__(self, cls, func_name):
        if cls.__name__ == 'MessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def implements(self, func_name):
        return self.__implements__(self.base_class.__class__, func_name)

    def distribute(self, func_name, kwargs):
        out = {}
        for key, param in self.params[func_name].items():
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out


def func_header_repr(func, keep_annotation=True):
    """func_header_repr"""
    source = inspect.getsource(func)
    signature = inspect.signature(func)

    if keep_annotation:
        return ''.join(re.split(r'(\).*?:.*?\n)', source,
                                maxsplit=1)[:2]).strip()

    params_repr = ['self']
    for param in signature.parameters.values():
        params_repr.append(param.name)
        if param.default is not inspect.Parameter.empty:
            params_repr[-1] += f'={param.default}'

    return f'def {func.__name__}({", ".join(params_repr)}):'


def func_body_repr(func, keep_annotation=True):
    """func_body_repr"""
    source = inspect.getsource(func)
    body_repr = re.split(r'\).*?:.*?\n', source, maxsplit=1)[1]
    if not keep_annotation:
        body_repr = re.sub(r'\s*# type:.*\n', '', body_repr)
    return body_repr
