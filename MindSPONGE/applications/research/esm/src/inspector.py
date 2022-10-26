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
"""Inspector"""

import inspect
import re
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import pyparsing as pp


def param_type_repr(param) -> str:
    """Return parameter type"""
    if param.annotation is inspect.Parameter.empty:
        return 'Tensor'
    return sanitize(re.split(r':|='.strip(), str(param))[1])


def split_types_repr(types_repr: str) -> List[str]:
    """Split type"""
    out = []
    i = depth = 0
    for j, char in enumerate(types_repr):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
        elif char == ',' and depth == 0:
            out.append(types_repr[i:j].strip())
            i = j + 1
    out.append(types_repr[i:].strip())
    return out


def return_type_repr(signature) -> str:
    """Return type"""
    return_type = signature.return_annotation
    if return_type is inspect.Parameter.empty:
        return 'torch.Tensor'
    if str(return_type)[:6] != '<class':
        return sanitize(str(return_type))
    if return_type.__module__ == 'builtins':
        return return_type.__name__
    return f'{return_type.__module__}.{return_type.__name__}'


def parse_types(func: Callable) -> List[Tuple[Dict[str, str], str]]:
    """Return parse type"""
    source = inspect.getsource(func)
    signature = inspect.signature(func)

    iterator = re.finditer(r'#\s*type:\s*\((.*)\)\s*->\s*(.*)\s*\n', source)
    matches = list(iterator)

    if matches:
        out = []
        args = list(signature.parameters.keys())
        for match in matches:
            arg_types_repr, return_type = match.groups()
            arg_types = split_types_repr(arg_types_repr)
            arg_types = OrderedDict((k, v) for k, v in zip(args, arg_types))
            return_type = return_type.split('#')[0].strip()
            out.append((arg_types, return_type))
        return out

    # Alternatively, parse annotations using the inspected signature.
    ps = signature.parameters
    arg_types = OrderedDict((k, param_type_repr(v)) for k, v in ps.items())
    return [(arg_types, return_type_repr(signature))]


def sanitize(type_repr: str):
    """Sanitize"""
    type_repr = re.sub(r'<class \'(.*)\'>', r'\1', type_repr)
    type_repr = type_repr.replace('typing.', '')
    type_repr = type_repr.replace('torch_sparse.tensor.', '')
    type_repr = type_repr.replace('Adj', 'Union[Tensor, SparseTensor]')

    # Replace `Union[..., NoneType]` by `Optional[...]`.
    sexp = pp.nestedExpr(opener='[', closer=']')
    tree = sexp.parseString(f'[{type_repr.replace(",", " ")}]').asList()[0]

    def union_to_optional_(tree):
        for i, _ in enumerate(tree):
            e, n = tree[i], tree[i + 1] if i + 1 < len(tree) else []
            if e == 'Union' and n[-1] == 'NoneType':
                tree[i] = 'Optional'
                tree[i + 1] = tree[i + 1][:-1]
            elif e == 'Union' and 'NoneType' in n:
                idx = n.index('NoneType')
                n[idx] = [n[idx - 1]]
                n[idx - 1] = 'Optional'
            elif isinstance(e, list):
                tree[i] = union_to_optional_(e)
        return tree

    tree = union_to_optional_(tree)
    type_repr = re.sub(r'\'|\"', '', str(tree)[1:-1]).replace(', [', '[')

    return type_repr


class Inspector:
    """Inspector"""

    def __init__(self, base_class: Any):
        self.base_class: Any = base_class
        self.params: Dict[str, Dict[str, Any]] = {}

    def __implements__(self, cls, func_name: str) -> bool:
        if cls.__name__ == 'MessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def inspect(self, func: Callable,
                pop_first: bool = False) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names: Optional[List[str]] = None) -> Set[str]:
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params.get(func, " ").keys()
        return set(keys)

    def implements(self, func_name: str) -> bool:
        return self.__implements__(self.base_class.__class__, func_name)

    def types(self, func_names: Optional[List[str]] = None) -> Dict[str, str]:
        """Return types"""

        out: Dict[str, str] = {}
        for func_name in func_names or list(self.params.keys()):
            func = getattr(self.base_class, func_name)
            arg_types = parse_types(func)[0][0]
            for key in self.params.get(func_name, " ").keys():
                if key in out and out.get(key, " ") != arg_types.get(key, " "):
                    raise ValueError(
                        (f'Found inconsistent types for argument {key}. '
                         f'Expected type {out.get(key, " ")} but found type '
                         f'{arg_types.get(key, " ")}.'))
                out[key] = arg_types.get(key, " ")
        return out

    def distribute(self, func_name, kwargs: Dict[str, Any]):
        """Distribute"""

        out = {}
        try:
            for key, param in self.params.get(func_name, " ").items():
                data = kwargs.get(key, inspect.Parameter.empty)
                if data is inspect.Parameter.empty:
                    data = param.default
                out[key] = data
        except KeyError:
            raise TypeError(f'Required parameter {key} is empty.')
        return out
