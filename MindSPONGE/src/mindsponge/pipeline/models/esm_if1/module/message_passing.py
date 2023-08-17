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
"""Message passing"""
import re
import inspect
from inspect import Parameter
from typing import List, Optional, Any, Callable, Dict, Set, Tuple
from collections import OrderedDict
import pyparsing as pp
from mindspore import Tensor
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import Size


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


def gather(params, indices, axis=None):
    """Gather"""
    if axis is None:
        axis = 0
    if axis < 0:
        axis = len(params.shape) + axis
    if axis == 0:
        return params[indices]
    if axis == 1:
        return params[:, indices]
    if axis == 2:
        return params[:, :, indices]
    if axis == 3:
        return params[:, :, :, indices]
    raise ValueError("Unknown axis selected")


def broadcast(src: ms.Tensor, other: ms.Tensor, dim: int):
    """Broadcast"""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = ops.ExpandDims()(src, 0)
    for _ in range(src.dim(), other.dim()):
        src = ops.ExpandDims()(src, -1)
    src = src.expand_as(other)
    return src


def tensor_scatter_add(out, index, src, dim):
    """Tensor scatter add"""
    if dim < 0:
        dim = out.ndim + dim
    if out.ndim == 1:
        out = ops.Cast()(out, ms.float32)
        index = index.reshape(index.shape[0], 1)
        src = ops.Cast()(src, ms.float32)
        out = ops.scatter_nd_add(out, index, src)
    elif out.ndim == 2:
        if dim == 0:
            m = index.shape[0]
            n = index.shape[1]
            index_new = index[:, :].reshape(-1)[:, None]
            index_j = mnp.arange(n).astype(mnp.int32)[None,]
            index_j = mnp.tile(index_j, (m, 1)).reshape(-1)[:, None]
            index = mnp.concatenate((index_new, index_j), -1)  # m*n, 2
            src = src[:, :].reshape(-1)  # m*n,
            out = ops.tensor_scatter_add(out, index, src)
    return out


def scatter_sum(src: Tensor, index: Tensor, dim: int = -1,
                out: Optional[Tensor] = None,
                dim_size: Optional[int] = None) -> Tensor:
    """Scatter sum"""
    index = broadcast(index, src, dim)
    index = index.astype(ms.int32)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.size() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = ops.Zeros()(tuple(size), src.dtype)
        out = tensor_scatter_add(out, index, src, dim)
        return out
    out = tensor_scatter_add(out, index, src, dim)
    return out


def scatter_mean(src: ms.Tensor, index: ms.Tensor, dim: int = -1,
                 out: Optional[ms.Tensor] = None,
                 dim_size: Optional[int] = None) -> ms.Tensor:
    """Scatter mean"""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.shape[dim]

    index_dim = dim

    if index_dim < 0:
        index_dim = 0
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = ops.Ones()(tuple(index.shape), ms.int32)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    out = ms.numpy.true_divide(out, count)
    return out


class MessagePassing(nn.Cell):
    """Message passing class"""

    special_args = {
        'edge_index', 'x', 'edge_weight'
    }

    def __init__(self, flow: str = "source_to_target", node_dim=-2):
        super().__init__()
        self.flow = flow
        self.node_dim = node_dim

        self.inspector = Inspector(self)
        self.inspector.inspect(self.message)
        self.__user_args__ = \
            self.inspector.keys(['message',]).difference(self.special_args)

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            if edge_index.dtype != ms.int32:
                raise TypeError("'edge_index' data type should be int32")
            if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                raise ValueError("'edge_index' dims or shape[0] should be 2")
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    def __set_size__(self, size: List[Optional[int]], dim: int, src: Tensor):
        the_size = size[dim]
        if the_size is None:
            size[dim] = src.shape[self.node_dim]
        elif the_size != src.shape[self.node_dim]:
            raise ValueError(
                (f'Encountered tensor with size {src.shape[self.node_dim]} in '
                 f'dimension {self.node_dim}, but expected size {the_size}.'))

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.gather(index, self.node_dim)
        raise ValueError

    def __collect__(self, args, edge_index, size, kwargs):
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        out = {}
        for arg in args:
            if arg[-2:] not in ['_i', '_j']:
                out[arg] = kwargs.get(arg, Parameter.empty)
            else:
                dim = j if arg[-2:] == '_j' else i
                data = kwargs.get(arg[:-2], Parameter.empty)

                if isinstance(data, (tuple, list)):
                    if len(data) != 2:
                        raise ValueError("data length should be 2")
                    if isinstance(data[1 - dim], Tensor):
                        self.__set_size__(size, 1 - dim, data[1 - dim])
                    data = data[dim]

                if isinstance(data, Tensor):
                    self.__set_size__(size, dim, data)
                    data = self.__lift__(data, edge_index, dim)

                out[arg] = data

        if isinstance(edge_index, Tensor):
            out['adj_t'] = None
            out['edge_index'] = edge_index
            out['edge_index_i'] = edge_index[i]
            out['edge_index_j'] = edge_index[j]
            out['ptr'] = None

        out['index'] = out.get('edge_index_i', " ")
        out['size'] = size
        out['size_i'] = size[1] if size[1] is not None else size[0]
        out['size_j'] = size[0] if size[0] is not None else size[1]
        out['dim_size'] = out.get('size_i', " ")
        return out

    def message_gvp(self, x, edge_index, edge_weight=None):
        msg = gather(x, edge_index[0, :])
        if edge_weight is not None:
            edge_weight = ops.ExpandDims()(edge_weight, -1)
            return msg * edge_weight
        return msg

    def aggregate(self, msg, edge_index, num_nodes=None, aggr='mean'):
        dst_index = edge_index[1, :]
        if aggr == 'mean':
            return scatter_mean(msg, dst_index, dim=self.node_dim, dim_size=num_nodes)
        raise NotImplementedError('Not support for this opearator')

    def update(self, x):
        return x

    def propagate(self, x, edge_index, aggr='sum', size: Size = None, **kwargs):
        """Propagate"""
        if 'num_nodes' not in kwargs.keys() or kwargs.get('num_nodes', ' ') is None:
            kwargs['num_nodes'] = x.shape[0]
        size = self.__check_input__(edge_index, size)
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        msg = self.message(**msg_kwargs)
        if aggr == 'mean':
            x = self.aggregate(msg, edge_index, num_nodes=kwargs.get('num_nodes', ' '), aggr=aggr)
        x = self.update(x)
        return x
