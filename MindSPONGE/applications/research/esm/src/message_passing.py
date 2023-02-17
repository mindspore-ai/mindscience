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

from inspect import Parameter
from typing import List, Optional
from src.inspector import Inspector
from mindspore import Tensor
import mindspore as ms
from mindspore import ops
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.ops import Size


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
            assert edge_index.dtype == ms.int32
            assert edge_index.dim() == 2
            assert edge_index.shape[0] == 2
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
                    assert len(data) == 2
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
