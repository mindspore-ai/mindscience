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
"""GVP operations, will be used in gvp_encoder.py"""

from message_passing import scatter_sum, MessagePassing
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn


def ms_transpose(x, index_a, index_b):
    """Transpose"""
    index = list(i for i in range(len(x.shape)))
    index[index_a] = index_b
    index[index_b] = index_a
    input_trans = x.transpose(index)
    return input_trans


def tuple_size(tp):
    """Return tuple size"""
    return tuple([0 if a is None else a.size() for a in tp])


def tuple_sum(tp1, tp2):
    """Return the sum of tuple"""
    s1, v1 = tp1
    s2, v2 = tp2
    if v2 is None and v2 is None:
        return (s1 + s2, None)
    return (s1 + s2, v1 + v2)


def tuple_cat(*args, dim=-1):
    """Return the concat of tuple"""
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    concat_op = ops.Concat(axis=dim)
    return concat_op(s_args), concat_op(v_args)


def tuple_index(x, idx):
    """Return the index of tuple"""
    return x[0][idx], x[1][idx]


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    square = ops.Square()
    ms_sum = ops.ReduceSum(keep_dims=keepdims)
    sqrt_1 = ops.Sqrt()
    out = ms_sum(square(x), axis) + eps
    return sqrt_1(out) if sqrt else out


def _split(x, nv):
    """Split"""
    reshape = ops.Reshape()
    v = reshape(x[..., -3 * nv:], x.shape[:-1] + (nv, 3))
    s = x[..., :-3 * nv]
    return s, v


def _merge(s, v):
    """Merge"""
    reshape = ops.Reshape()
    v = reshape(v, v.shape[:-2] + (3 * v.shape[-2],))
    concat_op = ops.Concat(axis=-1)
    a = concat_op((s, v))
    return a


class GVP(nn.Cell):
    """GVP"""

    def __init__(self, in_dims, out_dims, h_dim=None, vector_gate=False,
                 activations=(ops.ReLU(), ops.Sigmoid()), tuple_io=True,
                 eps=1e-8):
        super(GVP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.tuple_io = tuple_io
        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Dense(self.vi, self.h_dim, has_bias=False)
            self.ws = nn.Dense(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Dense(self.h_dim, self.vo, has_bias=False)
                if vector_gate:
                    self.wg = nn.Dense(self.so, self.vo)
        else:
            self.ws = nn.Dense(self.si, self.so)

        self.vector_gate = vector_gate
        self.scalar_act, self.vector_act = activations
        self.eps = eps

    def construct(self, x):
        """GVP construction"""

        if self.vi:
            s, v = x
            v = ms_transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2, eps=self.eps)
            concat_op = ops.Concat(axis=-1)
            s = self.ws(concat_op((s, vn)))
            if self.scalar_act:
                s = self.scalar_act(s)
            if self.vo:
                v = self.wv(vh)
                v = ms_transpose(v, -1, -2)
                if self.vector_gate:
                    unsqueeze = ops.ExpandDims()
                    g = unsqueeze(self.wg(s), -1)
                else:
                    g = _norm_no_nan(v, axis=-1, keepdims=True, eps=self.eps)
                if self.vector_act:
                    g = self.vector_act(g)
                    v = v * g
        else:
            if self.tuple_io:
                assert x[1] is None
                x = x[0]
            s = self.ws(x)
            if self.scalar_act:
                s = self.scalar_act(s)
            if self.vo:
                zeros = ops.Zeros()
                v = zeros(list(s.shape)[:-1] + [self.vo, 3])

        if self.vo:
            return (s, v)
        if self.tuple_io:
            return (s, None)
        return s


class _VDropout(nn.Cell):
    """Dropout"""

    def __init__(self, drop_rate):
        super(_VDropout, self).__init__()
        self.drop_rate = drop_rate

    def construct(self, x):
        """Dropout construction"""

        if x is None:
            return None
        if not self.training:
            return x

        ones = ops.Ones()
        mask = ops.bernoulli(
            (1 - self.drop_rate) * ones(x.shape[:-1], x.dtype))
        unsqueeze = ops.ExpandDims()
        mask = unsqueeze(mask, -1)
        x = mask * x / (1 - self.drop_rate)
        return x


class Dropout(nn.Cell):
    """Dropout"""

    def __init__(self, drop_rate):
        super(Dropout, self).__init__()
        self.sdropout = nn.Dropout(1 - drop_rate)
        self.vdropout = _VDropout(1 - drop_rate)

    def construct(self, x):
        if isinstance(x, ms.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Cell):
    """Layer normalization"""

    def __init__(self, dims, tuple_io=True, eps=1e-8):
        super(LayerNorm, self).__init__()
        self.tuple_io = tuple_io
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm([self.s])
        self.eps = eps

    def construct(self, x):
        """Layer normalization construction"""

        if not self.v:
            if self.tuple_io:
                return self.scalar_norm(x[0]), None
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False, eps=self.eps)
        nonzero_mask = (vn > 2 * self.eps)
        vn = (vn * nonzero_mask)
        nonzero_mask = ms.ops.Cast()(nonzero_mask, ms.float32)
        v_1 = ops.ReduceSum(keep_dims=True)(vn, axis=-2)
        v_2 = self.eps + ops.ReduceSum(keep_dims=True)(nonzero_mask, axis=-2)
        vn = v_1 / v_2
        sqrt = ops.Sqrt()
        vn = sqrt(vn + self.eps)
        v = nonzero_mask * (v / vn)
        return self.scalar_norm(s), v


class GVPConv(MessagePassing):
    """GVP Convolution"""

    def __init__(self, in_dims, out_dims, edge_dims, n_layers=3,
                 vector_gate=False, module_list=None, aggr="mean", eps=1e-8,
                 activations=(ops.ReLU(), ops.Sigmoid())):
        super(GVPConv, self).__init__()
        self.eps = eps
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.se, self.ve = edge_dims
        self.aggr = aggr

        module_list = module_list or []
        if not module_list:
            if n_layers == 1:
                module_list.append(
                    GVP((2 * self.si + self.se, 2 * self.vi + self.ve),
                        (self.so, self.vo), activations=(None, None)))
            else:
                module_list.append(
                    GVP((2 * self.si + self.se, 2 * self.vi + self.ve), out_dims,
                        vector_gate=vector_gate, activations=activations)
                )
                for _ in range(n_layers - 2):
                    module_list.append(GVP(out_dims, out_dims,
                                           vector_gate=vector_gate))
                module_list.append(GVP(out_dims, out_dims,
                                       activations=(None, None)))
        self.message_func = nn.SequentialCell(*module_list)

    def construct(self, x, edge_index, edge_attr):
        x_s, x_v = x
        message = self.propagate(x_s, edge_index, s=x_s, v=x_v.reshape(x_v.shape[0], 3 * x_v.shape[1]),
                                 edge_attr=edge_attr, aggr=self.aggr)
        return _split(message, self.vo)

    def message(self, s_i, v_i, s_j, v_j, edge_attr):
        v_j = v_j.view(v_j.shape[0], v_j.shape[1] // 3, 3)
        v_i = v_i.view(v_i.shape[0], v_i.shape[1] // 3, 3)
        message = tuple_cat((s_j, v_j), edge_attr, (s_i, v_i))
        message = self.message_func(message)
        return _merge(*message)


class GVPConvLayer(nn.Cell):
    """GVP Convolution layer"""

    def __init__(self, node_dims, edge_dims, vector_gate=False,
                 n_message=3, n_feedforward=2, drop_rate=.1,
                 autoregressive=False, attention_heads=0,
                 conv_activations=(ops.ReLU(), ops.Sigmoid()),
                 n_edge_gvps=0, layernorm=True, eps=1e-8):

        super(GVPConvLayer, self).__init__()
        if attention_heads == 0:
            self.conv = GVPConv(
                node_dims, node_dims, edge_dims, n_layers=n_message,
                vector_gate=vector_gate,
                aggr="add" if autoregressive else "mean",
                activations=conv_activations,
                eps=eps,
            )
        else:
            raise NotImplementedError
        if layernorm:
            self.norm = nn.CellList([LayerNorm(node_dims, eps=eps) for _ in range(2)])
        else:
            self.norm = nn.CellList([nn.Identity() for _ in range(2)])
        self.dropout = nn.CellList([Dropout(drop_rate) for _ in range(2)])

        ff_func = []
        if n_feedforward == 1:
            ff_func.append(GVP(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = 4 * node_dims[0], 2 * node_dims[1]
            ff_func.append(GVP(node_dims, hid_dims, vector_gate=vector_gate))
            for _ in range(n_feedforward - 2):
                ff_func.append(GVP(hid_dims, hid_dims, vector_gate=vector_gate))
            ff_func.append(GVP(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.SequentialCell(*ff_func)

        self.edge_message_func = None
        if n_edge_gvps > 0:
            si, vi = node_dims
            se, ve = edge_dims
            module_list = [
                GVP((2 * si + se, 2 * vi + ve), edge_dims, vector_gate=vector_gate)
            ]
            for _ in range(n_edge_gvps - 2):
                module_list.append(GVP(edge_dims, edge_dims,
                                       vector_gate=vector_gate))
            if n_edge_gvps > 1:
                module_list.append(GVP(edge_dims, edge_dims,
                                       activations=(None, None)))
            self.edge_message_func = nn.SequentialCell(*module_list)
            if layernorm:
                self.edge_norm = LayerNorm(edge_dims, eps=eps)
            else:
                self.edge_norm = nn.Identity()
            self.edge_dropout = Dropout(drop_rate)

    def construct(self, x, edge_index, edge_attr,
                  autoregressive_x=None, node_mask=None):
        """GVP Convolution layer construction"""

        if self.edge_message_func:
            src, dst = edge_index
            if autoregressive_x is None:
                x_src = x[0][src], x[1][src]
            else:
                unsqueeze = ops.ExpandDims()
                mask = (src < dst)
                mask = unsqueeze(mask, -1)
                x_src = (
                    ms.numpy.where(mask, x[0][src], autoregressive_x[0][src]),
                    ms.numpy.where(unsqueeze(mask, -1), x[1][src],
                                   autoregressive_x[1][src])
                )
            x_dst = x[0][dst], x[1][dst]

            x_edge = (
                ops.Concat(axis=-1)([x_src[0], edge_attr[0], x_dst[0]]),
                ops.Concat(axis=-2)([x_src[1], edge_attr[1], x_dst[1]])
            )
            edge_attr_dh = self.edge_message_func(x_edge)
            edge_attr = self.edge_norm(tuple_sum(edge_attr,
                                                 self.edge_dropout(edge_attr_dh)))

        if autoregressive_x is not None:
            src, dst = edge_index
            mask = src < dst
            edge_index_forward = edge_index[:, mask]
            edge_index_backward = edge_index[:, ~mask]
            edge_attr_forward = tuple_index(edge_attr, mask)
            edge_attr_backward = tuple_index(edge_attr, ~mask)

            dh = tuple_sum(
                self.conv(x, edge_index_forward, edge_attr_forward),
                self.conv(autoregressive_x, edge_index_backward, edge_attr_backward)
            )
            unsqueeze = ops.ExpandDims()

            src = ops.OnesLike()(dst)
            index = ms.Tensor(dst, ms.int32)
            count = scatter_sum(src, index, dim_size=dh[0].shape[0])

            min_value = ms.Tensor(1, ms.float32)
            count = ops.clip_by_value(count, clip_value_min=min_value)
            count = unsqueeze(count, -1)

            dh = dh[0] / count, unsqueeze((dh[1] / count), -1)


        else:
            dh = self.conv(x, edge_index, edge_attr)

        if node_mask is not None:
            x_ = x
            x, dh = tuple_index(x, node_mask), tuple_index(dh, node_mask)

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))

        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))

        if node_mask is not None:
            x_[0][node_mask], x_[1][node_mask] = x[0], x[1]
            x = x_

        return x, edge_attr
