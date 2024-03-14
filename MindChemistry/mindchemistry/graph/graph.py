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
"""graph"""
import mindspore as ms
from mindspore import ops, nn


def degree(index, dim_size, mask=None):
    r"""
    Computes the degree of a one-dimensional index tensor.
    """
    if index.ndim != 1:
        raise ValueError(f"the dimension of index {index.ndim} is not equal to 1")

    if mask is not None:
        if mask.shape[0] != index.shape[0]:
            raise ValueError(f"mask.shape[0] {mask.shape[0]} is not equal to index.shape[0] {index.shape[0]}")
        if mask.ndim != 1:
            st = [0] * mask.ndim
            slice_size = [1] * mask.ndim
            slice_size[0] = mask.shape[0]
            mask = ops.slice(mask, st, slice_size).squeeze()
        src = mask.astype(ms.int32)
    else:
        src = ops.ones(index.shape, ms.int32)

    index = index.unsqueeze(-1)
    out = ops.zeros((dim_size,), ms.int32)

    return ops.tensor_scatter_add(out, index, src)


class Aggregate(nn.Cell):
    r"""
    Easy-use version of scatter.

    Args:
        mode (str): {'add', 'sum', 'mean', 'avg'}, scatter mode.

    Raises:
        ValueError: If `mode` is not legal.

    Supported Platforms:
        ``CPU``, ``GPU``, ``Ascend``

    """

    def __init__(self, mode='add'):
        super().__init__()
        self.mode = mode
        if mode in ('add', 'sum'):
            self.scatter = self.scatter_sum
        elif mode in ('mean', 'avg'):
            self.scatter = self.scatter_mean
        else:
            raise ValueError(f"Unexpected scatter mode {mode}")

    @staticmethod
    def scatter_sum(src, index, out=None, dim_size=None, mask=None):
        r"""
        Computes the scatter sum of a source tensor. The index should be one-dimensional
        """
        if index.ndim != 1:
            raise ValueError(f"the dimension of index {index.ndim} is not equal to 1")
        if index.shape[0] != src.shape[0]:
            raise ValueError(f"index.shape[0] {index.shape[0]} is not equal to src.shape[0] {src.shape[0]}")
        if out is None and dim_size is None:
            raise ValueError(f"the out Tensor and out dim_size cannot be both None")

        index = index.unsqueeze(-1)

        if out is None:
            out = ops.zeros((dim_size,) + src.shape[1:], dtype=src.dtype)
        elif dim_size is not None and out.shape[0] != dim_size:
            raise ValueError(f"the out.shape[0] {out.shape[0]} is not equal to dim_size {dim_size}")

        if mask is not None:
            if mask.shape[0] != src.shape[0]:
                raise ValueError(f"mask.shape[0] {mask.shape[0]} is not equal to src.shape[0] {src.shape[0]}")
            if src.ndim != mask.ndim:
                if mask.size != mask.shape[0]:
                    raise ValueError(f"mask.ndim dose not match src.ndim, and cannot be broadcasted to the same")
                shape = [1] * src.ndim
                shape[0] = -1
                mask = ops.reshape(mask, shape)
            src = ops.mul(src, mask.astype(src.dtype))

        return ops.tensor_scatter_add(out, index, src)

    @staticmethod
    def scatter_mean(src, index, out=None, dim_size=None, mask=None):
        r"""
        Computes the scatter mean of a source tensor. The index should be one-dimensional
        """
        if out is None and dim_size is None:
            raise ValueError(f"the out Tensor and out dim_size cannot be both None")

        if dim_size is None:
            dim_size = out.shape[0]
        elif out is not None and out.shape[0] != dim_size:
            raise ValueError(f"the out.shape[0] {out.shape[0]} is not equal to dim_size {dim_size}")

        count = degree(index, dim_size, mask=mask)
        eps = 1e-5
        count = ops.maximum(count, eps)

        scatter_sum = Aggregate.scatter_sum(src, index, dim_size=dim_size, mask=mask)

        shape = [1] * scatter_sum.ndim
        shape[0] = -1
        count = ops.reshape(count, shape).astype(scatter_sum.dtype)
        res = ops.true_divide(scatter_sum, count)

        if out is not None:
            res = res + out

        return res


class AggregateNodeToGlobal(Aggregate):
    """AggregateNodeToGlobal"""

    def __init__(self, mode='add'):
        super().__init__(mode=mode)

    def construct(self, node_attr, batch, out=None, dim_size=None, mask=None):
        r"""
        Args:
            node_attr (Tensor): The source tensor of node attributes.
            batch (Tensor): The indices of sample to scatter to.
            out (Tensor): The destination tensor. Default: None.
            dim_size (int): If `out` is not given, automatically create output with size `dim_size`. Default: None.
                out and dim_size cannot be both None.
            mask (Tensor): The mask of the node_attr tensor
        Returns:
            Tensor.
        """
        return self.scatter(node_attr, batch, out=out, dim_size=dim_size, mask=mask)


class AggregateEdgeToGlobal(Aggregate):
    """AggregateEdgeToGlobal"""

    def __init__(self, mode='add'):
        super().__init__(mode=mode)

    def construct(self, edge_attr, batch_edge, out=None, dim_size=None, mask=None):
        r"""
        Args:
            edge_attr (Tensor): The source tensor of edge attributes.
            batch_edge (Tensor): The indices of sample to scatter to.
            out (Tensor): The destination tensor. Default: None.
            dim_size (int): If `out` is not given, automatically create output with size `dim_size`. Default: None.
                out and dim_size cannot be both None.
            mask (Tensor): The mask of the node_attr tensor
        Returns:
            Tensor.
        """
        return self.scatter(edge_attr, batch_edge, out=out, dim_size=dim_size, mask=mask)


class AggregateEdgeToNode(Aggregate):
    """AggregateEdgeToNode"""

    def __init__(self, mode='add', dim=0):
        super().__init__(mode=mode)
        self.dim = dim

    def construct(self, edge_attr, edge_index, out=None, dim_size=None, mask=None):
        r"""
        Args:
            edge_attr (Tensor): The source tensor of edge attributes.
            edge_index (Tensor): The indices of nodes in each edge.
            out (Tensor): The destination tensor. Default: None.
            dim_size (int): If `out` is not given, automatically create output with size `dim_size`. Default: None.
                out and dim_size cannot be both None.
            mask (Tensor): The mask of the node_attr tensor
        Returns:
            Tensor.
        """
        return self.scatter(edge_attr, edge_index[self.dim], out=out, dim_size=dim_size, mask=mask)


class Lift(nn.Cell):
    """Lift"""

    def __init__(self, mode="multi_graph"):
        super().__init__()
        self.mode = mode
        if mode not in ["multi_graph", "single_graph"]:
            raise ValueError(f"Unexpected lift mode {mode}")

    @staticmethod
    def lift(src, index, axis=0, mask=None):
        """lift"""
        res = ops.index_select(src, axis, index)

        if mask is not None:
            if mask.shape[0] != res.shape[0]:
                raise ValueError(f"mask.shape[0] {mask.shape[0]} is not equal to res.shape[0] {res.shape[0]}")
            if res.ndim != mask.ndim:
                if mask.size != mask.shape[0]:
                    raise ValueError(f"mask.ndim dose not match src.ndim, and cannot be broadcasted to the same")
                shape = [1] * res.ndim
                shape[0] = -1
                mask = ops.reshape(mask, shape)
            res = ops.mul(res, mask.astype(res.dtype))

        return res

    @staticmethod
    def repeat(src, num, axis=0, max_len=None):
        res = ops.repeat_elements(src, num, axis)

        if (max_len is not None) and (max_len > num):
            padding = ops.zeros((max_len - num,) + res.shape[1:], dtype=res.dtype)
            res = ops.cat((res, padding), axis=0)

        return res


class LiftGlobalToNode(Lift):
    """LiftGlobalToNode"""

    def __init__(self, mode="multi_graph"):
        super().__init__(mode=mode)

    def construct(self, global_attr, batch=None, num_node=None, mask=None, max_len=None):
        r"""
        Args:
            global_attr (Tensor): The source tensor of global attributes.
            batch (Tensor): The indices of samples to get.
            num_node (Int): The number of node in the graph, when there is only 1 graph.
            mask (Tensor): The mask of the output tensor.
            max_len (Int): The output length.
        Returns:
            Tensor.
        """
        if global_attr.shape[0] > 1 or self.mode == "multi_graph":
            return self.lift(global_attr, batch, mask=mask)
        return self.repeat(global_attr, num_node, max_len=max_len)


class LiftGlobalToEdge(Lift):
    """LiftGlobalToEdge"""

    def __init__(self, mode="multi_graph"):
        super().__init__(mode=mode)

    def construct(self, global_attr, batch_edge=None, num_edge=None, mask=None, max_len=None):
        r"""
        Args:
            global_attr (Tensor): The source tensor of global attributes.
            batch_edge (Tensor): The indices of samples to get.
            num_edge (Int): The number of edge in the graph, when there is only 1 graph.
            mask (Tensor): The mask of the output tensor.
            max_len (Int): The output length.
        Returns:
            Tensor.
        """
        if global_attr.shape[0] > 1 or self.mode == "multi_graph":
            return self.lift(global_attr, batch_edge, mask=mask)
        return self.repeat(global_attr, num_edge, max_len=max_len)


class LiftNodeToEdge(Lift):
    """LiftNodeToEdge"""

    def __init__(self, dim=0):
        super().__init__(mode="multi_graph")
        self.dim = dim

    def construct(self, global_attr, edge_index, mask=None):
        r"""
        Args:
            global_attr (Tensor): The source tensor of global attributes.
            edge_index (Tensor): The indices of nodes for each edge.
            mask (Tensor): The mask of the output tensor.
        Returns:
            Tensor.
        """
        return self.lift(global_attr, edge_index[self.dim], mask=mask)
