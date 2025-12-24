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
"""scatter"""
from mindspore import ops, nn
from mindspore.ops import operations as P


class Scatter(nn.Cell):
    r"""
    Easy-to-use wrapper for scatter operations: aggregates source values into a destination tensor according to index.

    Args:
        mode (str, optional): {'add', 'sum', 'div', 'max', 'min', 'mul'}, scatter mode.

            - 'add' or 'sum': element-wise addition.
            - 'div': element-wise division.
            - 'max': element-wise maximum.
            - 'min': element-wise minimum.
            - 'mul': element-wise multiplication. 

            Default: ``'add'``.

    Inputs:
        - **src** (Tensor) - The source tensor to scatter.
        - **index** (Tensor) - The indices of elements to scatter, must be int type.
        - **out** (Tensor, optional) - The destination tensor. If provided, scatter will be performed in-place.
            Default: ``None``.
        - **dim_size** (int, optional) - If `out` is not given, automatically create output with size `dim_size`.
            If `dim_size` is not given, a minimal sized output tensor is returned. Default: ``None``.

    Outputs:
        Tensor, the result after scatter operation.

    Raises:
        ValueError: If `mode` is not legal.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import Tensor
        >>> from mindscience.e3nn.nn import Scatter
        >>> scatter = Scatter('add')
        >>> src = Tensor([[1, 2], [3, 4], [5, 6]], ms.float32)
        >>> index = Tensor([0, 0, 1], ms.int32)
        >>> out = scatter(src, index)
        >>> print(out.shape)
        (3,2)
    """

    def __init__(self, mode='add'):
        super().__init__()
        self.mode = mode
        if mode in ('add', 'sum'):
            self.scatter = P.TensorScatterAdd()
        elif mode == 'div':
            self.scatter = P.TensorScatterDiv()
        elif mode == 'max':
            self.scatter = P.TensorScatterMax()
        elif mode == 'min':
            self.scatter = P.TensorScatterMin()
        elif mode == 'mul':
            self.scatter = P.TensorScatterMul()
        else:
            raise ValueError(f"Unexpected scatter mode {mode}")

        self.zeros = ops.Zeros()

    def construct(self, src, index, out=None, dim_size=None):
        r"""
        Args:
            src (Tensor): The source tensor.
            index (Tensor): The indices of elements to scatter.
            out (Tensor): The destination tensor. Default: None.
            dim_size (int): If `out` is not given, automatically create output with size `dim_size`.
                If `dim_size` is not given, a minimal sized output tensor is returned. Default: None.

        Returns:
            Tensor.
        """
        if index.ndim < 2:
            index = index.unsqueeze(-1)
        if out is not None:
            return self.scatter(out, index, src)
        dim_size = src.shape[0] if dim_size is None else dim_size
        zero = self.zeros((dim_size, src.shape[1]), src.dtype)
        return self.scatter(zero, index, src)

    def __repr__(self):
        return f'Scatter [{self.mode}]'
