# Copyright 2021 Huawei Technologies Co., Ltd
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
r"""DeepONet model."""
from mindspore import nn, Tensor, Parameter, ops
import mindspore.common.dtype as mstype

from ..basic_block import MLP, CoordPositionalEncoding


class DeepONet(nn.Cell):
    r"""
    The DeepONet model.
    DeepONet model is composed of branch net and trunk net, both of which are MLPs.
    The number of output neurons in branch net and trunk Net is the same, and
    the inner product of their outputs is the output of DeepONet.
    The details can be found in `Lu L, Jin P, Pang G, et al. Learning nonlinear
    operators via DeepONet based on the universal approximation theorem of operators[J].
    Nature machine intelligence, 2021, 3(3): 218-229.
    <https://www.nature.com/articles/s42256-021-00302-5>`.

    Args:
        trunk_dim_in (int): number of input neurons of trunk net.
        trunk_dim_hidden (int): number of neurons in hidden layers of trunk net.
        trunk_num_layers (int): number of layers of trunk net.
        branch_dim_in (int): number of input neurons of branch net.
        branch_dim_hidden (int): number of neurons in hidden layers of branch net.
        branch_num_layers (int): number of layers of branch net.
        dim_out (int): the number of output neurons of trunk net and branch net.

    Inputs:
        - **trunk_in** (Tensor) - input tensor of trunk net,
          shape is :math:`(num\_pdes, num\_points, trunk\_dim\_in)`.
        - **branch_in** (Tensor) - input tensor of branch net,
          shape is :math:`(num\_points, trunk\_dim\_in)`.

    Outputs:
        Output tensor, shape is :math:`(num\_pdes, num\_points, 1)`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from model.baseline.deeponet import DeepONet
        >>> trunk_dim_in = 2
        >>> trunk_dim_hidden = 128
        >>> trunk_num_layers = 5
        >>> branch_dim_in = 129
        >>> branch_dim_hidden = 128
        >>> branch_num_layers = 5
        >>> dim_out = 128
        >>> deeponet = DeepONet(trunk_dim_in, trunk_dim_hidden, trunk_num_layers, \
                     branch_dim_in, branch_dim_hidden, branch_num_layers, \
                     dim_out=dim_out)
        >>> num_pdes = 10
        >>> num_points = 8192
        >>> trunk_in = Tensor(np.random.rand(num_pdes, num_points, trunk_dim_in), dtype=mstype.float32)
        >>> branch_in = Tensor(np.random.rand(num_pdes, branch_dim_in), dtype=mstype.float32)
        >>> out = deeponet(trunk_in, branch_in)  # [num_pdes, num_points, 1]
        >>> print(out.shape)
        (10, 8192, 1)

    """

    def __init__(self, trunk_dim_in: int,
                 trunk_dim_hidden: int,
                 trunk_num_layers: int,
                 branch_dim_in: int,
                 branch_dim_hidden: int,
                 branch_num_layers: int,
                 dim_out: int = 256,
                 num_pos_enc: int = 5,
                 compute_dtype=mstype.float32) -> None:
        super().__init__()
        self.pos_enc = CoordPositionalEncoding(num_pos_enc)
        ext_dim_in = trunk_dim_in * (2 * num_pos_enc + 1)

        self.trunk_net = MLP(ext_dim_in, dim_out,
                             dim_hidden=trunk_dim_hidden,
                             num_layers=trunk_num_layers,
                             compute_dtype=compute_dtype)

        self.trunk_act = nn.ReLU()

        self.branch_net = MLP(branch_dim_in, dim_out,
                              dim_hidden=branch_dim_hidden,
                              num_layers=branch_num_layers,
                              compute_dtype=compute_dtype)

        self.b0 = Parameter(Tensor([0.0], dtype=compute_dtype))

        self.reduce_sum = ops.ReduceSum(keep_dims=True)

    def construct(self,
                  trunk_in: Tensor,
                  branch_in: Tensor) -> Tensor:
        r"""construct"""
        trunk_in = self.pos_enc(trunk_in)

        out_trunk = self.trunk_act(self.trunk_net(trunk_in))  # [num_pdes, num_points, dim_out]
        num_points = out_trunk.shape[1]

        out_branch = self.branch_net(branch_in)  # [num_pdes, dim_out]
        # [num_pdes, 1, dim_out] -> [num_pdes, num_points, dim_out]
        out_branch = out_branch.unsqueeze(1).repeat(num_points, axis=1)

        out = out_trunk * out_branch  # [num_pdes, num_points, dim_out]
        out = self.reduce_sum(out, -1)  # [num_pdes, num_points, 1]
        out = out + self.b0  # [num_pdes, num_points, 1]

        return out  # [num_pdes, num_points, 1]
