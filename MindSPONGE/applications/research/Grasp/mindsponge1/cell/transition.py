# Copyright 2021 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""Transition"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common.initializer import initializer
from .initializer import lecun_init
from .mask import MaskedLayerNormParallel#MaskedLayerNorm#
from ..common.utils import _memory_reduce, MemoryReduceCell


class Transition(nn.Cell):
    r"""
    This is 2-layer MLP where the intermediate layer expands number of channels
    of the input by a factor(num_intermediate_factor).

    .. math::
        Transition(\mathbf{act}) = Linear(Linear(\mathbf{act}))

    Args:
        num_intermediate_factor(float):    The expand factor of intermediate output
                                           channels compared to the input.
        input_dim(int):                    The channels of the input.
        batch_size(int):                   The batch size of parameters in Transition,
                                           used in while control flow. Default: "None".
        slice_num (int):                   The slice num used in transition layer
                                           when the memory is overflow. Default: 0.

    Inputs:
        - **act** (Tensor) - The input with channels equal to input_dim, shape is (..., input_dim).
        - **index** (Tensor) - The index of while loop, only used in case of while control
          flow. Default: "None".
        - **mask** (Tensor) - The mask of act when to do layernorm with shape :math:`(32, input_{dim})`,
          Default: "None".

    Outputs:
        Tensor, the float tensor of the output of the layer with shape (..., input_dim).

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import Transition
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = Transition(num_intermediate_factor=4, input_dim=128)
        >>> input = Tensor(np.ones((32, 128, 128)), mstype.float32)
        >>> output= model(input)
        >>> print(output.shape)
        (32, 128, 128)
    """

    def __init__(self, num_intermediate_factor, input_dim, device_num, batch_size=None, slice_num=0):
        super(Transition, self).__init__()
        self.matmul = P.MatMul(transpose_b=True)
        # self.batch_matmul = P.BatchMatMul(transpose_b=True).shard(((1, device_num, 1), (1, 1)))
        # self.biasadd = P.Add().shard(((1, device_num, 1), (1,)))
        self.input_dim = input_dim
        self.num_intermediate = int(input_dim * num_intermediate_factor)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.relu = nn.ReLU()
        self.idx = Tensor(0, mstype.int32)
        # self.masked_layer_norm = MaskedLayerNorm()
        self.masked_layer_norm = MaskedLayerNormParallel(device_num)
        # self.masked_layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5).shard(((1, device_num, 1), (1,), (1,)))
        self._init_parameter()
        self.gather3 = P.Gather().shard(((1, 1, 1), ()))
        self.gather2 = P.Gather().shard(((1, 1), ()))
        # concat = []
        # for i in range(slice_num):
        #     concat.append((1, device_num, 1))
        self.memory_reduce = MemoryReduceCell(self.slice_num, device_num, strategy=[((1, device_num, 1), (1,)),])
        # self.memory_reduce.concat.shard(tuple(concat))
        # self.mul = P.Mul().shard(((1, device_num, 1), (1, device_num, 1)))


    def construct(self, act, index=None, mask=None):
        '''Compute transition'''
        if self.batch_size:
            input_layer_norm_gamma = self.gather2(self.input_layer_norm_gammas, index, 0)
            input_layer_norm_beta = self.gather2(self.input_layer_norm_betas, index, 0)
            transition1_weight = self.gather3(self.transition1_weights, index, 0)
            transition1_bias = self.gather2(self.transition1_biases, index, 0)
            transition2_weight = self.gather3(self.transition2_weights, index, 0)
            transition2_bias = self.gather2(self.transition2_biases, index, 0)
        else:
            input_layer_norm_gamma = self.input_layer_norm_gammas
            input_layer_norm_beta = self.input_layer_norm_betas
            transition1_weight = self.transition1_weights
            transition1_bias = self.transition1_biases
            transition2_weight = self.transition2_weights
            transition2_bias = self.transition2_biases


        act = self.masked_layer_norm(act, input_layer_norm_gamma, input_layer_norm_beta, mask=mask)
        batched_inputs = (act,)
        nonbatched_inputs = (transition1_weight, transition1_bias, transition2_weight, transition2_bias)
        # act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        act = self.memory_reduce(self._compute, batched_inputs, nonbatched_inputs)
        return act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.input_layer_norm_gammas = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
            self.transition1_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate, self.input_dim)), mstype.float32))
            self.transition1_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.num_intermediate)), mstype.float32))
            self.transition2_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim, self.num_intermediate)), mstype.float32))
            self.transition2_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
        else:
            self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.input_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))
            self.transition1_weights = Parameter(initializer(lecun_init(self.input_dim, initializer_name='relu'),
                                                             [self.num_intermediate, self.input_dim]))
            self.transition1_biases = Parameter(Tensor(np.zeros((self.num_intermediate)), mstype.float32))
            self.transition2_weights = Parameter(
                Tensor(np.zeros((self.input_dim, self.num_intermediate)), mstype.float32))
            self.transition2_biases = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))

    def _compute(self, act, transition1_weight, transition1_bias, transition2_weight, transition2_bias):
        '''compute transition.'''

        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = self.relu(P.BiasAdd()(self.matmul(act, transition1_weight), transition1_bias))
        act = P.BiasAdd()(self.matmul(act, transition2_weight), transition2_bias)
        act = P.Reshape()(act, act_shape)
        # act = self.relu(self.biasadd(self.batch_matmul(act, transition1_weight), transition1_bias))
        # act = self.biasadd(self.batch_matmul(act, transition2_weight), transition2_bias)
        return act