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
"""Interface"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P
# from .mask import MaskedLayerNorm
from .sbr import lecun_normal


class AddInterface(nn.Cell):
    '''add interface information into msa representation or single representation'''

    def __init__(self, input_dim, batch_size=None):
        super(AddInterface, self).__init__()
        self.matmul = P.MatMul(transpose_b=True)
        self.input_dim = input_dim
        self.batch_size = batch_size
        # self.idx = Tensor(0, mstype.int32)
        # self.masked_layer_norm = MaskedLayerNorm()
        self._init_parameter()
        self.gather3 = P.Gather().shard(((1, 1, 1), ()))
        self.gather2 = P.Gather().shard(((1, 1), ()))

    def construct(self, interface_mask, act, index=None, mask=None):
        '''Compute linear'''
        if self.batch_size:
            # input_layer_norm_gamma = P.Gather()(self.input_layer_norm_gammas, index, 0)
            # input_layer_norm_beta = P.Gather()(self.input_layer_norm_betas, index, 0)
            linear_weight = self.gather3(self.linear_weights, index, 0)
            linear_bias = self.gather2(self.linear_biases, index, 0)
        else:
            # input_layer_norm_gamma = self.input_layer_norm_gammas
            # input_layer_norm_beta = self.input_layer_norm_betas
            linear_weight = self.linear_weights
            linear_bias = self.linear_biases
        # act = self.masked_layer_norm(act, input_layer_norm_gamma, input_layer_norm_beta, mask=mask)
        
        act_shape = P.Shape()(act)
        interface_mask = P.ExpandDims()(interface_mask, -1)
        while len(act_shape) > len(P.Shape()(interface_mask)):
            interface_mask = P.ExpandDims()(interface_mask, 0)
        mask1 = interface_mask
        interface_mask = P.Tile()(interface_mask, act_shape[: -2] + (1, 1))
        # act = P.Cast()(act, mstype.float16)
        act = P.Concat(-1)((act, interface_mask))
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]+1))
        act = P.BiasAdd()(self.matmul(act, linear_weight), linear_bias)
        act = P.Reshape()(act, act_shape)
        return act * mask1

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            # self.input_layer_norm_gammas = Parameter(
            #     Tensor(np.ones((self.batch_size, self.input_dim)), mstype.float32))
            # self.input_layer_norm_betas = Parameter(
            #     Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
            self.linear_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim, self.input_dim + 1)), mstype.float32))   
            self.linear_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
        else:
            # self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.input_dim)), mstype.float32))
            # self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))
            self.linear_weights = Parameter(Tensor(np.zeros((self.input_dim, self.input_dim + 1)), mstype.float32))
            self.linear_biases = Parameter(Tensor(np.zeros((self.input_dim,)), mstype.float32))