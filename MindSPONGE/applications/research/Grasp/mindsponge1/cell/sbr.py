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
"""Soft blurred restraints"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import Parameter
from mindspore.ops import operations as P
# from .mask import MaskedLayerNorm

def lecun_normal(dim_in, shape):
    stddev = 1./np.sqrt(dim_in)
    return np.random.normal(loc=0, scale=stddev, size=shape)

class ProcessSBR(nn.Cell):
    '''add inter-residue soft blurred restraints into pair representation'''

    def __init__(self, input_dim, output_dim, batch_size=None):
        super(ProcessSBR, self).__init__()
        self.matmul = P.MatMul(transpose_b=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        self.relu = nn.ReLU()
        self.layer_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self._init_parameter()
        self.gather3 = P.Gather().shard(((1, 1, 1), ()))
        self.gather2 = P.Gather().shard(((1, 1), ()))

    def construct(self, act, mask=None, index=None, useperm=False):
        '''Compute linear'''
        linear_bias=None
        if self.batch_size:
            input_layer_norm_gamma = self.gather2(self.input_layer_norm_gammas, index, 0)
            input_layer_norm_beta = self.gather2(self.input_layer_norm_betas, index, 0)
            linear_weight = self.gather3(self.linear_weights, index, 0)
            linear_bias = self.gather2(self.linear_biases, index, 0)
        else:
            input_layer_norm_gamma = self.input_layer_norm_gammas
            input_layer_norm_beta = self.input_layer_norm_betas
            linear_weight = self.linear_weights
            linear_bias = self.linear_biases
        act, _, _ = self.layer_norm(act, input_layer_norm_gamma, input_layer_norm_beta)

        act_shape = P.Shape()(act)
        if len(act_shape) != 2:
            act = P.Reshape()(act, (-1, act_shape[-1]))
        act = P.BiasAdd()(self.matmul(act, linear_weight), linear_bias)
      
        
        act = P.Reshape()(act, act_shape[:-1]+(-1,))
        if mask is not None:
            if not useperm:
                act *= P.ExpandDims()(mask, -1)
            else:
                act = P.Transpose()(act, (2, 0, 1))
                act *= mask[None, :, :]
                act = P.Transpose()(act, (1, 2, 0))
        return act
        

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.input_layer_norm_gammas = Parameter(
                Tensor(np.ones((self.batch_size, self.input_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(
                Tensor(np.zeros((self.batch_size, self.input_dim)), mstype.float32))
            self.linear_weights = Parameter(
                Tensor(np.zeros((self.batch_size, self.output_dim, self.input_dim)), mstype.float32))
            self.linear_biases = Parameter(
                Tensor(np.zeros((self.batch_size, self.output_dim)), mstype.float32))
        else:
            self.input_layer_norm_gammas = Parameter(Tensor(np.ones((self.input_dim)), mstype.float32))
            self.input_layer_norm_betas = Parameter(Tensor(np.zeros((self.input_dim)), mstype.float32))
            self.linear_weights = Parameter(Tensor(np.zeros((self.output_dim, self.input_dim)), mstype.float32))
            self.linear_biases = Parameter(Tensor(np.zeros((self.output_dim, )), mstype.float32))