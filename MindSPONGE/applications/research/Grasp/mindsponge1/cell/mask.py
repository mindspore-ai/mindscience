# Copyright 2022 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""Mask"""
# from mindspore.ops import operations as P
from mindspore import ops as P
from mindspore.ops import functional as F
import mindspore.nn as nn

class LayerNormProcess(nn.Cell):
    def __init__(self,):
        super(LayerNormProcess, self).__init__()
        self.layernorm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
    
    def construct(self, msa_act, query_norm_gamma, query_norm_beta):
        # print("debug LayerNormProcess msa_act", msa_act)
        # print("debug LayerNormProcess query_norm_gamma", query_norm_gamma[:])
        # print("debug LayerNormProcess query_norm_beta", query_norm_beta[:])
        output, _, _ = self.layernorm(msa_act, query_norm_gamma, query_norm_beta)
        # print("debug LayerNormProcess output", output)
        return output


class MaskedLayerNorm(nn.Cell):
    '''masked_layer_norm'''

    def __init__(self):
        super(MaskedLayerNorm, self).__init__()
        #self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.norm = LayerNormProcess()

    def construct(self, act, gamma, beta, mask=None):
        '''construct'''
        act = act
        gamma = gamma
        beta = beta
        # print("debug MaskedLayerNorm act", act)
        ones = P.Ones()(act.shape[:-1] + (1,), act.dtype)
        if mask is not None:
            mask = F.expand_dims(mask, -1)
            mask = mask * ones
        else:
            mask = ones
        # print("debug MaskedLayerNorm mask", mask)
        act = act * mask
        act = self.norm(act, gamma, beta)
        act = act * mask
        # print("debug MaskedLayerNorm act 54", act)
        return act

class MaskedLayerNormParallel(nn.Cell):
    '''masked_layer_norm'''

    def __init__(self, device_num):
        super(MaskedLayerNormParallel, self).__init__()
        self.norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5).shard(((1, device_num, 1), (1,), (1,)))
        self.expand = P.ExpandDims().shard(((1, device_num),))
        self.mul = P.Mul().shard(((1, device_num, 1), (1, device_num, 1)))
        # self.norm = LayerNormProcess()

    def construct(self, act, gamma, beta, mask=None):
        '''construct'''
        act = act
        gamma = gamma
        beta = beta
        # print("debug MaskedLayerNorm act", act)
        ones = P.Ones()(act.shape[:-1] + (1,), act.dtype)
        if mask is not None:
            # mask = F.expand_dims(mask, -1)
            # mask = mask * ones
            mask = self.expand(mask, -1)
            mask = self.mul(mask, ones)
        else:
            mask = ones
        # print("debug MaskedLayerNorm mask", mask)
        # act = act * mask
        # act = self.norm(act, gamma, beta)
        # act = act * mask
        # print("debug MaskedLayerNorm act 54", act)

        act = self.mul(act, mask)
        act, _, _ = self.norm(act, gamma, beta)
        act = self.mul(act, mask)
        return act