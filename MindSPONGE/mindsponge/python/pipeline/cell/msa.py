# Copyright 2023 The AIMM Group at Shenzhen Bay Laboratory & Peking University & Huawei Technologies Co., Ltd
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
"""MSA"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Parameter
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from .basic import Attention, GlobalAttention
from .mask import MaskedLayerNorm
from ...common.utils import _memory_reduce


class MSARowAttentionWithPairBias(nn.Cell):
    r"""
    MSA row attention. Information from pair action value is made as the bias of the matrix of MSARowAttention,
        in order to update the state of MSA using pair information.

    Reference:
        `Jumper et al. (2021) Suppl. Alg. 7 'MSARowAttentionWithPairBias'
            <https://www.nature.com/articles/s41586-021-03819-2>`_.

    Args:
        num_head (int):         The number of the attention head.
        key_dim (int):          The dimension of the attention hidden layer.
        gating (bool):          Indicator of if the attention is gated.
        msa_act_dim (int):      The dimension of the msa_act.
        pair_act_dim (int):     The dimension of the pair_act.
        batch_size (int):       The batch size of parameters in MSA row attention, used in while control flow.
                                Default: None.
        slice_num (int):        The number of slices to be made to reduce memory. Default: 0.

    Inputs:
        - **msa_act** (Tensor) - Tensor of msa_act with shape :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` .
        - **msa_mask** (Tensor) - The mask for MSA row attention matrix with shape :math:`(N_{seqs}, N_{res})` .
        - **pair_act** (Tensor) - Tensor of pair_act with shape :math:`(N_{res}, N_{res}, pair\_act\_dim)` .
          Data type is float.
        - **index** (Tensor) - The index of while loop, only used in case of while control flow. Default: "None".
        - **norm_msa_mask** (Tensor) - The mask of msa_act when to do layernorm with shape :math:`(N_{seqs}, N_{res})`,
          Default: "None".
        - **norm_pair_mask** (Tensor) - The mask of pair_act when to do layernorm with shape :math:`(N_{res}, N_{res})`,
          Default: "None".
        - **res_idx** (Tensor) - The residue index used to perform ROPE with shape :math:`(N_{res})`, Default: "None".

    Outputs:
        Tensor, the float tensor of the msa_act of the layer with shape :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import MSARowAttentionWithPairBias
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = MSARowAttentionWithPairBias(num_head=4, key_dim=4, gating=True,
        ...                                     msa_act_dim=64, pair_act_dim=128,
        ...                                     batch_size=None)
        >>> msa_act = Tensor(np.ones((4, 256, 64)), mstype.float32)
        >>> msa_mask = Tensor(np.ones((4, 256)), mstype.float16)
        >>> pair_act = Tensor(np.ones((256, 256, 128)), mstype.float32)
        >>> index = None
        >>> msa_out = model(msa_act, msa_mask, pair_act, index)
        >>> print(msa_out.shape)
        (4, 256, 64)
    """

    def __init__(self, num_head, key_dim, gating, msa_act_dim, pair_act_dim, batch_size=None, slice_num=0):
        super(MSARowAttentionWithPairBias, self).__init__()
        self.num_head = num_head
        self.batch_size = batch_size
        self.matmul = P.MatMul(transpose_b=True)
        self.attn_mod = Attention(num_head, key_dim, gating, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.msa_act_dim = msa_act_dim
        self.pair_act_dim = pair_act_dim
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.idx = Tensor(0, mstype.int32)
        self.masked_layer_norm = MaskedLayerNorm()
        self._init_parameter()

    def construct(self, msa_act, msa_mask, pair_act, index=None, norm_msa_mask=None, norm_pair_mask=None, res_idx=None):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            feat_2d_norm_gamma = P.Gather()(self.feat_2d_norm_gammas, index, 0)
            feat_2d_norm_beta = P.Gather()(self.feat_2d_norm_betas, index, 0)
            feat_2d_weight = P.Gather()(self.feat_2d_weights, index, 0)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            feat_2d_norm_gamma = self.feat_2d_norm_gammas
            feat_2d_norm_beta = self.feat_2d_norm_betas
            feat_2d_weight = self.feat_2d_weights

        q, k, _ = pair_act.shape
        input_bias = 1e9 * (msa_mask - 1.0)
        input_bias = P.ExpandDims()(P.ExpandDims()(input_bias, 1), 2)

        msa_act = self.masked_layer_norm(msa_act, query_norm_gamma, query_norm_beta, mask=norm_msa_mask)
        pair_act = self.masked_layer_norm(pair_act, feat_2d_norm_gamma, feat_2d_norm_beta, mask=norm_pair_mask)
        pair_act = P.Reshape()(pair_act, (-1, pair_act.shape[-1]))
        nonbatched_bias = P.Transpose()(P.Reshape()(self.matmul(pair_act, feat_2d_weight), (q, k, self.num_head)),
                                        (2, 0, 1))
        batched_inputs = (msa_act, input_bias)
        if res_idx is not None:
            nonbatched_inputs = (nonbatched_bias, res_idx)
        else:
            nonbatched_inputs = (index, nonbatched_bias)
        msa_act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        return msa_act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(
                Tensor(np.zeros([self.batch_size, self.pair_act_dim]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.zeros([self.batch_size, self.num_head, self.pair_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim]), mstype.float32))
            self.feat_2d_norm_gammas = Parameter(Tensor(np.ones([self.pair_act_dim]), mstype.float32))
            self.feat_2d_norm_betas = Parameter(Tensor(np.zeros([self.pair_act_dim]), mstype.float32))
            self.feat_2d_weights = Parameter(
                Tensor(np.random.normal(scale=1 / np.sqrt(self.pair_act_dim), size=[self.num_head, self.pair_act_dim]),
                       mstype.float32))

    def _compute(self, msa_act, mask, index, nonbatched_bias):
        """
        compute.

        Args:
            msa_act (Tensor):           Tensor of msa_act.
            mask (Tensor):              The mask for MSA row attention matrix.
            index (Tensor):             The index of while loop, only used in case of while control flow. Default: None
            nonbatched_bias(Tensor):    Tensor of non batched bias matrix.

        Outputs:
            - **msa_act** (Tensor)- Tensor, the float tensor of the msa_act of the attention layer.
        """
        msa_act = self.attn_mod(msa_act, msa_act, mask, index, nonbatched_bias)
        return msa_act


class MSAColumnAttention(nn.Cell):
    """
    MSA column-wise gated self attention.
    The column-wise attention lets the elements that belong to the same target residue exchange information.

    Reference:
        `Jumper et al. (2021) Suppl. Alg. 8 "MSAColumnAttention"
        <https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-021-03819-2/MediaObjects/41586_2021_3819_MOESM1_ESM.pdf>`_.

    Args:
        num_head (int):         The number of the heads.
        key_dim (int):          The dimension of the input.
        gating (bool):          Indicator of if the attention is gated.
        msa_act_dim (int):      The dimension of the msa_act. The intermediate variable after MSA retrieving
                                in AlphaFold.
        batch_size (int):       The batch size of parameters in MSAColumnAttention, used in while control flow,
                                Default: "None".
        slice_num (int):        The number of slices to be made to reduce memory, Default: 0.

    Inputs:
        - **msa_act** (Tensor) - Tensor of msa_act. The intermediate variable after MSA retrieving
          in AlphaFold, shape :math:`[N_{seqs}, N_{res}, C_m]` .
        - **msa_mask** (Tensor) - The mask for MSAColumnAttention matrix, shape :math:`[N_{seqs}, N_{res}]`.
        - **index** (Tensor) - The index of while loop, only used in case of while control flow. Default: "None".

    Outputs:
        Tensor, the float tensor of the msa_act of the layer, shape :math:`[N_{seqs}, N_{res}, C_m]`.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import MSAColumnAttention
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = MSAColumnAttention(num_head=8, key_dim=256, gating=True,
        ...                         msa_act_dim=256, batch_size=1, slice_num=0)
        >>> msa_act = Tensor(np.ones((512, 256, 256)), mstype.float32)
        >>> msa_mask = Tensor(np.ones((512, 256)), mstype.float32)
        >>> index = Tensor(0, mstype.int32)
        >>> attn_out = model(msa_act, msa_mask, index)
        >>> print(attn_out.shape)
        (512, 256, 256)
    """

    def __init__(self, num_head, key_dim, gating, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnAttention, self).__init__()
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.attn_mod = Attention(num_head, key_dim, gating, msa_act_dim, msa_act_dim, msa_act_dim, batch_size)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def construct(self, msa_act, msa_mask, index=None):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        msa_mask = P.Transpose()(msa_mask, (1, 0))

        input_mask = 1e9 * (msa_mask - 1.)
        input_mask = P.ExpandDims()(P.ExpandDims()(input_mask, 1), 2)
        msa_act, _, _ = self.query_norm(msa_act, query_norm_gamma, query_norm_beta)
        batched_inputs = (msa_act, input_mask)
        nonbatched_inputs = (index,)
        msa_act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act

    def _init_parameter(self):
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.batch_size, self.msa_act_dim]), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones([self.msa_act_dim]), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros([self.msa_act_dim]), mstype.float32))

    def _compute(self, msa_act, input_mask, index):
        '''compute'''
        msa_act = self.attn_mod(msa_act, msa_act, input_mask, index)
        return msa_act


class MSAColumnGlobalAttention(nn.Cell):
    r"""
    MSA column global attention. Transpose MSA information at sequence axis and residue axis, then use `GlobalAttention
    <https://www.mindspore.cn/mindsponge/docs/zh-CN/r1.0/cell/mindsponge.cell.GlobalAttention.html>`_ to
    do Attention between input sequences without dealing with the relationship between residues in sequence.
    Comparing with MSAColumnAttention, it uses GlobalAttention to deal with longer input sequence.

    Reference:
        `Jumper et al. (2021) Suppl. Alg. 19 'MSAColumnGlobalAttention'
        <https://www.nature.com/articles/s41586-021-03819-2>`_.

    Args:
        num_head (int):         The number of the attention heads.
        gating (bool):          Indicator of if the attention is gated.
        msa_act_dim (int):      The dimension of the msa_act.
        batch_size (int):       The batch size of parameters in MSAColumnGlobalAttention, used
                                in while control flow. Default: None.
        slice_num (int):        The number of slices to be made to reduce memory. Default: 0

    Inputs:
        - **msa_act** (Tensor) - Tensor of msa_act with shape :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` .
        - **msa_mask** (Tensor) - The mask for msa_act matrix with shape :math:`(N_{seqs}, N_{res})` .
        - **index** (Tensor) - The index of while loop, only used in case of while control flow. Default: "None".

    Outputs:
        Tensor, the float tensor of the msa_act of the layer with shape :math:`(N_{seqs}, N_{res}, msa\_act\_dim)` .

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindsponge.cell import MSAColumnGlobalAttention
        >>> from mindspore import dtype as mstype
        >>> from mindspore import Tensor
        >>> model = MSAColumnGlobalAttention(num_head=4, gating=True, msa_act_dim=64, batch_size=None)
        >>> msa_act = Tensor(np.ones((4, 256, 64)), mstype.float32)
        >>> msa_mask = Tensor(np.ones((4, 256)), mstype.float16)
        >>> index = None
        >>> msa_out = model(msa_act, msa_mask, index)
        >>> print(msa_out.shape)
        (4, 256, 64)
    """

    def __init__(self, num_head, gating, msa_act_dim, batch_size=None, slice_num=0):
        super(MSAColumnGlobalAttention, self).__init__()
        self.attn_mod = GlobalAttention(num_head, gating, msa_act_dim, msa_act_dim, batch_size)
        self.query_norm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)
        self.batch_size = batch_size
        self.slice_num = slice_num
        self.msa_act_dim = msa_act_dim
        self.idx = Tensor(0, mstype.int32)
        self._init_parameter()

    def construct(self, msa_act, msa_mask, index=None):
        '''construct'''
        if self.batch_size:
            query_norm_gamma = P.Gather()(self.query_norm_gammas, index, 0)
            query_norm_beta = P.Gather()(self.query_norm_betas, index, 0)
            msa_act = P.Transpose()(msa_act, (1, 0, 2))
            msa_mask = P.Transpose()(msa_mask, (1, 0))
        else:
            query_norm_gamma = self.query_norm_gammas
            query_norm_beta = self.query_norm_betas
            msa_act = P.Transpose()(msa_act, (1, 0, 2))
            msa_mask = P.Transpose()(msa_mask, (1, 0))

        input_mask = 1e9 * (msa_mask - 1.)
        input_mask = P.ExpandDims()(P.ExpandDims()(input_mask, 1), 2)

        msa_act, _, _ = self.query_norm(msa_act,
                                        query_norm_gamma,
                                        query_norm_beta)
        msa_mask = P.ExpandDims()(msa_mask, -1)
        batched_inputs = (msa_act, msa_mask)
        nonbatched_inputs = (index,)
        msa_act = _memory_reduce(self._compute, batched_inputs, nonbatched_inputs, self.slice_num)
        msa_act = P.Transpose()(msa_act, (1, 0, 2))
        return msa_act

    def _init_parameter(self):
        '''init parameter'''
        if self.batch_size:
            self.query_norm_gammas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.batch_size, self.msa_act_dim)), mstype.float32))
        else:
            self.query_norm_gammas = Parameter(Tensor(np.ones((self.msa_act_dim)), mstype.float32))
            self.query_norm_betas = Parameter(Tensor(np.zeros((self.msa_act_dim)), mstype.float32))

    def _compute(self, msa_act, msa_mask, index):
        """
        compute.

        Args:
            msa_act (Tensor):       Tensor of msa_act.
            msa_mask (Tensor):      The mask for msa_act matrix.
            index (Tensor):         The index of while loop, only used in case of while
                                    control flow. Default: None

        Outputs:
            - **msa_act** (Tensor)- Tensor, the float tensor of the msa_act of the attention layer.
        """
        msa_act = self.attn_mod(msa_act, msa_act, msa_mask, index)
        return msa_act
