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
"""Mask"""
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import nn


class LayerNormProcess(nn.Cell):
    def __init__(self,):
        super().__init__()
        self.layernorm = P.LayerNorm(begin_norm_axis=-1, begin_params_axis=-1, epsilon=1e-5)

    def construct(self, msa_act, query_norm_gamma, query_norm_beta):
        output, _, _ = self.layernorm(msa_act, query_norm_gamma, query_norm_beta)
        return output


class MaskedLayerNorm(nn.Cell):
    r"""
    Masked layer normalization. Applies layer normalization with mask to the input tensor.

    Inputs:
        - **act** (Tensor) - Tensor of shape :math:`(*, in\_channels)`.
        - **gamma** (Tensor) - Scale parameter of shape :math:`(in\_channels,)`.
        - **beta** (Tensor) - Offset parameter of shape :math:`(in\_channels,)`.
        - **mask** (Tensor, optional) - Mask tensor of shape :math:`(*, 1)`. Default: ``None``.

    Outputs:
        Tensor of shape :math:`(*, in\_channels)`.

    Examples:
        >>> import mindspore as ms
        >>> import mindspore.numpy as mnp
        >>> from mindspore import Tensor
        >>> from mindscience.models.layers import MaskedLayerNorm
        >>> ms.set_context(mode=ms.GRAPH_MODE, device_target="GPU")
        >>> x = Tensor(mnp.random.randn(2, 3, 4).astype(mnp.float32))
        >>> gamma = Tensor(mnp.ones((4,)).astype(mnp.float32))
        >>> beta = Tensor(mnp.zeros((4,)).astype(mnp.float32))
        >>> mask = Tensor(mnp.ones((2, 3)).astype(mnp.float32))
        >>> net = MaskedLayerNorm()
        >>> output = net(x, gamma, beta, mask)
        >>> print(output.shape)
        (2, 3, 4)
    """

    def __init__(self):
        super().__init__()
        self.norm = LayerNormProcess()

    def construct(self, act, gamma, beta, mask=None):
        """Forward pass for MaskedLayerNorm."""
        ones = P.Ones()(act.shape[:-1] + (1,), act.dtype)
        if mask is not None:
            mask = F.expand_dims(mask, -1)
            mask = mask * ones
        else:
            mask = ones

        act = act * mask
        act = self.norm(act, gamma, beta)
        act = act * mask
        return act
