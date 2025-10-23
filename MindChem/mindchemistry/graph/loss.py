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
"""loss"""
import mindspore as ms
from mindspore import ops, nn


class LossMaskBase(nn.Cell):
    """LossMaskBase"""

    def __init__(self, reduction='mean', dtype=None):
        super().__init__()
        self.reduction = reduction
        if reduction not in ["mean", "sum"]:
            raise ValueError(f"Unexpected reduction mode {reduction}")

        self.dtype = dtype if dtype is not None else ms.float32

    def construct(self, logits, labels, mask=None, num=None):
        """construct"""
        if logits.shape != labels.shape:
            raise ValueError(f"logits.shape {logits.shape} is not equal to labels.shape {labels.shape}")

        x = self.loss(logits.astype(self.dtype), labels.astype(self.dtype))

        if mask is not None:
            if mask.shape[0] != x.shape[0]:
                raise ValueError(f"mask.shape[0] {mask.shape[0]} is not equal to input.shape[0] {x.shape[0]}")
            if x.ndim != mask.ndim:
                if mask.size != mask.shape[0]:
                    raise ValueError(f"mask.ndim dose not match src.ndim, and cannot be broadcasted to the same")
                shape = [1] * x.ndim
                shape[0] = -1
                mask = ops.reshape(mask, shape)
            x = ops.mul(x, mask.astype(x.dtype))

        # pylint: disable=W0622
        sum = ops.sum(x)
        if self.reduction == "sum":
            return sum
        if num is None:
            num = x.size
        else:
            num_div = x.shape[0]
            if num_div != 0:
                num = x.size / num_div * num
            else:
                raise ValueError
        return ops.true_divide(sum, num)

class L1LossMask(LossMaskBase):

    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def loss(self, logits, labels):
        return ops.abs(logits - labels)


class L2LossMask(LossMaskBase):

    def __init__(self, reduction='mean'):
        super().__init__(reduction)

    def loss(self, logits, labels):
        return ops.square(logits - labels)
