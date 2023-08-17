# Copyright 2023 Huawei Technologies Co., Ltd
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
"proteinmpnnwracell"
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class CustomTrainOneStepCell(nn.Cell):
    """customlosscell"""

    def __init__(self, network_, optimizer_):
        """args：network，optimizer"""
        super(CustomTrainOneStepCell, self).__init__(auto_prefix=False)
        self.network = network_
        self.network.set_grad()
        self.optimizer_ = optimizer_
        self.weights = self.optimizer_.parameters
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, *inputs):
        x, s, mask, chain_m, residue_idx, chain_encoding_all, mask_for_loss = inputs
        input_ = x, s, mask, chain_m, residue_idx, chain_encoding_all
        loss_ = self.network(*inputs)
        grads = self.grad(self.network, self.weights)(*input_, mask_for_loss)
        self.optimizer_(grads)
        return loss_


class CustomWithLossCell(nn.Cell):
    """losscell"""

    def __init__(self, backbone, loss_fn):
        """init"""
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, x, s, mask, chain_m, residue_idx, chain_encoding_all, mask_for_loss):
        output = self.backbone(x, s, mask, chain_m, residue_idx, chain_encoding_all)
        loss_av = self.loss_fn(s, output, mask_for_loss)
        return loss_av


class LossSmoothed(nn.Cell):
    """loss_smoothed """

    def __init__(self, weight=0.1):
        """init"""
        super(LossSmoothed, self).__init__()
        self.weight = weight

    def construct(self, s, log_probs, mask):
        """ Negative log probabilities """
        s_onehot = ops.Cast()(nn.OneHot(depth=21)(s), ms.float32)

        # Label smoothing
        s_onehot = s_onehot + self.weight / float(s_onehot.shape[-1])
        s_onehot = s_onehot / ops.ReduceSum(keep_dims=True)(s_onehot, -1)

        loss = -(s_onehot * log_probs).sum(-1)
        loss_av = ops.ReduceSum()(loss * mask) / 2000.0
        return loss_av


class LRLIST:
    """LRLIST"""

    def __init__(self, model_size, factor, warmup):
        """init"""
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

    def cal_lr(self, total_step):
        """cal_lr"""
        lr = []
        for i in range(total_step):
            step = i
            if i == 0:
                lr.append(0.)
            else:
                lr.append(self.factor * (self.model_size ** (-0.5) *
                                         min(step ** (-0.5), step * self.warmup ** (-1.5))))
        lr = np.array(lr).astype(np.float32)
        return lr
