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
# ==============================================================================
"""Loss"""
from mindspore import ops
import mindspore.nn as nn



class WassersteinLoss(nn.Cell):
    """Wasserstein Loss"""
    def construct(self, y_pred, y_true):
        loss = y_true * y_pred
        loss_mean = ops.mean(loss, axis=(1, 2, 3, 4))
        return loss_mean


class GradLoss(nn.Cell):
    """Grad Loss"""
    def __init__(self, dxy):
        super(GradLoss, self).__init__()
        self.dx = dxy[0]
        self.dy = dxy[1]
        self.concat = ops.Concat(axis=2)

    def cal_grad(self, u_mat):
        """
        :param u_mat: the input, [batch_size,128,128,2] including u and v
        """
        group_00 = u_mat[:, :, :, 0:-2, 0:-2]
        group_01 = u_mat[:, :, :, 0:-2, 1:-1]
        group_02 = u_mat[:, :, :, 0:-2, 2:]
        group_10 = u_mat[:, :, :, 1:-1, 0:-2]
        group_12 = u_mat[:, :, :, 1:-1, 2:]
        group_20 = u_mat[:, :, :, 2:, 0:-2]
        group_21 = u_mat[:, :, :, 2:, 1:-1]
        group_22 = u_mat[:, :, :, 2:, 2:]

        grad_x = (group_20 + 2 * group_21 + group_22 - group_00 - 2 * group_01 - group_02) / (4 * 2 * self.dx)
        grad_y = (group_02 + 2 * group_12 + group_22 - group_00 - 2 * group_10 - group_20) / (4 * 2 * self.dy)
        grad_all = self.concat((grad_x, grad_y))
        return grad_all

    def construct(self, y_pred, y_true):
        grad_y_true = self.cal_grad(y_true)
        grad_y_fake = self.cal_grad(y_pred)
        grad_loss = ops.mean(ops.square(grad_y_true - grad_y_fake), axis=(1, 2, 3, 4))
        return grad_loss


class GradientPenaltyLoss(nn.Cell):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    first get the gradients:
    assuming: - that y_pred has dimensions (batch_size, patch_size)
              - averaged_samples has dimensions (batch_size, nbr_features)
    gradients afterward has dimension (batch_size, nbr_features), basically
    """
    def __init__(self):
        super(GradientPenaltyLoss, self).__init__()
        self.reducesum = ops.ReduceSum(keep_dims=False)
        self.square = ops.Square()
        self.sqrt = ops.Sqrt()
        self.grad_op = ops.GradOperation()

    def construct(self, d_model, interpolated_img, in_merge):
        gradient_function = self.grad_op(d_model)
        gradients = gradient_function(interpolated_img, in_merge)[0]
        gradients_sqr = self.square(gradients)
        gradients_sqr_sum = self.reducesum(gradients_sqr, axis=tuple(range(2, len(gradients_sqr.shape))))
        gradient_l2_norm = self.sqrt(gradients_sqr_sum)
        gradient_penalty = self.square(1 - gradient_l2_norm)
        return ops.mean(gradient_penalty)
