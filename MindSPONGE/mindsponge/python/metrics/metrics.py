# Copyright 2021-2022 @ Shenzhen Bay Laboratory &
#                       Peking University &
#                       Huawei Technologies Co., Ltd
#
# This code is a part of MindSPONGE:
# MindSpore Simulation Package tOwards Next Generation molecular modelling.
#
# MindSPONGE is open-source software based on the AI-framework:
# MindSpore (https://www.mindspore.cn/)
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
"""
Metrics for collective variables
"""
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.communication.management as D
import mindspore.nn as nn
import mindspore.numpy as mnp

from mindspore import Parameter, Tensor
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.nn import Metric

from ..colvar import Colvar


class CV(Metric):
    """Metric to output collective variables"""
    def __init__(self,
                 colvar: Colvar,
                 indexes: tuple = (2, 3),
                 ):

        super().__init__()
        self._indexes = indexes
        self.colvar = colvar
        self._cv_value = None

    def clear(self):
        self._cv_value = 0

    def update(self, *inputs):
        coordinate = inputs[self._indexes[0]]
        pbc_box = inputs[self._indexes[1]]
        self._cv_value = self.colvar(coordinate, pbc_box)

    def eval(self):
        return self._cv_value


class BalancedMSE(nn.Cell):
    """Balanced MSE error
    Compute Balanced MSE error between the prediction and the ground truth
     to solve unbalanced labels in regression task.
    Ren, Jiawei, et al. "Balanced MSE for Imbalanced Visual Regression."

    Args:
      prediction: A float tensor of size [batch, ...].
      target: A float tensor of size [batch, ...].

    Returns:
      A float tensor representing balanced error
    """

    def __init__(self, first_break, last_break, num_bins, beta=0.99):
        super(BalancedMSE, self).__init__()
        self.beta = beta
        self.first_break = first_break
        self.last_break = last_break
        self.num_bins = num_bins

        self.breaks = mnp.linspace(self.first_break, self.last_break, self.num_bins)
        self.width = self.breaks[1] - self.breaks[0]

        self.log_noise_scale = Parameter(Tensor([0.], mstype.float32))
        self.p_bins = Parameter(Tensor(np.ones((self.num_bins)) / self.num_bins, dtype=mstype.float32), \
                                   name='p_bins', requires_grad=False)

        self.softmax = nn.Softmax(-1)
        self.zero = Tensor([0.])

        self.onehot = nn.OneHot(depth=self.num_bins)
        self.allreduce = P.AllReduce()
        self.device_num = D.get_group_size()

    def construct(self, prediction, target, p_bins=None):
        """construct"""
        if p_bins is None:
            p_bins = self._compute_p_bins(prediction)

        log_sigma2 = self.log_noise_scale * 1.
        log_sigma2 = 5. * P.Tanh()(log_sigma2 / 5.)
        sigma2 = mnp.exp(log_sigma2) + 0.25 * self.width
        tau = 2. * sigma2
        a = - F.square(prediction - target) / tau

        ndim = prediction.ndim
        y_bins = mnp.reshape(self.centers * 1., ndim * (1,) + (-1,))
        b_term = - F.square(mnp.expand_dims(prediction, -1) - y_bins) / tau

        p_clip = mnp.clip(p_bins, 1e-8, 1 - 1e-8)
        log_p = mnp.log(p_clip)
        log_p = mnp.reshape(log_p, ndim * (1,) + (-1,))

        b_term += log_p
        b = nn.ReduceLogSumExp(-1, False)(b_term)

        err = -a + b
        return err

    def _compute_p_bins(self, y_gt):
        """compute bins"""
        ndim = y_gt.ndim
        breaks = mnp.reshape(self.breaks, (1,) * ndim + (-1,))
        y_gt = mnp.expand_dims(y_gt, -1)

        y_bins = (y_gt > breaks).astype(mstype.float32)
        y_bins = P.ReduceSum()(y_bins, -1).astype(mstype.int32)
        p_gt = self.onehot(y_bins)

        p_gt = P.Reshape()(p_gt, (-1, self.num_bins))
        p_bins = P.ReduceMean()(p_gt, 0)
        p_bins = self.allreduce(p_bins) / self.device_num

        p_bins = self.beta * self.p_bins + (1 - self.beta) * p_bins
        P.Assign()(self.p_bins, p_bins)

        return p_bins


class MultiClassFocal(nn.Cell):
    """Focal error for multi-class classifications.

    Compute the multiple classes focal error between `prediction` and the ground truth `target`.

    Lin, Tsung-Yi, et al. "Focal loss for dense object detection."

    Args:
      prediction: A float tensor of size [batch, ...].
      target: A float tensor of size [batch, ...].

    Returns:
      A scalar representing normalized total error.
    """

    def __init__(self, num_class=10, beta=0.99, gamma=2., e=0.1, neighbors=2, not_focal=False):
        super(MultiClassFocal, self).__init__()
        self.num_class = num_class
        self.beta = beta
        self.gamma = gamma
        self.e = e
        self.neighbors = neighbors
        self.not_focal = not_focal

        neighbor_mask = np.ones((self.num_class, self.num_class))
        neighbor_mask = neighbor_mask - np.triu(neighbor_mask, neighbors) - np.tril(neighbor_mask, -neighbors)
        neighbor_mask = neighbor_mask / (np.sum(neighbor_mask, axis=-1, keepdims=True) + 1e-10)
        self.neighbor_mask = Tensor(neighbor_mask, mstype.float32)

        self.class_weights = Parameter(Tensor(np.ones((self.num_class)) / self.num_class, dtype=mstype.float32), \
                                          name='class_weights', requires_grad=False)

        self.softmax = nn.Softmax(-1)
        self.cross_entropy = P.SoftmaxCrossEntropyWithLogits()
        self.zero = Tensor([0.])

        self.allreduce = P.AllReduce()
        self.device_num = D.get_group_size()

    def construct(self, prediction, target):
        """construct"""
        prediction_tensor = self.softmax(prediction)

        zeros = mnp.zeros_like(prediction_tensor)
        one_minus_p = mnp.where(target > 1e-5, target - prediction_tensor, zeros)
        ft = -1 * mnp.pow(one_minus_p, self.gamma) * mnp.log(mnp.clip(prediction_tensor, 1e-8, 1.0))

        classes_num = self._compute_classes_num(target)
        total_num = mnp.sum(classes_num)

        classes_w_t1 = total_num / classes_num
        sum_ = mnp.sum(classes_w_t1)
        classes_w_t2 = classes_w_t1 / sum_
        classes_w_tensor = F.cast(classes_w_t2, mstype.float32)

        weights = self.beta * self.class_weights + (1 - self.beta) * classes_w_tensor
        P.Assign()(self.class_weights, weights)

        classes_weight = mnp.broadcast_to(mnp.expand_dims(weights, 0), target.shape)
        alpha = mnp.where(target > zeros, classes_weight, zeros)

        balanced_fl = alpha * ft
        balanced_fl = mnp.sum(balanced_fl, -1)

        labels = P.MatMul()(target, self.neighbor_mask)
        xent, _ = self.cross_entropy(prediction, target)

        final_loss = (1 - self.e) * balanced_fl + self.e * xent

        if self.not_focal:
            softmax_xent, _ = self.cross_entropy(prediction, labels)
            final_loss = (1 - self.e) * softmax_xent + self.e * xent

        return final_loss

    def _compute_classes_num(self, target):
        "get global classes number"
        classes_num = mnp.sum(target, 0)
        classes_num = self.allreduce(classes_num)
        classes_num = F.cast(classes_num, mstype.float32)
        classes_num += 1.
        return classes_num


class BinaryFocal(nn.Cell):
    """Focal error for Binary classifications.
    Compute the binary classes focal error between `prediction` and the ground truth `target`.

    Lin, Tsung-Yi, et al. "Focal loss for dense object detection."

    Focal error = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p.

    Args:
      prediction: A float tensor of size [batch, ...].
      target: A float tensor of size [batch, ...].
      alpha: A float tensor specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      A scalar representing normalized total error.
    """

    def __init__(self, alpha=0.25, gamma=2., feed_in=False, not_focal=False):
        super(BinaryFocal, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.feed_in = feed_in
        self.not_focal = not_focal

        self.cross_entropy = P.BinaryCrossEntropy(reduction='none')
        self.sigmoid = P.Sigmoid()
        self.epsilon = 1e-8

    def construct(self, prediction, target):
        """construct"""
        epsilon = self.epsilon
        target = F.cast(target, mstype.float32)
        probs = F.cast(prediction, mstype.float32)
        if self.feed_in:
            probs = self.sigmoid(prediction)
        else:
            prediction = self._convert(prediction)

        ones_tensor = mnp.ones_like(target)
        positive_pt = mnp.where(target > 1e-5, probs, ones_tensor)
        negative_pt = mnp.where(target < 1e-5, 1 - probs, ones_tensor)

        focal_loss = -self.alpha * mnp.pow(1 - positive_pt, self.gamma) * \
                     mnp.log(mnp.clip(positive_pt, epsilon, 1.)) - (1 - self.alpha) * \
                     mnp.pow(1 - negative_pt, self.gamma) * mnp.log(mnp.clip(negative_pt, epsilon, 1.))
        focal_loss *= 2.

        if self.not_focal:
            focal_loss = self.cross_entropy(prediction, target, ones_tensor)

        return focal_loss

    def _convert(self, probs):
        """convert function"""
        probs = mnp.clip(probs, 1e-5, 1. - 1e-5)
        prediction = mnp.log(probs / (1 - probs))
        return prediction
