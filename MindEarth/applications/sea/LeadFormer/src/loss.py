# right 2020-2022 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore
import mindspore.nn as nn
import mindspore.common.dtype as mstype
import mindspore.ops.operations as F
from mindspore.ops import functional as F2
from mindspore.nn.cell import Cell
from mindspore import ops, Tensor

from .utils import warp, make_grid


class MyLoss(Cell):
    """
    Base class for other losses.
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction is None:
            reduction = "none"

        if reduction not in ("mean", "sum", "none"):
            raise ValueError(
                f"reduction method for {reduction.lower()} is not supported"
            )

        self.average = True
        self.reduce = True
        if reduction == "sum":
            self.average = False
        if reduction == "none":
            self.reduce = False

        self.reduce_mean = F.ReduceMean()
        self.reduce_sum = F.ReduceSum()
        self.mul = F.Mul()
        self.cast = F.Cast()

    def get_axis(self, x):
        shape = F2.shape(x)
        length = F2.tuple_len(shape)
        perm = F2.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)
        if self.reduce:
            axis = self.get_axis(x)
            if self.average:
                x = self.reduce_mean(x, axis)
            else:
                x = self.reduce_sum(x, axis)
        x = self.cast(x, input_dtype)
        return x

    def construct(self, logits, label):
        """unused"""
        raise NotImplementedError


class CrossEntropyWithLogits(MyLoss):
    """CrossEntropyWithLogits"""
    def __init__(self):
        super().__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.softmax_cross_entropy_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.cast = F.Cast()

    def construct(self, logits, label):
        """construct"""
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))
        _, _, _, c = F.Shape()(label)

        loss = self.reduce_mean(
            self.softmax_cross_entropy_loss(
                self.reshape_fn(logits, (-1, c)), self.reshape_fn(label, (-1, c))
            )
        )
        return self.get_loss(loss)


class MultiCrossEntropyWithLogits(nn.Cell):
    """MultiCrossEntropyWithLogits"""
    def __init__(self):
        super().__init__()
        self.loss = CrossEntropyWithLogits()
        self.squeeze = F.Squeeze(axis=0)

    def construct(self, logits, label):
        """construct"""
        total_loss = 0
        for i in range(len(logits)):
            total_loss += self.loss(self.squeeze(logits[i : i + 1]), label)
        return total_loss


class MSELoss(MyLoss):
    """MSEloss"""
    def __init__(self):
        super().__init__()
        self.transpose_fn = F.Transpose()
        self.reshape_fn = F.Reshape()
        self.mse_loss = nn.MSELoss()
        self.dice_loss = nn.DiceLoss(smooth=1e-5)
        self.cast = F.Cast()
        self.mae_loss = nn.MAELoss(reduction="mean")
        self.rmse_loss = nn.RMSELoss()

    def construct(self, logits, label):
        """construct"""
        print(logits.shape, label.shape)
        logits = self.transpose_fn(logits, (0, 2, 3, 1))
        logits = self.cast(logits, mindspore.float32)
        label = self.transpose_fn(label, (0, 2, 3, 1))
        label = label[:, :, :, 1]
        label = label.unsqueeze(dim=3)
        print(logits.shape, label.shape)
        _, _, _, c = F.Shape()(label)

        rmse_loss = self.rmse_loss(
            self.reshape_fn(logits, (-1, c)), self.reshape_fn(label, (-1, c))
        )
        return self.get_loss(rmse_loss)


class MultiMSELoss(nn.Cell):
    """MultiMSELoss"""
    def __init__(self):
        super().__init__()
        self.loss = MSELoss()
        self.squeeze = F.Squeeze(axis=0)

    def construct(self, logits, label):
        print(logits.shape, label.shape)
        total_loss = 0
        for i in range(len(logits)):
            total_loss += self.loss(self.squeeze(logits[i : i + 1]), label)
        return total_loss


class WeightDistance(nn.Cell):
    """Weighted L1 distance"""

    def construct(self, true_frame, pred_frame):
        loss = ops.mean(ops.abs(true_frame - pred_frame))
        return loss


class MotionLossNet(nn.Cell):
    """Motion regularization"""
    def __init__(self, in_channels=3, out_channels=3, kernel_size=3):
        super().__init__()
        kernel_v = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).astype(np.float32)
        kernel_h = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        self.weight_vt1 = self.get_kernels(kernel_v, out_channels)
        self.weight_vt2 = self.get_kernels(kernel_h, out_channels)
        self.conv_2d_v = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, has_bias=False
        )
        self.conv_2d_v.weight.set_data(Tensor(self.weight_vt1, mindspore.float32))
        for w in self.conv_2d_v.trainable_params():
            w.requires_grad = False
        self.conv_2d_h = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, has_bias=False
        )
        self.conv_2d_h.weight.set_data(Tensor(self.weight_vt2, mindspore.float32))
        for w in self.conv_2d_h.trainable_params():
            w.requires_grad = False

    @staticmethod
    def get_kernels(kernel, repeats):
        kernel = np.expand_dims(kernel, axis=(0, 1))
        kernels = [kernel] * repeats
        kernels = np.concatenate(kernels, axis=0)
        return kernels

    def calc_diff_v(self, image):
        diff_v1 = self.conv_2d_v(image)
        diff_v2 = self.conv_2d_h(image)
        lambda_v = diff_v1**2 + diff_v2**2
        loss = ops.sum(lambda_v)
        return loss

    def custom_2d_conv_sobel(self, image):
        motion_loss1 = self.calc_diff_v(image)
        motion_loss2 = self.calc_diff_v(image)
        loss = (motion_loss1 + motion_loss2) / (
            image.shape[0] * image.shape[-1] * image.shape[-2]
        )
        return loss

    def construct(self, motion):
        loss1 = self.custom_2d_conv_sobel(motion[:, :1])
        loss2 = self.custom_2d_conv_sobel(motion[:, 1:])
        return loss1 + loss2


class EvolutionLoss(nn.Cell):
    """Evolution loss definition"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn_accum = WeightDistance()
        self.loss_fn_motion = MotionLossNet(
            in_channels=1, out_channels=1, kernel_size=3
        )
        sample_tensor = np.zeros((1, 1, 2000, 2000)).astype(np.float32)
        self.grid = Tensor(make_grid(sample_tensor), mindspore.float32)
        self.lamb = float(1e-2)

    def construct(self, logits, labels):
        """construct"""
        intensity = logits[0]
        motion = logits[1]
        batch, _, height, width = logits[0].shape
        motion_ = motion.reshape(batch, 1, 2, height, width)
        intensity_ = intensity.reshape(batch, 1, 1, height, width)
        accum = 0
        last_frame = labels[:, 0, :, :]
        last_frame = ops.unsqueeze(last_frame, dim=1)
        grid = self.grid.tile((batch, 1, 1, 1))
        next_frame = labels[:, 1, :, :]
        next_frame = ops.unsqueeze(next_frame, dim=1)
        xt_1 = warp(
            last_frame, motion_[:, 0], grid, mode="bilinear", padding_mode="border"
        )
        accum += self.loss_fn_accum(next_frame, xt_1)
        last_frame = warp(
            last_frame, motion_[:, 0], grid, mode="nearest", padding_mode="border"
        )
        last_frame = last_frame + intensity_[:, 0]
        accum += self.loss_fn_accum(next_frame, last_frame)
        motion = self.loss_fn_motion(motion_[:, 0])
        loss = accum + self.lamb * motion
        return loss
