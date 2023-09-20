# ============================================================================
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
"""
Loss
"""
import mindspore.nn as nn
from mindspore import ops


class CustomWithLossCell2D(nn.Cell):
    """CustomWithLossCell2D"""
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell2D, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, sij, rs_value):
        """construct"""
        output = self._backbone(data)
        return self._loss_fn(output, label, sij, rs_value)


# 定义损失函数
class LossFunc2D(nn.Cell):
    """
    MindSpore自定义构建损失函数
    """
    def __init__(self):
        super(LossFunc2D, self).__init__()
        self.criterion = nn.MSELoss()
        self.abs = ops.Abs()
        self.sqrt = ops.Sqrt()
        self.e_value = 2.71828

    def rs_error(self, pred, sij, rs_value):
        """
        近壁面雷诺应力损失
        """
        rs1 = 2.0 * pred * sij
        error = self.criterion(rs1, rs_value)
        return error

    def construct(self, pred, label, sij, rs_value):
        """construct"""
        # 预测值不会小于0
        lx0 = ((self.abs(pred) - pred) / 2.0).mean()
        lx0 = lx0 ** 2
        # 数据之间的误差
        lx2 = self.criterion(pred, label)
        # 雷诺应力损失
        lx3 = 1e-2 * self.rs_error(pred, sij, rs_value)
        loss = lx0 + lx2 + lx3
        return loss


# 定义用于验证的损失函数
class LossToEval2D(nn.Cell):
    """
    MindSpore自定义构建损失函数
    """
    def __init__(self):
        super(LossToEval2D, self).__init__()
        self.criterion = nn.MSELoss()
        self.abs = ops.Abs()
        self.relu = ops.ReLU()
        self.sqrt = ops.Sqrt()
        self.e_value = 2.71828

    def rs_error(self, pred, sij, rs_value):
        """
        近壁面雷诺应力损失
        """
        rs1 = 2.0 * pred * sij
        error = self.criterion(rs1, rs_value)
        return error

    def construct(self, pred, label, sij, rs_value):
        # 预测值不会小于0
        lx0 = ((self.abs(pred) - pred) / 2.0).mean()
        lx0 = lx0 ** 2
        # 数据之间的误差
        lx2 = self.criterion(pred, label)
        # 雷诺应力损失
        lx3 = 1e-2 * self.rs_error(pred, sij, rs_value)
        loss = lx0 + lx2 + lx3
        return loss, lx3


class LossFunc3D(nn.Cell):
    """
    MindSpore自定义构建损失函数
    """
    def __init__(self, sca_min, sca_max):
        super(LossFunc3D, self).__init__()
        self.criterion = nn.MSELoss()
        self.abs = ops.Abs()
        self.sqrt = ops.Sqrt()
        self.sca_min = sca_min
        self.sca_max = sca_max
        self.e_value = 2.71828
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()

    def rs_error(self, pred, sij, rs_value):
        """
        近壁面雷诺应力损失
        """
        rs1 = 2.0 * pred * sij
        error = self.criterion(rs1, rs_value)
        return error

    def weighted_loss(self, dis, label, pred):
        """
        基于壁面距离权重的损失
        """
        loss = (dis * (pred - label) ** 2).mean()
        return loss

    def r2_loss(self, label, pred):
        """
        预测值与真实值之间的R2
        """
        y_mean = self.mean(label)
        ss_tot = self.sum((label - y_mean) ** 2)
        ss_res = self.sum((label - pred) ** 2)
        r2_value = 1.0 - (ss_res / ss_tot)
        loss = 1.0 - r2_value
        return loss

    def construct(self, pred, label, dis, sij, rs_value):
        """construct"""
        # 预测值不会小于0
        lx0 = ((self.abs(pred) - pred) / 2.0).mean()
        lx0 = lx0 ** 2
        # 数据之间的误差
        lx2 = self.weighted_loss(dis, label, pred)
        # 雷诺应力损失
        lx3 = 1.0e-2 * self.rs_error(pred, sij, rs_value)
        # r2损失
        lx4 = 1.0e-3 * self.r2_loss(label, pred)
        loss = lx0 + lx2 + lx3 + lx4
        return loss


# 定义用于验证的损失函数
class LossToEval3D(nn.Cell):
    """
    MindSpore自定义构建损失函数
    """
    def __init__(self, sca_min, sca_max):
        super(LossToEval3D, self).__init__()
        self.criterion = nn.MSELoss()
        self.abs = ops.Abs()
        self.sqrt = ops.Sqrt()
        self.sca_min = sca_min
        self.sca_max = sca_max
        self.e_value = 2.71828
        self.mean = ops.ReduceMean()
        self.sum = ops.ReduceSum()

    def rs_error(self, pred, sij, rs_value):
        """
        近壁面雷诺应力损失
        """
        rs1 = 2.0 * pred * sij
        error = self.criterion(rs1, rs_value)
        return error

    def weighted_loss(self, dis, label, pred):
        """
        基于壁面距离权重的损失
        """
        loss = (dis * (pred - label) ** 2).mean()
        return loss

    def r2_loss(self, label, pred):
        """
        预测值与真实值之间的R2
        """
        y_mean = self.mean(label)
        ss_tot = self.sum((label - y_mean) ** 2)
        ss_res = self.sum((label - pred) ** 2)
        r2_value = 1.0 - (ss_res / ss_tot)
        loss = 1.0 - r2_value
        return loss

    def construct(self, pred, label, dis, sij, rs_value):
        """construct"""
        # 预测值不会小于0
        lx0 = ((self.abs(pred) - pred) / 2.0).mean()
        lx0 = lx0 ** 2
        # 数据之间的误差
        lx2 = self.weighted_loss(dis, label, pred)
        # 雷诺应力损失
        lx3 = 1.0e-2 * self.rs_error(pred, sij, rs_value)
        # r2损失
        lx4 = self.r2_loss(label, pred)
        loss = lx0 + lx2 + lx3 + 1.0e-3 * lx4
        return loss, lx3, lx4, (lx0, lx2)


class CustomWithLossCell3D(nn.Cell):
    """CustomWithLossCell3D"""
    def __init__(self, backbone, loss_fn):
        super(CustomWithLossCell3D, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, data, label, dis, sij, rs_value):
        """construct"""
        output = self._backbone(data)
        return self._loss_fn(output, label, dis, sij, rs_value)
    