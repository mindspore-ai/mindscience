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
postprocess
"""
import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore import ops


class PostProcess2DMinMax(nn.Cell):
    """
    推理后处理
    """
    def __init__(self, reynolds, dis, sca_max, sca_min):
        super(PostProcess2DMinMax, self).__init__()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.pow = ops.Pow()
        self.low_value = 0.0
        self.up_value = 1.0
        self.sca_max = sca_max
        self.sca_min = sca_min
        self.reynolds = reynolds
        self.dis = dis

    def construct(self, pred):
        """construct"""
        # 1、 限制异值
        pred = msnp.clip(pred, self.low_value, self.up_value)
        # 2、 反归一化
        pred = pred[:, 0]
        pred = self.add(self.mul(pred, (self.sca_max - self.sca_min)), self.sca_min)
        # 3、 逆尺度变换
        trans = 1.0 / self.pow(self.dis, 0.6)
        reynolds = self.reynolds * 1e-6
        pred = pred / trans * reynolds
        return pred


class PostProcess2DStd(nn.Cell):
    """
    推理后处理
    """
    def __init__(self, reynolds, dis, df_mean, df_std):
        super(PostProcess2DStd, self).__init__()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.pow = ops.Pow()
        self.low_value = 0.0
        self.up_value = 1.0
        self.df_mean = df_mean
        self.df_std = df_std
        self.reynolds = reynolds
        self.dis = dis

    def construct(self, pred):
        """construct"""
        # 1、 反归一化
        pred = pred[:, 0]
        pred = self.add(self.mul(pred, self.df_std), self.df_mean)
        # 2、 逆尺度变换
        trans = 1.0 / self.pow(self.dis, 0.6)
        reynolds = self.reynolds * 1e-6
        pred = pred / trans * reynolds
        return pred


class PostProcess3DMinMax(nn.Cell):
    """
    推理后处理
    """
    def __init__(self, reynolds, sca_max, sca_min):
        super(PostProcess3DMinMax, self).__init__()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.low_value = 0.0
        self.up_value = 1.0
        self.sca_max = sca_max
        self.sca_min = sca_min
        self.reynolds = reynolds

    def construct(self, pred):
        """construct"""
        # 1、 限制异值
        pred = msnp.clip(pred, self.low_value, self.up_value)
        # 2、 反归一化
        pred = pred[:, 0]
        pred = self.add(self.mul(pred, (self.sca_max-self.sca_min)), self.sca_min)
        # 3、 逆尺度变换
        reynolds = self.reynolds * 1e-6
        pred = pred * reynolds
        return pred
    