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
build_feature
"""
import numpy as np

import mindspore.nn as nn
import mindspore.numpy as msnp
from mindspore import ops


class BuildFeature(nn.Cell):
    """BuildFeature"""
    def __init__(self):
        super(BuildFeature, self).__init__()
        self.mul = ops.Mul()
        self.div = ops.Div()
        self.sqrt = ops.Sqrt()
        self.add = ops.Add()
        self.square = ops.Square()
        self.tanh = ops.Tanh()
        self.pow = ops.Pow()
        self.concat = ops.Concat(1)
        self.abs = ops.Abs()
        self.sign = ops.Sign()
        self.tan = ops.Tan()

    def construct(self, mut_value, aoa, reynolds, dis, p_value, ru_value, y_value,
                  u_value, v_value, ux_value, uy_value, vx_value, vy_value):
        """construct"""
        # 涡量
        r_norm = self.sqrt(0.5 * self.square(uy_value - vx_value))

        # 与壁面距离有关的量
        da_r = self.mul(dis, self.mul(dis, self.mul(r_norm, (1 - self.tanh(dis)))))

        # 雷诺应力应变率
        s_norm = self.sqrt(self.square(ux_value) + self.square(vy_value) + 0.5 * self.square(
            uy_value + vx_value))

        # 熵
        entropy = 1.4 * self.div(p_value, self.pow(ru_value, 1.4)) - 1

        # exp func
        dref0 = 1.0 / (self.sqrt(reynolds))
        dref1 = ((dis + dref0) - self.abs(dis - dref0)) / 2
        dref2 = ((dis + dref0) + self.abs(dis - dref0)) / 2
        dsturb_min = dis.min()
        expfunc = self.pow(2.71828, self.sqrt(dref1 / dsturb_min))
        expfunc = self.mul(expfunc, self.sqrt(dref0 / dref2)) - 2

        # 速度的方向
        sig = self.sign(y_value)
        v_direct = msnp.arctan(self.mul(sig, v_value / u_value))
        # 速度投影
        proj_stream = self.mul(sig, self.add(-v_value, self.mul(u_value, self.tan(
            (np.pi * aoa) / 180.0))))

        # 涡黏系数处理
        mut_value = mut_value / (reynolds / 1e6)
        trans = 1.0 / self.pow(dis, 0.6)
        mut_value = self.mul(mut_value, trans)

        df_data = self.concat([msnp.expand_dims(u_value, 1), msnp.expand_dims(r_norm, 1),
                               msnp.expand_dims(s_norm, 1), msnp.expand_dims(entropy, 1),
                               msnp.expand_dims(da_r, 1), msnp.expand_dims(v_direct, 1),
                               msnp.expand_dims(proj_stream, 1), msnp.expand_dims(expfunc, 1),
                               msnp.expand_dims(mut_value, 1)])

        return df_data[:, :-1], df_data[:, -1]
