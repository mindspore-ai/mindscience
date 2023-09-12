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
"""Model definition"""
from mindspore import nn, ops, Tensor, float32


class Net(nn.Cell):
    """Network definition"""
    def __init__(self, fnn, odenn):
        super().__init__()
        self.fnn = fnn
        self.odenn = odenn

    def construct(self, t_train, t_bc_train):
        """Network forward pass"""
        u_pred = self.fnn(t_train)
        u_0_pred = self.fnn(t_bc_train)
        f_pred = self.odenn(t_train)
        return u_pred, u_0_pred, f_pred


class MyWithLossCell(nn.Cell):
    """Loss definition"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__()
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self, t_train, t_bc_train, u_0_train):
        """Network forward pass"""
        _, u_0_pred, f_pred = self._backbone(t_train, t_bc_train)
        return ops.add(self._loss_fn(f_pred, Tensor(0, dtype=float32)),
                       self._loss_fn(u_0_train, u_0_pred))
