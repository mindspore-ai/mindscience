# Copyright 2021 Huawei Technologies Co., Ltd
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
Define the network of PFNN
A penalty-free neural network method for solving a class of
second-order boundary-value problems on complex geometries
"""
from mindspore import Tensor, Parameter
from mindspore import dtype as mstype
from mindspore import nn, ops
from mindspore.common.initializer import Normal


class LenFac(nn.Cell):
    """
    Caclulate the length

    Args:
        bounds: Boundary of area
    """

    def __init__(self, bounds, mu):
        super(LenFac, self).__init__()
        self.bounds = bounds
        self.hx = self.bounds[0, 1] - self.bounds[0, 0]
        self.mu = mu

    def construct(self, x):
        """forward"""
        return 1.0 - (1.0 - (x - self.bounds[0, 0]) / self.hx) ** self.mu


class NetG(nn.Cell):
    """NetG"""

    def __init__(self):
        super(NetG, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc2 = nn.Dense(10, 1, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.w_tensor = Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], mstype.float32)
        self.w = Parameter(self.w_tensor, name="w", requires_grad=False)
        self.matmul = nn.MatMul()

    def construct(self, x, label=None):
        """forward"""
        z = self.matmul(x, self.w)
        h = self.sin(self.fc0(x))
        ret = self.fc2(self.sin(self.fc1(h)) + z)

        if label is not None:
            ret = ops.square(ret - label).mean()
        return ret


class NetF(nn.Cell):
    """NetF"""

    def __init__(self):
        super(NetF, self).__init__()
        self.sin = ops.Sin()
        self.fc0 = nn.Dense(2, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc1 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc2 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc3 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc4 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc5 = nn.Dense(10, 10, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.fc6 = nn.Dense(10, 1, weight_init=Normal(0.2),
                            bias_init=Normal(0.2))
        self.w_tensor = Tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]], mstype.float32)
        self.w = Parameter(self.w_tensor, name="w", requires_grad=False)
        self.matmul = nn.MatMul()

    def construct(self, x):
        """forward"""
        z = self.matmul(x, self.w)
        h = self.sin(self.fc0(x))
        x = self.sin(self.fc1(h)) + z
        h = self.sin(self.fc2(x))
        x = self.sin(self.fc3(h)) + x
        h = self.sin(self.fc4(x))
        x = self.sin(self.fc5(h)) + x
        return self.fc6(x)


class Loss(nn.Cell):
    """NetLoss"""

    def __init__(self, net):
        super(Loss, self).__init__()
        self.matmul = nn.MatMul()
        self.grad = ops.composite.GradOperation()
        self.sum = ops.ReduceSum()
        self.mean = ops.ReduceMean()
        self.net = net

    def get_variable(self, inset_g, inset_l, inset_gx, inset_lx, inset_a, inset_size,
                     inset_dim, inset_area, inset_c, bdset_nlength, bdset_nr, bdset_nl, bdset_ng):
        """Get Parameters for NetLoss"""
        self.inset_size = inset_size
        self.inset_dim = inset_dim
        self.inset_area = inset_area
        self.bdset_nlength = bdset_nlength

        self.inset_g = Parameter(
            Tensor(inset_g, mstype.float32), name="InSet_g", requires_grad=False)
        self.inset_l = Parameter(
            Tensor(inset_l, mstype.float32), name="InSet_l", requires_grad=False)
        self.inset_gx = Parameter(
            Tensor(inset_gx, mstype.float32), name="InSet_gx", requires_grad=False)
        self.inset_lx = Parameter(
            Tensor(inset_lx, mstype.float32), name="InSet_lx", requires_grad=False)
        self.inset_a = Parameter(
            Tensor(inset_a, mstype.float32), name="InSet_a", requires_grad=False)
        self.inset_c = Parameter(
            Tensor(inset_c, mstype.float32), name="InSet_c", requires_grad=False)
        self.bdset_nr = Parameter(
            Tensor(bdset_nr, mstype.float32), name="BdSet_nr", requires_grad=False)
        self.bdset_nl = Parameter(
            Tensor(bdset_nl, mstype.float32), name="BdSet_nl", requires_grad=False)
        self.bdset_ng = Parameter(
            Tensor(bdset_ng, mstype.float32), name="BdSet_ng", requires_grad=False)

    def construct(self, inset_x, bdset_x):
        """forward"""
        inset_f = self.net(inset_x)
        inset_fx = self.grad(self.net)(inset_x)
        inset_u = self.inset_g + self.inset_l * inset_f
        inset_ux = self.inset_gx + self.inset_lx * inset_f + self.inset_l * inset_fx
        inset_aux = self.matmul(self.inset_a, inset_ux.reshape(
            (self.inset_size, self.inset_dim, 1)))
        inset_aux = inset_aux.reshape(self.inset_size, self.inset_dim)
        bdset_nu = self.bdset_ng + self.bdset_nl * self.net(bdset_x)
        return 0.5 * self.inset_area * self.sum(self.mean((inset_aux * inset_ux), 0)) + \
            self.inset_area * self.mean(self.inset_c * inset_u) - \
            self.bdset_nlength * self.mean(self.bdset_nr * bdset_nu)
