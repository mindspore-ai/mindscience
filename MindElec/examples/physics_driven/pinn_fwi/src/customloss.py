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
"""define the loss"""

import mindspore as ms
from mindspore import nn, ops


def g(x, z, a, b, c, d):
    return (x - c) ** 2 / a ** 2 + (z - d) ** 2 / b ** 2


def alpha_true_func(data, args):
    """alpha_true_func"""
    dx = args['ax_spec'] / args['nx']
    dz = args['az_spec'] / args['nz']
    x = data[:, 0:1]
    z = data[:, 1:2]
    alpha_true = 3 - 0.25 * (1 + ops.tanh(
        100 * (1 - g(x * args['Lx'], z * args['Lz'], 0.18, 0.1,
                     1.0 - args['n_absx'] * dx, 0.3 - args['n_absz'] * dz))
    ))

    return alpha_true


class GradWrtX(nn.Cell):
    """gradient w.r.t. x"""

    def __init__(self, network):
        super(GradWrtX, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x, z, t):
        gout = self.grad(self.network)(x, z, t)
        gradient_x = gout[0]
        return gradient_x


class GradWrtZ(nn.Cell):
    """gradient w.r.t. z"""

    def __init__(self, network):
        super(GradWrtZ, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x, z, t):
        gout = self.grad(self.network)(x, z, t)
        gradient_z = gout[1]
        return gradient_z


class GradWrtT(nn.Cell):
    """gradient w.r.t. t"""

    def __init__(self, network):
        super(GradWrtT, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x, z, t):
        gout = self.grad(self.network)(x, z, t)
        gradient_t = gout[2]
        return gradient_t


class GradWrtXZT(nn.Cell):
    """gradient w.r.t. x,z,t"""

    def __init__(self, network):
        super(GradWrtXZT, self).__init__()
        self.grad = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x, z, t):
        gout = self.grad(self.network)(x, z, t)
        gradient_x = gout[0]
        gradient_z = gout[1]
        gradient_t = gout[2]
        return gradient_x, gradient_z, gradient_t


class GradSec(nn.Cell):
    """二阶求导"""

    def __init__(self, network):
        super(GradSec, self).__init__()
        self.grad_op = ops.GradOperation(get_all=True)
        self.network = network

    def construct(self, x, z, t):
        gradient_function = self.grad_op(self.network)
        gout = gradient_function(x, z, t)
        return gout


class CustomWithLossCell(nn.Cell):
    """define the training loss"""

    def __init__(self, args, neural_net, neural_net0, u_ini1x, u_ini1z, u_ini2x, u_ini2z, s_x, s_z, n_1, n_2, n_3, n_4):
        super(CustomWithLossCell, self).__init__(auto_prefix=False)
        dx = args['ax_spec'] / args['nx']
        dz = args['az_spec'] / args['nz']
        self.neural_net = neural_net
        self.neural_net0 = neural_net0

        self.z_st = 0.1 - args['n_absz'] * dz
        self.z_fi = 0.45 - args['n_absz'] * dz
        self.x_st = 0.7 - args['n_absx'] * dx
        self.x_fi = 1.25 - args['n_absx'] * dx
        self.lld = ms.Tensor(1000.0, dtype=ms.float32)
        self.l_z = ms.Tensor(args['Lz'], dtype=ms.float32)
        self.l_x = ms.Tensor(args['Lx'], dtype=ms.float32)

        self.fisrt_grad = GradWrtXZT(neural_net)
        self.secondgradxx = GradSec(GradWrtX(neural_net))
        self.secondgradzz = GradSec(GradWrtZ(neural_net))
        self.secondgradtt = GradSec(GradWrtT(neural_net))

        self.op_square = ops.Square()
        self.op_reduce_mean = ops.ReduceMean(keep_dims=False)

        self.u_ini1x = ms.Tensor(u_ini1x, dtype=ms.float32)
        self.u_ini1z = ms.Tensor(u_ini1z, dtype=ms.float32)
        self.u_ini2x = ms.Tensor(u_ini2x, dtype=ms.float32)
        self.u_ini2z = ms.Tensor(u_ini2z, dtype=ms.float32)
        self.s_x = ms.Tensor(s_x, dtype=ms.float32)
        self.s_z = ms.Tensor(s_z, dtype=ms.float32)

        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.n_4 = n_4

    def construct(self, data):
        """ wave equation"""

        x = data[:, 0:1]
        z = data[:, 1:2]
        t = data[:, 2:3]

        alpha_star = ops.tanh(self.neural_net0(x, z))
        alpha_bound = 0.5 * (1 + ops.tanh(self.lld * (z - self.z_st / self.l_z))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (-z + self.z_fi / self.l_z))) \
                      * 0.5 * (1 + ops.tanh(self.lld * (x - self.x_st / self.l_x))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (-x + self.x_fi / self.l_x)))
        alpha = 3 + 2 * alpha_star * alpha_bound

        ux, uz, _ = self.fisrt_grad(x, z, t)
        sg_xx = self.secondgradxx(x, z, t)[0]
        sg_zz = self.secondgradzz(x, z, t)[1]
        sg_tt = self.secondgradtt(x, z, t)[2]
        p = (1 / self.l_x) ** 2 * sg_xx + (1 / self.l_z) ** 2 * sg_zz
        eq = sg_tt - alpha ** 2 * p

        loss_pde = self.op_reduce_mean(self.op_square(eq[:self.n_1, 0:1]))
        loss_init_disp1_x = self.op_reduce_mean(self.op_square(ux[self.n_1:(self.n_1 + self.n_2), 0:1] - self.u_ini1x))
        loss_init_disp1_z = self.op_reduce_mean(self.op_square(uz[self.n_1:(self.n_1 + self.n_2), 0:1] - self.u_ini1z))
        loss_init_disp1 = loss_init_disp1_x + loss_init_disp1_z

        loss_init_disp2_x = self.op_reduce_mean(
            self.op_square(ux[(self.n_1 + self.n_2):(self.n_1 + self.n_2 + self.n_3), 0:1] - self.u_ini2x))
        loss_init_disp2_z = self.op_reduce_mean(
            self.op_square(uz[(self.n_1 + self.n_2):(self.n_1 + self.n_2 + self.n_3), 0:1] - self.u_ini2z))
        loss_init_disp2 = loss_init_disp2_x + loss_init_disp2_z

        loss_seism_x = self.op_reduce_mean(self.op_square(
            ux[(self.n_1 + self.n_2 + self.n_3):(self.n_1 + self.n_2 + self.n_3 + self.n_4), 0:1] - self.s_x))
        loss_seism_z = self.op_reduce_mean(self.op_square(
            uz[(self.n_1 + self.n_2 + self.n_3):(self.n_1 + self.n_2 + self.n_3 + self.n_4), 0:1] - self.s_z))
        loss_seism = loss_seism_x + loss_seism_z

        loss_bc = self.op_reduce_mean(self.op_square(p[(self.n_1 + self.n_2 + self.n_3 + self.n_4):, 0:1]))

        loss = 1e-1 * loss_pde + loss_init_disp1 + loss_init_disp2 + loss_seism + 1e-1 * loss_bc

        return loss


class CustomWithEvalCell(nn.Cell):
    """used for test process"""

    def __init__(self, args, neural_net, neural_net0, u_ini1x, u_ini1z, u_ini2x, u_ini2z, s_x, s_z, n_1, n_2, n_3, n_4):
        super(CustomWithEvalCell, self).__init__(auto_prefix=False)
        self.neural_net = neural_net
        self.neural_net0 = neural_net0
        dx = args['ax_spec'] / args['nx']
        dz = args['az_spec'] / args['nz']
        self.z_st = 0.1 - args['n_absz'] * dz
        self.z_fi = 0.45 - args['n_absz'] * dz
        self.x_st = 0.7 - args['n_absx'] * dx
        self.x_fi = 1.25 - args['n_absx'] * dx
        self.lld = ms.Tensor(1000.0, dtype=ms.float32)
        self.l_z = ms.Tensor(args['Lz'], dtype=ms.float32)
        self.l_x = ms.Tensor(args['Lx'], dtype=ms.float32)

        self.fisrt_grad = GradWrtXZT(neural_net)
        self.secondgradxx = GradSec(GradWrtX(neural_net))
        self.secondgradzz = GradSec(GradWrtZ(neural_net))
        self.secondgradtt = GradSec(GradWrtT(neural_net))

        self.op_square = ops.Square()
        self.op_reduce_mean = ops.ReduceMean(keep_dims=False)

        self.u_ini1x = ms.Tensor(u_ini1x, dtype=ms.float32)
        self.u_ini1z = ms.Tensor(u_ini1z, dtype=ms.float32)
        self.u_ini2x = ms.Tensor(u_ini2x, dtype=ms.float32)
        self.u_ini2z = ms.Tensor(u_ini2z, dtype=ms.float32)
        self.s_x = ms.Tensor(s_x, dtype=ms.float32)
        self.s_z = ms.Tensor(s_z, dtype=ms.float32)

        self.n_1 = n_1
        self.n_2 = n_2
        self.n_3 = n_3
        self.n_4 = n_4

    def construct(self, data):
        """ wave equation"""

        x = data[:, 0:1]
        z = data[:, 1:2]
        t = data[:, 2:3]

        alpha_star = ops.tanh(self.neural_net0(x, z))
        alpha_bound = 0.5 * (1 + ops.tanh(self.lld * (z - self.z_st / self.l_z))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (-z + self.z_fi / self.l_z))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (x - self.x_st / self.l_x))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (-x + self.x_fi / self.l_x)))
        alpha = 3 + 2 * alpha_star * alpha_bound

        ux, uz, _ = self.fisrt_grad(x, z, t)
        sg_xx = self.secondgradxx(x, z, t)[0]
        sg_zz = self.secondgradzz(x, z, t)[1]
        sg_tt = self.secondgradtt(x, z, t)[2]
        p = (1 / self.l_x) ** 2 * sg_xx + (1 / self.l_z) ** 2 * sg_zz
        eq = sg_tt - alpha ** 2 * p

        loss_pde = self.op_reduce_mean(self.op_square(eq[:self.n_1, 0:1]))

        loss_init_disp1 = self.op_reduce_mean(self.op_square(
            ux[self.n_1:(self.n_1 + self.n_2), 0:1] - self.u_ini1x)) \
                          + self.op_reduce_mean(self.op_square(uz[self.n_1:(self.n_1 + self.n_2), 0:1] - self.u_ini1z))

        loss_init_disp2 = self.op_reduce_mean(self.op_square(
            ux[(self.n_1 + self.n_2):(self.n_1 + self.n_2 + self.n_3), 0:1] - self.u_ini2x)) \
            + self.op_reduce_mean(self.op_square(
                uz[(self.n_1 + self.n_2):(self.n_1 + self.n_2 + self.n_3), 0:1] - self.u_ini2z))

        loss_seism = self.op_reduce_mean(self.op_square(
            ux[(self.n_1 + self.n_2 + self.n_3):(self.n_1 + self.n_2 + self.n_3 + self.n_4), 0:1] - self.s_x)) \
            + self.op_reduce_mean(self.op_square(
                uz[(self.n_1 + self.n_2 + self.n_3):(self.n_1 + self.n_2 + self.n_3 + self.n_4), 0:1] - self.s_z))

        loss_bc = self.op_reduce_mean(self.op_square(
            p[(self.n_1 + self.n_2 + self.n_3 + self.n_4):, 0:1]))

        loss = 0.1 * loss_pde + loss_init_disp1 + loss_init_disp2 + loss_seism + 0.1 * loss_bc

        loss_col = [loss, loss_pde, loss_init_disp1,
                    loss_init_disp2, loss_seism, loss_bc]

        return loss_col


class CustomWithEval2Cell(nn.Cell):
    """used for eval process"""

    def __init__(self, args, neural_net, neural_net0):
        super(CustomWithEval2Cell, self).__init__(auto_prefix=False)
        self.neural_net = neural_net
        self.neural_net0 = neural_net0
        dx = args['ax_spec'] / args['nx']
        dz = args['az_spec'] / args['nz']
        self.z_st = 0.1 - args['n_absz'] * dz
        self.z_fi = 0.45 - args['n_absz'] * dz
        self.x_st = 0.7 - args['n_absx'] * dx
        self.x_fi = 1.25 - args['n_absx'] * dx
        self.lld = ms.Tensor(1000.0, dtype=ms.float32)
        self.l_z = ms.Tensor(args['Lz'], dtype=ms.float32)
        self.l_x = ms.Tensor(args['Lx'], dtype=ms.float32)

        self.fisrt_grad = GradWrtXZT(neural_net)

    def construct(self, data):
        """计算方程中的传播速率与相应波势"""

        x = data[:, 0:1]
        z = data[:, 1:2]
        t = data[:, 2:3]

        alpha_star = ops.tanh(self.neural_net0(x, z))
        alpha_bound = 0.5 * (1 + ops.tanh(self.lld * (z - self.z_st / self.l_z))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (-z + self.z_fi / self.l_z))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (x - self.x_st / self.l_x))) * 0.5 \
                      * (1 + ops.tanh(self.lld * (-x + self.x_fi / self.l_x)))
        alpha = 3 + 2 * alpha_star * alpha_bound

        ux, uz, _ = self.fisrt_grad(x, z, t)
        return ux, uz, alpha
