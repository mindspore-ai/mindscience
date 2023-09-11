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

"""network common"""
import math
from abc import abstractmethod

import mindspore as ms
from mindspore import nn, ops
from sciai.architecture import MLP, SSE, Normalize
from sciai.common import TrainCellWithCallBack, XavierTruncNormal, lbfgs_train
from sciai.operators import grad
from sciai.utils import print_log


class IdnNetF(nn.Cell):
    """idn net f"""
    def __init__(self, idn_net_u, net_pde):
        super(IdnNetF, self).__init__()
        self.idn_net_u = idn_net_u
        self.net_pde = net_pde
        self.grad_net = grad(idn_net_u)
        self.grad_net_uxx = grad(self.grad_net, output_index=1, input_index=1)


class IdnNetU(nn.Cell):
    """idn net u"""
    def __init__(self, net_u, lb_idn, ub_idn):
        super(IdnNetU, self).__init__()
        self.net_u = net_u
        self.normalize = Normalize(lb_idn, ub_idn)

    def construct(self, t, x):
        """Network forward pass"""
        x_ = ops.concat([t, x], 1)
        h = self.normalize(x_)
        u = self.net_u(h)
        return u


class LossNetU(nn.Cell):
    """loss net u"""
    def __init__(self, idn_net_u):
        super(LossNetU, self).__init__()
        self.idn_net_u = idn_net_u
        self.sse = SSE()

    def construct(self, t, x, u):
        """Network forward pass"""
        idn_u_pred = self.idn_net_u(t, x)
        idn_u_loss = self.sse(idn_u_pred - u)
        return idn_u_loss


class LossNetF(nn.Cell):
    """loss net f"""
    def __init__(self, idn_net_f):
        super(LossNetF, self).__init__()
        self.idn_net_f = idn_net_f
        self.sse = SSE()

    def construct(self, t, x):
        """Network forward pass"""
        idn_f_pred = self.idn_net_f(t, x)
        idn_f_loss = self.sse(idn_f_pred)
        return idn_f_loss


class SolNetMLP(nn.Cell):
    """sol net mlp"""
    def __init__(self, layers, lb_sol, ub_sol):
        super(SolNetMLP, self).__init__()
        self.mlp = MLP(layers=layers, weight_init=XavierTruncNormal(), bias_init="zeros", activation=ops.Sin())
        self.normalize = Normalize(lb_sol, ub_sol)

    def construct(self, t, x):
        """Network forward pass"""
        x_ = ops.concat([t, x], 1)
        h = self.normalize(x_)
        u = self.mlp(h)
        return u


class SolNet(nn.Cell):
    """sol net"""
    def __init__(self, net, lb_sol, ub_sol, net_pde):
        super(SolNet, self).__init__()
        self.net = net
        self.lb_sol, self.ub_sol = lb_sol, ub_sol
        self.net_pde = net_pde

        self.grad_net = grad(self.net)
        self.grad_net_uxx = grad(self.grad_net, output_index=1, input_index=1)
        self.sse = SSE()

    @abstractmethod
    def sol_net_u(self, t, x):
        pass

    @abstractmethod
    def sol_net_f(self, t, x):
        pass


class DeepHPM(nn.Cell):  # inherit from Cell only for model saving
    """deep hpm"""
    def __init__(self, u_layers, pde_layers, layers, lb_idn, ub_idn, lb_sol, ub_sol):
        super(DeepHPM, self).__init__()
        # Init for Identification
        self.net_u = MLP(layers=u_layers, weight_init=XavierTruncNormal(), bias_init="zeros", activation=ops.Sin())
        self.net_pde = MLP(layers=pde_layers, weight_init=XavierTruncNormal(), bias_init="zeros", activation=ops.Sin())
        self.idn_net_u = IdnNetU(self.net_u, lb_idn, ub_idn)
        self.idn_net_f = self.case_idn_net_f()
        self.loss_net_u = LossNetU(self.idn_net_u)
        self.loss_net_f = LossNetF(self.idn_net_f)

        # Initialize NNs for Solution
        self.net = SolNetMLP(layers, lb_sol, ub_sol)
        self.sol_net = self.case_sol_net(lb_sol, ub_sol)

    @abstractmethod
    def case_idn_net_f(self) -> IdnNetF:
        pass

    @abstractmethod
    def case_sol_net(self, lb_sol, ub_sol) -> SolNet:
        pass

    def idn_u_train(self, args, *inputs) -> None:
        """idn u train"""
        optim = nn.Adam(self.net_u.trainable_params(), learning_rate=args.lr)
        train_cell_u = TrainCellWithCallBack(self.loss_net_u, optim,
                                             loss_interval=args.print_interval, time_interval=args.print_interval,
                                             amp_level=args.amp_level)
        idn_u_loss = math.inf
        for _ in range(args.train_epoch):
            idn_u_loss = train_cell_u(*inputs)
        print_log("idn_u_train adam:", idn_u_loss)
        lbfgs_train(self.loss_net_u, inputs, lbfgs_iter=args.train_epoch_lbfgs)
        idn_u_loss = self.loss_net_u(*inputs)
        print_log("idn_u_train lbfgs:", idn_u_loss)

    def idn_f_train(self, args, *inputs) -> None:
        """idn f train"""
        optim = nn.Adam(self.net_pde.trainable_params(), learning_rate=args.lr)
        train_cell_f = TrainCellWithCallBack(self.loss_net_f, optim,
                                             loss_interval=args.print_interval, time_interval=args.print_interval,
                                             amp_level=args.amp_level)
        idn_f_loss = math.inf
        for _ in range(args.train_epoch):
            idn_f_loss = train_cell_f(*inputs)
        print_log("idn_f_train adam:", idn_f_loss)
        if args.train_epoch_lbfgs:
            lbfgs_train(self.loss_net_f, inputs, lbfgs_iter=args.train_epoch_lbfgs)
        idn_f_loss = self.loss_net_f(*inputs)
        print_log("idn_f_train lbfgs:", idn_f_loss)

    def idn_predict(self, t_star, x_star) -> (ms.Tensor, ms.Tensor):
        """idn predict"""
        u_star = self.idn_net_u(t_star, x_star)
        f_star = self.idn_net_f(t_star, x_star)
        return u_star, f_star

    def sol_train(self, *inputs) -> None:
        """sol train"""
        x0, u0, tb, x_f_, lb_sol, ub_sol, args = inputs
        x0_ = ops.concat((0 * x0, x0), 1)  # (0, x0)
        x_lb_ = ops.concat((tb, 0 * tb + lb_sol[1]), 1)  # (tb, lb[1])
        x_ub_ = ops.concat((tb, 0 * tb + ub_sol[1]), 1)  # (tb, ub[1])

        t0, x0 = x0_[:, 0:1], x0_[:, 1:2]  # Initial data (time, space)
        t_lb, x_lb = x_lb_[:, 0:1], x_lb_[:, 1:2]  # Boundary data (time, space) -- lower boundary
        t_ub, x_ub = x_ub_[:, 0:1], x_ub_[:, 1:2]  # Boundary data (time, space) -- upper boundary
        t_f, x_f = x_f_[:, 0:1], x_f_[:, 1:2]  # Collocation Points (time, space)

        optim = nn.Adam(self.net.trainable_params(), learning_rate=args.lr)
        train_cell_sol = TrainCellWithCallBack(self.sol_net, optim,
                                               time_interval=args.print_interval, loss_interval=args.print_interval,
                                               amp_level=args.amp_level)
        input_data = t0, x0, u0, t_lb, x_lb, t_ub, x_ub, t_f, x_f
        sol_loss = math.inf
        for _ in range(args.train_epoch):
            sol_loss = train_cell_sol(*input_data)
        print_log("sol_train adam:", sol_loss)
        sol_loss = self.sol_net(*input_data)
        print_log("sol_train lbfgs:", sol_loss)

    def sol_predict(self, t_star, x_star) -> (ms.Tensor, ms.Tensor):
        """sol predict"""
        u_star, _, _ = self.sol_net.sol_net_u(t_star, x_star)
        f_star = self.sol_net.sol_net_f(t_star, x_star)
        return u_star, f_star

    @abstractmethod
    def eval_idn(self, t_idn_star, x_idn_star, u_idn_star):
        pass

    @abstractmethod
    def eval_sol(self, *inputs):
        pass
