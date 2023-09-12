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
burgers different case.
"""
from mindspore import ops
from sciai.utils import print_log
from sciai.architecture import SSE

from .network_common import DeepHPM, IdnNetF, SolNet


class IdnNetFBurgersDiff(IdnNetF):
    """
    idn_net_f burgers diff
    """

    def construct(self, t, x):
        """Network forward pass"""
        u = self.idn_net_u(t, x)
        u_t, u_x = self.grad_net(t, x)
        u_xx = self.grad_net_uxx(t, x)
        terms = ops.concat([u, u_x, u_xx], 1)
        f = u_t - self.net_pde(terms)
        return f


class DeepHPMBurgersDiff(DeepHPM):
    """
    Deep hpms burgers different.
    """

    def case_idn_net_f(self):
        return IdnNetFBurgersDiff(self.idn_net_u, self.net_pde)

    def case_sol_net(self, lb_sol, ub_sol):
        return SolNetBurgersDiff(self.net, lb_sol, ub_sol, self.net_pde)

    def sol_predict(self, t_star, x_star):
        u_star, _, _ = self.sol_net.sol_net_u(t_star, x_star)
        f_star = self.sol_net.sol_net_f(t_star, x_star)
        return u_star, f_star

    def eval_idn(self, t_idn_star, x_idn_star, u_idn_star):
        u_pred_identifier, _ = self.idn_predict(t_idn_star, x_idn_star)
        error_u_identifier = SSE()(u_idn_star - u_pred_identifier) / SSE()(u_idn_star)
        print_log(f'Error u: {error_u_identifier}')

    def eval_sol(self, *inputs):
        t_sol_star, u_sol_star, x_sol_star, _, _, _ = inputs
        u_pred, _ = self.sol_predict(t_sol_star, x_sol_star)
        error_u = SSE()(u_sol_star - u_pred) / SSE()(u_sol_star)
        print_log(f'Error u: {error_u}')
        return u_pred


class SolNetBurgersDiff(SolNet):
    """
    sol_net burgers different case.
    """

    def construct(self, *inputs):
        """Network forward pass"""
        t0, x0, u0, t_lb, x_lb, t_ub, x_ub, t_f, x_f = inputs
        u0_pred, _, _ = self.sol_net_u(t0, x0)
        u_lb_pred, u_x_lb_pred, _ = self.sol_net_u(t_lb, x_lb)
        u_ub_pred, u_x_ub_pred, _ = self.sol_net_u(t_ub, x_ub)
        sol_f_pred = self.sol_net_f(t_f, x_f)

        sol_loss = self.sse(u0 - u0_pred) + \
                   self.sse(u_lb_pred - u_ub_pred) + \
                   self.sse(u_x_lb_pred - u_x_ub_pred) + \
                   self.sse(sol_f_pred)
        return sol_loss

    def sol_net_u(self, t, x):
        u = self.net(t, x)
        u_t, u_x = self.grad_net(t, x)
        return u, u_x, u_t

    def sol_net_f(self, t, x):
        u, u_x, u_t = self.sol_net_u(t, x)
        u_xx = self.grad_net_uxx(t, x)
        terms = ops.concat([u, u_x, u_xx], 1)
        f = u_t - self.net_pde(terms)
        return f
