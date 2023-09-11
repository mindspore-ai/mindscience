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

"""network for kdv same"""
import mindspore as ms
from mindspore import ops
from sciai.operators import grad
from sciai.utils import print_log
from sciai.architecture import SSE

from .network_common import IdnNetF, SolNet, DeepHPM


def calc_2_norm_in_batch(x, batch_num=1):
    """calculate 2-norm of x, but split it in several batches"""
    length = x.shape[0]
    batch_size = length // batch_num
    if batch_size == 0 or batch_num == 1:
        res = SSE()(x)
    else:
        index_begin = ops.arange(start=0, end=((batch_num - 1) * batch_size + 1), step=batch_size)
        index_end = ops.arange(start=(batch_size - 1), end=(batch_num * batch_size + 1), step=batch_size)
        index_end[-1] = length - 1
        res = ms.Tensor(0, dtype=ops.dtype(x))
        for i in range(batch_num):
            res += ops.reduce_sum(ops.square(x[index_begin[i]:index_end[i]]))
        res = ops.sqrt(res)
    return res


class IdnNetFKdvSame(IdnNetF):
    """idn_net kdv same case"""
    def __init__(self, idn_net_u, net_pde):
        super(IdnNetFKdvSame, self).__init__(idn_net_u, net_pde)
        self.grad_net_uxxx = grad(self.grad_net_uxx, input_index=1)

    def construct(self, t, x):
        """Network forward pass"""
        u = self.idn_net_u(t, x)
        u_t, u_x = self.grad_net(t, x)
        u_xx = self.grad_net_uxx(t, x)
        u_xxx = self.grad_net_uxxx(t, x)
        terms = ops.concat([u, u_x, u_xx, u_xxx], 1)
        f = u_t - self.net_pde(terms)
        return f


class DeepHPMKdvSame(DeepHPM):
    """deep hpm kdv same case"""
    def case_idn_net_f(self):
        return IdnNetFKdvSame(self.idn_net_u, self.net_pde)

    def case_sol_net(self, lb_sol, ub_sol):
        return SolNetKdvSame(self.net, lb_sol, ub_sol, self.net_pde)

    def sol_predict(self, t_star, x_star):
        u_star, _, _, _ = self.sol_net.sol_net_u(t_star, x_star)
        f_star = self.sol_net.sol_net_f(t_star, x_star)
        return u_star, f_star

    def eval_idn(self, t_idn_star, x_idn_star, u_idn_star):
        u_pred_identifier, _ = self.idn_predict(t_idn_star, x_idn_star)
        norm_error_u = calc_2_norm_in_batch(u_idn_star - u_pred_identifier, 3)
        norm_u_idn_star = calc_2_norm_in_batch(u_idn_star, 3)
        error_u_identifier = norm_error_u / norm_u_idn_star
        print_log(f'Error u: {error_u_identifier}')

    def eval_sol(self, *inputs):
        t_sol_star, u_sol_star, x_sol_star, t_idn_star, x_idn_star, u_idn_star = inputs
        u_pred, _ = self.sol_predict(t_sol_star, x_sol_star)
        u_pred_idn, _ = self.sol_predict(t_idn_star, x_idn_star)

        norm_error_u = calc_2_norm_in_batch(u_sol_star - u_pred, 3)
        norm_u_sol_star = calc_2_norm_in_batch(u_sol_star, 3)
        error_u = norm_error_u / norm_u_sol_star
        print_log('Error u: %e' % (error_u))

        norm_error_u = calc_2_norm_in_batch(u_idn_star - u_pred_idn, 3)
        norm_u_idn_star = calc_2_norm_in_batch(u_idn_star, 3)
        error_u_idn = norm_error_u / norm_u_idn_star
        print_log('Error u (idn): %e' % (error_u_idn))

        return u_pred


class SolNetKdvSame(SolNet):
    """sol net for kdv same case"""
    def __init__(self, net, lb_sol, ub_sol, net_pde):
        super(SolNetKdvSame, self).__init__(net, lb_sol, ub_sol, net_pde)
        self.grad_net_uxxx = grad(self.grad_net_uxx, input_index=1)

    def construct(self, *inputs):
        """Network forward pass"""
        t0, x0, u0, t_lb, x_lb, t_ub, x_ub, t_f, x_f = inputs
        u0_pred, _, _, _ = self.sol_net_u(t0, x0)
        u_lb_pred, u_x_lb_pred, u_xx_lb_pred, _ = self.sol_net_u(t_lb, x_lb)
        u_ub_pred, u_x_ub_pred, u_xx_ub_pred, _ = self.sol_net_u(t_ub, x_ub)
        sol_f_pred = self.sol_net_f(t_f, x_f)

        sol_loss = self.sse(u0 - u0_pred) + \
                   self.sse(u_lb_pred - u_ub_pred) + \
                   self.sse(u_x_lb_pred - u_x_ub_pred) + \
                   self.sse(u_xx_lb_pred - u_xx_ub_pred) + \
                   self.sse(sol_f_pred)
        return sol_loss

    def sol_net_u(self, t, x):
        u = self.net(t, x)
        u_t, u_x = self.grad_net(t, x)
        u_xx = self.grad_net_uxx(t, x)
        return u, u_x, u_xx, u_t

    def sol_net_f(self, t, x):
        u, u_x, u_xx, u_t = self.sol_net_u(t, x)
        u_xxx = self.grad_net_uxxx(t, x)
        terms = ops.concat([u, u_x, u_xx, u_xxx], 1)
        f = u_t - self.net_pde(terms)
        return f
