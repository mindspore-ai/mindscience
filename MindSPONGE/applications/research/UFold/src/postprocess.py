# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Post-processing for optimization"""
import math
import mindspore.ops as ops
from mindspore import dtype as mstype


def constraint_matrix_batch(x):
    """
    this function is referred from e2efold utility function, located at
    https://github.com/ml4bio/e2efold/tree/master/e2efold/common/utils.py
    """
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    transpose = ops.Transpose()
    au = ops.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + transpose(au, (0, 2, 1))
    cg = ops.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + transpose(cg, (0, 2, 1))
    ug = ops.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + transpose(ug, (0, 2, 1))
    return au_ua + cg_gc + ug_gu


def constraint_matrix_batch_addnc(x):
    """constraint matrix batch with nc"""
    base_a = x[:, :, 0]
    base_u = x[:, :, 1]
    base_c = x[:, :, 2]
    base_g = x[:, :, 3]
    batch = base_a.shape[0]
    length = base_a.shape[1]
    transpose = ops.Transpose()
    au = ops.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))
    au_ua = au + transpose(au, (0, 2, 1))
    cg = ops.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))
    cg_gc = cg + transpose(cg, (0, 2, 1))
    ug = ops.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))
    ug_gu = ug + transpose(ug, (0, 2, 1))
    ## add non-canonical pairs
    ac = ops.matmul(base_a.view(batch, length, 1), base_c.view(batch, 1, length))
    ac_ca = ac + transpose(ac, (0, 2, 1))
    ag = ops.matmul(base_a.view(batch, length, 1), base_g.view(batch, 1, length))
    ag_ga = ag + transpose(ag, (0, 2, 1))
    uc = ops.matmul(base_u.view(batch, length, 1), base_c.view(batch, 1, length))
    uc_cu = uc + transpose(uc, (0, 2, 1))
    aa = ops.matmul(base_a.view(batch, length, 1), base_a.view(batch, 1, length))
    uu = ops.matmul(base_u.view(batch, length, 1), base_u.view(batch, 1, length))
    cc = ops.matmul(base_c.view(batch, length, 1), base_c.view(batch, 1, length))
    gg = ops.matmul(base_g.view(batch, length, 1), base_g.view(batch, 1, length))
    return au_ua + cg_gc + ug_gu + ac_ca + ag_ga + uc_cu + aa + uu + cc + gg


def contact_a(a_hat, m):
    """contact a_hat"""
    transpose = ops.Transpose()
    a = a_hat * a_hat
    a = (a + transpose(a, (0, 2, 1))) / 2
    a = a * m
    return a


def soft_sign(x):
    """softsigh function"""
    k = 1
    exp = ops.Exp()
    return 1.0/(1.0+exp(-2*k*x))


def postprocess_new(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    cast = ops.Cast()
    m = cast(constraint_matrix_batch(x), mstype.float32)
    u = soft_sign(u - s) * u

    sigmoid = ops.Sigmoid()
    relu = ops.ReLU()
    op = ops.ReduceSum(keep_dims=False)
    transpose = ops.Transpose()
    expand_dims = ops.ExpandDims()
    broadcast_to = ops.BroadcastTo(u.shape)
    op_abs = ops.Abs()

    a_hat = sigmoid(u) * soft_sign(u - s)
    lmbd = relu(op(contact_a(a_hat, m), -1) - 1)
    # gradient descent
    for _ in range(num_itr):
        grad_a = lmbd * soft_sign(op(contact_a(a_hat, m), -1) - 1)
        grad_a = expand_dims(grad_a, -1)
        grad_a = broadcast_to(grad_a) - u / 2

        grad = a_hat * m * (grad_a + transpose(grad_a, (0, 2, 1)))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = relu(op_abs(a_hat) - rho * lr_min)

        lmbd_grad = relu(op(contact_a(a_hat, m), -1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

    a = a_hat * a_hat
    a = (a + transpose(a, (0, 2, 1))) / 2
    a = a * m
    return a


def postprocess_new_nc(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)):
    """
    :param u: utility matrix, u is assumed to be symmetric, in batch
    :param x: RNA sequence, in batch
    :param lr_min: learning rate for minimization step
    :param lr_max: learning rate for maximization step (for lagrangian multiplier)
    :param num_itr: number of iterations
    :param rho: sparsity coefficient
    :param with_l1:
    :return:
    """
    cast = ops.Cast()
    m = cast(constraint_matrix_batch_addnc(x), mstype.float32)
    u = soft_sign(u - s) * u
    sigmoid = ops.Sigmoid()
    relu = ops.ReLU()
    op = ops.ReduceSum(keep_dims=False)
    transpose = ops.Transpose()
    expand_dims = ops.ExpandDims()
    broadcast_to = ops.BroadcastTo(u.shape)
    op_abs = ops.Abs()

    a_hat = sigmoid(u) * soft_sign(u - s)
    lmbd = relu(op(contact_a(a_hat, m), -1) - 1)
    # gradient descent
    for _ in range(num_itr):
        grad_a = lmbd * soft_sign(op(contact_a(a_hat, m), -1) - 1)
        grad_a = expand_dims(grad_a, -1)
        grad_a = broadcast_to(grad_a) - u / 2

        grad = a_hat * m * (grad_a + transpose(grad_a, (0, 2, 1)))
        a_hat -= lr_min * grad
        lr_min = lr_min * 0.99

        if with_l1:
            a_hat = relu(op_abs(a_hat) - rho * lr_min)

        lmbd_grad = relu(op(contact_a(a_hat, m), -1) - 1)
        lmbd += lr_max * lmbd_grad
        lr_max = lr_max * 0.99

    a = a_hat * a_hat
    a = (a + transpose(a, (0, 2, 1))) / 2
    a = a * m
    return a
    