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
'''utils for applying multi-gradient descent algorithm'''
import numpy as np

from mindspore import Tensor, ops
from mindspore import dtype as mstype

from mindflow.utils import print_log


def flatten_grads(grads):
    r"""
    Transform tuple of tensors to one dimension numpy array and save the list of shapes

    Args:
        grads (tuple) : input tuple of grads

    """
    shapes = []
    grads_numpy = np.array([])
    for grad in grads:
        shapes.append(grad.shape)  # save the shape of grad
        grad_array = grad.asnumpy()  # transform tensor to numpy
        grad_array_flatten = grad_array.flatten()  # flatten the numpy array
        grads_numpy = np.concatenate((grads_numpy, grad_array_flatten))

    return grads_numpy, shapes


def resize_array(grads_numpy, shapes):
    r"""
    Transform one dimension numpy array to standard tuple of tensors

    Args:
        grads_numpy (array) : input array
        shapes: shapes of tensors

    """
    nums = []  # amounts of figures in each tensor
    for shape in shapes:
        num = 1
        for tmp in shape:
            num = num * tmp
        nums.append(num)

    grads = []
    begin = 0
    end = 0
    len1 = len(shapes)
    for i in range(0, len1):
        begin = end
        end = end + nums[i]
        tmp_array = grads_numpy[begin:end]
        grad = Tensor(np.reshape(tmp_array, shapes[i]), mstype.float32)
        grads.append(grad)
    return tuple(grads)


def gradient_normalizers(grads, losses, normalization_type):
    r"""
    apply normalization

    """
    grads_count = len(grads)
    result_grads = []
    if normalization_type == 'l2':
        for t in range(grads_count):
            t_gn = np.sum(np.power(grads[t], 2))
            t_gn = np.sqrt(t_gn)
            result_grads.append(grads[t] / (t_gn + 1e-8))
    elif normalization_type == 'loss':
        for t in range(grads_count):
            result_grads.append(grads[t] / (losses[t] + 1e-8))
    elif normalization_type == 'loss+':
        for t in range(grads_count):
            t_gn = np.sum(np.power(grads[t], 2))
            t_gn = losses[t] * np.sqrt(t_gn)
            result_grads.append(grads[t] / (t_gn + 1e-8))
    elif normalization_type == 'none':
        for t in range(grads_count):
            result_grads.append(grads[t])
    else:
        print_log('ERROR: Invalid Normalization Type')
    return result_grads


class MinNormSolverNumpy:
    r"""
    class for finding min-norm coefficient

    """

    MAX_ITER = 250
    STOP_CRIT = 1e-6

    @staticmethod
    def find_min_norm_element(vecs):
        r"""
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric,
        and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
        the solution lies in (0, d_{i, j})
        Hence, we find the best 2-task solution, and then run the projected gradient descent until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolverNumpy._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolverNumpy.MAX_ITER:
            grad_dir = -1.0*np.dot(grad_mat, sol_vec)
            new_point = MinNormSolverNumpy._next_point(sol_vec, grad_dir, n)
            # Re-compute the inner products for line search
            v1v1 = 0.0
            v1v2 = 0.0
            v2v2 = 0.0
            for i in range(n):
                for j in range(n):
                    v1v1 += sol_vec[i]*sol_vec[j]*dps[(i, j)]
                    v1v2 += sol_vec[i]*new_point[j]*dps[(i, j)]
                    v2v2 += new_point[i]*new_point[j]*dps[(i, j)]
            nc, nd = MinNormSolverNumpy._min_norm_element_from2(
                v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec + (1-nc)*new_point
            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolverNumpy.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
        return sol_vec, nd

    @staticmethod
    def find_min_norm_element_fw(vecs):
        r"""
        Given a list of vectors (vecs), this method finds the minimum norm element in the convex hull
        as min |u|_2 st. u = \sum c_i vecs[i] and \sum c_i = 1.
        It is quite geometric,
        and the main idea is the fact that if d_{ij} = min |u|_2 st u = c x_i + (1-c) x_j;
        the solution lies in (0, d_{i, j})
        Hence, we find the best 2-task solution, and then run the Frank Wolfe until convergence
        """
        # Solution lying at the combination of two points
        dps = {}
        init_sol, dps = MinNormSolverNumpy._min_norm_2d(vecs, dps)

        n = len(vecs)
        sol_vec = np.zeros(n)
        sol_vec[init_sol[0][0]] = init_sol[1]
        sol_vec[init_sol[0][1]] = 1 - init_sol[1]

        if n < 3:
            # This is optimal for n=2, so return the solution
            return sol_vec, init_sol[2]

        iter_count = 0

        grad_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                grad_mat[i, j] = dps[(i, j)]

        while iter_count < MinNormSolverNumpy.MAX_ITER:
            t_iter = np.argmin(np.dot(grad_mat, sol_vec))

            v1v1 = np.dot(sol_vec, np.dot(grad_mat, sol_vec))
            v1v2 = np.dot(sol_vec, grad_mat[:, t_iter])
            v2v2 = grad_mat[t_iter, t_iter]

            nc, nd = MinNormSolverNumpy._min_norm_element_from2(
                v1v1, v1v2, v2v2)
            new_sol_vec = nc*sol_vec
            new_sol_vec[t_iter] += 1 - nc

            change = new_sol_vec - sol_vec
            if np.sum(np.abs(change)) < MinNormSolverNumpy.STOP_CRIT:
                return sol_vec, nd
            sol_vec = new_sol_vec
        return sol_vec, nd

    @staticmethod
    def _min_norm_element_from2(v1v1, v1v2, v2v2):
        r"""
        Analytical solution for min_{c} |cx_1 + (1-c)x_2|_2^2
        d is the distance (objective) optimzed
        v1v1 = <x1,x1>
        v1v2 = <x1,x2>
        v2v2 = <x2,x2>
        """
        if v1v2 >= v1v1:
            # Case: Fig 1, third column
            gamma = 0.999
            cost = v1v1
            return gamma, cost
        if v1v2 >= v2v2:
            # Case: Fig 1, first column
            gamma = 0.001
            cost = v2v2
            return gamma, cost
        # Case: Fig 1, second column
        gamma = -1.0 * ((v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2 + 1e-8))
        cost = v2v2 + gamma*(v1v2 - v2v2)
        return gamma, cost

    @staticmethod
    def _min_norm_2d(vecs, dps):
        r"""
        Find the minimum norm solution as combination of two points
        This solution is correct if vectors(gradients) lie in 2D
        ie. min_c |\sum c_i x_i|_2^2 st. \sum c_i = 1 , 1 >= c_1 >= 0 for all i, c_i + c_j = 1.0 for some i, j
        """
        dmin = 1e8
        sign = 0
        len1 = len(vecs)
        for i in range(len1):
            for j in range(i + 1, len1):
                if (i, j) not in dps:
                    dps[(i, j)] = 0.0
                    dps[(i, j)] = np.dot(vecs[i], vecs[j])
                    dps[(j, i)] = dps[(i, j)]
                if (i, i) not in dps:
                    dps[(i, i)] = 0.0
                    dps[(i, i)] = np.dot(vecs[i], vecs[i])
                if (j, j) not in dps:
                    dps[(j, j)] = 0.0
                    dps[(j, j)] = np.dot(vecs[j], vecs[j])
                c, d = MinNormSolverNumpy._min_norm_element_from2(
                    dps[(i, i)], dps[(i, j)], dps[(j, j)])
                if sign == 0:
                    dmin = d
                    sol = [(i, j), c, d]
                    sign = 1
                else:
                    if d < dmin:
                        dmin = d
                        sol = [(i, j), c, d]
        return sol, dps

    @staticmethod
    def _projection2simplex(y):
        r"""
        Given y, it solves argmin_z |y-z|_2 st \sum z = 1 , 1 >= z_i >= 0 for all i
        """
        m = len(y)
        sorted_y = np.flip(np.sort(y), axis=0)
        tmpsum = 0.0
        tmax_f = (np.sum(y) - 1.0) / m
        for i in range(m-1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1) / (i + 1.0)
            if tmax > sorted_y[i + 1]:
                tmax_f = tmax
                break
        return np.maximum(y - tmax_f, np.zeros(y.shape))

    @staticmethod
    def _next_point(cur_val, grad, n):
        r"""
        find next point
        """
        proj_grad = grad - (np.sum(grad) / n)
        tm1 = -1.0 * cur_val[proj_grad < 0] / (proj_grad[proj_grad < 0] + 1e-8)
        tm2 = (1.0 - cur_val[proj_grad > 0]) / (proj_grad[proj_grad > 0] + 1e-8)

        t = 1
        if len(tm1[tm1 > 1e-7]) >= 1:
            t = np.min(tm1[tm1 > 1e-7])
        if len(tm2[tm2 > 1e-7]) >= 1:
            t = min(t, np.min(tm2[tm2 > 1e-7]))

        next_point = proj_grad*t + cur_val
        next_point = MinNormSolverNumpy._projection2simplex(next_point)
        return next_point


def mgda_step(losses, origin_grads, optimizer):
    r"""multi-gradient descent
    given a list of losses and gradients,
    this method finds the minimum norm element in the convex hull
    """
    # transform tensors to numpy and save the shapes
    pde_grads_numpy = flatten_grads(origin_grads[0])[0]
    ic_bc_grads_numpy = flatten_grads(origin_grads[1])[0]
    numpy_grads = [pde_grads_numpy, ic_bc_grads_numpy]

    # execute normalization
    norm_grads = gradient_normalizers(numpy_grads, losses, 'l2')

    # executing mgda
    sol = MinNormSolverNumpy.find_min_norm_element_fw(norm_grads)[0]

    # combine gradients
    new_grads = []
    grad_count = len(origin_grads[0])
    for i in range(grad_count):
        grad = origin_grads[0][i] * sol[0] + origin_grads[1][i] * \
            sol[1]  # Gradient addition is performed
        new_grads.append(grad)
    t_grads = tuple(new_grads)

    # combine losses
    loss = losses[0] * sol[0] + losses[1] + sol[1]

    # update
    loss = ops.depend(loss, optimizer(t_grads))
    return loss
