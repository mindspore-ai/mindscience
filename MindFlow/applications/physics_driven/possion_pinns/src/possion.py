"""Define the Possion equation."""
import math

import mindspore as ms
from mindspore import nn, ops
from mindspore import numpy as ms_np
from mindflow.pde import Problem
from mindflow.operators import SecondOrderGrad


class Possion(Problem):
    """Possion equation."""
    def __init__(self, model, n_dim, domain_name, bc_name):
        super(Possion, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.grads = [SecondOrderGrad(model, input_idx1=i_dim, input_idx2=i_dim, output_idx=0) \
            for i_dim in range(n_dim)]
        self.prod = ops.ReduceProd(keep_dims=True)
        self.bc = AnalyticSolution(n_dim)

    def governing_equation(self, *output, **kwargs):
        x_in = kwargs[self.domain_name]
        output = 0.
        for d_xx in self.grads:
            output += d_xx(x_in)
        output += self.prod(ms_np.sin(4*math.pi*x_in), 1)
        return output

    def boundary_condition(self, *output, **kwargs):
        u = output[0]
        x = kwargs[self.bc_name]
        return u - self.bc(x)


class AnalyticSolution(nn.Cell):
    """Analytic solution."""
    def __init__(self, n_dim):
        super(AnalyticSolution, self).__init__()
        self.prod = ops.ReduceProd(keep_dims=True)
        self.factor = ms.Tensor(1/(16.*n_dim*math.pi*math.pi))

    def construct(self, x):
        return self.factor*self.prod(ms_np.sin(4*math.pi*x), 1)
