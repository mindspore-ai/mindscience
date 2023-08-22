"""Problem case - Helmholtz 2D"""
from mindspore import ops, ms_function

from mindelec.solver import Problem
from mindelec.operators import SecondOrderGrad as Hessian


class Helmholtz2D(Problem):
    """2D Helmholtz equation"""

    def __init__(self, domain_name, bc_name, net, wavenumber=2):
        super(Helmholtz2D, self).__init__()
        self.domain_name = domain_name
        self.bc_name = bc_name
        self.type = "Equation"
        self.wave_number = wavenumber
        self.grad_xx = Hessian(net, input_idx1=0, input_idx2=0, output_idx=0)
        self.grad_yy = Hessian(net, input_idx1=1, input_idx2=1, output_idx=0)
        self.reshape = ops.Reshape()

    @ms_function
    def governing_equation(self, *output, **kwargs):
        """governing equation"""
        u = output[0]
        x = kwargs[self.domain_name][:, 0]
        y = kwargs[self.domain_name][:, 1]
        x = self.reshape(x, (-1, 1))
        y = self.reshape(y, (-1, 1))

        u_xx = self.grad_xx(kwargs[self.domain_name])
        u_yy = self.grad_yy(kwargs[self.domain_name])

        return u_xx + u_yy + self.wave_number ** 2 * u

    @ms_function
    def boundary_condition(self, *output, **kwargs):
        """boundary condition"""
        u = output[0]
        x = kwargs[self.bc_name][:, 0]
        y = kwargs[self.bc_name][:, 1]
        x = self.reshape(x, (-1, 1))
        y = self.reshape(y, (-1, 1))

        test_label = ops.sin(self.wave_number * x)
        return 100 * (u - test_label)
