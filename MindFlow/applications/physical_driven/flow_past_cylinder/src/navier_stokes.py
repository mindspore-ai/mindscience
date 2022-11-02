"""
Navier-Stokes Problem domain description.
"""
from mindspore import ops
from mindspore import Tensor
import mindspore.common.dtype as mstype
from mindflow.solver import Problem
from mindflow.operators import Grad
from mindflow.operators import SecondOrderGrad as Hessian


class NavierStokes2D(Problem):
    """2D NavierStokes equation"""

    def __init__(self, model, domain_points, bc_points, ic_points, bc_label, ic_label, re=100):
        super(NavierStokes2D, self).__init__()
        self.domain_points = domain_points
        self.bc_points = bc_points
        self.ic_points = ic_points
        self.bc_label = bc_label
        self.ic_label = ic_label

        self.grad = Grad(model)
        self.gradux_xx = Hessian(model, input_idx1=0, input_idx2=0, output_idx=0)
        self.gradux_yy = Hessian(model, input_idx1=1, input_idx2=1, output_idx=0)
        self.graduy_xx = Hessian(model, input_idx1=0, input_idx2=0, output_idx=1)
        self.graduy_yy = Hessian(model, input_idx1=1, input_idx2=1, output_idx=1)
        self.split = ops.Split(1, 3)
        self.re = Tensor(re, mstype.float32)

    def governing_equation(self, *output, **kwargs):
        """governing equation"""
        flow_vars = output[0]
        u, v, _ = self.split(flow_vars)
        domain_data = kwargs[self.domain_points]

        du_dx, du_dy, du_dt = self.split(self.grad(domain_data, None, 0, flow_vars))
        dv_dx, dv_dy, dv_dt = self.split(self.grad(domain_data, None, 1, flow_vars))
        dp_dx, dp_dy, _ = self.split(self.grad(domain_data, None, 2, flow_vars))
        du_dxx = self.gradux_xx(domain_data)
        du_dyy = self.gradux_yy(domain_data)
        dv_dxx = self.graduy_xx(domain_data)
        dv_dyy = self.graduy_yy(domain_data)

        eq1 = du_dt + (u * du_dx + v * du_dy) + dp_dx - 1.0 / self.re * (du_dxx + du_dyy)
        eq2 = dv_dt + (u * dv_dx + v * dv_dy) + dp_dy - 1.0 / self.re * (dv_dxx + dv_dyy)
        eq3 = du_dx + dv_dy
        pde_residual = ops.Concat(1)((eq1, eq2, eq3))
        return pde_residual

    def boundary_condition(self, *output, **kwargs):
        """boundary condition"""
        flow_vars = output[0][:, :2]
        bc_label = kwargs[self.bc_label]
        bc_r = flow_vars - bc_label
        return bc_r

    def initial_condition(self, *output, **kwargs):
        """initial condition"""
        flow_vars = output[0]
        ic_label = kwargs[self.ic_label]
        ic_r = flow_vars - ic_label
        return ic_r
