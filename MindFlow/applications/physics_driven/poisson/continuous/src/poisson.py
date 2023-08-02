"""Define the Poisson equation."""
import sympy
from mindspore import numpy as ms_np
from mindflow import PDEWithLoss, MTLWeightedLoss, sympy_to_mindspore


class Poisson(PDEWithLoss):
    """Define the loss of the Poisson equation."""
    def __init__(self, model, n_dim):
        if n_dim == 1:
            var_str = 'x'
        elif n_dim == 2:
            var_str = 'x y'
        elif n_dim == 3:
            var_str = 'x y z'
        else:
            raise ValueError("`n_dim` can only be 2 or 3.")
        self.in_vars = sympy.symbols(var_str)
        self.out_vars = (sympy.Function('u')(*self.in_vars),)
        super(Poisson, self).__init__(model, self.in_vars, self.out_vars)
        self.bc_nodes = sympy_to_mindspore(self.bc(n_dim), self.in_vars, self.out_vars)
        self.loss_fn = MTLWeightedLoss(num_losses=2)

    def pde(self):
        """Define the gonvering equation."""
        poisson = 0
        src_term = 1
        sym_u = self.out_vars[0]
        for var in self.in_vars:
            poisson += sympy.diff(sym_u, (var, 2))
            src_term *= sympy.sin(4*sympy.pi*var)
        poisson += src_term
        equations = {"poisson": poisson}
        return equations

    def bc(self, n_dim):
        """Define the boundary condition."""
        bc_term = 1
        for var in self.in_vars:
            bc_term *= sympy.sin(4*sympy.pi*var)
        bc_term *= 1/(16*n_dim*sympy.pi*sympy.pi)
        bc_eq = self.out_vars[0] - bc_term
        equations = {"bc": bc_eq}
        return equations

    def get_loss(self, pde_data, bc_data):
        """Define the loss function."""
        res_pde = self.parse_node(self.pde_nodes, inputs=pde_data)
        res_bc = self.parse_node(self.bc_nodes, inputs=bc_data)
        loss_pde = ms_np.mean(ms_np.square(res_pde[0]))
        loss_bc = ms_np.mean(ms_np.square(res_bc[0]))
        return self.loss_fn((loss_pde, loss_bc))
