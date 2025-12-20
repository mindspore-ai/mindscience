## mindscience.pde

### Introduction

- The PDE module is a scientific computing operator library within the MindSpore Science framework, designed for solving partial differential equations (PDEs) in fields such as fluid dynamics and statics. It provides custom implementations of mathematical operations (`mindspore function`) and can convert symbolic computations from the `sympy`library into corresponding `mindspore function`operations. Additionally, the PDE module currently supports loss function computations for **fluid dynamics and statics equations** under different operator neural network frameworks (e.g., FNO, FFNO, SNO, PDENet, etc.). The PDE module defines mathematical operations such as addition, exponentiation, and differentiation as corresponding Node classes, offering users a concise **formal functional computation interface** through the `sympy_to_mindspore()`method. By integrating with other modules in `MindFlow`, users can more efficiently solve differential equations via neural networks and handle scientific computing tasks.

### PDEWithLoss

- This module is applied to neural network methods for solving single equations, such as Physics-Informed Neural Networks (PINNs).

- **Definition of Differential Equations:** Convert the mathematical symbolic string representation of the equation into a computable mindspore function.

- **Key Module:** `pde.sympy_to_mindspore`. This module defines the interface class `Node`(node) for converting specific `sympy` symbols into mindspore functions, representing various mathematical symbols such as addition, multiplication, exponentiation, partial derivatives, and more. Through `sympy_translation.py`, symbolic strings are transformed into a node graph, which is then translated into a fully computable mindspore function.

- Definition of Equation Internal Information: `self.pde_nodes` represents the mindsporefunction corresponding to the PDE. `pde(self)` returns a sympy mathematical symbolic string representing the left-hand side of the equation for the current PDE problem within the solution domain. The right-hand side defaults to zero.

- Definition of Equation Boundary Conditions: `bc(self)` returns a sympymathematical symbolic string representing the left-hand side of the boundary condition expression for the current PDE problem. The right-hand side defaults to zero. Note that this function is not mandatory for `PDEWithLoss`. `self.bc_nodes` represents the mindsporefunction corresponding to the boundary conditions. Note that this member is not mandatory for `PDEWithLoss`.

#### Numerical Examples

Users can define a custom subclass of `PDEWithLoss`containing the following information:

- A second-order elliptic equation defined on a two-dimensional domain.

  $$
  \begin{align}
  -\Delta u + u &= f = 4,~x \in \Omega \subset \mathbb{R}^2,\\
  \nabla u \cdot \mathbf{1} &= g = 2,~x\in \partial \Omega.
  \end{align}
  $$

- PINN loss function based on a fully connected neural network $u_\theta$ with two hidden layers.

  $$
  L(\theta) = \int_{\Omega} \left( -\Delta u_{\theta}(x) + u_{\theta}(x) -f(x) \right)^2 \mathrm{d}x + \int_{\partial \Omega}  \left( \nabla u_{\theta}(x) \cdot \mathbf{1} - g(x) \right)^2 \mathrm{d} S.
  $$

  Note that after discretization via Monte Carlo methods (with quadrature weights all set to 1), the integral form of the loss function is equivalent to the Mean Squared Error (MSE) form.

  ```python
  import numpy as np
  from sympy import symbols, Function, diff
  from mindspore import nn, ops, Tensor
  from mindspore import dtype as mstype
  from mindscience.pde import PDEWithLoss, sympy_to_mindspore
  # define a fully-connected neural network with tanh activation
  class Net(nn.Cell):
      def __init__(self, cin=2, cout=1, hidden=10):
          super().__init__()
          self.fc1 = nn.Dense(cin, hidden)
          self.fc2 = nn.Dense(hidden, hidden)
          self.fcout = nn.Dense(hidden, cout)
          self.act = ops.Tanh()
      def construct(self, x):
          x = self.act(self.fc1(x))
          x = self.act(self.fc2(x))
          x = self.fcout(x)
          return x
  model = Net()
  # user-defined class to describe Poisson's equation with pure Neumann's boundary condition.
  class MyProblem(PDEWithLoss):
      def __init__(self, model, loss_fn=nn.MSELoss()): # Take the MSE loss function
          self.x, self.y = symbols('x y')
          self.u = Function('u')(self.x, self.y)
          self.in_vars = [self.x, self.y]
          self.out_vars = [self.u]
          super(MyProblem, self).__init__(model, in_vars=self.in_vars, out_vars=self.out_vars)
          self.loss_fn = loss_fn
          self.bc_nodes = sympy_to_mindspore(self.bc(), self.in_vars, self.out_vars)
      # pde's info inside the domain
      def pde(self):
          my_eq = diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)) - self.u + 4.0
          equations = {"my_eq": my_eq}
          return equations
      # pde's info on the boundary
      def bc(self):
          bc_eq = diff(self.u, (self.x, 1)) + diff(self.u, (self.y, 1)) - 2.0
          equations = {"bc_eq": bc_eq}
          return equations
      # PINN's loss function
      def get_loss(self, pde_data, bc_data):
          pde_res = self.parse_node(self.pde_nodes, inputs=pde_data)
          pde_loss = self.loss_fn(pde_res[0], Tensor(np.array([0.0]), mstype.float32))
          bc_res = self.parse_node(self.bc_nodes, inputs=bc_data)
          bc_loss = self.loss_fn(bc_res[0], Tensor(np.array([0.0]), mstype.float32))
          return pde_loss + bc_loss
  problem = MyProblem(model)
  print(problem.pde())
  print(problem.bc())
  # predicted outputs:
  # my_eq: -u(x, y) + Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)) + 4.0
  #     Item numbers of current derivative formula nodes: 4
  # bc_eq: Derivative(u(x, y), x) + Derivative(u(x, y), y) - 2.0
  #     Item numbers of current derivative formula nodes: 3
  # {'my_eq': -u(x, y) + Derivative(u(x, y), (x, 2)) + Derivative(u(x, y), (y, 2)) + 4.0}
  # {'bc_eq': Derivative(u(x, y), x) + Derivative(u(x, y), y) - 2.0}
  ```

The equations currently supported by this module are as follows:

- One-dimensional viscous Burgers' equation
  $$
  \frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \epsilon \frac{\partial^2 u}{\partial x^2} = 0.
  $$

  ```python
  class Burgers(PDEWithLoss):
      def pde(self):
            """
            Define Burgers 1-D governing equations based on sympy, abstract method.
            Returns:
                dict, user defined sympy symbolic equations.
            """
            burgers_eq = diff(self.u, (self.t, 1)) + self.u * diff(self.u, (self.x, 1)) - \
                         self.mu * diff(self.u, (self.x, 2))
            equations = {"burgers": burgers_eq}
            return equations
  ```

- Two-dimensional incompressible Navier-Stokes equations
  $$
  \text{连续性方程：}\quad\quad  \frac{\partial u}{\partial x} + \frac{\partial u}{\partial y} = 0,\\
  x~\text{方向动量守恒：} \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} = -\frac{1}{\rho} \frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2}+\frac{\partial^2 u}{\partial y^2} \right), \\
  y~\text{方向动量守恒：} \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} = -\frac{1}{\rho}\frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2}+\frac{\partial^2 v}{\partial y^2} \right).
  $$

```python
class IncompressibleNavierStokes(PDEWithLoss):
    def pde(self):
          """
          Define governing equations based on sympy, abstract method.

          Returns:
              dict, user defined sympy symbolic equations.
          """

          # momentum convervation along x
          momentum_x = self.u.diff(self.t) + self.u * self.u.diff(self.x) + self.v * self.u.diff(self.y) + \
                       self.p.diff(self.x) - self.number * (diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)))

          # momentum conservation along y
          momentum_y = self.v.diff(self.t) + self.u * self.v.diff(self.x) + self.v * self.v.diff(self.y) + \
                       self.p.diff(self.y) - self.number * (diff(self.v, (self.x, 2)) + diff(self.v, (self.y, 2)))

          # continuity equation
          continuty = self.u.diff(self.x) + self.v.diff(self.y)

          equations = {"momentum_x": momentum_x, "momentum_y": momentum_y, "continuty": continuty}
          return equations
```

- Two-dimensional Poisson's equation
  $$
  -\Delta u = f = 1.
  $$

    ```python
  class Poisson(PDEWithLoss):
      def pde(self):
            """
            Define Poisson 2-D governing equations based on sympy, abstract method.
            Returns:
                dict, user defined sympy symbolic equations.
            """
            poisson = diff(self.u, (self.x, 2)) + diff(self.u, (self.y, 2)) + 1.0
            equations = {"poisson": poisson}
            return equations
    ```

### FlowWithLoss

- This base class defines the functions and data structures that a **complete, computable, and trainable** partial differential equation should possess.
- `get_loss`: Constructs a computable and trainable loss function based on the inputs (which can be sample points in the domain) and labels (which can be source term information of the equation).
- `step`: Returns a Tensor representing the prediction of the adopted model.
- This module currently supports steady flow and unsteady flow.
- It can be integrated with various types of neural networks as well as custom networks. Currently, this module is primarily **applied to operator learning methods**, and therefore does not include specific equation definitions.

#### Numerical Examples

Consider using FNO to solve the two-dimensional incompressible Navier-Stokes equations. https://atomgit.com/mindspore-lab/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/FNO2D.ipynb
