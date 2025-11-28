## mindscience.pde

### 模块介绍

- pde 模块是 MindSpore Science 框架中用于求解流体力学、静力学等领域中的偏微分方程 (Partial Differential Equations, PDEs) 的科学计算算子库，给出了自定义的数学运算实现（`mindspore function`），并可将 `sympy`  库中的符号计算转换成相应的 `mindspore function`。此外，pde 模块目前也支持在不同的算子神经网络框架下（如 FNO，FFNO，SNO，PDENet 等）给出一些**流体力学和静力学方程的损失函数计算**。pde 模块将形如加法、幂运算、微分等数学运算定义为相应的 Node 类，通过 `sympy_to_mindspore()` 方法为用户提供简洁的**形式化泛函计算接口**。结合 `mindflow` 中的其他模块，例如 `mindflow.cell`，用户能够更加高效的进行微分方程的神经网络求解和处理科学计算任务。

### PDEWithLoss

- 此模块被应用于神经网络求解单一方程的方法（如 PINNs）中；
- **微分方程的定义：**将方程的数学符号串转化成可计算的 mindspore function；
- 关键模块：`pde.sympy_to_mindspore` ，模块中定义了从特定 sympy 符号到 mindspore function 的接口类 Node（节点），对应于各种数学符号如加法、乘法、幂运算和偏导数等等；通过 `sympy_translation.py`  将符号串转换成节点图，随后翻译成完整可计算的 mindspore function；
- 方程内部信息定义：`self.pde_nodes`：表示对应于 `pde` 的 mindspore function；`pde(self)`返回一个 sympy 数学符号串，表示了当前 PDE 问题在求解区域内部的方程左端，默认右端项为零；
- 方程边界条件定义：`bc(self)`：返回一个 `sympy` 数学符号串，表示了当前 PDE 问题的边界条件表达式左端，默认右端项为零；注意此函数并非 `PDEWithLoss` 强制规定；`self.bc_nodes`：表示对应于 `bc` 的 mindspore function；注意此成员并非 `PDEWithLoss` 强制规定；

**使用示例：**用户自定义一个 `PDEWithLoss` 子类，并包含以下信息：

- 方程：定义在二维区域上的二阶椭圆型方程；
    $$
    \begin{align}
    -\Delta u + u &= f = 4,~x \in \Omega \subset \mathbb{R}^2,\\
    \nabla u \cdot \mathbf{1} &= g = 2,~x\in \partial \Omega.
    \end{align}
    $$

- 损失函数：基于两隐藏层全连接神经网络 $u_{\theta}$ 的 PINNs 损失函数；
    $$
    L(\theta) = \int_{\Omega} \left( -\Delta u_{\theta}(x) + u_{\theta}(x) -f(x) \right)^2 \mathrm{d}x + \int_{\partial \Omega}  \left( \nabla u_{\theta}(x) \cdot \mathbf{1} - g(x) \right)^2 \mathrm{d} S.
    $$
    注：积分形式的损失函数经过 Monte-Carlo 离散后（数值积分权值均为 1），等价于 MSE 形式。

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

此模块目前支持的方程如下:

- 一维有黏性的 Burgers' equation （目前初始条件和边界条件待补充）:
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

- 二维不可压 Navier-Stokes equation（目前初始条件和边界条件待补充）:
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

- 二维 Poisson's equation （目前边界条件待补充）:
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

- 该基类规定了一个**完整、可计算且可被训练求解**的偏微分方程应有的函数和数据结构。
- `get_loss(self, inputs, labels)`：根据输入（可以是区域的 sample 点）和标签（可以是方程的源项信息）来构建可被计算和训练的损失函数。`step(self, inputs)`：返回一个 Tensor 表示所采用模型的预测；
- 此模块目前支持的流体力学场景包括：稳态流（steady flow），即流场中任意一点的流体属性与时间无关；非稳态流（unsteady flow），即流场中至少存在一点的流体属性与时间相关。
- 可接入多种类型的神经网络以及自定义网络，目前该模块主要**被应用于算子学习**方法中，因此没有包含具体的方程定义。
- 示例：FNO 求解二维不可压 Navier-Stokes 方程 https://gitee.com/mindspore/mindscience/blob/master/MindFlow/applications/data_driven/navier_stokes/fno2d/train.py





