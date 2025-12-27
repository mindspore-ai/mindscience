## mindscience.solvers

### CBS 模块介绍

CBS 模块是一个基于收敛 Born 级数（Convergent Born Series, CBS）方法的 Helmholtz 方程求解器。Helmholtz 方程

$$
\Delta u + \kappa^2 u = f
$$

是描述波动现象（如声波、电磁波）在频域中的基本偏微分方程，广泛应用于声学、电磁学和量子力学等领域。Helmholtz 通常具有 Dirichlet，Neumann，Robin 和辐射边界条件等。CBS 方法通过 Born 级数迭代求解 Helmholtz 方程，具有良好的数值稳定性和收敛性。特别是引入合适的收敛因子 $\epsilon$ 后（[Osnabrugge et. al., 2016](https://doi.org/10.1016/j.jcp.2016.06.034)），CBS 格式具有无条件收敛的稳定性。迭代过程中涉及到一些原始空间中的卷积操作，变换到频域空间后改为乘积操作。对于无限区域上的声波模拟问题，CBS 模块会设置 PML 层（人工构造的吸收边界层）以实现在有界区域中模拟无限区域的结果。

#### CBS 迭代流程总结

##### 初始化：$u^{(0)} = 0$

##### 计算散射势：$V = \kappa^2 - \kappa_0^2$

##### 计算格林函数：$\hat{G}_0(\boldsymbol{r}) = 1/(|\boldsymbol{r}|- \kappa_0^2 + i\epsilon)$

##### 迭代更新（对每个 $n$）：

- 计算散射场：$S^{(n)} = V u^{(n)} + f$

- Fourier 变换：$\hat{S}^{(n)} = \mathcal{F}\lbrace S^{(n)}\rbrace$

- 格林滤波：$\hat{U}^{(n)} = \hat{G}_0 \cdot \hat{S}^{(n)}$

- Fourier 逆变换：$U^{(n)} = \mathcal{F}^{-1} \lbrace \hat{U}^{(n)} \rbrace$

- 更新步长：$\Delta u^{(n)} := \frac{i}{\epsilon} (U^{(n)} - u^{(n)})$

- 更新波场：$u^{(n+1)} = u^{(n)}+\Delta u^{(n)}$

##### 判断收敛：$\|\Delta u^{{n}}\| < \text{tol}$

#### 关键组件详解

- `MixedDSTDFTn`：混合离散正弦变换（DST）和离散 Fourier 叶变换（DFT）的组合方法，对应于自由表面边界（$z=0$）；

  $$
  \mathcal{F}_{mixed}\lbrace u \rbrace = \text{DST}^{(z)}\cdot \text{DFT}^{(y)}\cdot\text{DFT}^{(x)}\lbrace u \rbrace
  $$

- `MixedIDSTDFTn`：混合离散逆正弦变换和离散 Inverse Fourier 变换的组合方法，对应于自由表面边界（$z=0$）；

  $$
  \mathcal{F}^{-1}_{mixed}\lbrace U \rbrace = \text{IDFT}^{(z)}\cdot \text{IDFT}^{(y)}\cdot\text{IDST}^{(x)}\lbrace U \rbrace
  $$

- `CBSBlock`：执行一次 CBS 迭代的模块；

- `CBS`：完整的 CBS 方法；

#### CBS 参数解析

`CBS` 设有两个求解接口，分别是 `construct` 和 `solve`，分别对应不同设置下的求解过程：

- `construct`

| 参数名    | 参数类型           | 参数含义                       | 默认值 |
  | --------- | ------------------ | ------------------------------ | ------ |
  | `c_star`  | `mindspore.Tensor` | 无量纲化的声速场，表示介质属性 | -      |
  | `f_star`  | `mindspore.Tensor` | 归一化后的源项，波的能量注入点 | -      |
  | `ur_init` | `mindspore.Tensor` | 波场实部的初始条件             | `None` |
  | `ui_init` | `mindspore.Tensor` | 波场虚部的初始条件             | `None` |

- `solve`

| 参数名       | 参数类型           | 参数含义                         | 默认值  |
| ------------ | ------------------ | -------------------------------- | ------- |
| `c_star`     | `mindspore.Tensor` | 无量纲化的声速场，表示介质属性   | -       |
| `f_star`     | `mindspore.Tensor` | 归一化后的源项，波的能量注入点   | -       |
| `ur_init`    | `mindspore.Tensor` | 波场实部的初始条件               | `None`  |
| `ui_init`    | `mindspore.Tensor` | 波场虚部的初始条件               | `None`  |
| `tol`        | `float`            | 相对误差的收敛阈值，用于停机准则 | `1e-3`  |
| `max_iter`   | `int`              | 最大迭代次数，控制计算时间       | `10000` |
| `remove_pml` | `bool`             | 控制输出是否包含 PML 层          | `True`  |
| `print_info` | `bool`             | 控制输出收敛过程的信息           | `True`  |

#### 快速入门

考虑在二维矩形区域上使用 CBS 方法求解 Helmholtz 方程。以下示例的环境为 MindSpore 2.7.1 + CANN 8.2.RC1：

```python
import numpy as np
import mindspore as ms
from mindscience.solvers import CBS

# 2D case
shape = (128, 128) # 128 grid points on each axis
dxs = (1.0, 1.0)   # mesh size
n_iter = 20        # default number of iterations
max_iter = 1000    # maximum number of itertions
pml_size = 12      # pml layers
tol = 1e-4         # tolerance of error

# generate testing data
batch_size = 1
c_star = ms.Tensor(np.ones((batch_size, 1, *shape)) * 1500.0, dtype=ms.float32)
f_star = ms.Tensor(np.zeros((batch_size, 1, *shape)), dtype=ms.float32)
f_star[0, 0, shape[0]//2, shape[1]//2] = 1.0

# define CBS solver
solver = CBS(
    shape=shape,
    dxs=dxs,
    n_iter=n_iter,
    pml_size=pml_size,
    btype="pml",
    remove_pml=False
)

# get the real and imaginary part of solution
# get numerical errors
ur, ui, errs = solver.solve(
    c_star, f_star,
    tol=tol,
    max_iter=max_iter,
    print_info=True
)

# predicted outputs
print(f"Shape of the wave field: {ur.shape}")
print(f"Numerical error: {errs[-1]}")

# ......
# step 960, max error 0.013938, min error 0.013938, mean error 0.013938, mean step time 0.0039s
# step 980, max error 0.011450, min error 0.011450, mean error 0.011450, mean step time 0.0039s
# step 1000, max error 0.010027, min error 0.010027, mean error 0.010027, mean step time 0.0039s
# Shape of the wave field: (1, 1, 100, 100)
# Numerical error: [[0.00924161]]
```

### CFD 模块介绍

CFD 模块是集成了多种经典的有限差分和有限体积格式的计算流体求解器，可以求解形如

$$
\boldsymbol{U}_t + \boldsymbol{F}(\boldsymbol{U})_x = 0
$$

的守恒律方程。流体力学的守恒律方程是描述流体运动最核心的物理定律，它们本质上表达了质量、动量和能量这些基本物理量在流动中既不会凭空产生也不会消失的规律。具体来说，它包括三个方程：**质量守恒方程（连续性方程）** 描述了流入和流出某个微元体的质量必须等于该微元体内质量的增加，保证了流体是连续的；**动量守恒方程**（通常以 Navier-Stokes 方程的形式呈现） 是牛顿第二定律在流体上的应用，表明微元体动量的变化率等于作用在其上的所有力（包括压力、粘性力和重力等）之和；**能量守恒方程** 则基于热力学第一定律，指出微元体内能量的增加率等于传入的热量与外力对流体做功功率之和。这三个方程共同构成了一个封闭的方程组。

#### 关键组件详解

- `boundary_conditions`：设置守恒律方程的多种边界条件，包括自由边界、周期边界、对称边界和固定边界；
- `integrator`：时间步迭代器，即在某一时间离散格式下的下一时间步计算，包括向前 Euler 格式和 3 阶 Runge-Kutta 格式；
- `material`：不同流体物质的参数集合，为方程提供超参数设置，目前仅包含理想气体；
- `space_solver`： 空间离散器，即对空间导数的各种离散格式，包括了黎曼问题求解器、四阶 stencil 差分算子、WENO 数值通量（3，5，7 阶）、Godunov 格式和针对黏性项的计算；

#### 核心代码实现

- 代码主要位于 `space_solver` 和 `boundary_conditions` 文件夹：

  ```bash
  cfd
  ├── boundary_conditions
  │   ├── base.py
  │   ├── boundary_manager.py
  │   ├── __init__.py
  │   ├── neumann.py
  │   ├── periodic.py
  │   ├── __pycache__
  │   │   ├── base.cpython-311.pyc
  │   │   ├── boundary_manager.cpython-311.pyc
  │   │   ├── __init__.cpython-311.pyc
  │   │   ├── neumann.cpython-311.pyc
  │   │   ├── periodic.cpython-311.pyc
  │   │   ├── symmetry.cpython-311.pyc
  │   │   └── wall.cpython-311.pyc
  │   ├── symmetry.py
  │   └── wall.py
  ├── __init__.py
  ├── integrator
  │   ├── base.py
  │   ├── euler.py
  │   ├── __init__.py
  │   ├── __pycache__
  │   │   ├── base.cpython-311.pyc
  │   │   ├── euler.cpython-311.pyc
  │   │   ├── __init__.cpython-311.pyc
  │   │   └── runge_kutta3.cpython-311.pyc
  │   └── runge_kutta3.py
  ├── material
  │   ├── base.py
  │   ├── ideal_gas.py
  │   ├── __init__.py
  │   └── __pycache__
  │       ├── base.cpython-311.pyc
  │       ├── ideal_gas.cpython-311.pyc
  │       └── __init__.cpython-311.pyc
  ├── mesh_info.py
  ├── __pycache__
  │   ├── __init__.cpython-311.pyc
  │   ├── mesh_info.cpython-311.pyc
  │   ├── runtime.cpython-311.pyc
  │   ├── simulator.cpython-311.pyc
  │   ├── utils.cpython-311.pyc
  │   └── visualization.cpython-311.pyc
  ├── runtime.py
  ├── simulator.py
  ├── space_solver
  │   ├── derivative_computer
  │   │   ├── base.py
  │   │   ├── fourth_order_central_derivative_computer.py
  │   │   ├── fourth_order_face_derivative_computer.py
  │   │   ├── __init__.py
  │   │   └── __pycache__
  │   │       ├── base.cpython-311.pyc
  │   │       ├── fourth_order_central_derivative_computer.cpython-311.pyc
  │   │       ├── fourth_order_face_derivative_computer.cpython-311.pyc
  │   │       └── __init__.cpython-311.pyc
  │   ├── godunov_flux.py
  │   ├── __init__.py
  │   ├── interpolator
  │   │   ├── base.py
  │   │   ├── central_fourth_order_interpolator.py
  │   │   ├── __init__.py
  │   │   └── __pycache__
  │   │       ├── base.cpython-311.pyc
  │   │       ├── central_fourth_order_interpolator.cpython-311.pyc
  │   │       └── __init__.cpython-311.pyc
  │   ├── __pycache__
  │   │   ├── godunov_flux.cpython-311.pyc
  │   │   ├── __init__.cpython-311.pyc
  │   │   ├── space_solver.cpython-311.pyc
  │   │   └── viscous_flux.cpython-311.pyc
  │   ├── reconstructor
  │   │   ├── base.py
  │   │   ├── __init__.py
  │   │   ├── __pycache__
  │   │   │   ├── base.cpython-311.pyc
  │   │   │   ├── __init__.cpython-311.pyc
  │   │   │   ├── weno3.cpython-311.pyc
  │   │   │   ├── weno5.cpython-311.pyc
  │   │   │   └── weno7.cpython-311.pyc
  │   │   ├── weno3.py
  │   │   ├── weno5.py
  │   │   └── weno7.py
  │   ├── riemann_computer
  │   │   ├── base.py
  │   │   ├── hllc.py
  │   │   ├── __init__.py
  │   │   ├── __pycache__
  │   │   │   ├── base.cpython-311.pyc
  │   │   │   ├── hllc.cpython-311.pyc
  │   │   │   ├── __init__.cpython-311.pyc
  │   │   │   ├── roe.cpython-311.pyc
  │   │   │   ├── rusanov.cpython-311.pyc
  │   │   │   └── rusanov_net.cpython-311.pyc
  │   │   ├── roe.py
  │   │   ├── rusanov_net.py
  │   │   └── rusanov.py
  │   ├── space_solver.py
  │   └── viscous_flux.py
  ├── utils.py
  └── visualization.py
  ```

#### 快速入门

考虑求解一维可压缩的 Euler 方程，采用 5 阶 WENO 重构以及 Rusanov 近似黎曼求解器，时间离散采用 3 阶 Runge-Kutta 格式，边界条件为两侧的 Neumann 条件。

```python
from mindscience.solvers import cfd
config = {'mesh': {'dim': 1, 'nx': 100, 'gamma': 1.4, 'x_range': [0, 1], 'pad_size': 3},
          'material': {'type': 'IdealGas', 'heat_ratio': 1.4, 'specific_heat_ratio': 1.4,
          'specific_gas_constant': 1.0}, 'runtime': {'CFL': 0.9, 'current_time': 0.0, 'end_time': 0.2},
          'integrator': {'type': 'RungeKutta3'}, 'space_solver': {'is_convective_flux': True,
          'convective_flux': {'reconstructor': 'WENO5', 'riemann_computer': 'Rusanov'},
          'is_viscous_flux': False}, 'boundary_conditions': {'x_min': {'type': 'Neumann'},
          'x_max': {'type': 'Neumann'}}}
s = cfd.Simulator(config)
```