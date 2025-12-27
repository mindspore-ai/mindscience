## mindscience.solvers

### Introduction of CBS module

The CBS module is a Helmholtz equation solver based on the Convergent Born Series (CBS) method. The Helmholtz equation is a fundamental partial differential equation describing wave phenomena (such as acoustic and electromagnetic waves) in the frequency domain, and is widely used in acoustics, electromagnetics, quantum mechanics, and other fields. Helmholtz problems typically involve Dirichlet, Neumann, Robin, and radiation boundary conditions.

$$
\Delta u + \kappa^2 u = f
$$

The CBS method solves the Helmholtz equation through Born series iteration, offering good numerical stability and convergence. Particularly, with the introduction of a suitable convergence factor $\epsilon$ ([Osnabrugge et. al., 2016](https://doi.org/10.1016/j.jcp.2016.06.034)), the CBS scheme exhibits unconditionally stable convergence. The iteration process involves several convolution operations in the original spatial domain, which are transformed into multiplication operations in the frequency domain. For simulating acoustic waves in infinite domains, the CBS module incorporates a PML (Perfectly Matched Layer) boundary—an artificially constructed absorbing boundary layer—to emulate the behavior of an unbounded region within a finite computational domain.

#### Summary of the CBS Iteration

##### Initialization: $u^{(0)} = 0$

##### Calculate the scattering potential: $V = \kappa^2 - \kappa_0^2$

##### Green's function: $\hat{G}_0(\boldsymbol{r}) = 1/(|\boldsymbol{r}|- \kappa_0^2 + i\epsilon)$

##### Iteration (for each $n$)

- Scattered field: $S^{(n)} = V u^{(n)} + f$

- Fourier transform: $\hat{S}^{(n)} = \mathcal{F}\lbrace S^{(n)}\rbrace$

- Green's function filtering: $\hat{U}^{(n)} = \hat{G}_0 \cdot \hat{S}^{(n)}$

- Inverse Fourier transform: $U^{(n)} = \mathcal{F}^{-1} \lbrace \hat{U}^{(n)} \rbrace$

- New step: $\Delta u^{(n)} := \frac{i}{\epsilon} (U^{(n)} - u^{(n)})$

- Update: $u^{(n+1)} = u^{(n)}+\Delta u^{(n)}$

##### Stopping criterion: $\|\Delta u^{{n}}\| < \text{tol}$

#### Key Components

- `MixedDSTDFTn`: Hybrid method combining Discrete Sine Transform (DST) and Discrete Fourier Transform (DFT), corresponding to free-surface boundary conditions at $z=0$;

  $$
  \mathcal{F}_{mixed}\lbrace u \rbrace = \text{DST}^{(z)}\cdot \text{DFT}^{(y)}\cdot\text{DFT}^{(x)}\lbrace u \rbrace
  $$

- `MixedIDSTDFTn`: Hybrid method combining the Inverse Discrete Sine Transform (IDST) and the Inverse Discrete Fourier Transform (IDFT), corresponding to the free-surface boundary condition at $z=0$;

  $$
  \mathcal{F}^{-1}_{mixed}\lbrace U \rbrace = \text{IDFT}^{(z)}\cdot \text{IDFT}^{(y)}\cdot\text{IDST}^{(x)}\lbrace U \rbrace
  $$

- `CBSBlock`: Module for executing one iteration of the CBS (Convergent Born Series) method.

- `CBS`: The complete CBS (Convergent Born Series) method.

#### Parameters

`CBS` provides two solving interfaces: `construct` and `solve`, corresponding to the solving processes under different configurations.

- `construct`

| Name      | Type               | Definitions                                                  | Default |
| --------- | ------------------ | ------------------------------------------------------------ | ------- |
| `c_star`  | `mindspore.Tensor` | The dimensionless sound speed field, representing medium properties. | -       |
| `f_star`  | `mindspore.Tensor` | The normalized source term, the energy injection point of the wave. | -       |
| `ur_init` | `mindspore.Tensor` | Initial condition for the real part of the wavefield.        | `None`  |
| `ui_init` | `mindspore.Tensor` | Initial condition for the imaginary part of the wavefield.   | `None`  |

- `solve`

| Name         | Type               | Definitions                                                  | Default |
| ------------ | ------------------ | ------------------------------------------------------------ | ------- |
| `c_star`     | `mindspore.Tensor` | The dimensionless sound speed field, representing medium properties. | -       |
| `f_star`     | `mindspore.Tensor` | The normalized source term, the energy injection point of the wave. | -       |
| `ur_init`    | `mindspore.Tensor` | Initial condition for the real part of the wavefield.        | `None`  |
| `ui_init`    | `mindspore.Tensor` | Initial condition for the imaginary part of the wavefield.   | `None`  |
| `tol`        | `float`            | Convergence threshold for relative error, used as a stopping criterion. | `1e-3`  |
| `max_iter`   | `int`              | Maximum number of iterations, controlling computation time.  | `10000` |
| `remove_pml` | `bool`             | Controls whether the output includes the PML region.         | `True`  |
| `print_info` | `bool`             | Controls the output of convergence process information.      | `True`  |

#### Quick Start

Consider solving the Helmholtz equation on a two-dimensional rectangular domain using the CBS (Convergent Born Series) method. In the following example we use the environment MindSpore 2.7.1 + CANN 8.2.RC1:

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

#
# step 960, max error 0.013938, min error 0.013938, mean error 0.013938, mean step time 0.0039s
# step 980, max error 0.011450, min error 0.011450, mean error 0.011450, mean step time 0.0039s
# step 1000, max error 0.010027, min error 0.010027, mean error 0.010027, mean step time 0.0039s
# Shape of the wave field: (1, 1, 100, 100)
# Numerical error: [[0.00924161]]
```

### Indroduction of CFD module

The CFD module is a computational fluid dynamics solver that integrates various classical finite difference and finite volume schemes. It can solve conservation law equations of the form:

$$
\boldsymbol{U}_t + \boldsymbol{F}(\boldsymbol{U})_x = 0.
$$

The conservation laws of fluid dynamics are the most fundamental physical principles describing fluid motion, expressing that basic physical quantities—mass, momentum, and energy—are neither created nor destroyed in the flow. Specifically, the system consists of three equations:

1. Mass Conservation Equation (Continuity Equation): Describes that the mass flowing into and out of a fluid element must equal the increase in mass within that element, ensuring the continuity of the fluid.
2. Momentum Conservation Equation (typically expressed in the Navier-Stokes form): Applies Newton’s second law to fluid motion, stating that the rate of change of momentum of a fluid element is equal to the sum of all forces (including pressure, viscous forces, gravity, etc.) acting on it.
3. Energy Conservation Equation: Based on the first law of thermodynamics, it states that the rate of increase of energy within a fluid element equals the sum of the heat transferred into the element and the work done on it by external forces.

Together, these three equations form a closed system.

#### Key Components

- `boundary_conditions`: Set various boundary conditions for the conservation law equations, including free boundaries, periodic boundaries, symmetric boundaries, and fixed boundaries.
- `integrator`: Time step iterator, which computes the next time step under a specific temporal discretization scheme, including the forward Euler method and the third-order Runge-Kutta method.
- `material`: Parameter sets for different fluid materials, providing hyperparameter configurations for the equations. Currently, only the ideal gas is included.；
- `space_solver`: Spatial discretizer, which encompasses various discretization schemes for spatial derivatives, including Riemann problem solvers, fourth-order stencil differencing and interpolation, WENO numerical fluxes (3rd, 5th, and 7th order), the Godunov method, and computations for viscous terms.

#### Implementation

- The code is primarily located in the `space_solver` and `boundary_conditions` folders:

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

#### Quick Start

Consider solving the one-dimensional compressible Euler equations using a 5th-order WENO reconstruction scheme and the Rusanov Riemann solver, with time discretization via the 3rd-order Runge-Kutta method and Neumann boundary conditions applied on both sides. The solver allows users to use a `config` dictionary as input.

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