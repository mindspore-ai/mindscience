# Writing SPH Equations

PySPH equations define particle interactions in a restricted but powerful syntax.

## Equation Structure

```python
from pysph.sph.equation import Equation

class MyEquation(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)

    def initialize(self, d_idx, d_rho):
        # Called once per destination particle

    def loop(self, d_idx, d_rho, s_m, s_idx, WIJ):
        # Called for each particle pair

    def post_loop(self, d_idx, d_rho):
        # Called after all neighbor loops

    def reduce(self, dst, t, dt):
        # Called once per destination array
```

## Method Execution Order

For equation `MyEquation(dest='fluid', sources=['fluid', 'solid'])`:

1. `py_initialize(dst, t, dt)` - Pure Python, once per destination
2. `initialize(d_idx, ...)` - Once per particle, for each source
3. `initialize_pair(d_idx, ...)` - Once per particle, all sources available
4. `loop_all(d_idx, NBRS, N_NBRS, ...)` - Once per particle, all neighbors
5. `loop(d_idx, s_idx, ...)` - For each particle pair
6. `post_loop(d_idx, ...)` - Once per particle, after all sources
7. `reduce(dst, t, dt)` - Pure Python, once per destination

## Available Precomputed Quantities

Inside equation methods, these are automatically available:

**Geometric quantities**:
- `HIJ`: 0.5*(d_h[d_idx] + s_h[s_idx])
- `XIJ[0,1,2]`: d_x - s_x, d_y - s_y, d_z - s_z
- `R2IJ`: XIJ[0]^2 + XIJ[1]^2 + XIJ[2]^2
- `RIJ`: sqrt(R2IJ)

**Kernel quantities**:
- `WIJ`: KERNEL(XIJ, RIJ, HIJ)
- `WJ`: KERNEL(XIJ, RIJ, s_h[s_idx])
- `DWIJ, DWJ, DWI`: Kernel gradients

**Other quantities**:
- `RHOIJ`: 0.5*(d_rho[d_idx] + s_rho[s_idx])
- `VIJ[0,1,2]`: d_u - s_u, d_v - s_v, d_w - s_w
- `EPS`: 0.01 * HIJ * HIJ

**Math constants**:
- `M_PI`: π value
- `M_E`: e value
- `M_SQRT2`: √2
- `M_2_PI`: 2/π

## Variable Declarations

Declare temporary variables for performance:

```python
def loop(self, d_idx, d_rho, s_m, s_idx, WIJ):
    i, j = declare('int', 2)              # Two integers
    vec = declare('matrix(3)')               # 3-element vector
    mat = declare('matrix((3,3)')          # 3x3 matrix
    result = 0.0                          # Scalar
```

## Loop All Method

For non-pairwise computations or complex neighbor operations:

```python
class DensityEquation(Equation):
    def initialize(self, d_idx, d_rho):
        d_rho[d_idx] = 0.0

    def loop_all(self, d_idx, d_rho, d_x, d_y,
                     s_m, s_x, s_y, SPH_KERNEL,
                     NBRS, N_NBRS):
        i = declare('int')
        s_idx = declare('long')
        xij = declare('matrix(3)')
        rij = 0.0
        sum_val = 0.0

        for i in range(N_NBRS):
            s_idx = NBRS[i]
            xij[0] = d_x[d_idx] - s_x[s_idx]
            xij[1] = d_y[d_idx] - s_y[s_idx]
            rij = sqrt(xij[0]*xij[0] + xij[1]*xij[1])
            sum_val += s_m[s_idx]*SPH_KERNEL.kernel(xij, rij, 0.5*(s_h[s_idx] + d_h[d_idx]))

        d_rho[d_idx] += sum_val
```

**Use when**:
- Non-pairwise interactions
- Matrix operations per particle
- Complex neighbor algorithms
- Molecular dynamics

## Reduce Method

Perform global reductions across all particles:

```python
class GlobalStats(Equation):
    def reduce(self, dst, t, dt):
        m = serial_reduce_array(dst.m, 'sum')
        max_u = serial_reduce_array(dst.u, 'max')
        dst.total_mass[0] = parallel_reduce_array(m, 'sum')
        dst.max_u[0] = parallel_reduce_array(max_u, 'max')
```

**Operations**: `sum`, `prod`, `max`, `min`

**Use when**:
- Computing total mass/momentum
- Finding maximum/minimum values
- Convergence criteria
- Global statistics

## User-Defined Helper Functions

Call custom Python functions from equations:

```python
def my_helper(x=1.0):
    return x * 2.0 + 1.0

class MyEquation(Equation):
    def loop(self, d_idx, d_val, s_idx):
        d_val[d_idx] += my_helper(s_m[s_idx])

    def _get_helpers_(self):
        return [my_helper]
```

**Array helper example**:
```python
def vector_sum(x=[1.0, 1.0, 1.0], n=3):
    i = declare('int')
    result = 0.0
    for i in range(n):
        result += x[i]
    return result
```

## Groups and Equation Execution

Equations are organized into groups:

```python
from pysph.sph.equation import Group

equations = [
    Group(
        equations=[
            DensityEquation(dest='fluid', sources=['fluid']),
            PressureEquation(dest='fluid', sources=['fluid'])
        ]
    ),
    Group(
        equations=[
            AccelerationEquation(dest='fluid', sources=['fluid'])
        ]
    )
]
```

**Execution order**:
1. All equations in first group complete
2. All equations in second group complete
3. Continue sequentially

**Different destinations** are processed separately even in same group.

## Conditional Group Execution

Execute groups based on conditions:

```python
def every_20_steps(t, dt):
    return int(t/dt) % 20 == 0

equations = [
    Group(equations=[DensityEquation(...)]),
    Group(
        equations=[FilterEquation(...)],
        condition=every_20_steps
    )
]
```

## Pre and Post Functions

Execute arbitrary Python code before/after groups:

```python
def my_pre_function():
    print("Before group execution")

def my_post_function():
    print("After group execution")

equations = [
    Group(
        equations=[DensityEquation(...)],
        pre=my_pre_function,
        post=my_post_function
    )
]
```

**Use when**:
- Complex Python operations
- File I/O during simulation
- Custom algorithms
- Debugging

## Particle Subset Iteration

Iterate over subset of particles:

```python
Group(
    equations=[MyEquation(dest='fluid', sources=['fluid'])],
    start_idx=10,      # Start at particle 10
    stop_idx=20       # Stop at particle 20
)
```

**Using constants**:
```python
Group(
    equations=[MyEquation(...)],
    stop_idx='n_body'    # Use constant value
)
```

## Real vs Ghost Particles

Control which particles are operated on:

```python
# Only real particles (default)
Group(equations=[Eq1(...)], real=True)

# Include ghost particles
Group(equations=[Eq2(...)], real=False)
```

**Particle tags**:
- `Local = 0`: Real particles owned by processor
- `Remote = 1`: Real particles owned by other processor
- `Ghost = 2`: Boundary condition particles

## Convergence in Iterated Groups

For groups with `iterate=True`:

```python
class IterativeEquation(Equation):
    def converged(self):
        return dst.max_error[0] < 1e-6

equations = [
    Group(
        equations=[IterativeEquation(...)],
        iterate=True
    )
]
```

**Return value**:
- `> 0`: Converged
- `<= 0`: Not converged

## Performance Tips

1. **Use precomputed quantities**: Avoid recomputing XIJ, RIJ, WIJ
2. **Declare variables properly**: Use `declare()` for temporaries
3. **Minimize operations**: Combine computations where possible
4. **Use loop_all for complex operations**: More efficient than pairwise
5. **Avoid Python functions in loops**: Use helper functions sparingly
6. **Use stride properties**: For multiple values per particle

## Common Patterns

### Density Computation

```python
class DensityEquation(Equation):
    def loop(self, d_idx, d_rho, d_h, s_m, s_h, s_idx, WIJ):
        d_rho[d_idx] += s_m[s_idx] * WIJ
```

### Pressure Computation (Tait equation of state)

```python
class PressureEquation(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)
        self.gamma = 1.4
        self.c0 = 1400.0

    def post_loop(self, d_idx, d_rho, d_p, d_h):
        d_p[d_idx] = self.c0 * (d_rho[d_idx]/1000.0)**self.gamma
```

### Acceleration Computation

```python
class AccelerationEquation(Equation):
    def loop(self, d_idx, d_au, d_p, s_m, s_p, s_idx, DWIJ):
        d_au[d_idx] += -s_m[s_idx] * (d_p[d_idx] + s_p[s_idx]) * DWIJ
```

### Artificial Viscosity

```python
class ViscosityEquation(Equation):
    def __init__(self, dest, sources):
        super().__init__(dest, sources)
        self.alpha = 0.1
        self.beta = 0.1

    def loop(self, d_idx, d_au, d_u, d_v, s_m, s_u, s_v, s_idx, DWIJ):
        vij_x = d_u[d_idx] - s_u[s_idx]
        vij_y = d_v[d_idx] - s_v[s_idx]
        rij2 = XIJ[0]*XIJ[0] + XIJ[1]*XIJ[1]
        dot = vij_x*XIJ[0] + vij_y*XIJ[1]
        visc = (self.alpha * d_h[d_idx] * dot + self.beta * d_h[d_idx]**2) / (rij2 + 0.01*d_h[d_idx]**2)
        d_au[d_idx] += s_m[s_idx] * visc * DWIJ
```

## Debugging Equations

1. **Check generated code**: Look at `~/.pysph/source` directory
2. **Print intermediate values**: Use temporary variables and check values
3. **Validate properties**: Ensure arrays have correct properties
4. **Test with small particle count**: Easier to debug
5. **Use reduce for statistics**: Check global values
6. **Compare with known solutions**: Validate equation correctness

## Advanced Topics

- **Multi-stage equations**: See [integrators.md](integrators.md)
- **Adaptive timesteps**: Use `dt_adapt` property
- **Particle shifting**: Correct particle positions
- **Boundary conditions**: See [boundaries.md](boundaries.md)
- **Parallel execution**: See [parallel.md](parallel.md)
