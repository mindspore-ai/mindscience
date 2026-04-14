# Boundary Conditions

Comprehensive guide to boundary condition types in FiPy.

## Dirichlet (Fixed Value)

Specify fixed value at boundary:

```python
# Fixed value on left boundary
phi.constrain(0., where=mesh.facesLeft)

.

# Fixed value on right boundary
phi.conphi.constrain(1., where=mesh.facesRight)

# Fixed value on all exterior faces
phi.constrain(0., where=mesh.exteriorFaces)
```

**Use cases:**
- Fixed temperature at walls
- Prescribed concentration at inlets
- Voltage at electrodes

## Neumann (Fixed Gradient)

Specify fixed gradient (flux) at boundary:

```python
# Zero gradient (insulated/no flux)
phi.faceGrad.constrain(0., where=mesh.facesTop)

# Fixed gradient in normal direction
phi.faceGrad.constrain(2.0, where=mesh.facesRight)

# Fixed gradient vector (2D)
phi.faceGrad.constrain(((0.,), (2.,)), where=mesh.facesTop)

# Normal gradient using face normals
phi.faceGrad.constrain(1.0 * mesh.faceNormals, where=mesh.exteriorFaces)
```

**Use cases:**
- Insulated boundaries (zero flux)
- Prescribed heat flux
- Symmetry conditions

## Robin (Mixed) Boundary Condition

Combines value and gradient:

$$\hat{n} \cdot (a\phi + b\nabla\phi) = g$$

**Heat transfer example:**
$$-k \hat{n} \cdot \nabla T = h(T - T_\infty)$$

**Implementation:**
```python
from fipy import FaceVariable, DiffusionTerm, PowerLawConvectionTerm

# Parameters
k = 1.0          # Thermal conductivity
h = 10.0         # Heat transfer coefficient
T_inf = 25.0     # Ambient temperature

# Create coefficient variables
a = FaceVariable(mesh=mesh, value=0., rank=1)
b = FaceVariable(mesh=mesh, value=k)
g = FaceVariable(mesh=mesh, value=h * T_inf)

# Apply Robin condition on right boundary
mask = mesh.facesRight
a.setValue(h * mesh.faceNormals, where=mask)

# Modified equation with Robin BC
eq = (TransientTerm()
      == PowerLawConvectionTerm(coeff=a)
      + DiffusionTerm(coeff=b)
      + (g * mask * mesh.faceNormals).divergence)
```

**General Robin condition:**
```python
# For condition: n·(aφ + b∇φ) = g on boundary S_R
mask = mesh.facesRight  # Boundary where Robin applies

# Zero out diffusion/convection on Robin boundary
diffCoeff = FaceVariable(mesh=mesh, value=D)
diffCoeff.setValue(0., where=mask)

convCoeff = FaceVariable(mesh=mesh, value=(0., 0.), rank=1)
convCoeff.setValue((0., 0.), where=mask)

# Add Robin term
dPf = FaceVariable(mesh=mesh, value=mesh._faceToCellDistanceRatio * mesh.cellDistanceVectors)
n = mesh.faceNormals
a = FaceVariable(mesh=mesh, value=a_val, rank=1)
b = FaceVariable(mesh=mesh, value=b_val)
g = FaceVariable(mesh=mesh, value=g_val)

RobinCoeff = mask * D * n / (dPf.dot(a) + b)

eq = (TransientTerm()
      == DiffusionTerm(coeff=diffCoeff)
      + (RobinCoeff * g).divergence
      - ImplicitSourceTerm((RobinCoeff * n.dot(a)).divergence))
```

## Spatially Varying Boundary Conditions

**Varying Dirichlet:**
```python
X, Y = mesh.faceCenters

# φ = x*y on top boundary
phi.constrain(X * Y, where=mesh.facesTop)

# φ = sin(x) on right boundary
phi.constrain(np.sin(X), where=mesh.facesRight)
```

**Varying Neumann:**
```python
# Gradient varies with position
X, Y = mesh.faceCenters
grad_value = 1.0 + 0.5 * np.sin(2 * np.pi * X)
phi.faceGrad.constrain(grad_value * mesh.faceNormals, where=mesh.facesTop)
```

**Complex spatial variation:**
```python
# φ = 1 on top-right quadrant, φ = 0 elsewhere
mask = (X > 0.5) & (Y > 0.5)
phi.constrain(1., where=mesh.exteriorFaces & mask)
phi.faceGrad.constrain(0., where=mesh.exteriorFaces & ~mask)
```

## Periodic Boundaries

Periodic boundaries are the default for structured grids:

```python
# No explicit constraints needed
# Grid meshes automatically have periodic boundaries
```

For explicit periodic behavior:
```python
# Left-right periodic
phi.constrain(phi, where=mesh.facesLeft)
phi.constrain(phi, where=mesh.facesRight)
```

## Outlet/Inlet Conditions

For convection-dominated flows:

```python
from fipy import ConvectionTerm

velocity = FaceVariable(mesh=mesh, value=(1., 0.), rank=1)

# Inlet (fixed value) on left
phi.constrain(1., where=mesh.facesLeft)

# Outlet (zero gradient) on right
phi.faceGrad.constrain(0., where=mesh.facesRight)

eq = TransientTerm() + ConvectionTerm(coeff=velocity) == DiffusionTerm()
```

## Internal Boundary Conditions

**Fixed value at internal region:**
```python
# Fix φ = 0.5 in region x < 0.3
mask = mesh.x < 0.3
large_value = 1e10

eq = (TransientTerm()
      == DiffusionTerm()
      - ImplicitSourceTerm(mask * large_value)
      + mask * large_value * 0.5)
```

**Fixed gradient at internal interface:**
```python
# Fixed gradient at x = 0.5
mask = (mesh.x > 0.49) & (mesh.x < 0.51)
large_value = 1e10

Gamma = FaceVariable(mesh=mesh, value=D)
Gamma.setValue(0., where=mask)

eq = (TransientTerm()
      == DiffusionTerm(coeff=Gamma)
      + DiffusionTerm(coeff=large_value * mask)
      - ImplicitSourceTerm(mask * large_value * gradient * mesh.faceNormals).divergence)
```

## Default Boundary Conditions

If no constraints are applied, FiPy uses **zero flux** (Neumann) by default:

$$\hat{n} \cdot (a\phi + b\nabla\phi) = 0$$

This means:
- No mass/energy enters or leaves the domain
- Conservative by default
- Insulated boundaries

## Common Patterns

**Insulated box:**
```python
# No constraints needed - zero flux by default
```

**Fixed temperature on all walls:**
```python
T_wall = 300.0
T.constrain(T_wall, where=mesh.exteriorFaces)
```

**Left-right gradient, top-bottom insulated:**
```python
phi.constrain(0., where=mesh.facesLeft)
phi.constrain(1., where=mesh.facesRight)
# Top and bottom: zero flux (default)
```

**Inlet-outlet with side walls insulated:**
```python
phi.constrain(1.0, where=mesh.facesLeft)        # Inlet
phi.faceGrad.constrain(0., where=mesh.facesRight)  # Outlet
# Top and bottom: zero flux (default)
```

## Troubleshooting

**Boundary conditions not working:**
- Ensure constraints are applied before solving
- Check that `where` condition is correct
- Verify face selection (facesLeft, facesRight, etc.)

**Unexpected behavior at boundaries:**
- Check for conflicting constraints
- Ensure mesh has correct boundary faces
- Verify constraint values are appropriate

**Robin condition issues:**
- Ensure coefficient variables have correct rank
- Check face normal directions
- Verify mask identifies correct boundary faces
