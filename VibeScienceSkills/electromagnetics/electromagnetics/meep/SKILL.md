---
name: meep
description: MIT's open-source FDTD electromagnetic simulation solver. Use for time-domain Maxwell's equations simulation including: (1) Waveguide and photonic-crystal simulations, (2) Transmission/reflection spectra computation, (3) Resonant mode analysis, (4) Scattering problems, (5) Nonlinear and dispersive materials, (6) Near-to-far field transformations. Supports Python, Scheme, and C++ interfaces with 1D/2D/3D simulations, PML boundaries, symmetries, and parallel computing via MPI.
---

# Meep: MIT Electromagnetic Equation Propagation

Meep is a free and open-source software package for electromagnetic simulation via the finite-difference time-domain (FDTD) method.

## Quick Start

Basic simulation workflow:

```python
import meep as mp

# Define computational cell
cell = mp.Vector3(16, 8, 0)  # 2D simulation (z=0)

# Define geometry
geometry = [mp.Block(mp.Vector3(mp.inf, 1, mp.inf),
                     center=mp.Vector3(),
                     material=mp.Medium(epsilon=12))]

# Define source
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0))]

# Define PML boundaries
pml_layers = [mp.PML(1.0)]

# Set resolution (pixels per unit distance)
resolution = 10

# Create simulation
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# Run simulation
sim.run(until=200)
```

## Core Concepts

### Units in Meep

Meep uses dimensionless units where the speed of light c = 1. Choose a characteristic length scale and set it to 1.

- **Frequency**: Specified in units of 2πc (inverse vacuum wavelength)
- **Wavelength**: λ = 1/f (in vacuum units)
- **Time**: t (in units where c=1)
- **Distance**: User-defined unit (e.g., μm, nm)

Example: If unit distance = 1 μm, then frequency=0.15 corresponds to λ ≈ 6.67 μm.

### Simulation Dimensions

- **1D**: `cell = mp.Vector3(0, 0, sz)` - Only Ex and Hy components
- **2D**: `cell = mp.Vector3(sx, sy, 0)` - Optimized for planar problems
- **3D**: `cell = mp.Vector3(sx, sy, sz)` - Full 3D simulation
- **Cylindrical**: `dimensions=mp.CYLINDRICAL` - For rotationally symmetric problems

### Materials

See [materials.md](references/materials.md) for comprehensive material definitions.

Basic material:
```python
mp.Medium(epsilon=12)  # Dielectric constant
```

Complex materials with dispersion, conductivity, nonlinearity:
```python
# Conductive material (loss)
mp.Medium(epsilon=3.4, D_conductivity=0.5)

# Lorentzian dispersion
mp.Medium(epsilon=3.4,
            susc=mp.LorentzianSusceptibility(frequency=0.3,
                                            gamma=0.1,
                                            sigma=0.5))

# Nonlinear (Kerr effect)
mp.Medium(epsilon=2.0, chi3=0.1)
```

Predefined materials from library:
```python
from meep.materials import Si, Au, Al
geometry = [mp.Block(material=Si, ...)]
```

### Geometric Objects

See [geometry.md](references/geometry.md) for complete object types.

Common primitives:
```python
# Block (parallelepiped)
mp.Block(size=mp.Vector3(10, 5, mp.inf),
          center=mp.Vector3(0, 0, 0),
          material=mp.Medium(epsilon=12))

# Cylinder
mp.Cylinder(radius=2, height=mp.inf,
            center=mp.Vector3(0, 0, 0),
            axis=mp.Z,
            material=mp.Medium(epsilon=12))

# Sphere
mp.Sphere(radius=1.5,
          center=mp.Vector3(0, 0, 0),
          material=mp.Medium(epsilon=12))

# Ellipsoid
mp.Ellipsoid(size=mp.Vector3(3, 2, 1),
             center=mp.Vector3(0, 0, 0),
             material=mp.Medium(epsilon=12))
```

### Sources

See [sources.md](references/sources.md) for detailed source configurations.

Source types:
```python
# Continuous wave (CW)
mp.ContinuousSource(frequency=0.15, width=20)

# Gaussian pulse
mp.GaussianSource(fcen=0.15, fwidth=0.1)

# Custom source
mp.CustomSource(src_func=my_function)
```

Field components: `mp.Ex`, `mp.Ey`, `mp.Ez`, `mp.Hx`, `mp.Hy`, `mp.Hz`

### Boundary Conditions

See [boundary_conditions.md](references/boundary_conditions.md) for complete options.

PML (absorbing boundaries):
```python
pml_layers = [mp.PML(thickness=1.0)]
pml_layers = [mp.PML(thickness=1.0, direction=mp.X, side=mp.High)]
```

Perfect conductor:
```python
sim.set_boundary(mp.Low, mp.X, mp.Metallic)
```

Bloch-periodic:
```python
sim = mp.Simulation(..., k_point=mp.Vector3(0.1, 0, 0))
```

### Symmetries

Exploit symmetries to reduce computation:
```python
symmetries = [mp.Mirror(mp.Y),  # Mirror symmetry in Y
              mp.Mirror(mp.Z, phase=-1)]  # Mirror with phase shift in Z
```

## Common Workflows

### Workflow 1: Basic Field Simulation

Simulate field propagation and visualize:

```python
import meep as mp
import numpy as np
import matplotlib.pyplot as plt

# Setup simulation
cell = mp.Vector3(16, 8, 0)
geometry = [mp.Block(mp.Vector3(mp.inf, 1, mp.inf),
                     material=mp.Medium(epsilon=12))]
sources = [mp.Source(mp.ContinuousSource(frequency=0.15),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0))]
pml_layers = [mp.PML(1.0)]
resolution = 10

sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# Run simulation
sim.run(until=200)

# Get dielectric function
eps_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)

# Get electric field
ez_data = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)

# Visualize
plt.figure()
plt.imshow(eps_data.transpose(), cmap='binary')
plt.imshow(ez_data.transpose(), cmap='RdBu', alpha=0.9)
plt.axis('off')
plt.show()
```

### Workflow 2: Transmission/Reflection Spectra

Compute broadband spectra from single simulation:

```python
import meep as mp
import numpy as np

# Setup parameters
fcen = 0.15  # Center frequency
df = 0.1     # Frequency width
nfreq = 100  # Number of frequency points

# Gaussian source for broadband excitation
sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                     component=mp.Ez,
                     center=mp.Vector3(-7, 0))]

# Create simulation
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    sources=sources,
                    resolution=resolution)

# Add flux monitors
refl_fr = mp.FluxRegion(center=mp.Vector3(-5, 0), size=mp.Vector3(0, 2))
tran_fr = mp.FluxRegion(center=mp.Vector3(5, 0), size=mp.Vector3(0, 2))

refl = sim.add_flux(fcen, df, nfreq, refl_fr)
tran = sim.add_flux(fcen, df, nfreq, tran_fr)

# Run until fields decay
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(5, 0), 1e-3))

# Get flux spectra
refl_flux = mp.get_fluxes(refl)
tran_flux = mp.get_fluxes(tran)
freqs = mp.get_flux_freqs(refl)

# Compute transmittance and reflectance
incident_flux = sum(tran_flux)  # For normalization
transmittance = tran_flux / incident_flux
reflectance = -refl_flux / incident_flux
```

For normalization (two-run method), see [flux_analysis.md](references/flux_analysis.md).

### Workflow 3: Resonant Mode Analysis

Extract resonant frequencies and Q factors:

```python
import meep as mp

# Setup cavity simulation
sim = mp.Simulation(...)

# Add Harminv mode monitor
harminv = mp.Harminv(component=mp.Ez,
                       frequency=0.15,
                       decay_by=0.001)

# Run simulation
sim.run(until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, mp.Vector3(0, 0), 1e-3))

# Get modes
modes = harminv.modes
for mode in modes:
    print(f"Frequency: {mode.freq}")
    print(f"Decay rate: {mode.decay}")
    print(f"Q factor: {mode.freq / (2*mode.decay)}")
```

### Workflow 4: Scattering Problems

Compute scattering cross section:

```python
import meep as mp

# Surround scatterer with flux box
box_x1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(x=-r), size=mp.Vector3(0, 2*r, 2*r)))
box_x2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(x=+r), size=mp.Vector3(0, 2*r, 2*r)))
box_y1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(y=-r), size=mp.Vector3(2*r, 0, 2*r)))
box_y2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(y=+r), size=mp.Vector3(2*r, 0, 2*r)))
box_z1 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=-r), size=mp.Vector3(2*r, 2*r, 0)))
box_z2 = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(center=mp.Vector3(z=+r), size=mp.Vector3(2*r, 2*r, 0)))

# Run normalization (empty cell)
sim.run(until_after_sources=10)
box_x1_flux0 = mp.get_fluxes(box_x1)
# ... save all fluxes

# Run with scatterer
sim.reset_meep()
geometry = [mp.Sphere(radius=r, material=mp.Medium(index=2.0))]
sim = mp.Simulation(..., geometry=geometry)
# ... re-add flux boxes

sim.run(until_after_sources=10)

# Compute scattered power
scattered_power = (sum(mp.get_fluxes(box_x1)) - box_x1_flux0 + \
                 (sum(mp.get_fluxes(box_x2)) - box_x2_flux0 + \
                 (sum(mp.get_fluxes(box_y1)) - box_y1_flux0 + \
                 (sum(mp.get_fluxes(box_y2)) - box_y2_flux0 + \
                 (sum(mp.get_fluxes(box_z1)) - box_z1_flux0 + \
                 (sum(mp.get_fluxes(box_z2)) - box_z2_flux0)

scattering_cross_section = scattered_power / incident_intensity
```

### Workflow 5: Mode Decomposition

Decpose fields into waveguide modes:

```python
import meep as mp

# Get waveguide mode profile
k = mp.Vector3(0.15, 0, 0)  # Propagation constant
mode = sim.get_eigenmode(k, mp.Ez, mp.Vector3(0, 0), mp.Vector3(0, 1))

# Add mode decomposition monitor
d = sim.add_mode_monitor(k, mode, mp.FluxRegion(center=mp.Vector3(5, 0), size=mp.Vector3(0, 2)))

# Run simulation
sim.run(until=200)

# Get mode coefficients
mode_coeffs = sim.get_mode_coeffs(d)
```

## Advanced Features

### Dispersive Materials

Model frequency-dependent materials:

```python
# Lorentzian resonance
susceptibility = mp.LorentzianSusceptibility(frequency=0.3,
                                             gamma=0.1,
                                             sigma=0.5)
material = mp.Medium(epsilon=3.4, susc=susceptibility)

# Drude model (for metals)
susceptibility = mp.DrudeSusceptibility(gamma=0.1, sigma=0.5)
material = mp.Medium(epsilon=1.0, susc=susceptibility)

# Multiple resonances
susceptibilities = [mp.LorentzianSusceptibility(frequency=0.3, gamma=0.1, sigma=0.5),
                    mp.LorentzianSusceptibility(frequency=0.5, gamma=0.2, sigma=0.3)]
material = mp.Medium(epsilon=3.4, susc=susceptibilities)
```

### Nonlinear Materials

Kerr and Pockels nonlinearities:

```python
# Kerr nonlinearity (χ³)
material = mp.Medium(epsilon=2.0, chi3=0.1)

# Pockels nonlinearity (χ²)
material = mp.Medium(epsilon=2.0, chi2_diag=[0.1, 0.1, 0.1])
```

### Near-to-Far Field Transformation

Compute far-field radiation pattern:

```python
import meep as mp

# Define near-field surface
n2f_region = [mp.FluxRegion(center=mp.Vector3(x=-5), size=mp.Vector3(0, 10, 10)),
                mp.FluxRegion(center=mp.Vector3(x=5), size=mp.Vector3(0, 10, 10)),
                mp.FluxRegion(center=mp.Vector3(y=-5), size=mp.Vector3(10, 0, 10)),
                mp.FluxRegion(center=mp.Vector3(y=5), size=mp.Vector3(10, 0, 10))]

# Add near-to-far monitor
n2f = sim.add_near2far(fcen, df, nfreq, n2f_region)

# Run simulation
sim.run(until=200)

# Compute far-field
ff = sim.get_farfield(n2f, mp.Vector3(10, 0, 0))  # Far field at (10, 0, 0)
```

### Frequency-Domain Solver

Find steady-state response to CW source:

```python
import meep as mp

# Use frequency-domain solver
sim = mp.Simulation(...)

# Run frequency-domain simulation
sim.run(until=200)

# Get fields at specific frequency
freq = 0.15
sim.solve_cw(freq)
```

### Eigensolver

Find resonant modes directly:

```python
import meep as mp

# Setup simulation without sources
sim = mp.Simulation(cell_size=cell,
                    boundary_layers=pml_layers,
                    geometry=geometry,
                    resolution=resolution)

# Run eigensolver
num_modes = 10
target_freq = 0.15
k = mp.Vector3(0, 0, 0)  # k-point for periodic structures

sim.run(mp.eigensolver(num_modes, target_freq, k))
```

## Output and Visualization

### HDF5 Output

```python
# Output dielectric function
sim.run(mp.at_beginning(mp.output_epsilon))

# Output electric field at every time step
sim.run(mp.at_every(0.6, mp.output_efield_z))

# Output to single HDF5 file with time dimension
sim.run(mp.to_appended("ez", mp.at_every(0.6, mp.output_efield_z)))
```

### PNG Output

```python
# Output PNG images directly
sim.run(mp.at_every(0.6, mp.output_png(mp.Ez, "-Zc dkbluered")))
```

### Custom Output Functions

```python
def custom_output(sim):
    ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
    # Process data...

sim.run(mp.at_every(0.6, custom_output))
```

### Output Directory

```python
# Put all output files in subdirectory
sim.use_output_directory("my_simulation")
```

## Best Practices

### Resolution Guidelines

- Minimum: 8 pixels per wavelength in highest-index material
- Recommended: 10-20 pixels per wavelength
- High accuracy: 20+ pixels per wavelength

### PML Thickness

- Minimum: 1-2 wavelengths
- Recommended: 2-3 wavelengths
- Ensure PML overlaps structures for proper absorption

### Convergence Testing

Always test convergence by:
1. Doubling resolution
2. Increasing cell size
3. Reducing PML reflections

### Symmetry Exploitation

Use symmetries when possible:
- Mirror planes for symmetric structures
- Rotation symmetries for periodic structures
- Can reduce computation by 2x, 4x, or 8x

### Source Placement

- Keep sources at least 1 unit from PML boundaries
- Use smooth turn-on (width parameter) to reduce high-frequency content
- For waveguides, use line sources or eigenmode sources

### Flux Monitor Placement

- Place flux monitors at least 1 unit from PML
- For normalization, use same monitor positions in both runs
- Ensure monitors span entire mode profile

## Troubleshooting

### Simulation Diverges

Common causes:
1. Courant factor too high (reduce by 10-20%)
2. Dispersive materials at too high frequency
3. Gain materials without saturation
4. Overlapping PML with dispersive materials

### Unexpected Reflections

Check:
1. PML thickness sufficient
2. Sources not too close to boundaries
3. Resolution adequate
4. Symmetries consistent with structure and sources

### Slow Convergence

Solutions:
1. Increase resolution
2. Use subpixel averaging (eps_averaging=True)
3. Reduce PML thickness if possible
4. Check for resonant trapping

### Incorrect Flux Values

Verify:
1. Normalization runs use identical setup
2. Flux monitors properly positioned
3. Source properly characterized
4. Sufficient run time for field decay

## Resources

### References

- [materials.md](references/materials.md) - Material definitions and properties
- [sources.md](references/sources.md) - Source types and configurations
- [geometry.md](references/geometry.md) - Geometric object types
- [boundary_conditions.md](references/boundary_conditions.md) - Boundary condition options
- [flux_analysis.md](references/flux_analysis.md) - Flux spectrum computation details
- [output_visualization.md](references/output_visualization.md) - Output and visualization techniques

### Scripts

- [visualize_fields.py](scripts/visualize_fields.py) - Field visualization utilities
- [compute_spectrum.py](scripts/compute_spectrum.py) - Spectrum computation helpers
- [analyze_modes.py](scripts/analyze_modes.py) - Mode analysis tools

### External Resources

- Official documentation: https://meep.readthedocs.io/
- GitHub repository: https://github.com/NanoComp/meep
- Discussion forum: https://github.com/NanoComp/meep/discussions
- Materials library: https://github.com/NanoComp/meep/blob/master/python/materials.py