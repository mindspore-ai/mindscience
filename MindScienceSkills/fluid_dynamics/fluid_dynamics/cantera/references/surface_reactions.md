# Surface Reactions in Cantera

Complete guide to heterogeneous chemistry and catalysis.

## Surface Phases

### Creating Surface Phases

```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')

# Create surface phase
surf = ct.Interface('surface.yaml', 'Pt_surf', [gas])

# Set coverages
surf.coverages = {'H(s)': 0.5, 'O(s)': 0.3, 'Pt(s)': 0.2}
```

### Surface Properties

```python
# Site density (sites/m²)
site_density = surf.site_density

# Surface area
area = 0.01  # m²

# Coverages
coverages = surf.coverages
print(f"Coverages: {coverages}")
```

## Surface Kinetics

### Surface Reactions

```python
# Number of surface reactions
n_surf_reactions = surf.n_reactions
print(f"Surface reactions: {n_surf_reactions}")

# Reaction equations
for i in range(surf.n_reactions):
    eq = surf.reaction_equation(i)
    print(f"Reaction {i}: {eq}")
```

### Surface Reaction Rates

```python
# Forward and reverse rates
w_f = surf.forward_rates_of_progress  # kmol/m²/s
w_r = surf.reverse_rates_of_progress  # kmol/m²/s
w_net = surf.net_rates_of_progress  # kmol/m²/s

print(f"Surface rates: {w_net}")
```

### Production Rates

```python
# Species production rates (kmol/m²/s)
R = surf.net_production_rates

# Production rate for specific species
R_H_s = R[surf.species_index('H(s)')]
print(f"H(s) production: {R_H_s:.6e} kmol/m²/s")
```

## Surface Reactors

### Basic Surface Reactor

```python
import cantera as ct

# Create gas and surface phases
gas = ct.Solution('gri30.yaml', 'gas')
surf = ct.Interface('surface.yaml', 'Pt_surf', [gas])

# Set initial conditions
gas.TPX = 600, 101325, 'H2:1'
surf.coverages = {'H(s)': 0.5, 'Pt(s)': 0.5}

# Create gas and surface reactors
r_gas = ct.IdealGasConstPressureReactor(gas)
r_surf = ct.SurfaceReactor(surf, A=0.01)

# Create reactor network
sim = ct.ReactorNet([r_gas, r_surf])

# Advance
sim.advance(1e-3)

print(f"Time: {sim.time} s")
print(f"Gas T: {gas.T} K")
print(f"Coverages: {surf.coverages}")
```

### Surface Reactor with Mass Transfer

```python
# Wall with mass transfer to surface
r = ct.Wall(gas, A=0.01, left=surf)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

### Multiple Surface Sites

```python
# Multiple surface phases
surf1 = ct.Interface('surface1.yaml', 'Pt_surf', [gas])
surf2 = ct.Interface('surface2.yaml', 'Pt_surf', [gas])

# Create surface reactors
r_surf1 = ct.SurfaceReactor(surf1, A=0.01)
r_surf2 = ct.SurfaceReactor(surf2, A=0.01)

sim = ct.ReactorNet([r_gas, r_surf1, r_surf2])
sim.advance(1e-3)
```

## Coverage Analysis

### Coverage Evolution

```python
# Track coverage over time
times = []
coverages_H = []
coverages_O = []

for step in range(100):
    sim.advance(1e-3)
    times.append(sim.time)
    coverages_H.append(surf.coverages['H(s)'])
    coverages_O.append(surf.coverages['O(s)'])
```

### Steady-State Coverage

```python
# Advance to steady state
sim.advance_to_steady_state(max_steps=10000)

print(f"Steady coverages: {surf.coverages}")
```

### Coverage vs. Temperature

```python
# Vary temperature and find steady coverage
temperatures = np.linspace(300, 1000, 50)

for T in temperatures:
    gas.T = T
    surf.coverages = {'H(s)': 0.5, 'Pt(s)': 0.5}
    
    sim.advance_to_steady_state(max_steps=1000)
    print(f"T = {T:.0f} K, Coverage = {surf.coverages}")
```

## Catalysis Applications

### Heterogeneous Catalysis

```python
# Platinum catalyst
gas = ct.Solution('gri30.yaml', 'gas')
surf = ct.Interface('pt_surface.yaml', 'Pt_surf', [gas])

gas.TPX = 600, 101325, 'H2:1, O2:0.5'
surf.coverages = {'H(s)': 0.5, 'O(s)': 0.3, 'Pt(s)': 0.2}

r_gas = ct.IdealGasConstPressureReactor(gas)
r_surf = ct.SurfaceReactor(surf, A=0.01)

sim = ct.ReactorNet([r_gas, r_surf])
sim.advance(1e-3)
```

### Autoignition on Surface

```python
# Surface autoignition
gas.TPX = 300, 101325, 'H2:2, O2:1'
surf.coverages = {'H(s)': 0.5, 'O(s)': 0.3, 'Pt(s)': 0.2}

sim.advance(1e-3)

# Check for ignition
if gas.T > 1000:
    print("Ignition occurred!")
```

### Surface Oxidation

```python
# Surface oxidation
gas.TPX = 800, 101325, 'O2:1'
surf.coverages = {'O(s)': 0.5, 'Pt(s)': 0.5}

sim.advance(1e-3)
```

## Surface Thermodynamics

### Surface Energy

```python
# Surface energy (J/m²)
surface_energy = surf.coverage_energy

# Temperature dependence
for T in [300, 600, 900]:
    surf.T = T
    print(f"T = {T} K, Surface energy = {surf.coverage_energy} J/m²")
```

### Adsorption Energy

```python
# Adsorption energy for species
# See mechanism definition
# Typically specified in surface YAML file
```

## Numerical Considerations

### Time Step Selection

**Guidelines:**
- Use smaller steps for fast surface reactions
- Monitor coverage changes
- Check for stiffness
- Use appropriate solver

### Convergence

**Steady-state convergence:**
- Monitor coverage changes
- Check reaction rates
- Use appropriate max_steps
- Verify steady state

### Mass Transfer

**Mass transfer coefficients:**
- Specify in wall reactor
- Use appropriate values
- Check for mass transfer limitations
- Verify species balance

## Troubleshooting

### Coverage Issues

**Issue:** Coverages out of range

**Solutions:**
- Check initial coverages
- Verify surface reactions
- Check mass transfer
- Monitor reaction rates

### Negative Coverages

**Issue:** Negative site coverages

**Solutions:**
- Reduce time step
- Check reaction rates
- Verify mechanism
- Use appropriate solver

### Slow Convergence

**Issue:** Slow steady-state convergence

**Solutions:**
- Increase max_steps
- Improve initial guess
- Check reaction rates
- Verify mass transfer

### Surface Species Errors

**Issue:** Surface species not found

**Solutions:**
- Check surface YAML file
- Verify species definitions
- Check phase coupling
- Consult mechanism documentation

## Advanced Topics

### Multiple Surface Phases

```python
# Multiple surface phases on same surface
# Different site types
```

### Surface Transport

```python
# Surface diffusion
# Site-to-site transport
```

### Electrochemistry

```python
# Electrochemical surface reactions
# Charged species
# See electrochemistry examples
```

## Resources

- Cantera surface reaction reference: https://cantera.org/stable/reference/reactors/surface.html
- Surface reaction examples: https://cantera.org/stable/examples/python/surface/
- Catalysis examples: https://cantera.org/stable/examples/python/catalysis/
