# Reactor Models in Cantera

Complete guide to reactor models and reactor networks.

## Basic Reactors

### Ideal Gas Reactors

**Constant pressure reactor:**
```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 1500, 101325, 'H2:2, O2:1'

# Create constant pressure reactor
r = ct.IdealGasConstPressureReactor(gas)

# Create reactor network
sim = ct.ReactorNet([r])

# Advance in time
sim.advance(1e-3)

print(f"Time: {sim.time} s")
print(f"Temperature: {gas.T} K")
print(f"Pressure: {gas.P} Pa")
```

**Constant volume reactor:**
```python
# Create constant volume reactor
r = ct.IdealGasConstVolumeReactor(gas)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

**General ideal gas reactor:**
```python
# Specify energy equation type
r = ct.IdealGasReactor(gas, energy='on')  # 'on' or 'off'
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

### Flow Reactors

**Plug flow reactor:**
```python
# 1D steady flow
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Create plug flow reactor
area = 0.01  # m²
r = ct.FlowReactor(gas, area=area)

# Create reactor network
sim = ct.ReactorNet([r])

# Set initial conditions
sim.set_initial_time(0.0)

# Advance to steady state
sim.advance_to_steady_state(max_steps=1000)

print(f"Steady state reached")
print(f"Temperature: {gas.T} K")
print(f"Pressure: {gas.P} Pa")
```

**Reservoirs:**
```python
# Infinite reservoir
r1 = ct.Reservoir(gas)
r2 = ct.FlowReactor(gas, area=0.01)

# Connect reactors
sim = ct.ReactorNet([r1, r2])
sim.set_initial_time(0.0)
sim.advance(1e-3)
```

### Wall Reactors

**Adiabatic wall:**
```python
# Heat transfer wall
r = ct.Wall(gas, A=0.01)  # Area (m²)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

**Heat transfer wall:**
```python
# With heat transfer coefficient
r = ct.Wall(gas, A=0.01, U=100.0)  # W/m²/K
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

**Diabatic wall with mass transfer:**
```python
# Mass transfer to surface phase
surf = ct.Interface('surface.yaml', 'Pt_surf', [gas])
r = ct.Wall(gas, A=0.01, left=surf)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

## Reactor Networks

### Connecting Reactors

```python
import cantera as ct

# Create gas phase
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

# Create reactors
r1 = ct.Reservoir(gas)
r2 = ct.FlowReactor(gas, area=0.01)
r3 = ct.IdealGasConstPressureReactor(gas)

# Connect reactors
sim = ct.ReactorNet([r1, r2, r3])

# Set initial conditions
sim.set_initial_time(0.0)

# Advance
sim.advance(1e-3)
```

### Multiple Inlets

```python
# Mix two streams
gas1 = ct.Solution('gri30.yaml', 'gas')
gas1.TPX = 300, 101325, 'H2:1'

gas2 = ct.Solution('gri30.yaml', 'gas')
gas2.TPX = 300, 101325, 'O2:1'

# Create reservoirs
r1 = ct.Reservoir(gas1)
r2 = ct.Reservoir(gas2)

# Create mixing reactor
r3 = ct.IdealGasConstPressureReactor(gas)

# Connect
sim = ct.ReactorNet([r1, r2, r3])
sim.advance(1e-3)
```

### Flow Splitting

```python
# Split flow
r1 = ct.Reservoir(gas)
r2 = ct.FlowReactor(gas, area=0.01)
r3 = ct.FlowReactor(gas, area=0.005)  # Half area
r4 = ct.FlowReactor(gas, area=0.005)

# Connect
sim = ct.ReactorNet([r1, r2, r3, r4])
sim.advance(1e-3)
```

## Surface Reactors

### Basic Surface Reactor

```python
# Create gas and surface phases
gas = ct.Solution('gri30.yaml', 'gas')
surf = ct.Interface('surface.yaml', 'Pt_surf', [gas])

# Set initial conditions
gas.TPX = 300, 101325, 'H2:2, O2:1'
surf.coverages = {'H(s)': 0.5, 'O(s)': 0.3, 'Pt(s)': 0.2}

# Create reactors
r_gas = ct.IdealGasConstPressureReactor(gas)
r_surf = ct.SurfaceReactor(surf, A=0.01)

# Connect
sim = ct.ReactorNet([r_gas, r_surf])
sim.advance(1e-3)
```

### Surface Kinetics

```python
# Surface reactions
# See references/surface_reactions.md for complete details
```

## Advanced Reactors

### Pressure Controlled Reactor

```python
# Reactor with specified pressure
r = ct.IdealGasConstPressureReactor(gas, P=101325)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

### Volume Controlled Reactor

```python
# Reactor with specified volume
r = ct.IdealGasConstVolumeReactor(gas, V=0.001)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

### Reactor with Mass Flow

```python
# Specify mass flow rate
r = ct.FlowReactor(gas, area=0.01, mass_flow_rate=0.01)
sim = ct.ReactorNet([r])
sim.advance(1e-3)
```

## Reactor Controls

### Time Stepping

```python
# Advance by time step
sim.advance(dt=1e-4)

# Advance to specific time
sim.advance(time=1e-3)

# Advance to steady state
sim.advance_to_steady_state(max_steps=1000)
```

### Output Control

```python
# Set output interval
sim.set_initial_time(0.0)
sim.advance(1e-3)

# Get reactor data
print(f"Time: {sim.time} s")
print(f"Temperature: {gas.T} K")
print(f"Pressure: {gas.P} Pa")
```

### Sensitivity Analysis

```python
# Enable sensitivity
sim.algebraic_sensitivity_on()

# Advance
sim.advance(1e-3)

# Get sensitivity coefficients
sens = sim.sensitivity()
```

## Common Applications

### Batch Reactor

```python
# Constant volume batch reactor
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

r = ct.IdealGasConstVolumeReactor(gas)
sim = ct.ReactorNet([r])

# Simulate reaction
sim.advance(1e-3)
```

### Continuous Stirred Tank Reactor

```python
# CSTR approximation
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

r = ct.IdealGasConstPressureReactor(gas)
sim = ct.ReactorNet([r])

# Residence time
residence_time = 1.0  # s
sim.advance(residence_time)
```

### Plug Flow Reactor

```python
# 1D steady flow
gas = ct.Solution('gri30.yaml', 'gas')
gas.TPX = 300, 101325, 'H2:2, O2:1'

r = ct.FlowReactor(gas, area=0.01)
sim = ct.ReactorNet([r])

sim.set_initial_time(0.0)
sim.advance_to_steady_state(max_steps=1000)
```

### Heat Exchanger

```python
# Heat transfer wall
r1 = ct.IdealGasConstPressureReactor(gas)
r2 = ct.Wall(gas, A=0.01, U=100.0)
r3 = ct.IdealGasConstPressureReactor(gas)

sim = ct.ReactorNet([r1, r2, r3])
sim.advance(1e-3)
```

## Numerical Considerations

### Time Step Selection

**Guidelines:**
- Use smaller steps for fast reactions
- Larger steps for slow chemistry
- Monitor temperature/pressure changes
- Check for stiffness

### Solver Selection

**Stiff systems:**
- Use CVODE solver for stiff problems
- Enable analytical Jacobian
- Consider QSSA for fast species

**Non-stiff systems:**
- Use standard ODE solver
- Larger time steps possible

### Conservation

**Check conservation:**
- Mass conservation
- Element conservation
- Energy conservation (if energy equation on)

## Troubleshooting

### Numerical Instability

**Issue:** Solution diverges

**Solutions:**
- Reduce time step
- Use stiff solver
- Check initial conditions
- Verify mechanism

### Poor Convergence

**Issue:** Slow convergence to steady state

**Solutions:**
- Increase max_steps
- Check flow rates
- Verify boundary conditions
- Improve initial guess

### Conservation Errors

**Issue:** Mass/energy not conserved

**Solutions:**
- Check element conservation
- Verify solver tolerances
- Check reactor type
- Monitor conservation

### Memory Issues

**Issue:** Out of memory for large mechanisms

**Solutions:**
- Use compiled mechanisms (CTI)
- Reduce mechanism size
- Use QSSA for fast species
- Consider mechanism reduction

## Advanced Topics

### Custom Reactors

```python
# Define custom reactor behavior
# See Cantera documentation for details
```

### Reactor Arrays

```python
# Multiple reactors in array
# Reactor networks with complex topologies
```

### Parallel Execution

```python
# Parallel reactor networks
# See Cantera documentation for details
```

## Resources

- Cantera reactor reference: https://cantera.org/stable/reference/reactors/index.html
- Reactor examples: https://cantera.org/stable/examples/python/reactors/
- 1D flame models: https://cantera.org/stable/examples/python/onedim/
