# Parameter Sweep and Optimization

## Parameterization

CST allows full parameterization of models:

### Defining Parameters

```
# VBA syntax
StoreParameter "length", 50e-3
StoreParameter "width", 10e-3
StoreParameter "freq", 2.4e9
```

### Using Parameters

```
# Geometry
Brick "substrate", "length", "width", "thickness"

# Material
Material "substrate", "Epsilon", "eps_r"

# Boundary
Boundary "port", "distance", "lambda/4"
```

## Parameter Sweep

### Basic Sweep

```
Sweep type: Parameter sweep
Parameter: "length"
Values: 40mm, 45mm, 50mm, 55mm, 60mm
```

### Multi-Parameter Sweep

```
Parameters: "length", "width"
Values:
  length: [40, 50, 60] mm
  width: [8, 10, 12] mm
Total runs: 3 × 3 = 9
```

### Sweep Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| All combinations | Full factorial | Small parameter space |
| Sequential | One at a time | Quick exploration |
| Random | Random sampling | Large parameter space |

## Optimization

### Optimization Types

| Type | Description |
|------|-------------|
| Local | Gradient-based, fast but may find local minimum |
| Global | Genetic algorithm, particle swarm, slower but thorough |
| DOE | Design of experiments, systematic exploration |

### Local Optimizers

```
Optimizer: Trust Region
Maximum iterations: 100
Convergence: 1e-6
```

### Global Optimizers

```
Optimizer: Genetic Algorithm
Population: 20
Generations: 50
Mutation rate: 0.1
```

```
Optimizer: Particle Swarm
Particles: 30
Iterations: 100
Inertia: 0.7
```

### Objective Functions

| Goal | Objective |
|------|-----------|
| Minimize | Return loss (S11) |
| Maximize | Gain |
| Target | Center frequency |
| Multi-objective | Weighted combination |

### Example: Antenna Optimization

```
# Goal: Minimize S11 at 2.4 GHz
Objective: Min(S11 @ 2.4GHz)
Parameters:
  - patch_length: 30-40 mm
  - feed_position: 5-15 mm
Constraints:
  - S11 < -10 dB at 2.4 GHz
  - Gain > 5 dBi
```

## Sensitivity Analysis

### Purpose
- Identify critical parameters
- Understand parameter impact
- Guide optimization

### Method

```
Analysis: Sensitivity
Parameter: "length"
Variation: ±5%
Output: S11, gain, bandwidth
```

## DOE (Design of Experiments)

### Types

| Type | Description |
|------|-------------|
| Full factorial | All combinations |
| Latin hypercube | Space-filling |
| Taguchi | Orthogonal arrays |

### Example

```
DOE type: Latin Hypercube
Parameters: 5
Samples: 50
```

## Post-Processing Sweep Results

### S-Parameter Extraction

```
# Extract S11 for all sweep points
Result: S11
Format: Magnitude (dB)
Export: CSV
```

### Field Extraction

```
# Extract E-field at specific frequency
Result: E-field
Frequency: 2.4 GHz
Export: 3D data
```

## Best Practices

1. **Start with sweep** before optimization
2. **Use sensitivity** to identify key parameters
3. **Set reasonable bounds** for parameters
4. **Monitor convergence** during optimization
5. **Validate results** with fine sweep

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Slow convergence | Poor initial guess | Use sweep first |
| Local minimum | Local optimizer | Use global optimizer |
| Too many runs | Large parameter space | Use DOE |
| Infeasible design | Constraints too strict | Relax constraints |

## Example: Filter Optimization

```
# Bandpass filter optimization
Parameters:
  - resonator_length: 10-20 mm
  - coupling_gap: 0.5-2 mm
  - feed_position: 1-5 mm

Objectives:
  - Center frequency: 5 GHz
  - Bandwidth: 500 MHz
  - Return loss: < -15 dB

Optimizer: Genetic Algorithm
Population: 30
Generations: 100
```
