# Materials in gprMax

gprMax supports various material types for electromagnetic simulation.

## Built-in Materials

### Free Space

```python
# Free space (air)
#material: 6 0 1 0 free_space
```

### Perfect Electric Conductor

```python
# Perfect electric conductor
#material: 6 0 1 0 pec
```

## Dielectric Materials

### Simple Dielectric

```python
# Simple dielectric material
#material: 6 0 1 0 half_space epsilon_r=6.0 sigma=0.0
```

Parameters:
- `epsilon_r`: Relative permittivity
- `sigma`: Conductivity (S/m)

### Conductive Dielectric

```python
# Conductive dielectric material
#material: 6 0 1 0 conductive_dielectric epsilon_r=4.0 sigma=0.01
```

## Magnetic Materials

### Simple Magnetic

```python
# Simple magnetic material
#material: 6 0 1 0 magnetic_material epsilon_r=1.0 mu_r=2.0 sigma=0.0
```

Parameters:
- `epsilon_r`: Relative permittivity
- `mu_r`: Relative permeability
- `sigma`: Conductivity (S/m)

### Conductive Magnetic

```python
# Conductive magnetic material
#material: 6 0 1 0 conductive_magnetic epsilon_r=1.0 mu_r=2.0 sigma=0.01
```

## Dispersive Materials

### Drude Material

```python
# Drude material (metal-like)
#material: 6 0 1 0 drude epsilon_inf=1.0 plasma=1.0 gamma=0.05 sigma=0.1
```

Parameters:
- `epsilon_inf`: Infinite-frequency permittivity
- `plasma`: Plasma frequency (Hz)
- `gamma`: Damping rate (Hz)
- `sigma`: Conductivity (S/m)

### Lorentzian Material

```python
# Lorentzian dispersive material
#material: 6 0 1 0 lorentzian epsilon_inf=4.0 epsilon_r=1.0 plasma=1e12 gamma=1e11 sigma=0.1
```

Parameters:
- `epsilon_inf`: Infinite-frequency permittivity
- `epsilon_r`: Relative permittivity at resonance
- `plasma`: Resonance frequency (Hz)
- `gamma`: Damping rate (Hz)
- `sigma`: Conductivity (S/m)

### Multiple Lorentzian Terms

```python
# Multiple Lorentzian resonances
#material: 6 0 1 0 multi_lorentzian epsilon_inf=4.0 epsilon_r=[1.0 2.0] plasma=[1e12 2e12] gamma=[1e11 1e11] sigma=[0.1 0.1]
```

## Nonlinear Materials

### Kerr Nonlinearity

```python
# Kerr nonlinearity (third-order)
#material: 6 0 1 0 kerr chi3=0.1
```

Parameters:
- `chi3`: Third-order nonlinear susceptibility

### Pockels Nonlinearity

```python
# Pockels nonlinearity (second-order)
#material: 6 0 1 0 pockels chi2=[0.1 0.1 0.1]
```

Parameters:
- `chi2`: Second-order nonlinear susceptibility tensor

## Material Properties

### Frequency-Dependent Properties

```python
# Material properties vary with frequency
# Check material response at different frequencies
```

### Loss and Gain

```python
# Lossy materials (positive sigma)
# Gain materials (negative sigma)
#material: 6 0 1 0 lossy_material epsilon_r=4.0 sigma=0.01
#material: 6 0 1 0 gain_material epsilon_r=4.0 sigma=-0.001
```

### Anisotropic Materials

```python
# Anisotropic materials (tensor permittivity)
#material: 6 0 1 0 anisotropic epsilon_r=[[12 0 0] [0 12 0] [0 0 12]]
```

## Best Practices

### Material Selection

1. **Use appropriate models**: Choose correct material model for application
2. **Frequency range**: Ensure material valid over simulation frequency range
3. **Stability check**: Verify material doesn't cause numerical instability

### Resolution Requirements

1. **High-index materials**: Higher resolution needed
2. **Dispersive materials**: Higher resolution near resonances
3. **Source regions**: Higher resolution near sources

### Stability Considerations

1. **Courant number**: Reduce Courant number for dispersive materials
2. **Time step**: Ensure time step resolves material dynamics
3. **PML interaction**: Be careful with dispersive materials near PML

## Troubleshooting

### Material Instability

**Problem**: Simulation diverges with material

**Solutions**:
1. Reduce Courant number
2. Increase resolution
3. Check material parameters
4. Use subpixel averaging

### Unexpected Loss

**Problem**: Excessive or unexpected loss

**Solutions**:
1. Check conductivity values
2. Verify frequency range
3. Check material model
4. Increase resolution

### Poor Convergence

**Problem**: Results don't converge with resolution

**Solutions**:
1. Increase resolution
2. Use subpixel averaging
3. Check Courant number
4. Verify material parameters