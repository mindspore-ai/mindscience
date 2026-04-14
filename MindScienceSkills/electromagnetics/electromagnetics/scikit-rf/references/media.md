# Media and Transmission Lines in scikit-rf

Media objects represent transmission lines and waveguide structures.

## Media Overview

Media objects provide electromagnetic properties needed for network analysis:

- **Propagation constant** (γ)
- **Characteristic impedance** (Z₀)
- **Electrical length** (θ)
- **Effective dielectric constant** (εᵣ)

## CPW Media

### Basic CPW

```python
from skrf.media import CPW
import skrf as rf

# Create frequency range
freq = rf.Frequency(75, 110, 101, 'GHz')

# Create CPW media
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)

# Access properties
print(f"Propagation constant: {cpw.gamma}")
print(f"Characteristic impedance: {cpw.z0}")
print(f"Effective dielectric: {cpw.ep_reff}")
```

### CPW Parameters

```python
# CPW with all parameters
cpw = CPW(
    frequency=freq,     # Frequency object
    w=10e-6,            # Width (meters)
    s=5e-6,             # Substrate height (meters)
    ep_r=10.6,           # Relative permittivity
    mu_r=1.0,            # Relative permeability
    rho=1.0,            # Resistivity (Ohm-m)
    tand=0.0,           # Dielectric loss tangent
    sigma=0.0            # Conductivity (S/m)
)
```

### CPW Line

```python
# Create CPW transmission line
line = cpw.line(d=90, unit='deg', name='90deg_line')

# Plot characteristic impedance
line.plot_z_re(m=1, n=0)
plt.show()
```

## Coaxial Media

### Basic Coaxial

```python
from skrf.media import Coaxial

# Create frequency range
freq = rf.Frequency(1, 10, 101, 'GHz')

# Create coaxial media
coax = Coaxial(frequency=freq, Dint=1e-3, Dout=2e-3)

# Access properties
print(f"Propagation constant: {coax.gamma}")
print(f"Characteristic impedance: {coax.z0}")
```

### Coaxial Parameters

```python
# Coaxial with parameters
coax = Coaxial(
    frequency=freq,     # Frequency object
    Dint=1e-3,          # Inner diameter (meters)
    Dout=2e-3,          # Outer diameter (meters)
    ep_r=2.08,           # Dielectric constant
    mu_r=1.0,            # Relative permeability
    tand=0.0,           # Dielectric loss tangent
    sigma=0.0            # Conductivity
)
```

### Coaxial Line

```python
# Create coaxial transmission line
line = coax.line(d=90, unit='deg', name='coax_line')

# Plot propagation constant
line.plot_gamma()
plt.show()
```

## Rectangular Waveguide

### Basic Rectangular Waveguide

```python
from skrf.media import RectangularWaveguide

# Create frequency range
freq = rf.Frequency(10, 20, 101, 'GHz')

# Create rectangular waveguide
rwg = RectangularWaveguide(freq, a=2.54e-3, b=1.27e-3)

# Access properties
print(f"Propagation constant: {rwg.gamma}")
print(f"Characteristic impedance: {rwg.z0}")
```

### Rectangular Waveguide Parameters

```python
# Rectangular waveguide with parameters
rwg = RectangularWaveguide(
    frequency=freq,     # Frequency object
    a=2.54e-3,          # Width (meters)
    b=1.27e-3,          # Height (meters)
    ep_r=1.0,            # Relative permittivity
    mu_r=1.0,            # Relative permeability
    rho=0.0,            # Resistivity
    tand=0.0,           # Dielectric loss tangent
    sigma=0.0            # Conductivity
)
```

### Rectangular Waveguide Line

```python
# Create waveguide line
line = rwg.line(d=100, unit='mm', name='rwg_line')

# Plot characteristic impedance
line.plot_z_re(m=1, n=0)
plt.show()
```

## Circular Waveguide

### Basic Circular Waveguide

```python
from skrf.media import CircularWaveguide

# Create frequency range
freq = rf.Frequency(10, 20, 101, 'GHz')

# Create circular waveguide
cwg = CircularWaveguide(freq, r=1e-3)

# Access properties
print(f"Propagation constant: {cwg.gamma}")
print(f"Characteristic impedance: {cwg.z0}")
```

### Circular Waveguide Parameters

```python
# Circular waveguide with parameters
cwg = CircularWaveguide(
    frequency=freq,     # Frequency object
    r=1e-3,             # Radius (meters)
    ep_r=1.0,            # Relative permittivity
    mu_r=1.0,            # Relative permeability
    rho=0.0,            # Resistivity
    tand=0.0,           # Dielectric loss tangent
    sigma=0.0            # Conductivity
)
```

### Circular Waveguide Line

```python
# Create waveguide line
line = cwg.line(d=100, unit='mm', name='cwg_line')

# Plot propagation constant
line.plot.plot_gamma()
plt.show()
```

## Free Space

### Basic Free Space

```python
from skrf.media import FreeSpace

# Create frequency range
freq = rf.Frequency(1, 10, 101, 'GHz')

# Create free space
fs = FreeSpace(freq)

# Access properties
print(f"Propagation constant: {fs.gamma}")
print(f"Characteristic impedance: {fs.z0}")
```

### Free Space Parameters

```python
# Free space with parameters
fs = FreeSpace(
    frequency=freq,     # Frequency object
    ep_r=1.0,            # Relative permittivity
    mu_r=1.0,            # Relative permeability
    rho=0.0,            # Resistivity
    sigma=0.0            # Conductivity
)
```

## Dielectric

### Basic Dielectric

```python
from skrf.media import Dielectric

# Create frequency range
freq = rf.Frequency(1, 10, 101, 'GHz')

# Create dielectric medium
dielectric = Dielectric(freq, ep_r=4.0)

# Access properties
print(f"Propagation constant: {dielectric.gamma}")
print(f"Characteristic impedance: {dielectric.z0}")
```

### Dielectric Parameters

```python
# Dielectric with parameters
dielectric = Dielectric(
    frequency=freq,     # Frequency object
    ep_r=4.0,            # Relative permittivity
    mu_r=1.0,            # Relative permeability
    rho=0.0,            # Resistivity
    tand=0.0,           # Dielectric loss tangent
    sigma=0.0            # Conductivity
)
```

## Media Operations

### Media Arithmetic

```python

# Create media objects
cpw1 = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)
cpw2 = CPW(freq, w=10e-6, s=5e-6, ep_r=4.0)

# Add media (series connection)
combined = cpw1 + cpw2

# Plot combined propagation constant
combined.plot_gamma()
plt.show()
```

### Media Comparison

```python
# Compare media objects
if cpw1 == cpw2:
    print("Media are identical")
else:
    print("Media are different")
```

## Transmission Line Design

### Cascaded Lines

```python
# Create multiple line segments
line1 = cpw.line(d=10, unit='mm', name='segment1')
line2 = cpw.line(d=20, unit='mm', name='segment2')
line3 = cpw.line(d=30, unit='mm', name='segment3')

# Cascade lines
total_line = line1 ** line2 ** line3

# Plot total line response
total_line.s21.plot_s_db()
plt.show()
```

### Tapered Lines

```python
from skrf.taper import line_taper

# Create tapered line
tapered_line = line_taper(
    media_start=cpw_start,    # Starting media
    media_end=cpw_end,        # Ending media
    d=100,                   # Length (mm)
    unit='mm',               # Length unit
    num_sections=10          # Number of sections
)

# Plot tapered line response
tapered_line.s21.plot_s_db()
plt.show()
```

## Best Practices

### Media Selection

1. **Match application**: Use appropriate media for your application
2. **Consider losses**: Include dielectric and conductor losses
3. **Verify frequency range**: Ensure media is valid over frequency range

### Line Design

1. **Use realistic dimensions**: Physical dimensions must be feasible
2. **Consider dispersion**: Media properties vary with frequency
3. **Account for discontinuities**: Waveguide cutoffs and resonances

### Parameter Selection

1. **Use standard materials**: Common materials (FR4, Rogers, etc.)
2. **Include losses**: tand and sigma for realistic models
3. **Verify parameters**: Check for physically impossible values

## Troubleshooting

### Invalid Media Parameters

**Problem**: Media creation fails with invalid parameters

**Solutions**:
1. Check frequency range is valid
2. Verify physical dimensions are positive
3. Ensure material parameters are realistic

### Propagation Constant Issues

**Problem**: Unexpected propagation constant values

**Solutions**:
1. Check for waveguide cutoff frequencies
2. Verify material parameters are correct
3. Consider numerical precision issues

### Characteristic Impedance Issues

**Problem**: Unexpected impedance values

**Solutions**:
1. Check for waveguide mode cutoffs
2. Verify material parameters are correct
3. Consider conductor losses

### Line Response Issues

**Problem**: Line response doesn't match expectations

**Solutions**:
1. Verify media properties are correct
2. Check line length and units
3. Consider frequency dispersion effects