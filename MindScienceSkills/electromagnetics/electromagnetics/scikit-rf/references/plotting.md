# Plotting and Visualization in scikit-rf

Plotting functions for network parameters and analysis.

## Network Plotting

### S-Parameter Plots

Plot scattering parameters:

```python
import skrf as rf
import matplotlib.pyplot as plt

# Load network
network = rf.Network('data/device.s2p')

# Apply skrf plotting style
rf.stylely()

# Plot magnitude (dB)
network.plot_s_db(m=1, n=0, label='S11')
network.plot_s_db(m=0, n=1, label='S12')
plt.legend()
plt.show()

# Plot phase (degrees)
network.plot_s_deg(m=1, n=0, label='S11 phase')
plt.legend()
plt.show()

# Plot magnitude (linear)
network.plot_s_mag(m=1, n=0, label='S11')
plt.legend()
plt.show()
```

### Z-Parameter Plots

Plot impedance parameters:

```python
# Plot impedance magnitude
network.plot_z_re(m=1, n=0, label='Z11')
plt.legend()
plt.show()

# Plot impedance phase
network.plot_z_im(m=1, n=0, label='Z11')
plt.legend()
plt.show()

# Plot impedance angle
network.plot_z_ang(m=1, n=0, label='Z11')
plt.legend()
plt.show()
```

### Y-Parameter Plots

Plot admittance parameters:

```python
# Plot admittance magnitude
network.plot_y_db(m=1, n=0, label='Y11')
plt.legend()
plt.show()

# Plot admittance phase
network.plot_y_deg(m=1, n=0, label='Y11 phase')
plt.legend()
plt.show()
```

## Smith Chart Plots

### Basic Smith Chart

Plot S-parameters on Smith chart:

```python
# Plot S11 on Smith chart
network.s11.plot_s_smith()
plt.title('S11 Smith Chart')
plt.show()

# Plot S21 on Smith chart
network.s21.plot_s_smith()
plt.title('S21 Smith Chart')
plt.show()

# Plot S12 on Smith chart
network.s12.plot_s_smith()
plt.title('S12 Smith Chart')
plt.show()
```

### Multiple Smith Charts

Plot multiple parameters on Smith chart:

```python
# Plot multiple S-parameters
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

network.s11.plot_s_smith(ax=axes[0, 0])
axes[0, 0].set_title('S11')

network.s21.plot_s_smith(ax=axes[0, 1])
axes[0, 1].set_title('S21')

network.s12.plot_s_smith(ax=axes[1, 0])
axes[1, 0].set_title('S12')

network.s22.plot_s_smith(ax=axes[1, 1])
axes[1, 1].set_title('S22')

plt.tight_layout()
plt.show()
```

## NetworkSet Plots

### Uncertainty Bounds

Plot uncertainty bounds for network sets:

```python
from skrf.networkSet import NetworkSet

# Load multiple networks
networks = rf.io.read_all('data/', contains='measurement')
ns = NetworkSet(networks, name='measurement set')

# Plot uncertainty bounds
ns.plot_uncertainty_bounds_s_db(m=1, n=0, label='S11')
plt.legend()
plt.title('Uncertainty Bounds for S11')
plt.show()
```

### Statistical Plots

Plot statistical properties:

```python
# Plot mean response
ns.mean_s.plot_s_db(m=1, n=0, label='Mean S11')
plt.legend()
plt.title('Mean S11')
plt.show()

# Plot standard deviation
ns.std_s.plot_s_db(m=1, n=0, label='Std S11')
plt.legend()
plt.title('Standard Deviation of S11')
plt.show()
```

## Custom Plotting

### Custom S-Parameter Plot

Create custom S-parameter plots:

```python
import matplotlib.pyplot as plt

# Apply skrf style
rf.stylely()

# Create custom plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot S-parameters
network.s11.plot_s_db(ax=ax, m=1, n=0, label='|S11|')
network.s21.plot_s_db(ax=ax, m=1, n=0, label='|S21|')
network.s12.plot_s_db(ax=ax, m=1, n=0, label='|S12|')
network.s22.plot_s_db(ax=ax, m=1, n=0, label='|S22|')

ax.set_xlabel('Frequency (GHz)')
ax.set_ylabel('Magnitude (dB)')
ax.set_title('S-parameters')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Multi-Panel Plot

Create multi-panel plots:

```python
# Create 2x2 panel plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# S11 magnitude
network.s11.plot_s_db(ax=axes[0, 0], m=1, n=0)
axes[0, 0].set_title('|S11| (dB)')
axes[0, 0].grid(True, alpha=0.3)

# S21 magnitude
network.s21.plot_s_db(ax=axes[0, 1], m=1, n=0)
axes[0, 1].set_title('|S21| (dB)')
axes[0, 1].grid(True, alpha=0.3)

# S11 phase
network.s11.plot_s_deg(ax=axes[1, 0], m=1, n=0)
axes[1, 0].set_title('∠S11| (degrees)')
axes[1, 0].grid(True, alpha=0.3)

# S21 phase
network.s21.plot_s_deg(ax=axes[1, 1], m=1, n=0)
axes[1, 1].set('∠S21| (degrees)')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Frequency Range Plot

Plot specific frequency range:

```python
# Plot frequency subset
network['1-5ghz'].s11.plot_s_db(label='1-5 GHz')
network['5-10ghz'].s11.plot_s_db(label='5-10 GHz')
plt.legend()
plt.title('S11 by Frequency Range')
plt.show()
```

## Quality Factor Plots

### Q-Factor Plot

Plot quality factors:

```python
from skrf.qfactor import Qfactor

# Load network
network = rf.Network('data/resonator.s2p')

# Calculate Q-factors
qf = Qfactor(network)

# Plot Q-factors
qf.plot_q_unloaded()
plt.title('Unloaded Q-Factor')
plt.show()

qf.plot_q_loaded()
plt.title('Loaded Q-Factor')
plt.show()

qf.plot_q_external()
plt.title('External Q-Factor')
plt.show()
```

### Resonator Plot

Plot resonator parameters:

```python
# Calculate resonator parameters
resonance = qf.resonance

# Plot resonator frequency
plt.figure(figsize=(10, 6))
plt.plot(network.frequency.f_scaled, resonance.f0)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Resonance Frequency (GHz)')
plt.title('Resonance Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# Plot bandwidth
plt.figure(figsize=(10, 6))
plt.plot(network.frequency.f_scaled, resonance.bw)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Bandwidth (GHz)')
plt.title('Resonator Bandwidth')
plt.grid(True, alpha=0.3)
plt.show()
```

## Media Plots

### Propagation Constant Plot

Plot propagation constant:

```python
from skrf.media import CPW

# Create CPW media
freq = rf.Frequency(75, 110, 101, 'GHz')
cpw = CPW(freq, w=10e-6, s=5e-6, ep_r=10.6)

# Plot propagation constant
cpw.plot_gamma()
plt.title('CPW Propagation Constant')
plt.show()
```

### Characteristic Impedance Plot

Plot characteristic impedance:

```python
# Plot characteristic impedance
cpw.plot_z_re()
plt.title('CPW Characteristic Impedance')
plt.show()
```

## Best Practices

### Plotting Style

1. **Apply skrf style**: Use `rf.stylely()` for consistent appearance
2. **Use descriptive labels**: Clear labels for legends
3. **Include grid lines**: `plt.grid(True, alpha=0.3)`
4. **Set axis labels**: Clear axis labels and titles

### Plot Organization

1. **Use subplots**: Organize related plots in subplots
2. **Consistent sizing**: Use consistent figure sizes
3. **Color schemes**: Use consistent color schemes
4. **Legend placement**: Place legends appropriately

### Frequency Slicing

1. **Plot frequency ranges**: Use human-readable frequency strings
2. **Highlight regions**: Use different colors for regions of interest
3. **Compare responses**: Overlay multiple responses

## Troubleshooting

### Plot Fails

**Problem**: Plotting functions fail

**Solutions**:
1. Check network has valid data
2. Verify frequency range is valid
3. Check matplotlib is installed
4. Verify network has required parameters

### Smith Chart Issues

**Problem**: Smith chart plots don't display correctly

**Solutions**:
1. Check S-parameters are valid
2. Verify network is 2-port
3. Check frequency range is appropriate
4. Try different plot parameters

### Uncertainty Bounds Issues

**Problem**: Uncertainty bounds don't display

**Solutions**:
1. Verify NetworkSet has multiple networks
2. Check frequency ranges overlap
3. Verify networks are compatible
4. Check network count is sufficient

### Quality Factor Issues

**Problem**: Q-factor calculations fail

**Solutions**:
1. Verify network represents a resonator
2. Check frequency range includes resonance
3. Verify network has valid S-parameters
4. Check for numerical issues