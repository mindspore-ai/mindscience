# Quality Factor Analysis in scikit-rf

Quality factor calculations and resonator analysis.

## Q-Factor Calculation

### Basic Q-Factor

```python
import skrf as rf
from skrf.qfactor import Qfactor

# Load network representing a resonator
network = rf.Network('data/resonator.s2p')

# Calculate Q-factors
qf = Qfactor(network)

# Get Q-factors
print(f"Unloaded Q: {qf.q_unloaded}")
print(f"Loaded Q: {qf.q_loaded}")
print(f"External Q: {qf.q_external}")
```

### Q-Factor Components

```python
# Get individual Q-factor components
print(f"Q-factor from S-parameters: {qf.q_s}")
print(f"Q-factor from Z-parameters: {qf.q_z}")
print(f"Q-factor from Y-parameters: {qf.q_y}")
```

### Q-Factor from Specific Frequency

```python
# Calculate Q-factor at specific frequency
freq = 2.4e9  # 2.4 GHz
q_at_freq = qf.q_at_frequency(freq)
print(f"Q-factor at {freq/1e9:.2f} GHz: {q_at_freq}")
```

## Resonator Analysis

### Resonator Parameters

```python
# Get resonator parameters
resonance = qf.resonance

print(f"Resonance frequency: {resonance.f0}")
print(f"Bandwidth: {resonance.bw}")
print(f"Coupling: {resonance.k}")
print(f"Group velocity: {resonance.vg}")
```

### Resonator Quality Metrics

```python
# Calculate quality metrics
print(f"Unloaded Q: {resonance.q_unloaded}")
print(f"Loaded Q: {resonance.q_loaded}")
print(f"External Q: {resonance.q_external}")
print(f"Dissipation factor: {resonance.df}")
```

### Resonator Plotting

```python
import matplotlib.pyplot as plt

# Plot Q-factors vs frequency
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
qf.plot_q_unloaded()
plt.title('Unloaded Q-Factor')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
qf.plot_q_loaded()
plt.title('Loaded Q-Factor')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Coupling Analysis

### Coupling Coefficient

```python
# Calculate coupling coefficient
coupling = qf.coupling_coefficient
print(f"Coupling coefficient: {coupling}")
```

### Critical Coupling

```python
# Calculate critical coupling
critical_coupling = qf.critical_coupling
print(f"Critical coupling: {critical_coupling}")
```

### CouplingQ Analysis

```python
# Analyze coupling vs Q-factor
plt.figure(figsize=(10, 6))

plt.plot(qf.q_loaded, coupling, 'bo-')
plt.xlabel('Loaded Q-Factor')
plt.ylabel('Coupling Coefficient')
plt.title('Coupling vs Q-Factor')
plt.grid(True, alpha=0.3)
plt.show()
```

## Advanced Q-Factor Analysis

### Frequency-Dependent Q-Factor

```python
# Calculate Q-factor vs frequency
freqs = network.frequency.f_scaled
q_loaded = qf.q_loaded

plt.figure(figsize=(10, 6))
plt.plot(freqs, q_loaded, 'b-', linewidth=2)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Q-Factor')
plt.title('Frequency-Dependent Q-Factor')
plt.grid(True, alpha=0.3)
plt.show()
```

### Q-Factor Comparison

```python
# Compare Q-factors of multiple networks
networks = rf.io.read_all('data/', contains='resonator')

q_factors = []
for name, network in networks.items():
    qf = Qfactor(network)
    q_factors.append(qf.q_loaded)
    plt.plot(network.frequency.f_scaled, qf.q_loaded, 
             label=name)

plt.xlabel('Frequency (GHz)')
plt.ylabel('Q-Factor')
plt.title('Q-Factor Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Q-Factor Statistics

```python
# Calculate Q-factor statistics
import numpy as np

q_values = qf.q_loaded
print(f"Mean Q: {np.mean(q_values):.2f}")
print(f"Std Q: {np.std(q_values):.2f}")
print(f"Min Q: {np.min(q_values):.2f}")
print(f"Max Q: {np.max(q_values):.2f}")
```

## Best Practices

### Q-Factor Calculation

1. **Use appropriate network**: Network must represent a resonator
2. **Check frequency range**: Ensure resonance is within frequency range
3. **Verify network quality**: High-quality measurements give accurate Q-factors

### Resonator Analysis

1. **Identify resonance mode**: Know which mode you're analyzing
2. **Consider loading effects**: Account for external loading
3. **Analyze coupling**: Understand coupling to external circuits

### Quality Metrics

1. **Use multiple Q-factors**: Compare unloaded, loaded, and external Q
2. **Check consistency**: Q-factors should be consistent across calculations
3. **Verify physical limits**: Q-factors should be physically reasonable

## Troubleshooting

### Invalid Q-Factor

**Problem**: Q-factor is infinite or NaN

**Solutions**:
1. Check network represents a resonator

2. Verify frequency range includes resonance
3. Check for numerical issues in network parameters

### Poor Q-Factor Accuracy

**Problem**: Q-factor values seem inaccurate

**Solutions**:
1. Improve measurement quality
2. Increase frequency resolution near resonance
3. Use appropriate calibration
4. Check for external loading effects

### Resonance Not Found

**Problem**: Resonance parameters are not calculated

**Solutions**:
1. Verify network has a clear resonance
2. Check frequency range covers resonance
3. Ensure network has sufficient frequency resolution
4. Check for multiple resonances

### Coupling Issues

**Problem**: Coupling analysis gives unexpected results

**Solutions**:
1. Verify network port configuration
2. Check for proper termination
3. Consider external loading effects
4. Verify network represents coupled resonator