# Sources in gprMax

gprMax supports various source types for electromagnetic excitation.

## Source Types

### Hertzian Dipole

Point source with Ricker waveform:

```python
# Hertzian dipole source
#source: 0 0.0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#waveform: ricker 1.5e9 my_ricker
```

Parameters:
- `f`: Center frequency (Hz)
- `polarization`: Field polarization (x, y, z)
- `amplitude`: Source amplitude
- `phase`: Source phase (degrees)

### Gaussian Pulse

Time-domain Gaussian pulse source:

```python
# Gaussian pulse source
#source: 0 0.0 0.100 0.170 0 gaussian 1.5e9 0.2e9
#waveform: gaussian 1.5e9 0.2e9
```

Parameters:
- `f_center`: Center frequency (Hz)
- `f_width`: Frequency width (Hz)
- `amplitude`: Source amplitude
- `phase`: Source phase (degrees)

### Plane Wave

Extended plane wave source:

```python
# Plane wave source
#source: 0 0.0 0.100 0.170 0 plane_wave
#waveform: plane_wave
```

### Custom Source

User-defined source function:

```python
# Custom source (in Python code)
import gprMax

def my_source(t, position):
    # User-defined source function
    return source_value

# Add custom source
gprMax.source(my_source, position=[0, 0, 0])
```

## Source Configuration

### Source Parameters

```python
# Source position and polarization
#source: 0 0.0 0.100 0.170 0 my_ricker
#polarization: z
#amplitude: 1.0
#phase: 0.0
```

### Source Frequency

```python
# Source frequency parameters
#source: 0 0.0 0.100 0.170 0 gaussian 1.5e9 0.2e9
#f_center: 1.5e9
#f_width: 0.2e9
```

### Multiple Sources

```python
# Multiple sources
#source: 0 0.0 0.100 0.170 0 my_ricker
#source: 0.0.0 0.200 0.170 0 my_ricker_2
```

## Source Placement

### Positioning Guidelines

1. **Distance from boundaries**: Keep sources at least 10 cells from PML
2. **Material interfaces**: Consider material interfaces for coupling
3. **Resolution requirements**: Higher resolution near sources
4. **Polarization matching**: Match source polarization to mode

### Source Excitation

1. **Frequency content**: Ensure source contains desired frequencies
2. **Pulse duration**: Include full pulse for time-domain analysis
3. **Amplitude selection**: Use appropriate amplitude for application
4. **Phase considerations**: Consider phase for coherent combining

## Source Examples

### GPR A-Scan Source

```python
# Hertzian dipole for GPR A-scan
#source: 0 0.0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#waveform: ricker 1.5e9 my_ricker
```

### B-Scan Source

```python
# Multiple sources for B-scan
#source: 0 0.0 0.100 0.170 0 hertzian_dipole z 0.100 0.170 0 my_ricker
#source: 0.0.0 0.200 0.170 0 hertzian_dipole z 0.200 0.170 0 my_ricker_2
#src_steps: 0.002 0
```

### Wideband Source

```python
# Wideband source for frequency sweep
#source: 0 0.0 0.100 0.170 0 gaussian 1.0e9 0.5e9
#f_center: 1.0e9
#f_width: 0.5e9
```

## Best Practices

### Source Selection

1. **Match application**: Choose appropriate source type for application
2. **Frequency range**: Ensure source covers frequency range of interest
3. **Polarization**: Match source polarization to expected modes
4. **Bandwidth**: Choose appropriate source bandwidth

### Source Parameters

1. **Amplitude**: Use appropriate amplitude for application
2. **Phase**: Consider phase for coherent combining
3. **Frequency**: Use correct frequency for application
4. **Polarization**: Specify correct field component

### Source Placement

1. **Boundary distance**: Keep sources away from PML boundaries
2. **Material coupling**: Consider material interfaces
3. **Mode matching**: Position sources to excite desired modes
4. **Resolution**: Higher resolution near sources

## Troubleshooting

### Source Not Exciting Modes

**Problem**: Source doesn't excite expected modes

**Solutions**:
1. Check source frequency
2. Verify source polarization
3. Check source position
4. Increase source amplitude

### Poor Coupling

**Problem**: Source couples poorly to structure

**Solutions**:
1. Check source position
2. Verify source polarization
3. Consider material interfaces
4. Adjust source frequency

### Numerical Instability

**Problem**: Simulation diverges with source

**Solutions**:
1. Reduce source amplitude
2. Check material stability
3. Verify boundary conditions
4. Reduce time step