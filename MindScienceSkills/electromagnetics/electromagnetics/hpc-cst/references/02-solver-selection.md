# Solver Selection

## Available Solvers

CST Studio Suite offers multiple electromagnetic solvers:

| Solver | Domain | Best For |
|--------|--------|----------|
| Transient | Time Domain | Wideband, pulsed signals |
| Frequency Domain | Frequency Domain | Narrowband, high-Q structures |
| Eigenmode | Frequency Domain | Resonant cavities, modes |
| Integral Equation | Frequency Domain | Large structures, scattering |
| Asymptotic | High Frequency | Electrically large structures |

## Transient Solver

**Best for:**
- Wideband simulations
- Time-domain reflectometry
- Pulse propagation
- EMC/EMI analysis

**Settings:**
```
Solver: Transient
Frequency range: 0 - 10 GHz
Mesh: Hexahedral
Accuracy: -50 dB (default)
```

**Advantages:**
- Single run for entire frequency range
- Efficient for broadband
- Natural for time-domain signals

**Limitations:**
- May need many mesh cells
- Memory intensive for large structures

## Frequency Domain Solver

**Best for:**
- Narrowband devices
- High-Q resonators
- Periodic structures
- S-parameter extraction

**Settings:**
```
Solver: Frequency Domain
Frequency range: 2.4 - 2.5 GHz
Mesh: Tetrahedral
Adaptive mesh: Yes
```

**Advantages:**
- Efficient for narrowband
- Better for high-Q structures
- Direct S-parameter calculation

**Limitations:**
- Multiple runs for broadband
- Slower for wide frequency range

## Eigenmode Solver

**Best for:**
- Resonant cavities
- Waveguide modes
- Filter design
- Particle accelerator structures

**Settings:**
```
Solver: Eigenmode
Modes: 10
Frequency range: 0 - 20 GHz
```

**Outputs:**
- Resonant frequencies
- Mode patterns (E, H fields)
- Q-factors (with losses)

## Integral Equation Solver

**Best for:**
- Antenna radiation
- Radar cross-section
- EMC analysis
- Large structures

**Settings:**
```
Solver: Integral Equation
Method: MLFMM (Multi-Level Fast Multipole)
Frequency: Single or sweep
```

## Solver Selection Guide

| Application | Recommended Solver |
|-------------|-------------------|
| Antenna (wideband) | Transient |
| Antenna (narrowband) | Frequency Domain |
| Filter design | Eigenmode + Frequency Domain |
| EMC/EMI | Transient |
| Waveguide | Eigenmode |
| RCS calculation | Integral Equation |
| High-frequency optics | Asymptotic |

## Multi-Physics Coupling

CST supports coupled simulations:

| Coupling | Description |
|----------|-------------|
| EM-Thermal | Joule heating, thermal effects |
| EM-Mechanical | Deformation, stress |
| EM-Particle | Charged particle dynamics |

## Performance Tips

1. **Transient**: Use mesh refinement for accuracy
2. **Frequency Domain**: Enable adaptive mesh
3. **Eigenmode**: Set appropriate mode count
4. **Integral Equation**: Use MLFMM for large problems
