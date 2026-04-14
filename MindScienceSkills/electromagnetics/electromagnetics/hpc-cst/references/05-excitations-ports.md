# Excitations and Ports

## Port Types

CST supports various excitation types:

| Port Type | Use Case | Solver |
|-----------|----------|--------|
| Waveguide Port | Waveguide, coax, microstrip | All |
| Discrete Port | Lumped element excitation | All |
| Plane Wave | Scattering, RCS | Transient, IE |
| Farfield Source | Antenna excitation | All |
| Current Source | Wire excitation | Transient |

## Waveguide Port

**Best for:**
- Waveguide structures
- Coaxial cables
- Microstrip lines
- CPW (Coplanar Waveguide)

### Settings

```
Port type: Waveguide
Number of modes: 1 (fundamental) or more
Reference impedance: 50 Ω (default) or custom
```

### Mode Types

| Structure | Fundamental Mode |
|-----------|-----------------|
| Rectangular waveguide | TE10 |
| Circular waveguide | TE11 |
| Coaxial cable | TEM |
| Microstrip | Quasi-TEM |
| CPW | Quasi-TEM |

### Port Placement

```
Position: At waveguide cross-section
Size: Cover entire cross-section
Distance: λ/4 from discontinuity (minimum)
```

### Example: Rectangular Waveguide

```
# WR-90 waveguide (X-band)
Port: "Port1"
Type: Waveguide
Face: "input_face"
Modes: 1 (TE10)
Frequency: 8-12 GHz
```

### Example: Microstrip Line

```
# Microstrip port
Port: "Port1"
Type: Waveguide
Face: "port_face"
Width: 2.5 mm (line width)
Ground: "ground_plane"
Reference impedance: 50 Ω
```

## Discrete Port

**Best for:**
- Lumped element sources
- Simple excitation
- Quick simulations

### Settings

```
Port type: Discrete
Impedance: 50 Ω (default)
Location: Between two points
```

### Types

| Type | Description |
|------|-------------|
| S-Parameter | Excitation with impedance |
| Voltage | Voltage source |
| Current | Current source |

### Example: Dipole Feed

```
Port: "Feed"
Type: Discrete
Location: Gap between dipole arms
Impedance: 50 Ω
```

## Plane Wave Excitation

**Best for:**
- Radar cross-section (RCS)
- Scattering analysis
- EMC testing

### Settings

```
Excitation: Plane Wave
Direction: Propagation vector
Polarization: E-field direction
Frequency: Single or sweep
```

### Example: RCS Simulation

```
Excitation: Plane Wave
Direction: θ = 0°, φ = 0° (normal incidence)
Polarization: Linear (Ex)
Frequency: 10 GHz
```

## Multiple Ports

### S-Parameter Matrix

For N ports, CST computes N×N S-parameter matrix:

```
S = [S11  S12 ... S1N]
    [S21  S22 ... S2N]
    [...  ... ... ...]
    [SN1  SN2 ... SNN]
```

### Port De-embedding

```
Reference plane shift: Distance from port
Phase adjustment: Automatic
```

### Example: 4-Port Coupler

```
Port1: Input
Port2: Through
Port3: Coupled
Port4: Isolated

S-parameters of interest:
- S21: Through transmission
- S31: Coupling
- S41: Isolation
```

## Excitation Signals

### Transient Solver

| Signal | Use Case |
|--------|----------|
| Gaussian | Broadband |
| Rectangular | Pulse |
| User-defined | Custom |

### Frequency Domain Solver

| Signal | Use Case |
|--------|----------|
| Single frequency | Narrowband |
| Frequency sweep | Broadband |

## Port Best Practices

1. **Port size**: Cover entire cross-section
2. **Port distance**: λ/4 from discontinuities
3. **Reference impedance**: Match to system
4. **Mode count**: Use fundamental unless higher modes needed
5. **De-embedding**: Set correct reference plane

## Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Port reflection | Impedance mismatch | Adjust reference impedance |
| Mode mismatch | Wrong mode selected | Check mode pattern |
| Port coupling | Ports too close | Increase separation |
| Inaccurate S21 | Port de-embedding wrong | Adjust reference plane |
