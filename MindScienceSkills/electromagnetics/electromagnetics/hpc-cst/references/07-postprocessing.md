# Postprocessing

## Result Types

CST provides various postprocessing capabilities:

| Result Type | Description |
|-------------|-------------|
| S-parameters | Scattering parameters |
| Fields | E, H, current distributions |
| Far-field | Radiation patterns |
| SAR | Specific absorption rate |
| Energy | Stored, dissipated energy |

## S-Parameters

### Extraction

```
Result: S-parameters
Format: Touchstone (.snp) or CST native
Frequency range: As simulated
```

### Display Options

| Format | Description |
|--------|-------------|
| Magnitude (dB) | |Sij| in dB |
| Magnitude (linear) | |Sij| |
| Phase | ∠Sij in degrees |
| Smith chart | Complex S11, S22 |
| Polar plot | Complex representation |

### Example: Return Loss

```
# Plot S11
Result: S11
Format: Magnitude (dB)
Frequency: 2-3 GHz
Marker: Minimum at 2.45 GHz
```

### Example: Insertion Loss

```
# Plot S21
Result: S21
Format: Magnitude (dB)
Frequency: 2-3 GHz
Target: < 1 dB in passband
```

## Field Visualization

### Electric Field

```
Field: E-field
Component: |E|, Ex, Ey, Ez
Display: 2D slice or 3D volume
Scale: Linear or dB
```

### Magnetic Field

```
Field: H-field
Component: |H|, Hx, Hy, Hz
Display: Vector or magnitude
```

### Surface Current

```
Field: Surface current
Display: Vector plot
Arrow size: Proportional to magnitude
```

### Field Animations

```
Animation: Field vs. phase
Phases: 0° to 360°
Frames: 36
Export: GIF or video
```

## Far-Field Results

### Radiation Pattern

```
Result: Far-field
Type: 2D or 3D pattern
Cuts: E-plane, H-plane
Frequency: Single or sweep
```

### Antenna Parameters

| Parameter | Description |
|-----------|-------------|
| Gain | Directive gain (dBi) |
| Directivity | Angular distribution |
| Efficiency | Radiation efficiency |
| HPBW | Half-power beamwidth |
| SLL | Side-lobe level |

### Example: Antenna Gain

```
# 3D gain pattern
Result: Gain
Type: 3D
Frequency: 2.4 GHz
Peak gain: 8.2 dBi
```

### Example: Radiation Pattern Cuts

```
# E-plane and H-plane cuts
Cut: φ = 0° (E-plane)
Cut: φ = 90° (H-plane)
Frequency: 2.4 GHz
```

## SAR (Specific Absorption Rate)

### Definition

SAR measures power absorption in biological tissue:

```
SAR = σ|E|² / ρ
```
- σ: Conductivity (S/m)
- E: Electric field (V/m)
- ρ: Mass density (kg/m³)

### Standards

| Standard | Limit |
|----------|-------|
| FCC (US) | 1.6 W/kg (1g) |
| ICNIRP (EU) | 2.0 W/kg (10g) |

### Calculation

```
Result: SAR
Averaging: 1g or 10g
Frequency: As simulated
Standard: FCC or ICNIRP
```

## Energy and Power

### Stored Energy

```
Result: Stored energy
Type: Electric, Magnetic, Total
Frequency: As simulated
```

### Dissipated Power

```
Result: Dissipated power
Source: Conductor loss, dielectric loss
```

## Export Options

### Data Export

| Format | Description |
|--------|-------------|
| ASCII | Text file |
| CSV | Comma-separated |
| Touchstone | S-parameters |
| HDF5 | Binary, large data |

### Field Export

```
Export: E-field
Format: ASCII or HDF5
Location: 2D slice or 3D volume
```

### Image Export

```
Export: Plot image
Format: PNG, JPG, PDF
Resolution: 300 DPI
```

## Postprocessing Templates

### S-Parameter Template

```
1. Plot S11 (return loss)
2. Plot S21 (insertion loss)
3. Smith chart for S11
4. Export Touchstone file
```

### Antenna Template

```
1. 3D gain pattern
2. E-plane cut
3. H-plane cut
4. Efficiency calculation
5. Export radiation pattern
```

### Filter Template

```
1. S11 and S21 vs. frequency
2. Group delay
3. Passband ripple
4. Stopband attenuation
```

## Best Practices

1. **Check convergence** before postprocessing
2. **Use appropriate format** for data export
3. **Document results** with annotations
4. **Compare with measurements** when available
5. **Save templates** for repeated analysis
