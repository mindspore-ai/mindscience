# Boundary Conditions

## Boundary Types

CST supports various boundary conditions for different simulation scenarios:

| Boundary | Symbol | Description |
|----------|--------|-------------|
| Open | - | Free space, radiation |
| Electric (PEC) | Et = 0 | Perfect electric conductor |
| Magnetic (PMC) | Ht = 0 | Perfect magnetic conductor |
| Open (add space) | - | Open with padding |
| Periodic | - | Unit cell, array |
| Symmetry | - | Exploit symmetry planes |

## Open Boundary

**Use for:**
- Antenna radiation
- Scattering problems
- Open structures

**Settings:**
```
Boundary: Open
Distance: λ/4 to λ/2 from structure
Open add space: 10-20% of wavelength
```

**PML (Perfectly Matched Layer):**
```
PML layers: 8-16 (default: 8)
Reflection: -80 dB (default)
Distance from structure: λ/4 minimum
```

## Electric Boundary (PEC)

**Use for:**
- Metal surfaces
- Ground planes
- Waveguide walls

**Properties:**
```
Tangential E = 0
Normal H = 0
Surface current allowed
```

**Example:**
```
# Ground plane
Boundary: Electric
Face: "ground_plane"
```

## Magnetic Boundary (PMC)

**Use for:**
- Symmetry planes (H-plane)
- Magnetic walls
- Ideal magnetic conductors

**Properties:**
```
Tangential H = 0
Normal E = 0
```

## Periodic Boundary

**Use for:**
- Antenna arrays
- Frequency selective surfaces (FSS)
- Metamaterials
- Photonic crystals

**Settings:**
```
Boundary: Periodic
Unit cell: x × y × z
Phase shift: 0° (broadside) or scan angle
```

**Types:**
| Type | Description |
|------|-------------|
| Floquet mode | Unit cell analysis |
| Array pattern | Full array factor |

## Symmetry Boundary

**Use for:**
- Reducing simulation domain
- Exploiting geometric symmetry

**Types:**
| Symmetry | Condition | Reduction |
|----------|-----------|-----------|
| Electric (Et=0) | H-plane symmetry | 1/2 domain |
| Magnetic (Ht=0) | E-plane symmetry | 1/2 domain |
| Both | Quarter symmetry | 1/4 domain |

**Example: Dipole Antenna**
```
# Dipole along z-axis
Symmetry plane: xy-plane
Type: Magnetic (Ht=0)
Domain: Upper half only
```

## Thermal Boundary

**Use for:**
- EM-Thermal coupling
- Heat dissipation

**Types:**
| Type | Description |
|------|-------------|
| Fixed temperature | Constant T |
| Convection | Heat transfer coefficient |
| Radiation | Stefan-Boltzmann |

## Boundary Selection Guide

| Application | Recommended Boundary |
|-------------|----------------------|
| Antenna | Open (PML) |
| Waveguide | Electric (PEC) |
| Cavity | Electric or Magnetic |
| Array element | Periodic |
| Symmetric structure | Symmetry |

## Common Mistakes

| Mistake | Consequence | Solution |
|---------|-------------|----------|
| Open too close | Reflections | Add λ/4 space |
| Wrong symmetry | Incorrect results | Check field pattern |
| Missing ground | Floating metal | Add PEC boundary |
| Periodic mismatch | Phase errors | Verify unit cell |

## Example: Patch Antenna

```
# Boundary setup for patch antenna
Xmin: Open (radiation)
Xmax: Open (radiation)
Ymin: Open (radiation)
Ymax: Open (radiation)
Zmin: Electric (ground plane)
Zmax: Open (radiation)

# Ground plane
Face: "ground"
Boundary: Electric
```

## Example: Waveguide Filter

```
# Boundary setup for waveguide
Walls: Electric (PEC)
Input port: Waveguide port
Output port: Waveguide port

# Symmetry (if applicable)
Symmetry plane: H-plane
Type: Magnetic
```
