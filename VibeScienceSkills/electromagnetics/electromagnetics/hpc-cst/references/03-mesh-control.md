# Mesh Control

## Mesh Types

CST uses different mesh types for different solvers:

| Solver | Mesh Type | Description |
|--------|-----------|-------------|
| Transient | Hexahedral | Cartesian grid cells |
| Frequency Domain | Tetrahedral | Triangular elements |
| Integral Equation | Surface | Triangular surface mesh |

## Hexahedral Mesh (Transient)

### Basic Settings

```
Mesh type: Hexahedral
Cells per wavelength: 10-20 (default: 15)
Smallest mesh cell: Auto or manual
```

### Mesh Lines

```
Mesh lines: Automatic or Manual
- Automatic: Based on geometry and wavelength
- Manual: User-defined mesh lines
```

### Mesh Refinement

| Refinement Type | Purpose |
|-----------------|---------|
| Automatic | Based on field energy |
| Manual | User-specified regions |
| Edge refinement | Small features |
| Curved surfaces | Smooth representation |

### Convergence Criteria

```
Energy accuracy: -50 dB (default)
Mesh adaptation: 5-10 passes
Convergence: ΔS < 0.01 (1%)
```

## Tetrahedral Mesh (Frequency Domain)

### Basic Settings

```
Mesh type: Tetrahedral
Maximum element size: λ/6 to λ/10
Minimum element size: Based on geometry
```

### Adaptive Mesh

```
Adaptive mesh: Enabled
Refinement passes: 5-10
Convergence: ΔS < 0.02 (2%)
```

### Mesh Quality

| Metric | Target |
|--------|--------|
| Aspect ratio | < 10 |
| Skewness | < 0.8 |
| Orthogonality | > 0.1 |

## Surface Mesh (Integral Equation)

### Settings

```
Mesh type: Surface (triangular)
Edge length: λ/8 to λ/12
Curved surfaces: Finer mesh
```

## Mesh Control Strategies

### 1. Wavelength-Based

```
Cells per wavelength: 15-30
- 15: Quick simulation
- 20: Standard accuracy
- 30: High accuracy
```

### 2. Geometry-Based

```
- Small features: Increase refinement
- Thin layers: Use mesh layers
- Curved surfaces: Enable curvature refinement
```

### 3. Field-Based

```
- High field regions: Finer mesh
- Low field regions: Coarser mesh
- Adaptive refinement: Automatic
```

## Common Mesh Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Too many cells | Small features | Use local mesh |
| Poor convergence | Mesh too coarse | Increase cells/λ |
| Memory overflow | Mesh too fine | Reduce mesh density |
| Inaccurate results | Mesh not converged | Enable adaptive mesh |

## Mesh Optimization Tips

1. **Start coarse**, refine as needed
2. **Use local mesh** for small features
3. **Enable adaptive mesh** for accuracy
4. **Check convergence** before final run
5. **Balance accuracy vs. time**

## Example: Antenna Mesh

```
# Global mesh
Cells per wavelength: 20

# Local mesh on antenna
Mesh region: "antenna"
Cells per wavelength: 30

# Local mesh on feed
Mesh region: "feed"
Maximum cell size: 0.5 mm
```

## Example: Waveguide Mesh

```
# Global mesh
Cells per wavelength: 15

# Mesh lines at boundaries
Mesh line: x = 0, a/2, a
Mesh line: y = 0, b/2, b

# Inside waveguide
Mesh region: "waveguide"
Cells per wavelength: 20
```
