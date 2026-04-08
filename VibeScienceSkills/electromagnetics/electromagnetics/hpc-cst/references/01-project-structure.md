# CST Project Structure

## Project File Types

CST Studio Suite uses several file types:

| Extension | Description |
|-----------|-------------|
| `.cst` | Main project file (XML-based) |
| `.m3d` | 3D model data |
| `.res` | Result files |
| `.hdf` | HDF5 result data |
| `.log` | Simulation log |

## Project Organization

```
project/
├── model.cst           # Main project file
├── History/            # Undo history
├── Result/             # Simulation results
│   ├── 3D/            # Field results
│   ├── 1D/            # S-parameters, time signals
│   └── 0D/            # Scalar results
└── Export/            # Exported data
```

## Key Components

### 1. Geometry
- Parametric 3D models
- Import from CAD (STEP, IGES, SAT)
- Boolean operations

### 2. Materials
- Built-in material library
- Custom material definitions
- Frequency-dependent properties

### 3. Mesh
- Hexahedral (Time Domain)
- Tetrahedral (Frequency Domain)
- Surface mesh (Surface Integral)

### 4. Solver
- Transient (Time Domain)
- Frequency Domain
- Eigenmode
- Integral Equation

### 5. Postprocessing
- S-parameter extraction
- Field visualization
- Far-field patterns
- SAR calculations

## VBA Macro Structure

CST projects can be automated via VBA:

```vba
' Create new project
NewProject

' Define parameter
StoreParameter "freq", 2.4e9

' Create geometry
Brick "substrate", "dx", "dy", "dz"

' Set material
Material "FR4", "Epsilon", 4.4

' Add port
Port "Port1", "face", "x", "ymin"
```

## Best Practices

1. **Use parameters** for all critical dimensions
2. **Name components** descriptively
3. **Organize history tree** logically
4. **Save checkpoints** before major changes
5. **Document assumptions** in project notes
