# ParaView pvpython pvbatch And Traces

## Purpose

Use this reference when deciding between GUI trace generation, `pvpython`, and `pvbatch`.

## Tool selection

| Mode | Best fit |
| --- | --- |
| GUI trace | prototype a workflow from interactive actions |
| `pvpython` | interactive scripting or serial scripted execution |
| `pvbatch` | non-interactive batch processing, including MPI-capable builds |

## Practical rules

- start from GUI trace when pipeline is easier to discover interactively
- use `pvpython` when you want an interactive Python-driven client
- use `pvbatch` for non-interactive scripted execution and cluster-side batch processing

## Critical distinction

`pvbatch` acts as its own server for the script it runs. Do not try to build a `pvbatch` script that also connects to another server with `Connect()`.

## Python API quick reference

### Core imports

```python
from paraview.simple import *
from paraview.servermanager import *
from paraview import modules
```

### Common operations

| Operation | Code | Notes |
|-----------|------|-------|
| Open file | `OpenDataFile("path")` | Auto-detects format |
| Get active view | `GetActiveViewOrCreate("RenderView")` | Creates if needed |
| Get reader | `GetActiveViewOrCreate().GetActiveSource()` | Reader object |
| Get data arrays | `reader.PointData`, `reader.CellData` | Available arrays |
| Apply filter | `filter = Input(reader)` | Creates filter object |
| Get output | `filter.GetOutput()` | Output port |
| Save screenshot | `SaveScreenshot("file.png", view, resolution)` | Capture view |
| Save state | `SaveState("file.pvsm")` | Save pipeline state |
| Write data | `writer.Write()` | Export data |

### Data inspection

```python
# List all point data arrays
reader = GetActiveViewOrCreate().GetActiveSource()
for i in range(reader.GetNumberOfPointArrays()):
    arr = reader.GetPointArray(i)
    print(f"Array {i}: {arr.GetName()} ({arr.GetNumberOfTuples()} tuples)")
```

### Filter operations

```python
# Create and apply a threshold filter
threshold = Threshold(Input=reader)
threshold.ThresholdBetween(lower, upper)
threshold.UpdatePipeline()

# Create a contour filter
contour = Contour(Input=threshold)
contour.SetValue(0, contour_value)  # Contour value
contour.UpdatePipeline()
```

### View and display

```python
# Reset camera and render
view = GetActiveViewOrCreate()
view.ResetCamera()
view.Reset()

# Set background color
ren1 = view.GetRenderer()
ren1.SetBackground(0.1, 0.2, 0.3)  # RGB

# Force render
view.Render()
```

## MPI parallel processing

### pvbatch with MPI

`pvbatch` supports MPI parallel processing when built with MPI support. Use for:

- Distributed data rendering on large datasets
- Parallel filter operations
- Multi-node visualization

### MPI usage pattern

```bash
# MPI batch execution
mpirun -np 4 pvbatch script.py
```

### MPI considerations

- Each rank processes a subset of data
- Use `GenerateProcessIds` for distributed processing
- Use `GhostCellsGenerator` for distributed rendering
- Configure `KDTREE` or `LOD` for efficient spatial queries

## Scripting baseline

Common baseline:

```python
from paraview.simple import *

# Always use explicit imports
# Always get explicit view and reader
# Always use explicit filter names
```

### Error handling

```python
try:
    reader = OpenDataFile("input.vtu")
except Exception as e:
    print(f"Error loading file: {e}")
    sys.exit(1)
```

### Memory management

```python
# For large datasets, use streaming
from paraview import core
core.vtkProcess.GetMemoryUsage()

# Or use specific readers that support streaming
```

## Trace generation best practices

### Start from GUI

1. Open ParaView GUI
2. Load test dataset
3. Build desired pipeline interactively
4. Use **Tools → Trace → Generate Python Script**
5. Save generated script as starting point

### Clean up trace

Remove unnecessary elements:
- GUI-specific state
- Absolute paths
- Interactive prompts
- View-specific camera settings (unless intentional)

### Add error handling

```python
# Wrap critical operations
try:
    reader = OpenDataFile("input.vtu")
    # Process data
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
```

### Parameterize

Make script flexible:
- Use command-line arguments for input/output paths
- Use environment variables for configuration
- Use config files for complex pipelines

## Failure patterns

| Symptom | Likely cause | First repair |
|----------|--------------|-------------|
| traced script is noisy or brittle | too much GUI state captured | simplify to trace into a cleaner script |
| `pvbatch` script fails with remote-connect logic | wrong execution model | use `pvserver` with `pvpython` or remove connect path |
| script works in one environment but not another | missing dependencies or version mismatch | check ParaView version and available modules |
| MPI script has poor scaling | inefficient data distribution | use `GenerateProcessIds` and spatial queries |
| memory errors on large datasets | loading entire dataset | use streaming readers or MPI parallelization |