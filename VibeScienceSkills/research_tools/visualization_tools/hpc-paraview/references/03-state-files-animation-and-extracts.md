# ParaView State Files Animation And Extracts

## Purpose

Use this reference when a workflow depends on state files, screenshots, animations, or extracted data products.

## State files

### Common state forms

- `.pvsm` state files
- Python state files

### .pvsm state files

Binary state files that capture the complete ParaView state including:

- Pipeline configuration
- View settings
- Data array properties
- Animation settings
- Camera positions

**Advantages:**
- Binary format, fast to load
- Complete state preservation
- Portable across machines

**Limitations:**
- Binary format, not human-readable
- Cannot be easily edited
- Version compatibility concerns

### Python state files

Text-based Python scripts repr() the ParaView state:

- Human-readable
- Can be version-controlled
- Can be manually edited
- Easy to debug

**Disadvantages:**
- Slower to load than .pvsm
- May not capture all GUI state
- Python version dependencies

### State file usage patterns

```python
# Save .pvsm state
SaveState("workflow_state.pvsm")

# Load .pvsm state
LoadState("workflow_state.pvsm")

# Save Python state
SaveState("workflow_state.py")
```

## Practical rules

- Use `.pvsm` when robust state persistence matters
- Use Python state or trace when manual editing is part of workflow
- Avoid committing brittle absolute data paths into shared state unless intentional
- Test state file portability before sharing across machines

## Animation generation

### Creating animations

```python
# from paraview.simple import *
from paraview import animation

# Create animation scene
scene = animation.AnimationScene()
scene.SetFileName("animation_%t.png")
scene.SetFrameRate(10)  # 10 FPS
scene.SetStartFrame(0)
scene.SetEndFrame(100)

# Add view to scene
view = GetActiveViewOrCreate("RenderView")
scene.AddView(view)

# Write animation
scene.Write()
```

### Animation formats

| Format | Extension | Use Case |
|---------|------------|----------|
| PNG sequence | .png | Frame-by-frame images |
| AVI | .avi | Windows-compatible video |
| MPEG | .mpg | Cross-platform video |
| OGG/Theora | .ogv | High-quality video |
| VTK XML | .pvtu | VTK animation file |

### Animation best practices

- Set appropriate frame rate (typically 10-30 FPS)
- Use consistent resolution across frames
- Consider compression for long animations
- Test playback on target systems

## Time step control

### Controlling time steps in animations

```python
# Set time step for animation
scene.SetTimeStep(10)  # Integer frame/index, not a physical time value

# Use temporal cache for time-series data
reader = OpenDataFile("time_series.vtu")
cache = TemporalCache(reader)
cache.UpdatePipeline()

# Get specific time step
reader.SetTimeStep(time_value)
reader.UpdatePipeline()
```

### Temporal data handling

```python
# List available time steps
reader = OpenDataFile("data.vtu")
time_steps = reader.TimestepValues

# Jump to specific time step
reader.SetTimeStep(time_steps[50])
reader.UpdatePipeline()
```

## Screenshot and animation logic

### Before generating image products

- Pick exact view or layout
- Set target resolution intentionally
- Confirm time step and coloring are intended ones

### Screenshot parameters

```python
# High-resolution screenshot
SaveScreenshot("high_res.png", ImageResolution=[1920, 1080])

# Specific view screenshot
SaveScreenshot("view1.png", view=GetActiveViewOrCreate("View1"))

# Multiple views
for view_name in ["View1", "View2", "View3"]:
    view = GetActiveViewOrCreate(view_name)
    SaveScreenshot(f"{view_name}.png", view=view)
```

### Multi-view layouts

```python
# Create layout with multiple views
layout = CreateLayout("MultiView")

# Configure each view
view1 = GetActiveViewOrCreate("View1")
view1.SetPosition(0, 0)
view1.SetSize(0.5, 0.5)

view2 = GetActiveViewOrCreate("View2")
view2.SetPosition(0.5, 0)
view2.SetSize(0.5, 0.5)

# Save layout
SaveLayout("multi_view_layout.pvsm")
```

## Save data and extracts

### Common export formats

| Format | Extension | Use Case |
|---------|------------|----------|
| CSV | .csv | Tabular data |
| VTK | .vtk, .vtu | VTK format |
| Exodus II | .e, .ex2 | FEM results |
| PLY | .ply | Point clouds |
| STL | .stl | 3D printing |

### Extracting data

```python
# Extract to CSV
writer = CreateWriter("output.csv", "CSV")
writer.SetInputConnection(reader)
writer.UpdatePipeline()

# Extract specific arrays
writer = CreateWriter("selected_data.csv", "CSV")
writer.SetInputConnection(reader)
writer.AddArray(reader.GetPointData().GetArray("Pressure"))
writer.AddArray(reader.GetPointData().GetArray("Velocity"))
writer.UpdatePipeline()
```

### Data sampling

```python
# Sample point data
sample = Sample(Input=reader)
sample.SetSamplingMode(0)  # Random sampling
sample.SetNumberOfSamples(1000)
sample.UpdatePipeline()

# Uniform sampling
sample.SetSamplingMode(2)  # Uniform sampling
sample.SetSamplingDimensions(3)  # 3D uniform
sample.UpdatePipeline()
```

### Field calculations

```python
# Compute derived fields
calculator = Calculator(Input=reader)
calculator.SetFunction("Magnitude")
calculator.SetInputArray([reader.GetPointData().GetArray("Velocity")])
calculator.SetResultArrayName("VelocityMagnitude")
calculator.UpdatePipeline()
```

## Failure patterns

| Symptom | Likely cause | First repair |
|----------|--------------|-------------|
| loading state cannot find files | saved paths are stale or machine-specific | remap data paths or rebuild a portable state |
| exported image is inconsistent | view, layout, or timestep was implicit | make a saved target explicit |
| output data is missing fields | wrong pipeline object was exported | export from intended source or filter |
| animation has wrong timing | time step not set correctly | set explicit time step and verify temporal cache |
| memory errors during export | large dataset, no streaming | use streaming writers or reduce data subset |

