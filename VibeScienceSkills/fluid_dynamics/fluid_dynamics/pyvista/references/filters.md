# Filters

PyVista provides powerful filtering capabilities for data transformation and analysis.

## Geometric Filters

### Contouring

```python
import pyvista as pv

# Extract contours
contours = mesh.contour(isosurfaces=[0.5], scalars='scalars')

# Display
contours.plot()
```

### Decimation

```python
import pyvista as pv

# Decimate mesh
decimated = mesh.decimate(target_reduction=0.5)

# Display
decimated.plot()
```

### Extract Surface

```python
import pyvista as pv

# Extract surface
surface = mesh.extract_surface()

# Display
surface.plot()
```

### Extract Edges

```python
import pyvista as pv

# Extract edges
edges = mesh.extract_edges()

# Display
edges.plot()
```

## Spatial Filters

### Clipping

```python
import pyvista as pv

# Clip with plane
clipped = mesh.clip(normal=(1, 0, 0), origin=(0, 0, 0))

# Clip with box
clipped = mesh.clip(bounds=(0, 1, 0, 1, 0, 1))

# Clip with surface
clipped = mesh.clip(surface=clip_surface)
```

### Slicing

```python
import pyvista as pv

# Slice with plane
sliced = mesh.slice(normal=(1, 0, 0), origin=(0, 0, 0))

# Slice with box
sliced = mesh.slice(bounds=(0, 1, 0, 1, 0, 1))
```

### Thresholding

```python
import pyvista as pv

# Threshold scalars
thresholded = mesh.threshold(scalars='scalars', value=0.5)

# Display
thresholded.plot()
```

## Data Filters

### Cell Centers

```python
import pyvista as pv

# Extract cell centers
cell_centers = mesh.cell_centers()

# Display
cell_centers.plot()
```

### Point Data

```python
import pyvista as pv

# Extract point data
point_data = mesh.points

# Display
point_data.plot()
```

### Cell Data

```python
import pyvista as pv

# Extract cell data
cell_data = mesh.cell_data()

# Display
cell_data.plot()
```

## Smoothing Filters

### Gaussian Smoothing

```python
import pyvista as pv

# Gaussian smoothing
smoothed = mesh.gaussian_smooth(n_iter=20, relaxation_factor=0.01)

# Display
smoothed.plot()
```

### Windowed Sinc Filter

```python
import pyvista as pv

# Windowed sinc filter
smoothed = mesh.windowed_sinc_filter()
```

## Transform Filters

### Rotation

```python
import pyvista as pv

# Rotate mesh
rotated = mesh.rotate_y(45)
rotated = mesh.rotate_x(30)
rotated = mesh.rotate_z(60)
```

### Translation

```python
import pyvista as pv

# Translate mesh
translated = mesh.translate((1, 2, 3))
```

### Scaling

```python
import pyvista as pv

# Scale mesh
scaled = mesh.scale((2, 2, 2))
```

### Reflection

```python
import pyvista as pv

# Reflect mesh
reflected = mesh.reflect((1, 0, 0))
```

## Analysis Filters

### Compute Normals

```python
import pyvista as pv

# Compute surface normals
normals = mesh.compute_normals()

# Display
normals.plot()
```

### Compute Gradients

```python
import pyvista as pv

# Compute gradient
gradient = mesh.compute_gradient(scalars='scalars')

# Display
gradient.plot()
```

### Compute Curvature

```python
import pyvista as pv

# Compute curvature
curvature = mesh.compute_curvature()

# Display
curvature.plot()
```

## Common Issues and Solutions

### Filter Failures

**Problem**: Filter fails to execute

**Solutions**:
- Check input data validity
- Verify filter parameters
- Check data type compatibility
- Try alternative filter

### Memory Issues

**Problem**: Out of memory with large datasets

**Solutions**:
- Use decimation before filtering
- Process in chunks
- Reduce data resolution
- Use appropriate data types

### Performance Issues

**Problem**: Filtering is slow

**Solutions**:
- Use vectorized operations
- Process in parallel
- Optimize filter parameters
- Use appropriate filter type

## Best Practices

1. **Check data validity**: Verify data before filtering
2. **Use appropriate filters**: Match filter to data type
3. **Handle edge cases**: Account for boundary conditions
4. **Document filter parameters**: Keep track of filter settings
5. **Test incrementally**: Start simple, validate, then expand
6. **Use appropriate output**: Choose output format for application
