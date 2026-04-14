# Mesh Operations

PyVista provides comprehensive mesh operations for creating, modifying, and analyzing meshes.

## Mesh Creation

### Basic Geometric Objects

```python
import pyvista as pv

# Create sphere
sphere = pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30)

# Create cube
cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)

# Create cylinder
cylinder = pv.Cylinder(radius=0.5, height=2.0, resolution=50)
```

### Parametric Objects

```python
import pyvista as pv

# Create parametric sphere
sphere = pv.ParametricSpline(radius=1.0)

# Create parametric cube
cube = pv.ParametricCube(x_length=1.0, y_length=1.0, z_length=1.0)

# Create parametric cylinder
cylinder = pv.ParametricCylinder(radius=0.5, height=2.0)
```

### Structured Grids

```python
import pyvista as pv

# Create explicit structured grid
grid = pv.ExplicitStructuredGrid(dimensions=(10, 10, 10))

# Create rectilinear grid
grid = pv.RectilinearGrid(dimensions=(10, 10, 10))

# Create curvilinear grid
grid = pv.CurvilinearGrid(dimensions=(10, 10, 10))
```

### Unstructured Meshes

```python
import pyvista as pv
import numpy as np

# Create triangle strip
vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
faces = np.array([[0, 1, 2]])
mesh = pv.PolyData(vertices, faces)

# Create polygon
vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2, 3]])
mesh = pv.PolyData(vertices, faces)
```

## Mesh Loading

### VTK Files

```python
import pyvista as pv

# Load VTK file
mesh = pv.read('mesh.vtk')

# Load legacy VTK file
mesh = pv.read('mesh.legacy.vtk')
```

### STL Files

```python
import pyvista as pv

# Load STL file
mesh = pv.read('model.stl')

# Load binary STL
mesh = pv.read('model_binary.stl')
```

### PLY Files

```python
import pyvista as pv

# Load PLY file
mesh = pv.read('pointcloud.ply')

# Load binary PLY
mesh = pv.read('pointcloud_binary.ply')
```

### OBJ Files

```python
import pyvista as pv

# Load OBJ file
mesh = pv.read('model.obj')
```

## Mesh Transformation

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

## Mesh Analysis

### Geometric Properties

```python
import pyvista as pv

# Compute volume
volume = mesh.volume()

# Compute surface area
area = mesh.area()

# Compute bounds
bounds = mesh.bounds
```

### Quality Metrics

```python
import pyvista as pv

# Compute mesh quality
quality = mesh.compute_mesh_quality()

# Get quality metrics
metrics = {
    'aspect_ratio': quality['aspect_ratio'],
    'skewness': quality['skewness'],
    'non_orthogonality': quality['non_orthogonality']
}
```

### Connectivity

```python
import pyvista as pv

# Extract connectivity
contours = mesh.contour()

# Extract edges
edges = mesh.extract_edges()

# Extract surface
surface = mesh.extract_surface()
```

## Mesh Modification

### Decimation

```python
import pyvista as pv

# Decimate mesh
decimated = mesh.decimate(target_reduction=0.5)
```

### Smoothing

```python
import pyvista as pv

# Gaussian smoothing
smoothed = mesh.gaussian_smooth(n_iter=20, relaxation_factor=0.01)
```

### Subdivision

```python
import pyvista as pv

# Subdivide cells
subdivided = mesh.subdivide(nsubdivs=2)
```

## Mesh Comparison

### Distance Between Meshes

```python
import pyvista as pv

# Compute distance between meshes
distance = mesh1.distance_to(mesh2)
```

### Collision Detection

```python
import pyvista as pv

# Check collision
collision = mesh1.collide_with(mesh2)
```

## Common Issues and Solutions

### Mesh Loading Failures

**Problem**: Cannot load mesh file

**Solutions**:
- Check file format compatibility
- Verify file path and permissions
- Try different reader
- Check file corruption

### Memory Issues

**Problem**: Out of memory with large meshes

**Solutions**:
- Use decimation for large meshes
- Process in chunks
- Use appropriate data types
- Reduce mesh resolution

### Quality Issues

**Problem**: Poor mesh quality affects analysis

**Solutions**:
- Apply smoothing
- Improve mesh generation
- Check and fix degenerate cells
- Use appropriate mesh operations

## Best Practices

1. **Use appropriate mesh type**: Match mesh type to application
2. **Check mesh quality**: Validate before analysis
3. **Handle large meshes**: Use decimation and streaming
4. **Document transformations**: Keep track of mesh operations
5. **Use appropriate file formats**: Choose format for application
