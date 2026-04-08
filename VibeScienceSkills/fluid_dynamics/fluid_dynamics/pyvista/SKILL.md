---
name: pyvista
description: 3D visualization and data analysis library for scientific visualization, mesh processing, and computational fluid dynamics post-processing. Use when visualizing 3D data, processing meshes, analyzing CFD results, or creating interactive visualizations. Supports point clouds, meshes, volume data, and various file formats.
---

# PyVista

PyVista is a 3D visualization and data analysis library for scientific computing.

## Quick Start

### Basic Plotting

```python
import pyvista as pv

# Create a sphere
sphere = pv.Sphere(radius=1.0)

# Create plotter and display
plotter = pv.Plotter()
plotter.add_mesh(sphere)
plotter.show()
```

### Loading Data

```python
import pyvista as pv

# Load from file
mesh = pv.read('mesh.vtk')

# Display
mesh.plot()
```

### Filtering

```python
import pyvista as pv

# Create and filter
sphere = pv.Sphere(radius=1.0)
contours = sphere.contour()

# Display
contours.plot()
```

## Core Concepts

### Data Types

**Mesh Data**: Unstructured grids (triangles, quads, etc.)
```python
import pyvista as pv

# Create unstructured mesh
mesh = pv.PolyData(vertices, faces)
```

**Grid Data**: Structured grids (rectilinear, curvilinear, etc.)
```python
# Create structured grid
grid = pv.StructuredGrid(dimensions=(10, 10, 10))
```

**Point Clouds**: Unconnected points
```python
# Create point cloud
points = pv.PolyData(points)
```

**Volume Data**: 3D image data
```python
# Create volume data
volume = pv.UniformGrid(dimensions=(10, 10, 10))
```

### Common Operations

**Filters**: Transform and analyze data
```python
# Apply filter
contours = mesh.contour()
decimated = mesh.decimate()
```

**Mappers**: Map data to visual properties
```python
# Set mapper properties
mesh['scalars'] = mesh['scalars'] * 2.0
```

**Plotting**: Render and display
```python
# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show()
```

## Mesh Operations

### Mesh Creation

```python
import pyvista as pv

# Create sphere
sphere = pv.Sphere(radius=1.0, theta_resolution=30)

# Create cube
cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)

# Create cylinder
cylinder = pv.Cylinder(radius=0.5, height=2.0)
```

### Mesh Loading

```python
import pyvista as pv

# Load VTK file
mesh = pv.read('mesh.vtk')

# Load STL file
mesh = pv.read('model.stl')

# Load PLY file
mesh = pv.read('pointcloud.ply')
```

### Mesh Filtering

```python
import pyvista as pv

# Contour extraction
contours = mesh.contour()

# Decimation
decimated = mesh.decimate(target_reduction=0.5)

# Surface normals
normals = mesh.compute_normals()
```

## Data Analysis

### Scalar Field Analysis

```python
import pyvista as pv

# Extract scalar field
scalars = mesh['scalars']

# Compute gradient
gradient = mesh.compute_gradient(scalars)

# Compute curvature
curvature = mesh.compute_curvature()
```

### Vector Field Analysis

```python
import pyvista as pv

# Extract vector field
vectors = mesh['vectors']

# Compute divergence
divergence = mesh.compute_divergence(vectors)

# Compute vorticity
vorticity = mesh.compute_vorticity(vectors)
```

### Geometric Analysis

```python
import pyvista as pv

# Compute volume
volume = mesh.volume()

# Compute surface area
area = mesh.area()

# Compute bounds
bounds = mesh.bounds
```

## Visualization

### Basic Plotting

```python
import pyvista as pv

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show()
```

### Advanced Plotting

```python
import pyvista as pv

# Create plotter with background
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.background_color = 'white'
plotter.show()
```

### Multiple Meshes

```python
import pyvista as pv

# Add multiple meshes
plotter = pv.Plotter()
plotter.add_mesh(mesh1)
plotter.add_mesh(mesh2)
plotter.show()
```

## File I/O

### Writing Data

```python
import pyvista as pv

# Write to VTK format
mesh.save('output.vtk')

# Write to STL format
mesh.save('output.stl')

# Write to PLY format
mesh.save('output.ply')
```

### Reading Data

```python
import pyvista as pv

# Read from file
mesh = pv.read('input.vtk')

# Read with specific reader
reader = pv.XMLPolyDataReader()
reader.set_file_name('input.vtu')
reader.update()
mesh = reader.get_output()
```

## Resources

- **Mesh operations**: See [mesh_operations.md](references/mesh_operations.md)
- **Data analysis**: See [data_analysis.md](references/data_analysis.md)
- **Visualization**: See [visualization.md](references/visualization.md)
- **Filters**: See [filters.md](references/filters.md)
- **File I/O**: See [file_io.md](references/file_io.md)
- **Advanced features**: See [advanced.md](references/advanced.md)
