# Visualization

PyVista provides comprehensive visualization capabilities for 3D data.

## Basic Plotting

### Simple Plotting

```python
import pyvista as pv

# Create sphere
sphere = pv.Sphere(radius=1.0)

# Create plotter
plotter = pv.Plotter()
plotter.add_mesh(sphere)
plotter.show()
```

### Multiple Meshes

```python
import pyvista as pv

# Create multiple meshes
sphere1 = pv.Sphere(radius=1.0)
sphere2 = pv.Sphere(radius=0.5, center=(2, 0, 0))

# Add to plotter
plotter = pv.Plotter()
plotter.add_mesh(sphere1)
plotter.add_mesh(sphere2)
plotter.show()
```

## Advanced Plotting

### Scalar Bar Plotting

```python
import pyvista as pv

# Create mesh with scalars
mesh = pv.read('mesh.vtk')

# Add scalar bar
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='scalars')
plotter.add_scalar_bar('scalars', title='Scalar Values')
plotter.show()
```

### Vector Glyphs

```python
import pyvista as pv

# Create mesh with vectors
mesh = pv.read('flow.vtk')

# Add vector glyphs
arrows = mesh.glyph(orient='vectors')
arrows.glyph.scale_by('scalars')

plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='scalars')
plotter.add_mesh(arrows)
plotter.show()
```

### Streamlines

```python
import pyvista as pv

# Create mesh with vectors
mesh = pv.read('flow.vtk')

# Add streamlines
streamlines = mesh.streamlines('vectors')

plotter = pv.Plotter()
plotter.add_mesh(streamlines)
plotter.show()
```

## Color Mapping

### Scalar Colormap

```python
import pyvista as pv

# Create mesh with scalars
mesh = pv.read('mesh.vtk')

# Set colormap
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='scalars', cmap='viridis')
plotter.show()
```

### Custom Colormap

```python
import pyvista as pv

# Create custom colormap
colors = [(0, 'blue'), (0.5, 'green'), (1, 'red')]
cmap = pv.LookupTable([c for c in colors])

# Apply colormap
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='scalars', cmap=cmap)
plotter.show()
```

## Lighting

### Basic Lighting

```python
import pyvista as pv

# Create mesh
mesh = pv.read('mesh.vtk')

# Add lighting
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.enable_lighting()
plotter.show()
```

### Light Actors

```python
import pyvista as pv

# Create light
light = pv.Light(position=(10, 10, 10), focal_point=(0, 0, 0))

# Add to plotter
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.add_light(light)
plotter.show()
```

## Camera Control

### Camera Position

```python
import pyvista as pv

# Set camera position
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.camera_position = (10, 10, 10)
plotter.show()
```

### Camera Movement

```python
import pyvista as pv

# Orbit camera
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.show(interactive_update=True)
```

## Background

### Background Color

```python
import pyvista as pv

# Set background color
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.background_color = 'white'
plotter.show()
```

### Background Image

```python
import pyvista as pv

# Set background image
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.set_background_image('background.png')
plotter.show()
```

## Axes

### Add Axes

```python
import pyvista as pv

# Add axes
plotter = pv.Plotter()
plotter.add_mesh(mesh)
plotter.add_axes()
plotter.show()
```

### Customize Axes

```python
import pyvista as pv

# Customize axes
plotter = pv.Plotter()
plotter.add_mesh(mesh)
axes = plotter.add_axes()
axes.x_label = 'X Axis'
axes.y_label = 'Y Axis'
axes.z_label = 'Z Axis'
plotter.show()
```

## Colorbar

### Add Colorbar

```python
import pyvista as pv

# Add colorbar
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='scalars')
plotter.add_scalar_bar('scalars', title='Scalar Field')
plotter.show()
```

### Customize Colorbar

```python
import pyvista as pv

# Customize colorbar
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars='scalars')
cbar = plotter.add_scalar_bar('scalars')
cbar.title = 'Scalar Field'
cbar.label_text_color = 'black'
cbar.title_text_color = 'black'
plotter.show()
```

## Common Issues and Solutions

### Rendering Issues

**Problem**: Mesh doesn't render correctly

**Solutions**:
- Check mesh validity
- Verify scalar/vector data
- Check camera position
- Ensure proper lighting

### Performance Issues

**Problem**: Rendering is slow

**Solutions**:
- Use decimation for large meshes
- Reduce mesh resolution
- Disable expensive features
- Use appropriate rendering backend

### Display Issues

**Problem**: Display window doesn't appear

**Solutions**:
- Check plotter configuration
- Verify mesh data
- Check display backend
- Try different display options

## Best Practices

1. **Start simple**: Begin with basic plotting
2. **Use appropriate colormaps**: Match colormap to data
3. **Add legends**: For clarity
4. **Set proper camera**: For good viewing angle
5. **Use lighting**: For 3D perception
6. **Add colorbars**: For scalar fields
7. **Test interactively**: Use interactive plots for exploration
