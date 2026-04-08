# File I/O

PyVista provides comprehensive file I/O capabilities for various data formats.

## Writing Data

### VTK Format

```python
import pyvista as pv

# Create mesh
sphere = pv.Sphere(radius=1.0)

# Write to VTK format
sphere.save('output.vtk')

# Write to legacy VTK format
sphere.save('output.legacy.vtk')
```

### STL Format

```python
import pyvista as pv

# Create mesh
cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)

# Write to STL format
cube.save('output.stl')

# Write to binary STL
cube.save('output_binary.stl')
```

### PLY Format

```python
import pyvista as pv

# Create point cloud
points = pv.PolyData(points)

# Write to PLY format
points.save('output.ply')

# Write to binary PLY
points.save('output_binary.ply')
```

### OBJ Format

```python
import pyvista as pv

# Create mesh
sphere = pv.Sphere(radius=1.0)

# Write to OBJ format
sphere.save('output.obj')
```

### VTK XML Format

```python
import pyvista as pv

# Create unstructured grid
grid = pv.UnstructuredGrid(points, cells)

# Write to VTK XML format
grid.save('output.vtu')
```

## Reading Data

### VTK Format

```python
import pyvista as pv

# Read VTK file
mesh = pv.read('input.vtk')

# Read legacy VTK file
mesh = pv.read('input.legacy.vtk')
```

### STL Format

```python
import pyvista as pv

# Read STL file
mesh = pv.read('input.stl')

# Read binary STL
mesh = pv.read('input_binary.stl')
```

### PLY Format

```python
import pyvista as pv

# Read PLY file
mesh = pv.read('input.ply')

# Read binary PLY
mesh = pv.read('input_binary.ply')
```

### OBJ Format

```python
import pyvista as pv

# Read OBJ file
mesh = pv.read('input.obj')
```

### VTK XML Format

```python
import pyvista as pv

# Read VTK XML file
mesh = pv.read('input.vtu')
```

## Advanced Reading

### Using Specific Readers

```python
import pyvista as pv

# Use XML poly data reader
reader = pv.XMLPolyDataReader()
reader.set_file_name('input.vtu')
reader.update()
mesh = reader.get_output()

# Use STL reader
reader = pv.STLReader()
reader.set_file_name('input.stl')
reader.update()
mesh = reader.get_output()
```

### Parallel File Reading

```python
import pyvista as pv

# Read multiple files in parallel
files = ['file1.vtk', 'file2.vtk', 'file3.vtk']
meshes = [pv.read(f) for f in files]

# Combine meshes
combined = pv.merge(meshes)
```

### Streaming Large Files

```python
import pyvista as pv

# Stream large file
reader = pv.XMLUnstructuredGridReader()
reader.set_file_name('large_file.vtu')
reader.update()

# Process in chunks
while reader.can_read():
    mesh = reader.get_output()
    # Process mesh
```

## File Format Selection

| Format | Extension | Use Case | Features |
|--------|-----------|---------|---------|
| VTK | .vtk | General VTK format | Binary, ASCII |
| VTK XML | .vtu | Modern VTK format | Compressed, metadata |
| STL | .stl | Stereolithography | Binary, ASCII |
| PLY | .ply | Point clouds | Binary, ASCII |
| OBJ | .obj | 3D models | Text format |

## File Format Conversion

```python
import pyvista as pv

# Read one format
mesh = pv.read('input.stl')

# Write to another format
mesh.save('output.vtk')
```

## Common Issues and Solutions

### File Reading Failures

**Problem**: Cannot read file

**Solutions**:
- Check file path and permissions
- Verify file format
- Check file corruption
- Try different reader

### File Writing Failures

**Problem**: Cannot write file

**Solutions**:
- Check write permissions
- Verify disk space
- Check file path validity
- Try different format

### Format Compatibility

**Problem**: Data not compatible with format

**Solutions**:
- Check data type requirements
- Convert data to appropriate type
- Use compatible format
- Check format specifications

### Memory Issues

**Problem**: Out of memory with large files

**Solutions**:
- Use streaming for large files
- Process in chunks
- Reduce data resolution
- Use appropriate data types

## Best Practices

1. **Use appropriate format**: Match format to application
2. **Check file validity**: Verify before reading/writing
3. **Handle errors**: Catch file I/O exceptions
4. **Document format choices**: Keep track of format decisions
5. **Use binary formats**: For large files
6. **Validate data**: Check data after reading
7. **Use streaming**: For large files
8. **Handle metadata**: Preserve important information
