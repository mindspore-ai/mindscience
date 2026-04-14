# Complex Meshing with Gmsh

Guide to creating complex geometries using Gmsh with FiPy.

## Basic Gmsh Integration

**Inline geometry specification:**
```python
from fipy import Gmsh2D

mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
''')
```

**From .geo file:**
```python
mesh = Gmsh2D('geometry.geo')
```

**From .msh file:**
```python
mesh = Gmsh2D('mesh.msh')
```

## Common Geometries

### Circle

```python
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {0, 1, 0, 0.1};
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 2};
Line Loop(3) = {1, 2};
Plane Surface(4) = {3};
''')
```

### Annulus (ring)

```python
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {0.5, 0, 0, 0.1};
Point(3) = {0, 0.5, 0, 0.1};
Point(4) = {-0.5, 0, 0, 0.1};
Point(5) = {0, -0.5, 0, 0.1};
Point(6) = {1, 0, 0, 0.1.5};
Point(7) = {0, 1, 0, 0.1.5};
Point(8) = {-1, 0, 0, 0.1.5};
Point(9) = {0, -1, 0, 0.1.5};

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};
Circle(5) = {6, 1, 7};
Circle(6) = {7, 1, 8};
Circle(7) = {8, 1, 9};
Circle(8) = {9, 1, 6};

Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {5, 6, 7, 8};

Plane Surface(11) = {10, 9};
''')
```

### Rectangle with Hole

```python
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Point(5) = {0.4, 0.4, 0, 0.05};
Point(6) = {0.6, 0.4, 0, 0.05};
Point(7) = {0.6, 0.6, 0, 0.05};
Point(8) = {0.4, 0.6, 0, 0.05};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {5, 6, 7, 8};

Plane Surface(11) = {9, 10};
''')
```

### Channel with Obstacle

```python
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {2, 0, 0, 0.1};
Point(3) = {2, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Point(5) = {0.5, 0.3, 0, 0.05};
Point(6) = {0.7, 0.3, 0, 0.05};
Point(7) = {0.7, 0.7, 0, 0.05};
Point(8) = {0.5, 0.7, 0, 0.05};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line Loop(9) = {1, 2, 3, 4};
Line Loop(10) = {5, 6, 7, 8};

Plane Surface(11) = {9, 10};
''')
```

## Mesh Refinement

**Local refinement around points:**
```python
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.01};    // Fine mesh at origin
Point(2) = {1, 0, 0, 0.1};     // Coarser mesh
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
''')
```

**Size field:**
```python
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

Field[1] = MathEval;
Field[1].F = "0.05 + 0.1*x";  // Size varies with x
Background Field = 1;
''')
```

## 3D Meshing

**Simple cube:**
```python
from fipy import Gmsh3D

mesh = Gmsh3D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Point(5) = {0, 0, 1, 0.1};
Point(6) = {1, 0, 1, 0.1};
Point(7) = {1, 1, 1, 0.1};
Point(8) = {0, 1, 1, 0.1};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};
Line(9) = {1, 5};
Line(10) = {2, 6};
Line(11) = {3, 7};
Line(12) = {4, 8};

Line Loop(13) = {1, 2, 3, 4};
Line Loop(14) = {5, 6, 7, 8};
Line Loop(15) = {1, 10, -5, -9};
Line Loop(16) = {2, 11, -6, -10};
Line Loop(17) = {3, 12, -7, -11};
Line Loop(18) = {4, 9, -8, -12};

Plane Surface(19) = {13};
Plane Surface(20) = {14};
Plane Surface(21) = {15};
Plane Surface(22) = {16};
Plane Surface(23) = {17};
Plane Surface(24) = {18};

Surface Loop(25) = {19, 20, 21, 22, 23, 24};
Volume(26) = {25};
''')
```

## Boundary Identification

**Named physical groups for boundary conditions:**
```python
mesh = Gmsh2D('''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

Physical Line("inlet") = {1};
Physical Line("outlet") = {3};
Physical Line("walls") = {2, 4};
Physical Surface("domain") = {6};
''')

# Access physical groups
inlet_faces = mesh.physicalGroups["inlet"]
outlet_faces = mesh.physicalGroups["outlet"]
```

## Parallel Mesh Partitioning

**Automatic partitioning with MPI:**
```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
mesh = Gmsh2D(geometry_string, communicator=comm)
```

Mesh is automatically partitioned across processors.

## Mesh Quality

**Check mesh quality:**
```python
# Cell volumes
print(f"Min cell volume: {mesh.cellVolumes.min()}")
print(f"Max cell volume: {mesh.cellVolumes.max()}")

# Cell aspect ratios
aspect_ratios = mesh.cellVolumes.max() / mesh.cellVolumes.min()
print(f"Aspect ratio: {aspect_ratios:.2f}")

# Orthogonality (important for accuracy)
# Compute face normals vs. cell center vectors
```

**Mesh quality guidelines:**
- Aspect ratio < 10: Good
- Aspect ratio 10-100: Acceptable
- Aspect ratio > 100: May affect accuracy
- Highly non-orthogonal meshes reduce accuracy

## Working with Gmsh Files

**Save geometry to .geo file:**
```python
geometry = '''
Point(1) = {0, 0, 0, 0.1};
Point(2) = {1, 0, 0, 0.1};
...
'''

with open('my_geometry.geo', 'w') as f:
    f.write(geometry)

mesh = Gmsh2D('my_geometry.geo')
```

**Generate mesh externally:**
```bash
# Generate mesh with Gmsh
gmsh -2 my_geometry.geo -o my_mesh.msh

# Use in FiPy
python script.py
```

```python
# In script.py
from fipy import Gmsh2D
mesh = Gmsh2D('my_mesh.msh')
```

## Common Issues

**Gmsh not found:**
```bash
conda install -c conda-forge gmsh
```

**Mesh generation fails:**
- Check geometry syntax
- Ensure points are defined before lines
- Verify line loops are closed
- Check surface orientation

**Poor mesh quality:**
- Refine mesh in critical regions
- Use size fields for controlled refinement
- Check for highly skewed elements

**Parallel partitioning issues:**
- Ensure MPI is properly configured
- Check that mesh is partitionable
- Verify communicator is passed correctly

## Advanced Features

**Boolean operations:**
```python
# Union, intersection, difference of geometries
# See Gmsh documentation for boolean operations
```

**Transfinite interpolation:**
```python
# Create structured-like meshes on complex geometries
# Useful for boundary layer meshes
```

**Periodic meshes:**
```python
# Create periodic boundary conditions
# Requires matching boundary definitions
```

## Resources

- Gmsh documentation: https://gmsh.info/doc/
- Gmsh tutorial: https://gmsh.info/doc/tutorials/
- FiPy Gmsh examples: `examples.diffusion.circle`
