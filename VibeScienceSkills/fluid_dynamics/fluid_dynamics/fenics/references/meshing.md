# Complex Meshing with Gmsh

Guide to creating complex geometries using Gmsh with FEniCS.

## Basic Gmsh Integration

**Inline geometry:**
```python
import dolfinx as dfx
from mpi4py import MPI

# Define geometry in Gmsh format
geometry = """
// Gmsh script
SetFactory("OpenCASCADE");
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
"""

# Convert to DOLFINx mesh
mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

**From .geo file:**
```python
# Create geometry.geo file
# Then load
mesh = dfx.io.gmsh_to_dolfinx("geometry.geo", MPI.COMM_WORLD)
```

**From .msh file:**
```python
# Generate mesh externally: gmsh -2 geometry.geo -o mesh.msh
# Then load
mesh = dfx.io.gmsh_to_dolfinx("mesh.msh", MPI.COMM_WORLD)
```

## Common Geometries

### Circle

```python
geometry = """
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {0, 1, 0};
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 2};
Line Loop(3) = {1, 2};
Plane Surface(4) = {3};
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

### Annulus (Ring)

```python
geometry = """
Point(1) = {0, 0, 0};
Point(2) = {0.5, 0, 0};
Point(3) = {0, 0.5, 0};
Point(4) = {-0.5, 0, 0};
Point(5) = {0, -0.5, 0};
Point(6) = {1, 0, 0};
Point(7) = {0, 1, 0};
Point(8) = {-1, 0, 0};
Point(9) = {0, -1, 0};

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
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

### Rectangle with Hole

```python
geometry = """
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0.4, 0.4, 0};
Point(6) = {0.6, 0.4, 0};
Point(7) = {0.6, 0.6, 0};
Point(8) = {0.4, 0.6, 0};

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
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

### Channel with Obstacle

```python
geometry = """
Point(1) = {0, 0, 0};
Point(2) = {2, 0, 0};
Point(3) = {2, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0.5, 0.3, 0};
Point(6) = {0.7, 0.3, 0};
Point(7) = {0.7, 0.7, 0};
Point(8) = {0.5, 0.7, 0};

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
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

## Mesh Refinement

### Local Refinement

```python
geometry = """
// Fine mesh near origin
Point(1) = {0, 0, 0, 0.01};    // Small characteristic length
Point(2) = {1, 0, 0, 0.1};     // Coarser mesh
Point(3) = {1, 1, 0, 0.1};
Point(4) = {0, 1, 0, 0.1};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};
"""
```

### Size Field

```python
geometry = """
// Size varies with position
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

Field[1] = MathEval;
Field[1].F = "0.05 + 0.1*x";  // Size varies with x
Background Field = 1;
"""
```

### Boundary Layer Mesh

```python
geometry = """
// Boundary layer refinement
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// Boundary layer on bottom
Field[1] = BoundaryLayer;
Field[1].EdgesList = {1};
Field[1].SizeMin = 0.001;
Field[1].SizeMax = 0.1;
Field[1].Thickness = 0.1;
Background Field = 1;
"""
```

## 3D Meshing

### Simple Cube

```python
geometry = """
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Point(5) = {0, 0, 1};
Point(6) = {1, 0, 1};
Point(7) = {1, 1, 1};
Point(8) = {0, 1, 1};

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
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

## Physical Groups

### Named Boundaries

```python
geometry = """
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// Name physical groups
Physical Line("inlet") = {1};
Physical Line("outlet") = {3};
Physical Line("walls") = {2, 4};
Physical Surface("domain") = {6};
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)

# Access physical groups
inlet_facets = mesh.topology.meshtags("inlet")
outlet_facets = mesh.topology.meshtags("outlet")
```

### Boundary Conditions with Tags

```python
# Use physical tags for boundary conditions
def inlet_boundary(x):
    return np.isclose(x[0], 0.0)

# Or use mesh tags for complex geometries
inlet_dofs = dfx.fem.locate_dofs_topological(V, inlet_facets)
```

## Parallel Meshing

### Automatic Partitioning

```python
from mpi4py import MPI

# Mesh is automatically partitioned across processors
mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)

print(f"Rank: {MPI.COMM_WORLD.rank}")
print(f"Local cells: {mesh.topology.index_map(mesh.topology.dim, 0).size_local}")
```

### Load Balancing

```python
# Gmsh provides load balancing
# Specify number of partitions in Gmsh script
```

## Mesh Quality

### Checking Quality

```python
# Cell volumes
volumes = mesh.geometry.cell_volumes
print(f"Min volume: {volumes.min():.6e}")
print(f"Max volume: {volumes.max():.6e}")
print(f"Aspect ratio: {volumes.max()/volumes.min():.2f}")

# Cell quality (requires additional tools)
# See FEniCS documentation for quality metrics
```

### Quality Guidelines

- **Aspect ratio**: < 10 for good quality
- **Jacobian**: Should be positive
- **Angles**: Avoid very small or very large angles
- **Orthogonality**: Important for accuracy

## Advanced Features

### Boolean Operations

```python
geometry = """
// Union, intersection, difference
SetFactory("OpenCASCADE");

// Create two circles
Circle(1) = {0, 0, 0, 0.5};
Circle(2) = {1, 0, 0, 0.5};

// Boolean operations
Boolean Difference(3) = {2, 1};
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

### Extrusion

```python
geometry = """
// Extrude 2D surface to 3D
Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {1, 1, 0};
Point(4) = {0, 1, 0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Line Loop(5) = {1, 2, 3, 4};
Plane Surface(6) = {5};

// Extrude in z-direction
Extrude {7, {8}} = {6, {0, 0, 1}};
"""

mesh = dfx.io.gmsh_to_dolfinx(geometry, MPI.COMM_WORLD)
```

### Periodic Boundaries

```python
geometry = """
// Periodic mesh
// Requires careful setup
// See Gmsh documentation for periodic boundaries
"""
```

## Workflow

### Recommended Workflow

1. **Create geometry** in .geo file
2. **Generate mesh** with Gmsh: `gmsh -2 geometry.geo -o mesh.msh`
3. **Load in FEniCS**: `mesh = dfx.io.gmsh_to_dolfinx("mesh.msh", MPI.COMM_WORLD)`
4. **Check quality**: Verify mesh metrics
5. **Apply tags**: Name physical groups
6. **Solve**: Use mesh in FEniCS simulation

### External Mesh Generation

```python
# Generate mesh externally
# 1. Create geometry.geo
# 2. Run: gmsh -2 geometry.geo -o mesh.msh -3
# 3. Load in FEniCS
mesh = dfx.io.gmsh_to_dolfinx("mesh.msh", MPI.COMM_WORLD)
```

## Common Issues

### Gmsh Not Found

```bash
# Install Gmsh
conda install -c conda-forge gmsh
# or
brew install gmsh
```

### Mesh Generation Fails

**Check:**
- Geometry syntax
- Point ordering
1. Line loops are closed
- Surface orientation

### Poor Mesh Quality

**Solutions:**
- Refine critical regions
- Use size fields
- Check geometry
- Try different mesh algorithms

### Partitioning Issues

**Check:**
- MPI configuration
- Mesh is partitionable
- Load balancing

## Resources

- Gmsh documentation: https://gmsh.info/doc/
- Gmsh tutorial: https://gmsh.info/doc/tutorials/
- FEniCS Gmsh demo: https://docs.fenicsproject.org/dolfinx/v0.10.0.post1/python/demos/demo_gmsh.html
- OpenCASCADE: https://www.opencascade.com/
