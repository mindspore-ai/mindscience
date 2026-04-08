# Visualization in NGsolve

NGsolve provides various options for visualizing simulation results.

## Field Visualization

### Electric Field

```python
import ngsolve as ns
import matplotlib.pyplot as plt

# Get electric field
E = V.get_subfunction('E')
E_array = E.vector().get_array()

# Plot electric field
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(E_array, cmap='RdBu', origin='lower')
ax.set_title('Electric Field')
ax.colorbar(im, label='E field')
plt.show()
```

### Magnetic Field

```python
# Get magnetic field
H = V.get_subfunction('H')
H_array = H.vector().get_array()

# Plot magnetic field
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(H_array, cmap='RdBu', origin='lower')
ax.set_title('Magnetic Field')
ax.colorbar(im, label='H field')
plt.show()
```

### Poynting Vector

```python
# Calculate Poynting vector
# S = E.cross(H)

# Plot Poynting vector
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(S, cmap='seismic', origin='lower')
ax.set_title('Poynting Vector')
ax.colorbar(im, label='Poynting Vector')
plt.show()
```

## Material Visualization

### Permittivity Distribution

```python
# Get permittivity
epsilon = V.get_subfunction('epsilon')
epsilon_array = epsilon.vector().get_array()

# Plot permittivity
fig, = plt.figure(figsize=(10, 8))
plt.imshow(epsilon_array, cmap='viridis', origin='lower')
plt.colorbar(label='Permittivity')
plt.title('Permittivity Distribution')
plt.show()
```

### Material Boundaries

```python
# Visualize material boundaries
# Get material indicator
mat = V.get_subfunction('material')
mat_array = mat.vector().get_array()

# Plot material boundaries
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mat_array, cmap='nipy_spectral', origin='lower')
ax.set_title('Material Boundaries')
ax.colorbar(im, label='Material')
plt.show()
```

## 3D Visualization

### Slice Visualization

```python
# Get 3D field slice
E = V.get_subfunction('E')
E_slice = E.vector().get_array(z=10)

# Plot field slice
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(E_slice, cmap='RdBu', origin='lower')
ax.set_title('Field Slice at z=10')
ax.colorbar(im, label='E field')
plt.show()
```

### 2D Contour Plot

```python
# Get 2D contour
E = V.get_subfunction('E')
E_2d = E.vector().get_array()

# Plot 2D contour
fig, ax = plt.subplots(figsize=(10, 8))
contour = ax.contourf(E_2d, levels=20)
ax.set_title('Electric Field Contour')
plt.colorbar(contour, label='E field')
plt.show()
```

## Time-Domain Visualization

### Field Evolution

```python
# Get field at multiple time steps
# Requires time-stepping during simulation
# See post_processing.md for details
```

### Field Animation

```python
# Create field animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Get field data
E = V.get_subfunction('E')
E_array = E.vector().get_array()

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(E_array[0], cmap='RdBu', origin='lower')
ax.set_title('Electric Field Evolution')

def update(frame):
    im.set_array(E_array[frame], cmap='RdBu', origin='lower')
    return [im]

ani = FuncAnimation(fig, update, frames=len(E_array), interval=50)
plt.show()
```

## Vector Field Visualization

### Vector Field Magnitude

```python
# Calculate vector field magnitude
S = E.cross(H)
S_mag = np.sqrt(np.abs(S)**2 + np.abs(H)**2)

# Plot vector field magnitude
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(S_mag, cmap='hot', origin='lower')
ax.set_title('Vector Field Magnitude')
ax.colorbar(im, label='|S|')
plt.show()
```

### Vector Field Direction

```python
# Calculate vector field direction
S = E.cross(H)
S_dir = S / np.abs(S)

# Plot vector field direction
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(S_dir, cmap='coolwarm', origin='lower')
ax.set_title('Vector Field Direction')
ax.colorbar(im, label='Direction')
plt.show()
```

## Multi-Field Plots

### Combined Field Plot

```python
# Plot multiple fields
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Electric field
E = V.get_subfunction('E')
E_array = E.vector().get_array()
im1 = axes[0, 0].imshow(E_array, cmap='RdBu', origin='lower')
axes[0, 0].set_title('Electric Field')
axes[0, 0].colorbar(im1, label='E field')

# Magnetic field
H = V.get_subfunction('H')
H_array = H.vector().get_array()
im2 = axes[0, 1].imshow(H_array, cmap='RdBu', origin='lower')
axes[0, 1].set_title('Magnetic Field')
axes[0, 1].colorbar(im2, label='H field')

plt.tight_layout()
plt.show()
```

## Advanced Visualization

### Field Slice Animation

```python
# Create field slice animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Get field data
E = V.get_subfunction('E')
E_array = E.vector().get_array()

# Create animation
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(E_array[0], cmap='RdBu', origin='lower')
ax.set_title('Field Slice Animation')

def update(frame):
    im.set_array(E_array[frame], cmap='RdBu', origin='lower')
    return [im]

ani = FuncAnimation(fig, update, frames=len(E_array), interval=50)
plt.show()
```

### 3D Volume Visualization

```python
# Create 3D volume visualization
from mpl_toolkits.mplot3d import Axes3D

# Get field data
E = V.get_subfunction('E')
E_array = E.vector().get_array()

# Create 3D visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot field at different z slices
for z_idx in range(0, E_array.shape[2], 5):
    ax.plot_surface(E_array[:, :, z_idx], 
                       z=z_idx*0.01, 
                       cmap='RdBu',
                       shade='auto')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Field Visualization')
plt.colorbar(label='E field')
plt.show()
```

## Best Practices

### Visualization Strategy

1. **Use appropriate colormaps**: Choose colormaps for field type
2. **Consider field range**: Focus on region of interest
3. **Use logarithmic scale**: Use dB scale for wide dynamic range
4. **Add colorbars**: Include colorbars for reference

### Performance Optimization

1. **Reduce output frequency**: Balance resolution and storage
2. **Use appropriate resolution**: Match mesh resolution
3. **Consider subampling**: Use subampling for smooth plots
4. **Use vectorization**: Use vectorized operations for efficiency

### Animation Considerations

1. **Frame rate**: Choose appropriate frame rate
2. **Memory management**: Handle large arrays carefully
3. **File format**: Consider saving to video formats
4. **Interactive vs. non-interactive**: Choose appropriate method

## Troubleshooting

### Visualization Errors

**Problem**: Plotting fails or produces incorrect results

**Solutions**:
1. Check field data is valid
2. Verify mesh and solution are compatible
3. Check visualization libraries are installed
4. Try different visualization methods

### Memory Issues

**Problem**: Out of memory errors during visualization

**Solutions**:
1. Reduce output frequency
2. Reduce field resolution
3. Use appropriate visualization method
4. Consider downsampling large arrays

### Performance Issues

**Problem**: Visualization is too slow

**Solutions**:
1. Reduce output frequency
2. Use appropriate resolution
3. Use vectorized operations
4. Consider using faster visualization libraries

### Display Issues

**Problem**: Plots don't display correctly

**Solutions**:
1. Check field data format
2. Verify matplotlib configuration
3. Try different plotting parameters
4. Check colorbar and colormap settings