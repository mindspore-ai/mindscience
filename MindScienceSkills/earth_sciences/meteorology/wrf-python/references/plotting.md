# Plotting with WRF-Python

Complete guide to plotting WRF model output with cartopy, basemap, and PyNGL.

## Plotting Packages

### Cartopy

**Basic plot:**
```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Create map
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Add coastlines
ax.coastlines()
ax.gridlines()

# Plot temperature
ax.contourf(lon, lat, t[0, :, :], levels=20, cmap='coolwarm')
ax.set_title('Temperature (K)')
plt.colorbar(label='Temperature (K)')
plt.show()
```

**With features:**
```python
# Add features
ax.add_feature(ccrs.BORDERS, scale='10m')
ax.add_feature(ccrs.COASTLINES)
ax.add_feature(ccrs.STATES, scale='10m')

# Add country borders
ax.add_feature(ccrs.BORDERS.with_scale('10m'), facecolor='none', edgecolor='black')
```

### Basemap

**Basic plot:**
```python
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Create map
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection=basemap.Basemap())

# Plot temperature
ax.contourf(lon, lat, t[0, :, :], levels=20, cmap='coolwarm')
ax.set_title('Temperature (K)')
plt.colorbar(label='Temperature (K)')
plt.show()
```

**With features:**
```python
# Add features
ax.drawcoastlines()
ax.drawcountries()
ax.drawstates()
```

### PyNGL

**Basic 3D plot:**
```python
import pyngl as gl
from pyngl.gl import GL
import numpy as np

# Initialize
gl.glutInit()
gl.glutInitDisplayMode(800, 600, 'Temperature')

# Create figure
fig = gl.GLFigure(800, 600)
ax = fig.add_subplot(111, projection='3d')

# Plot 3D temperature
ax.plot_surface(lon, lat, z, t[0, :, :], cmap='coolwarm')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Pressure (hPa)')
ax.set_title('Temperature (K)')

plt.show()
```

## Common Plot Types

### Horizontal Maps

**Temperature map:**
```python
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# Plot temperature
im = ax.contourf(lon, lat, t[0, :, :], levels=20, cmap='coolwarm')
plt.colorbar(im, ax=ax, label='Temperature (K)')

ax.set_title('Temperature at 500 hPa')
ax.coastlines()
ax.gridlines()
plt.show()
```

**Wind speed map:**
```python
# Plot wind speed
im = ax.contourf(lon, lat, spd[0, :, :], levels=20, cmap='viridis')
plt.colorbar(im, ax=ax, label='Wind Speed (m/s)')

ax.set_title('Wind Speed at 500 hPa')
```

**CAPE map:**
```python
# Plot CAPE
im = ax.contourf(lon, lat, cape[0, :, :], levels=20, cmap='plasma')
plt.colorbar(im, ax=ax, label='CAPE (J/kg)')

ax.set_title('CAPE (J/kg)')
```

### Vertical Cross Sections

**Temperature cross section:**
```python
import matplotlib.pyplot as plt

# Extract cross section
lat_idx = 50
lon_idx = 50

t_cross = t_ml[:, :, lat_idx, lon_idx]
h_cross = h[:, :, lat_idx, lon_idx]

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(h_cross[0, :], t_cross[0, :])
ax.set_xlabel('Geopotential Height (m)')
ax.set_ylabel('Temperature (K)')
ax.set_title('Vertical Cross Section')
ax.grid(True)
plt.show()
```

**Wind speed cross section:**
```python
# Plot wind speed cross section
spd_cross = spd_ml[:, :, lat_idx, lon_idx]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(h_cross[0, :], spd_cross[0, :])
ax.set_xlabel('Geopotential Height (m)')
ax.set_ylabel('Wind Speed (m/s)')
ax.set_title('Wind Speed Cross Section')
ax.grid(True)
plt.show()
```

### Skew-T Plots

**Temperature skew-T:**
```python
import matplotlib.pyplot as plt

# Plot temperature skew-T
fig, ax = plt.subplots(figsize=(10, 6))
ax.contourf(lon, t[0, 10, :, :], levels=20, cmap='coolwarm')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Temperature Skew-T')
plt.colorbar(label='Temperature (K)')
plt.show()
```

**Wind speed skew-T:**
```python
# Plot wind speed skew-T
fig, ax = plt.subplots(figsize=(10, 6))
ax.contourf(lon, spd[0, 10, :, :], levels=20, cmap='viridis')
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Pressure (hPa)')
ax.set_title('Wind Speed Skew-T')
plt.colorbar(label='Wind Speed (m/s)')
plt.show()
```

### Wind Vectors

**Quiver plot:**
```python
import matplotlib.pyplot as plt

# Plot wind vectors
fig, ax = plt.subplots(figsize=(12, 8))
skip = 10  # Skip every 10th point
ax.quiver(lon[::skip, ::skip], lat[::skip, ::skip], 
          u[0, 10, ::skip, ::skip], v[0, 10, ::skip, ::skip])
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.set_title('Wind Vectors at 500 hPa')
plt.show()
```

**Streamlines:**
```python
# Plot streamlines
fig, ax = plt.subplots(figsize=(12, 8))
ax.streamplot(lon, lat, u[0, 10, :, :], v[0, 10, :, :], 
              density=2, color='blue')
ax.set_title('Streamlines at 500 hPa')
plt.show()
```

## Multi-Panel Figures

**Multiple variables:**
```python
import matplotlib.pyplot as plt

# Create multi-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Temperature
im1 = axes[0, 0].contourf(lon, lat, t[0, :, :], levels=20, cmap='coolwarm')
axes[0, 0].set_title('Temperature (K)')
plt.colorbar(im1, ax=axes[0, 0])

# Wind speed
im2 = axes[0, 1].contourf(lon, lat, spd[0, :, :], levels=20, cmap='viridis')
axes[0, 1].set_title('Wind Speed (m/s)')
plt.colorbar(im2, ax=axes[0, 1])

# CAPE
im3 = axes[1, 0].contourf(lon, lat, cape[0, :, :], levels=20, cmap='plasma')
axes[1, 0].set_title('CAPE (J/kg)')
plt.colorbar(im3, ax=axes[1, 0])

# SRH
im4 = axes[1, 1].contourf(lon, lat, srh[0, :, :], levels=20, cmap='autumn')
axes[1, 1].set_title('Storm Relative Helicity')
plt.colorbar(im4, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

## Animation

**Time series animation:**
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Animation function
def animate(frame):
    ax.clear()
    im = ax.contourf(lon, lat, t[frame, :, :], levels=20, cmap='coolwarm')
    ax.set_title(f'Temperature at frame {frame}')
    return im

# Create animation
anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=200)
anim.save('temperature_animation.mp4', writer='ffmpeg', fps=5)
plt.show()
```

## Best Practices

### 1. Map Projections
- Use appropriate projection for region
- Set map extent explicitly
- Add relevant features (coastlines, borders)
- Use appropriate scale

### 2. Color Maps
- Use perceptually uniform colormaps (viridis, plasma, coolwarm)
- Avoid rainbow colormaps
- Choose appropriate range for data
- Use diverging colormaps for signed data

### 3. Figure Size
- Use appropriate aspect ratio
- Consider publication requirements
- Ensure readable labels
- Use consistent font sizes

### 4. Labels and Titles
- Include units in labels
- Use descriptive titles
- Add colorbar labels
- Include time information

### 5. Performance
- Use appropriate resolution
- Consider downsampling for large datasets
- Use efficient plotting methods
- Save to appropriate formats

## Resources

- Cartopy documentation: https://scitools.org/cartopy/
- Basemap documentation: https://matplotlib.org/basemap/stable/api.html
- PyNGL documentation: https://pyngl.sourceforge.net/
- WRF-Python plotting: https://wrf-python.readthedocs.io/en/latest/plot.html
