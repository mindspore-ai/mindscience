# Visualization with Py-ART

This guide covers visualization techniques and plotting examples for radar data.

## Basic Plots

### PPI Plot

Create Plan Position Indicator (PPI) plot:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Create display
display = pyart.graph.RadarDisplayar(radar)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot reflectivity
display.plot('reflectivity', ax=ax, vmin=-30, vmax=70, cmap='NWSpectral')

# Add colorbar
plt.colorbar(ax.collections[0], ax=ax, label='Reflectivity (dBZ)')

# Add title
ax.set_title('Radar Reflectivity PPI')

plt.tight_layout()
plt.savefig('ppi.png', dpi=300, bbox_inches='tight')
plt.show()
```

### CAPPI Plot

Create Constant Altitude PPI plot:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Create display
display = pyart.graph.RadarDisplayar(radar)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot CAPPI at 2 km altitude
display.plot_ppi('reflectivity', 2000, ax=ax, vmin=-30, vmax=70, cmap='NWSpectral')

# Add colorbar
plt.colorbar(ax.collections[0], ax=ax, label='Reflectivity (dBZ)')

# Add title
ax.set_title('Radar Reflectivity CAPPI (2 km)')

plt.tight_layout()
plt.savefig('cappi.png', dpi=300, bbox_inches='tight')
plt.show()
```

### RHI Plot

Create Range-Height Indicator (RHI) plot:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Create display
display = pyart.graph.RadarDisplayar(radar)

# Create figure
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111)

# Plot RHI for sweep 0
display.plot_rhi('reflectivity', 0, ax=ax, vmin=-30, vmax=70, cmap='NWSpectral')

# Add colorbar
plt.colorbar(ax.collections[0], ax=ax, label='Reflectivity (dBZ)')

# Add title
ax.set_title('Radar Reflectivity RHI (Sweep 0)')

plt.tight_layout()
plt.savefig('rhi.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Polarimetric Plots

### ZDR Plot

Plot differential reflectivity:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for ZDR field
if 'differential_reflectivity' not in radar.fields:
    print("Error: ZDR field not found")
    exit(1)

# Create display
display = pyart.graph.RadarDisplayar(radar)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot ZDR
display.plot('differential_reflectivity', ax=ax, vmin=-5, vmax=5, cmap='coolwarm')

# Add colorbar
plt.colorbar(ax.collections[0], ax=ax, label='ZDR (dB)')

# Add title
ax.set_title('Differential Reflectivity (ZDR)')

plt.tight_layout()
plt.savefig('zdr.png', dpi=300, bbox_inches='tight')
plt.show()
```

### RHOHV Plot

Plot cross-correlation ratio:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for RHOHV field
if 'cross_correlation_ratio' not in radar.fields:
    print("Error: RHOHV field not found")
    exit(1)

# Create display
display = pyart.graph.RadarDisplayar(radar)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot RHOHV
display.plot('cross_correlation_ratio', ax=ax, vmin=0.5, vmax=1.0, cmap='viridis')

# Add colorbar
plt.colorbar(ax.collections[0], ax=ax, label='RHOHV')

# Add title
ax.set_title('Cross-Correlation Ratio (RHOHV)')

plt.tight_layout()
plt.savefig('rhohv.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Derived Quantities Plots

### Rain Rate Plot

Plot retrieved rain rate:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Estimate rain rate
radar_rain = pyart.retrieve.est_rain_rate_zr(radar)

# Create display
display = pyart.graph.RadarDisplayar(radar_rain)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot rain rate
display.plot('rain_rate', ax=ax, vmin=0, vmax=100, cmap='Blues')

# Add colorbar
plt.colorbar(ax.collections[0], ax=ax, label='Rain Rate (mm/hr)')

# Add title
ax.set_title('Rain Rate (Z-R Relationship)')

plt.tight_layout()
plt.savefig('rain_rateRE.png', dpi=300, bbox_inches='tight')
plt.show()
```

### KDP Plot

Plot specific differential phase:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Compute KDP
radar_kdp = pyart.retrieve.compute_kdp(radar)

# Create display
display = pyart.graph.RadarDisplayar(radar_kdp)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot KDP
display.plot('specific_differential_phase', ax=ax, vmin=-2, vmax=2, cmap='seismic')

# Add colorbar
plt.colorbar(ax.collections[0], ax=ax, label='KDP (deg/km)')

# Add title
ax.set_title('Specific Differential Phase (KDP)')

plt.tight_layout()
plt.savefig('kdp.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Multi-Field Plots

### Field Comparison

Compare multiple radar fields:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Create display
display = pyart.graph.RadarDisplayar(radar)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot reflectivity
display.plot('reflectivity', ax=axes[0, 0], vmin=-30, vmax=70, cmap='NWSpectral')
axes[0, 0].set_title('Reflectivity')

# Plot velocity
display.plot('velocity', ax=axes[0, 1], vmin=-30, vmax=30, cmap='RdBu_r')
raxes[0, 1].set_title('Velocity')

# Plot spectrum width
display.plot('spectrum_width', ax=axes[1, 0], vmin=0, vmax=10, cmap='viridis')
axes[1, 0].set_title('Spectrum Width')

# Plot differential reflectivity (if available)
if 'differential_reflectivity' in radar.fields:
    display.plot('differential_reflectivity', ax=axes[1, 1], vmin=-5, vmax=5, cmap='coolwarm')
    axes[1, 1].set_title('Differential Reflectivity')

plt.tight_layout()
plt.savefig('multi_field.png', dpi=300, bbox_inches='tight')
plt.show()
```

### Time Series Comparison

Compare multiple sweeps:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Create display
display = pyart.graph.RadarDisplayar(radar)

# Create figure
fig = plt.figure(figsize=(15, 6))
ax = fig.add_subplot(111)

# Plot first 3 sweeps
for i in range(min(3, radar.nsweeps)):
    display.plot('reflectivity', i, ax=ax, vmin=-30, vmax=70, 
              alpha=0.7, label=f'Sweep {i}')

ax.set_xlabel('Range (km)')
ax.set_ylabel('Azimuth (degrees)')
ax.set_title('Reflectivity - Multiple Sweeps')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('multi_sweep.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Hydrometeor Classification Plots

### Classification Plot

Plot hydrometeor classification:

```python
import pyart
import matplotlib.pyplot as plt
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Classify hydrometeors
radar_hydro = pyart.retrieve.hydroclass_hs(
    radar, 
    field_name='reflectivity',
    hydro_class='HS'
)

# Create display
display = pyart.graph.RadarDisplayar(radar_hydro)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot classification
display.plot('hydro_class', ax=ax, vmin=0, vmax=10, cmap='tab10')

# Add colorbar
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Hydrometeor Class')
cbar.set_ticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
cbar.set_ticklabels(['Drizzle', 'Rain', 'Snow', 'Ice Crystals', 
                   'Mixed', 'Graupel', 'Dry Snow', 'Wet Snow', 'Ice',
                   'Unknown'])

ax.set_title('Hydrometeor Classification')

plt.tight_layout()
plt.savefig('hydro_class.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Cartesian Grid Plots

### Grid Plot

Plot converted Cartesian grid:

```python
import pyart
import matplotlib.pyplot as plt
import xarray as xr

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert to Cartesian grid
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    edge_factor=0.0
)

# Convert to xarray
ds = pyart.io.to_xarray(grid)

# Create figure
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111)

# Plot grid
ds['reflectivity'].plot(ax=ax, cmap='NWSpectral', vmin=-30, vmax=70)

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_title('Reflectivity Grid')

plt.tight_layout()
plt.savefig('grid.png', dpi=300, bbox_inches='tight')
plt.show()
```

## Best Practices

### 1. Use Appropriate Color Scales

```python
# Reflectivity
display.plot('reflectivity', ax=ax, cmap='NWSpectral')

# Velocity
display.plot('velocity', ax=ax, cmap='RdBu_r')

# Differential reflectivity
display.plot('differential_reflectivity', ax=ax, cmap='coolwarm')

# Spectrum width
display.plot('spectrum_width', ax=ax, cmap='viridis')
```

### 2. Set Appropriate Value Ranges

```python
# Reflectivity (dBZ)
display.plot('reflectivity', ax=ax, vmin=-30, vmax=70)

# Velocity (m/s)
display.plot('velocity', ax=ax, vmin=-30, vmax=30)

# Spectrum width (m/s)
display.plot('spectrum_width', ax=ax, vmin=0, vmax=10)

# ZDR (dB)
display.plot('differential_reflectivity', ax=ax, vmin=-5, vmax=5)
```

### 3. Add Colorbars

```python
# Add colorbar with label
plt.colorbar(ax.collections[0], ax=ax, label='Reflectivity (dBZ)')
```

### 4. Use High DPI for Publication

```python
# Save with high DPI
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

### 5. Add Descriptive Titles

```python
# Add title with field information
ax.set_title(f'{field_name} - {radar.radar_type}')
```

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf
- matplotlib Documentation: https://matplotlib.org/stable/contents.html