# Advanced WRF-Python Usage

Complete guide to advanced features and performance tips.

## Performance Optimization

### Memory Management

**Using xarray for large files:**
```python
import xarray as xr

# Open netCDF file with xarray
ds = xr.open_dataset('wrfout_d01.nc')

# Lazy loading
t = ds['T']  # Not loaded until accessed

# Process in chunks
for time_idx in range(0, len(ds.time), 10):
    t_chunk = ds['T'][time_idx:time_idx+10]
    # Process chunk
```

**Chunk processing:**
```python
from wrf import getvar

# Process in time chunks
chunk_size = 10
n_chunks = len(time) // chunk_size

for chunk_idx in range(n_chunks):
    start = chunk_idx * chunk_size
    end = (chunk_idx + 1) * chunk_size
    
    # Get chunk
    t_chunk = getvar(ncfile, 'T', True, timeidx=slice(start, end))
    # Process chunk
```

### Parallel Processing

**Using OpenMP:**
```python
from wrf import getvar
import multiprocessing as mp

def process_time_step(time_idx):
    t = getvar(ncfile, 'T', True, timeidx=time_idx)
    # Process time step
    return t

# Create pool
pool = mp.Pool(processes=4)

# Process in parallel
results = pool.map(process_time_step, range(len(time)))
```

**Using MPI:**
```python
from mpi4py import MPI

# Get MPI rank
rank = MPI.COMM_WORLD.rank
size = MPI.COMM_WORLD.size

# Distribute work
local_time_steps = time_steps[rank::size]

# Process local time steps
for time_idx in local_time_steps:
    t = getvar(ncfile, 'T', True, timeidx=time_idx)
    # Process time step
```

### Vectorized Operations

**Using numpy vectorization:**
```python
import numpy as np

# Vectorized operations are faster
t_mean = np.mean(t, axis=(1, 2, 3))
t_std = np.std(t, axis=(1, 2, 3))
```

**Using dask for large datasets:**
```python
import dask.array as da

# Create dask array
t_da = da.from_array(t, chunks=(1, 1, 1, 10))

# Compute mean
t_mean = t_da.mean(axis=(1, 2, 3)).compute()
```

## Advanced Diagnostics

### Custom Diagnostics

**Defining custom diagnostic:**
```python
# See WRF-Python documentation
# Define custom diagnostic calculation
```

### Time Series Diagnostics

**Calculate diagnostics over time:**
```python
from wrf import getcape, srhel, uvmet

# Calculate time series
cape_series = []
srh_series = []
spd_series = []

for time_idx in range(len(time)):
    cape = getcape(ncfile, True, timeidx=time_idx)
    srh = srhel(ncfile, True, timeidx=time_idx)
    spd = uvmet(ncfile, 'uvmet', True, timeidx=time_idx)
    
    cape_series.append(cape.max())
    srh_series.append(srh.max())
    spd_series.append(spd.max())

# Plot time series
import matplotlib.pyplot as plt
plt.plot(time, cape_series, label='CAPE')
plt.plot(time, srh_series, label='SRH')
plt.plot(time, spd_series, label='Wind Speed')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
```

### Spatial Averaging

**Average over regions:**
```python
import numpy as np

# Define region
lat_slice = slice(30, 40)
lon_slice = slice(30, 40)

# Average over region
t_region = t[:, :, lat_slice, lon_slice].mean(axis=(1, 2))
print(f"Regional average temperature: {t_region:.2f} K")
```

**Weighted averaging:**
```python
# Weighted by area
lat = getvar(ncfile, 'XLAT', False)
weights = np.cos(np.radians(lat))

# Weighted average
t_weighted = (t * weights[np.newaxis, :, :]).sum(axis=(1, 2)) / weights.sum()
```

## Advanced Interpolation

### Multi-Variable Interpolation

**Interpolate multiple variables together:**
```python
from wrf import getvar, vertcross

# Get variables
t_p = getvar(ncfile, 'T', True)
u_p = getvar(ncfile, 'U', True)
v_p = getvar(ncfile, 'V', True)
h = getvar(ncfile, 'GHT', True)

# Interpolate all to model levels
t_ml = vertcross(t_p, h)
u_ml = vertcross(u_p, h)
v_ml = vertcross(v_p, h)
```

### Time-Dependent Interpolation

**Interpolate at each time step:**
```python
for time_idx in range(len(time)):
    t_p = getvar(ncfile, 'T', True, timeidx=time_idx)
    h = getvar(ncfile, 'GHT', True)
    
    t_ml = vertcross(t_p, h)
    # Process interpolated data
```

### Custom Interpolation

**Define custom interpolation method:**
```python
# See WRF-Python documentation
# Define custom interpolation routine
```

## Data Validation

### Quality Control

**Check for missing values:**
```python
import numpy as np

# Check for missing values
missing_count = np.sum(np.isnan(t))
print(f"Missing values: {missing_count}")

# Check for out-of-range values
invalid_count = np.sum(t < 100) + np.sum(t > 400)
print(f"Invalid temperature values: {invalid_count}")
```

**Check for physical consistency:**
```python
# Check temperature range
assert t.min() >= 100, "Temperature below minimum"
assert t.max() <= 400, "Temperature above maximum"

# Check pressure range
p = getvar(ncfile, 'P', True)
assert p.min() >= 100, "Pressure below minimum"
assert p.max() <= 110000, "Pressure above maximum"
```

### Cross-Validation

**Compare with observations:**
```python
# Load observations
obs_t = load_observations('temperature_obs.nc')

# Compare with model
bias = t.mean() - obs_t.mean()
rmse = np.sqrt(((t - obs_t) ** 2).mean())

print(f"Bias: {bias:.2f} K")
print(f"RMSE: {rmse:.2f} K")
```

## File I/O

### Reading Multiple Files

**Process ensemble of simulations:**
```python
import glob

# Find all WRF output files
files = glob.glob('wrfout_*.nc')

# Process each file
for file in files:
    ncfile = nc.Dataset(file, 'r')
    t = getvar(ncfile, 'T', True)
    # Process file
    ncfile.close()
```

### Writing NetCDF Files

**Create custom netCDF files:**
```python
import netCDF4 as nc

# Create output file
outfile = nc.Dataset('output.nc', 'w', format='NETCDF4')

# Create dimensions
outfile.createDimension('time', len(time))
outfile.createDimension('lat', len(lat))
outfile.createDimension('lon', len(lon))

# Create variables
t_var = outfile.createVariable('T', 'f4', ('time', 'lat', 'lon'))
t_var[:] = t

# Close file
outfile.close()
```

## Advanced Plotting

### Custom Color Maps

**Create custom colormap:**
```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create custom colormap
colors = ['#000080', '#0000ff', '#00ff00', '#ffff00', '#ff0000']
cmap = mcolors.ListedColormap(colors, N=100)

# Plot with custom colormap
im = ax.contourf(lon, lat, t[0, :, :], levels=20, cmap=cmap)
plt.colorbar(im, ax=ax)
plt.show()
```

### Interactive Plots

**Using matplotlib widgets:**
```python
from matplotlib.widgets import Slider, Button

# Create interactive plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.contourf(lon, lat, t[0, :, :], levels=20, cmap='coolwarm')
plt.colorbar(im, ax=ax)

# Add time slider
ax_time = plt.axes([0.1, 0.1, 0.1, 0.1])
slider = Slider(ax_time, 'Time', 0, len(time)-1, valinit=0, valstep=1)

def update(val):
    time_idx = int(val)
    ax.clear()
    im = ax.contourf(lon, lat, t[time_idx, :, :], levels=20, cmap='coolwarm')
    plt.colorbar(im, ax=ax)

slider.on_changed(update)
plt.show()
```

### Animation with Multiple Variables

**Animate multiple variables:**
```python
import matplotlib.animation as animation

def animate(frame):
    ax1.clear()
    ax2.clear()
    
    im1 = ax1.contourf(lon, lat, t[frame, :, :], levels=20, cmap='coolwarm')
    im2 = ax2.contourf(lon, lat, spd[frame, :, :], levels=20, cmap='viridis')
    
    ax1.set_title(f'Temperature at frame {frame}')
    ax2.set_title(f'Wind Speed at frame {frame}')
    
    return im1, im2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
anim = animation.FuncAnimation(fig, animate, frames=len(t), interval=200)
anim.save('multi_variable_animation.mp4', writer='ffmpeg', fps=5)
plt.show()
```

## Best Practices

### 1. Memory Management
- Use xarray for large files
- Process in chunks
- Close files after reading
- Use appropriate data types

### 2. Performance
- Use vectorized operations
- Process multiple time steps together
- Use parallel processing for large datasets
- Consider dask for very large datasets

### 3. Data Validation
- Check for missing values
- Validate physical ranges
- Cross-validate with observations
- Check array shapes and dimensions

### 4. Coordinate Systems
- Be aware of grid staggering
- Check coordinate ranges
- Validate interpolation results
- Handle cyclic boundaries

### 5. Documentation
- Document custom code
- Add comments for complex operations
- Keep track of variable names
- Note coordinate systems used

## Troubleshooting

### Performance Issues

**Issue:** Slow processing

**Solutions:**
- Use xarray for large files
- Process in chunks
- Use vectorized operations
- Consider parallel processing

### Memory Issues

**Issue:** Out of memory

**Solutions:**
- Use xarray for lazy loading
- Process in smaller chunks
- Close files after reading
- Reduce spatial resolution

### Interpolation Errors

**Issue:** Interpolation fails

**Solutions:**
- Check geopotential height availability
- Verify coordinate systems
- Check array shapes
- Review WRF model configuration

### Plotting Issues

**Issue:** Plots don't display correctly

**Solutions:**
- Check coordinate arrays
- Verify data ranges
- Check for missing values
- Use appropriate map projection

## Advanced Topics

### Custom Diagnostics

```python
# Define custom diagnostic calculations
# See WRF-Python documentation for details
```

### Ensemble Processing

```python
# Process ensemble of simulations
# Analyze statistics across ensemble
```

### Data Assimilation

```python
# Assimilate observations with model output
# Statistical analysis
```

## Resources

- WRF-Python advanced usage: https://wrf-python.readthedocs.io/en/latest/basic%5usage.html
- Performance tips: https://wrf-python.readthedocs.io/en/latest/basic%5usage.html#performance-tips
- FAQ: https://wrf-python.readthedocs.io/en/latest/faq.html
