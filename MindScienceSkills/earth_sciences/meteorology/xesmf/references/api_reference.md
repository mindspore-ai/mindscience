# xesmf and xarray API Reference

## Overview

This reference provides a comprehensive guide to the xesmf and xarray APIs for working with large meteorological datasets.

## xarray API

### Dataset Operations

#### Opening Datasets

```python
import xarray as xr

# Open NetCDF file
ds = xr.open_dataset('file.nc')

# Open with chunking
ds = xr.open_dataset('file.nc', chunks='auto')

# Open with manual chunking
ds = xr.open_dataset('file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Open with engine specification
ds = xr.open_dataset('file.nc', engine='netcdf4')

# Open Zarr dataset
ds = xr.open_zarr('data.zarr')

# Open multiple files
ds = xr.open_mfdataset('file_*.nc', chunks='auto')
```

#### Saving Datasets

```python
import xarray as xr

# Save to NetCDF
ds.to_netcdf('output.nc')

# Save with encoding
ds.to_netcdf('output.nc', encoding={
    'temperature': {
        'zlib': True,
        'complevel': 5,
        'chunksizes': (10, 90, 180)
    }
})

# Save to Zarr
ds.to_zarr('output.zarr')

# Save to Zarr with encoding
ds.to_zarr('output.zarr', encoding={
    'temperature': {
        'chunksizes': (10, 90, 180)
    }
})
```

### DataArray Operations

#### Selection

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks='auto')

# Select by coordinate
data = ds['temperature'].sel(latitude=40.0, longitude=-100.0)

# Select by index
data = ds['temperature'].isel(time=0, latitude=100, longitude=200)

# Select with slicing
data = ds['temperature'].sel(
    latitude=slice(50, 30),
    longitude=slice(-120, -90)
)

# Select with method
data = ds['temperature'].sel(latitude=40.0, method='nearest')
```

#### Computation

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks='auto')

# Arithmetic operations
result = ds['temperature'] + 273.15  # Convert to Kelvin
result = ds['temperature'] * 1.8 + 32  # Convert to Fahrenheit

# Mathematical operations
result = np.sqrt(ds['temperature'])
result = np.log(ds['temperature'])

# Aggregation operations
mean = ds['temperature'].mean()
std = ds['temperature'].std()
min_val = ds['temperature'].min()
max_val = ds['temperature'].max()
sum_val = ds['temperature'].sum()

# Dimension-specific aggregation
mean_time = ds['temperature'].mean(dim='time')
mean_spatial = ds['temperature'].mean(dim=['latitude', 'longitude'])
```

#### Grouping

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks='auto')

# Group by month
monthly = ds.groupby('time.month')

# Group by season
seasonal = ds.groupby('time.season')

# Group by custom function
def custom_group(time):
    return time.dt.month % 3

grouped = ds.groupby(custom_group)
```

#### Rolling Operations

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks='auto')

# Rolling mean
rolling_mean = ds['temperature'].rolling(time=7, center=True).mean()

# Rolling standard deviation
rolling_std = ds['temperature'].rolling(time=7, center=True).std()

# Rolling sum
rolling_sum = ds['precipitation'].rolling(time=7, center=True).sum()
```

### Coordinate Operations

#### Working with Coordinates

```python
import xarray as xr

ds = = xr.open_dataset('file.nc', chunks='auto')

# Access coordinates
time_coords = ds.time
lat_coords = ds.latitude
lon_coords = ds.longitude

# Add coordinates
ds = ds.assign_coords(new_coord=range(len(ds.time)))

# Drop coordinates
ds = ds.drop_coords('old_coord')

# Rename coordinates
ds = ds.rename({'old_name': 'new_name'})
```

#### Time Coordinates

```python
import xarray as xr
import pandas as pd

ds = xr.open_dataset('file.nc', chunks='auto')

# Access time properties
years = ds.time.dt.year
months = ds.time.dt.month
days = ds.time.dt.day
hours = ds.time.dt.hour

# Select by time
data = ds.sel(time='2020-01-01')
data = ds.sel(time=slice('2020-01-01', '2020-12-31'))

# Convert time units
ds['time'] = pd.to_datetime(ds['time'].values)
```

## xesmf API

### Regridding

#### Basic Regridding

```python
import xarray as xr
import xesmf as xe

# Open source dataset
ds = xr.open_dataset('source.nc', chunks='auto')

# Define target grid
target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(30, 50, 50)),
    'lon': (['lon'], np.linspace(-120, -90, 50))
})

# Create regridder
regridder = xe.Regridder(ds, target_grid, method='bilinear')

# Regrid data
regridded = regridder(ds['temperature'])
```

#### Regridding Methods

```python
import xesmf as xe

# Bilinear interpolation
regridder = xe.Regridder(ds, target_grid, method='bilinear')

# Conservative regridding
regridder = xe.Regridder(ds, target_grid, method='conservative')

# Nearest neighbor
regridder = xe.Regridder(ds, target_grid, method='nearest')

# Patch recovery
regridder = xe.Regridder(ds, target_grid, method='patch')
```

#### Regridding Options

```python
import xesmf as xe

# Create regridder with options
regridder = xe.Regridder(
    ds, 
    target_grid, 
    method='bilinear',
    periodic=False,
    ignore_degenerate=True,
    reuse_weights=True
)

# Regrid with options
regridded = regridder(
    ds['temperature'],
    keep_attrs=True
)
```

### Spatial Operations

#### Spatial Averaging

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks='auto')

# Global spatial mean
global_mean = ds['temperature'].mean(dim=['latitude', 'longitude'])

# Zonal mean
zonal_mean = ds['temperature'].mean(dim='longitude')

# Meridional mean
meridional_mean = ds['temperature'].mean(dim='latitude')

# Regional mean
regional_mean = ds['temperature'].sel(
    latitude=slice(50, 30),
    longitude=slice(-120, -90)
).mean(dim=['latitude', 'longitude'])
```

#### Spatial Interpolation

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks='auto')

# Define new coordinates
new_lat = np.linspace(30, 50, 50)
'new_lon = np.linspace(-120, -90, 50)

# Nearest neighbor interpolation
interp_nearest = ds['temperature'].interp(
    latitude=new_lat,
    longitude=new_lon,
    method='nearest'
)

# Linear interpolation
interp_linear = ds['temperature'].interp(
    latitude=new_lat,
    longitude=new_lon,
    method='linear'
)

# Cubic interpolation
interp_cubic = ds['temperature'].interp(
    latitude=new_lat,
    longitude=new_lon,
    method='cubic'
)
```

## Dask Integration

### Dask Arrays

```python
import xarray as xr
import dask.array as da

# Open with Dask
ds = xr.open_dataset('file.nc', chunks='auto')

# Access Dask array
dask_array = ds['temperature'].data

# Check if data is Dask array
is_dask = isinstance(ds['temperature'].data, da.Array)

# Compute Dask array
result = ds['temperature'].compute()
```

### Dask Operations

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks='auto')

# Lazy operations
mean = ds['temperature'].mean()
std = ds['temperature'].std()

# Compute operations
mean_result = mean.compute()
std_result = std.compute()

# Persist operations
persisted = ds['temperature'].persist()
```

### Dask Scheduler

```python
from dask.distributed import Client
import xarray as xr

# Start Dask client
client = Client(n_workers=4,
                threads_per_worker=1,
                memory_limit='2GB',
                dashboard_address=':8787')

# Process with Dask
ds = xr.open_dataset('file.nc', chunks='auto')
result = ds['temperature'].mean().compute()

# Close client
client.close()
```

## Utility Functions

### Memory Management

```python
import xarray as xr
import psutil
import os

# Get memory usage
def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

# Monitor memory
before = get_memory_usage()
ds = xr.open_dataset('file.nc', chunks='auto')
result = = ds['temperature'].mean().compute()
after = get_memory_usage()

print(f"Memory usage: {after - before:.2f} GB")
```

### Progress Monitoring

```python
import xarray as xr
from tqdm import tqdm

# Process with progress bar
ds = xr.open_dataset('file.nc', chunks={'time': 10})

results = []
for i in tqdm(range(0, len(ds.time), 10)):
    chunk = ds.isel(time=slice(i, i+10))
    result = chunk['temperature'].mean().compute()
    results.append(result)
```

### Error Handling

```python
import xarray as xr

try:
    ds = xr.open_dataset('file.nc', chunks='auto')
    result = ds['temperature'].mean().compute()
except Exception as e:
    print(f"Error: {e}")
    # Handle error
```

## Common Patterns

### Pattern 1: Process Large File

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(0, i+10))
    result = chunk['temperature'].mean().compute()
    print(f"Chunk {i//10 + 1}: {result:.2f}")
```

### Pattern 2: Parallel Processing

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_chunk(ds, chunk_idx):
    chunk = ds.isel(time=slice(chunk_idx, chunk_idx + 10))
    return chunk['temperature'].mean().compute()

ds = xr.open_dataset('large_file.nc', chunks='auto')

with ThreadPoolExecutor(max_workers=4) as executor:
    chunk_indices = range(0, len(ds.time), 10)
    results = list(executor.map(process_chunk, [ds] * len(chunk_indices), chunk_indices))
```

### Pattern 3: Time Series Analysis

```python
import xarray as xr

ds = xr.open_dataset('file.nc', chunks={'time': 10})

ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

for i in range(0, len(ts.time), 10):
    = chunk = ts.isel(time=slice(i, i+10))
    chunk_mean = chunk.mean().compute()
    print(f"Time step {i}: {chunk_mean:.2f}")
```

## Additional Resources

- xarray documentation: https://xarray.pydata.org/
- xesmf documentation: https://xesmf.readthedocs.io/
- Dask documentation: https://docs.dask.org/
