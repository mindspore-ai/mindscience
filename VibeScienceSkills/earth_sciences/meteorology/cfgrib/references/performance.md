# Performance Optimization with cfgrib

This guide covers performance optimization and memory management techniques for cfgrib and xarray.

## Lazy Evaluation

### Understanding Lazy Evaluation

cfgrib uses lazy evaluation - data is not loaded until needed:

```python
import xarray as xr

# Data is not loaded yet
ds = xr.open_dataset('file.grib', engine='cfgrib')

# Operations are lazy
mean_temp = ds['t2m'].mean()  # Still not loaded

# Data is loaded when you access values
print(mean_temp.values)  # Now data is loaded
```

### Explicit Loading

Control when data is loaded:

```python
import xarray as xr

# Load entire dataset
ds = xr.open_dataset('file.grib', engine='cfgrib')
ds_loaded = ds.load()

# Load specific variable
ds = xr.open_dataset('file.grib', engine='cfgrib')
temp_loaded = ds['t2m'].load()

# Load specific slice
ds = xr.open_dataset('file.grib', engine='cfgrib')
temp_slice = ds['t2m'].isel(time=0).load()
```

## Chunking

### Automatic Chunking

Let xarray determine optimal chunks:

```python
import xarray as xr

# Automatic chunking
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')

# Check chunks
print(ds['t2m'].chunks)
```

### Manual Chunking

Specify chunk sizes manually:

```python
import xarray as xr

# Manual chunking
ds = xr.open_dataset('file.grib', engine='cfgrib', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Check chunks
print(ds['t2m'].chunks)
```

### Rechunking

Change chunk sizes after loading:

```python
import xarray as xr

# Load with chunking
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')

# Rechunk
ds_chunked = ds.chunk({'time': 5, 'latitude': 45, 'longitude': 90})

# Check new chunks
print(ds_chunked['t2m'].chunks)
```

## Memory Management

### Processing Large Files

Process large files in chunks:

```python
import xarray as xr

# Load with chunking
ds = xr.open_dataset('large_file.grib', engine='cfgrib', 
                      chunks={'time': 10})

# Process in chunks
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    
    # Process chunk
    result = chunk['t2m'].mean()
    
    # Save or process result
    print(f"Processed chunk {i//10 + 1}")
```

### Reducing Memory Usage

Reduce memory footprint:

```python
import xarray as xr

# Load only needed variables
ds = xr.open_dataset('file.grib', engine='cfgrib')
temp = ds['t2m'].load()  # Only loads t2m

# Use smaller data types
ds = xr.open_dataset('file.grib', engine='cfgrib')
temp = ds['t2m'].astype('float32')  # 32-bit instead of 64-bit

# Drop unused coordinates
ds = xr.open_dataset('file.grib', engine='cfgrib')
temp = ds['t2m'].reset_coords(drop=True)
```

### Freeing Memory

Explicitly free memory:

```python
import xarray as xr
import gc

# Load and process
ds = xr.open_dataset('file.grib', engine='cfgrib')
result = ds['t2m'].mean().load()

# Close dataset
ds.close()

# Force garbage collection
gc.collect()
```

## Using Dask

### Dask Integration

Use Dask for parallel processing:

```python
import xarray as xr

# Load with Dask
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')

# Operations are lazy
mean_temp = ds['t2m'].mean()

# Compute with Dask
result = mean_temp.compute()
```

### Dask Scheduler

Configure Dask scheduler:

```python
import xarray as xr
import dask

# Configure Dask
dask.config.set(scheduler='threads', num_workers=4)

# Load with Dask
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')

# Compute
result = ds['t2m'].mean().compute()
```

### Dask Dashboard

Monitor Dask computations:

```python
import xarray as xr
from dask.distributed import Client

# Start Dask client
client = Client(processes=False)

# Load with Dask
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')

# Compute (dashboard available at http://localhost:8787)
result = ds['t2m'].mean().compute()

# Close client
client.close()
```

## Performance Optimization

### Vectorized Operations

Use vectorized operations instead of loops:

```python
import xarray as xr

# Good: Vectorized
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    mean = ds['t2m'].mean(dim=['latitude', 'longitude'])

# Bad: Loop-based
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    total = 0
    count = 0
    for lat in ds.latitude:
        for lon in ds.longitude:
            total += ds['t2m'].sel(latitude=lat, longitude=lon)
            count += 1
    mean = total / count
```

### Efficient Selection

Use efficient selection methods:

```python
import xarray as xr

# Good: Coordinate-based selection
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    region = ds['t2m'].sel(
        latitude=slice(50, 30),
        longitude=slice(-120, -90)
    )

# Bad: Boolean mask
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    mask = (ds.latitude >= 30) & (ds.latitude <= 50) & \
            (ds.longitude >= -120) & (ds.longitude <= -90)
    region = ds['t2m'].where(mask)
```

### Groupby Optimization

Optimize groupby operations:

```python
import xarray as xr

# Good: Pre-select data
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    temp = ds['t2m'].sel(latitude=slice(50, 30))
    monthly = temp.groupby('time.month').mean()

# Bad: Groupby on entire dataset
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    monthly = ds['t2m'].groupby('time.month').mean()
```

## Caching

### Enable Caching

Cache frequently accessed data:

```python
import xarray as xr

# Enable caching
xr.set_options(file_cache_maxsize=1e9)  # 1 GB

# Load dataset
ds = xr.open_dataset('file.grib', engine='cfgrib')

# Data will be cached on first access
temp = ds['t2m'].load()

# Subsequent accesses use cache
temp_again = ds['t2m'].load()
```

### Disable Caching

Disable caching for one-time reads:

```python
import xarray as xr

# Disable caching
xr.set_options(file_cache_maxsize=0)

# Load dataset
ds = xr.open_dataset('file.grib', engine='cfgrib')
temp = ds['t2m'].load()
```

## Parallel Processing

### Multi-File Processing

Process multiple files in parallel:

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_file(filename):
    """Process a single file."""
    with xr.open_dataset(filename, engine='cfgrib') as ds:
        return ds['t2m'].mean().load()

# Process multiple files
files = ['file1.grib', 'file2.grib', 'file3.grib']

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))

print(f"Processed {len(results)} files")
```

### Parallel Operations

Use parallel operations with Dask:

```python
import xarray as xr

# Load with Dask
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')

# Parallel operations
mean = ds['t2m'].mean(dim='time')
std = ds['t2m'].std(dim='time')

# Compute in parallel
mean_result = mean.compute()
std_result = std.compute()
```

## I/O Optimization

### NetCDF Optimization

Optimize NetCDF writing:

```python
import xarray as xr

# Read GRIB
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Write with compression
    encoding = {
        var: {
            'zlib': True,
            'complevel': 5,
            'shuffle': True
        }
        for var in ds.data_vars
    }
    
    ds.to_netcdf('output.nc', encoding=encoding)
```

### Zarr Optimization

Optimize Zarr writing:

```python
import xarray as xr

# Read GRIB
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Write with optimal chunking
    encoding = {
        var: {
            'chunksizes': (10, 90, 180)
        }
        for var in ds.data_vars
    }
    
    ds.to_zarr('output.zarr', encoding=encoding)
```

## Performance Profiling

### Time Operations

Measure operation time:

```python
import xarray as xr
import time

# Time operation
start = time.time()

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    result = ds['t2m'].mean().load()

end = time.time()
print(f"Operation took {end - start:.2f} seconds")
```

### Memory Profiling

Profile memory usage:

```python
import xarray as xr
import psutil
import os

def get_memory_usage():
    """Get current memory usage."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Profile memory
mem_before = get_memory_usage()

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    result = ds['t2m'].mean().load()

mem_after = get_memory_usage()
print(f"Memory used: {mem_after - mem_before:.2f} MB")
```

## Best Practices

### 1. Use Context Managers

```python
# Good: Context manager
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    result = ds['t2m'].mean()

# Bad: Manual cleanup
ds = xr.open_dataset('file.grib', engine='cfgrib')
result = ds['t2m'].mean()
ds.close()
```

### 2. Use Appropriate Chunking

```python
# Good: Appropriate chunking
ds = xr.open_dataset('file.grib', engine='cfgrib', 
                      chunks={'time': 10})

# Bad: No chunking
ds = xr.open_dataset('file.grib', engine='cfgrib')
```

### 3. Load Only Needed Data

```python
# Good: Load only needed variable
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    temp = ds['t2m'].load()

# Bad: Load entire dataset
ds = xr.open_dataset('file.grib', engine='cfgrib').load()
temp = ds['t2m']
```

### 4. Use Vectorized Operations

```python
# Good: Vectorized
mean = ds['t2m'].mean(dim=['latitude', 'longitude'])

# Bad: Loop-based
# (see example above)
```

### 5. Monitor Memory Usage

```python
# Monitor memory
import psutil
import os

process = psutil.Process(os.getpid())
mem_usage = process.memory_info().rss / 1024 / 1024
print(f"Memory usage: {mem_usage:.2f} MB")
```

## Common Performance Issues

### Issue: High Memory Usage

**Solution:** Use chunking and process in pieces
```python
ds = xr.open_dataset('large_file.grib', engine='cfgrib', 
                      chunks={'time': 10})
```

### Issue: Slow Performance

**Solution:** Use Dask for parallel processing
```python
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')
result = ds['t2m'].mean().compute()
```

### Issue: Slow Repeated Access

**Solution:** Enable caching
```python
xr.set_options(file_cache_maxsize=1e9)
```

### Issue: Slow File Writing

**Solution:** Use appropriate compression and chunking
```python
encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
ds.to_netcdf('output.nc', encoding=encoding)
```

## References

- xarray Performance: https://xarray.pydata.org/stable/performance.html
- Dask Documentation: https://docs.dask.org/
- psutil Documentation: https://psutil.readthedocs.io/