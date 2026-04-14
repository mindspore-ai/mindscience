# Memory Management for Large Datasets

## Overview

Memory management is critical when working with large meteorological datasets. This guide covers techniques for optimizing memory usage and avoiding out-of-memory errors.

## Memory Monitoring

### Check Memory Usage

```python
import psutil
import os

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

print(f"Memory usage: {get_memory_usage():.2f} GB")
```

### Monitor During Processing

```python
import xarray as xr
import psutil
import os

def monitor_memory(func):
    """Decorator to monitor memory usage."""
    def wrapper(*args, **kwargs):
        before = get_memory_usage()
        result = func(*args, **kwargs)
        after = get_memory_usage()
        print(f"Memory change: {after - before:.2f} GB")
        return result
    return wrapper

@monitor_memory
def process_data(ds):
    return ds['temperature'].mean().compute()
```

## Memory Reduction Techniques

### Selective Loading

Load only needed variables:

```python
import xarray as xr

# Load only specific variables
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10},
                      drop_variables=['variable1', 'variable2'])

# Load only needed region
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10})
ds = ds.sel(latitude=slice(50, 30), longitude=slice(-120, -90))
```

### Data Type Conversion

Use smaller data types:

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Convert to float32
ds = ds.astype({'temperature': 'float32'})

# Convert to int16 for categorical data
ds = ds.astype({'land_mask': 'int16'})
```

### Lazy Evaluation

Use lazy evaluation to avoid loading data unnecessarily:

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Operations are lazy
mean = ds['temperature'].mean()
std = ds['temperature'].std()

# Only compute when needed
mean_result = mean.compute()
std_result = std.compute()
```

## Chunking for Memory

### Small Chunks

Use small chunks for limited memory:

```python
import xarray as xr

# Very small chunks for limited memory
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 5, 'latitude': 45, 'longitude': 90})
```

### Time-Based Chunking

Process one time step at a time:

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 1})

# Process one time step at a time
for i in range(len(ds.time)):
    time_step = ds.isel(time=i)
    result = time_step['temperature'].mean().compute()
    print(f"Time step {i}: {result:.2f}")
```

## Processing in Chunks

### Manual Chunk Processing

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Process in chunks
chunk_means = []
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    chunk_mean = chunk['temperature'].mean().compute()
    chunk_means.append(chunk_mean)
    print(f"Processed chunk {i//10 + 1}")
```

### Accumulate Results

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Accumulate statistics
sum_temp = 0
count = 0

for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    chunk_sum = chunk['temperature'].sum().compute()
    sum_temp += chunk_sum
    count += chunk['temperature'].size

overall_mean = sum_temp / count
print(f"Overall mean: {overall_mean:.2f}")
```

## Memory Cleanup

### Explicit Cleanup

```python
import xarray as xr
import gc

# Process data
ds = xr.open_dataset('large_file.nc', chunks='auto')
result = ds['temperature'].mean().compute()

# Explicit cleanup
del ds
gc.collect()
```

### Context Manager

```python
import xarray as xr

def process_large_file(filename):
    """Process large file with automatic cleanup."""
    with xr.open_dataset(filename, chunks='auto') as ds:
        result = ds['temperature'].mean().compute()
    return result
```

## Advanced Techniques

### Memory-Mapped Files

Use memory-mapped files for very large datasets:

```python
import xarray as xr

# Open with memory mapping
ds = xr.open_dataset('large_file.nc', 
                      chunks='auto',
                      engine='h5netcdf')
```

### Zarr Format

Use Zarr for cloud-optimized chunked storage:

```python
import xarray as xr

# Convert to Zarr
ds = xr.open_dataset('large_file.nc', chunks='auto')
ds.to_zarr('output.zarr')

# Open Zarr dataset
ds_zarr = xr.open_zarr('output.zarr')
```

### Dask Scheduler

Configure Dask scheduler for memory management:

```python
import dask
from dask.distributed import Client

# Start Dask client with memory limits
client = Client(n_workers=4, 
                threads_per_worker=1,
                memory_limit='2GB')

# Process with Dask
import xarray as xr
ds = xr.open_dataset('large_file.nc', chunks='auto')
result = ds['temperature'].mean().compute()

client.close()
```

## Troubleshooting

### Out of Memory Errors

**Problem**: Out of memory errors during processing

**Solutions**:
1. Reduce chunk size
2. Process fewer chunks in parallel
3. Drop unused variables
4. Use selective loading
5. Convert to smaller data types

### Memory Leaks

**Problem**: Memory usage increases over time

**Solutions**:
1. Explicitly delete variables
2. Use garbage collection
3. Use context managers
4. Process in smaller chunks
5. Monitor memory usage

### Slow Performance

**Problem**: Processing is slow due to memory constraints

**Solutions**:
1. Increase chunk size
2. Use parallel processing
3. Optimize chunking strategy
4. Use memory-mapped files
5. Consider Zarr format

## Best Practices

1. **Monitor memory usage** - Track memory consumption during processing
2. **Use appropriate chunking** - Choose chunk sizes based on available memory
3. **Load only needed data** - Select regions and variables to reduce memory
4. **Process in chunks** - Avoid loading entire datasets into memory
5. **Use smaller data types** - Convert to float32 or int16 when possible
6. **Explicit cleanup** - Delete variables and call garbage collection
7. **Use context managers** - Ensure resources are cleaned up properly
8. **Consider Zarr format** - Use Zarr for cloud-optimized storage
