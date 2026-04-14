# Performance Optimization with xesmf

## Overview

Performance optimization is crucial when working with large meteorological datasets. This guide covers techniques and best practices for optimizing xesmf performance.

## Chunking Optimization

### Optimal Chunk Size

```python
import xarray as xr
import numpy as np

# Calculate optimal chunk size based on memory
available_memory_gb = 8
target_memory_gb = available_memory_gb * 0.7

# For float32 data (4 bytes per value)
bytes_per_value = 4

# Calculate chunk size for 3D data
chunk_size = 10 * 90 * 180 * bytes_per_value
chunk_size_gb = chunk_size / (1024**3)

print(f"Chunk size: {chunk_size_gb:.2f} GB")
```

### Test Chunk Sizes

```python
import xarray as xr
import time

# Test different chunk sizes
chunk_sizes = [
    {'time': 5, 'latitude': 45, 'longitude': 90},
    {'time': 10, 'latitude': 90, 'longitude': 180},
    {'time': 20, 'latitude': 180, 'longitude': 360}
]

for chunks in chunk_sizes:
    ds = xr.open_dataset('large_file.nc', chunks=chunks)
    
    start_time = time.time()
    mean = ds['temperature'].mean().compute()
    elapsed_time = time.time() - start_time
    
    print(f"Chunks {chunks}: {elapsed_time:.2f} seconds")
```

### Automatic Chunking

```python
import xarray as xr

# Let xesmf determine optimal chunk sizes
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Check resulting chunks
print(ds['temperature'].chunks)
```

## I/O Optimization

### Use Efficient File Formats

```python
import xarray as xr

# Open Zarr for better performance
ds = xr.open_zarr('data.zarr')

# Or use NetCDF4 with chunking
ds = xr.open_dataset('data.nc', engine='netcdf4', chunks='auto')
```

### Optimize NetCDF Encoding

```python
import xarray as xr

ds = xr.open_dataset('input.nc', chunks='auto')

# Optimize encoding for performance
encoding = {
    'temperature': {
        'zlib': True,
        'complevel': 5,
        'chunksizes': (10, 90, 180),
        'fletcher32': True
    }
}

ds.to_netcdf('output.nc', encoding=encoding)
```

### Use Memory Mapping

```python
import xarray as xr

# Open with memory mapping
ds = xr.open_dataset('large_file.nc', 
                      chunks='auto',
                      engine='h5netcdf')
```

## Memory Optimization

### Reduce Memory Footprint

```python
import xarray as xr

# Open with small chunks
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 5, 'latitude': 45, 'longitude': 90})

# Convert to smaller data types
ds = ds.astype({'temperature': 'float32'})
```

### Selective Loading

```python
import xarray as xr

# Load only needed variables
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10},
                      drop_variables=['variable1', 'variable2'])

# Load only needed region
ds = ds.sel(latitude=slice(50, 30), longitude=slice(-120, -90))
```

### Lazy Evaluation

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

## Parallel Processing

### Dask Configuration

```python
from dask.distributed import Client
import xarray as xr

# Configure Dask for optimal performance
client = Client(n_workers=4,
                threads_per_worker=1,
                memory_limit='2GB',
                dashboard_address=':8787')

# Process with Dask
ds = xr.open_dataset('large_file.nc', chunks='auto')
result = ds['temperature'].mean().compute()

client.close()
```

### ThreadPoolExecutor

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_chunk(ds, chunk_idx):
    """Process a single chunk."""
    chunk = ds.isel(time=slice(chunk_idx, chunk_idx + 10))
    return chunk['temperature'].mean().compute()

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Process chunks in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    chunk_indices = range(0, len(ds.time), 10)
    results = list(executor.map(process_chunk, [ds] * len(chunk_indices), chunk_indices))

print(f"Processed {len(results)} chunks")
```

### ProcessPoolExecutor

```python
import xarray as xr
from concurrent.futures import ProcessPoolExecutor

def process_file(filename):
    """Process a single file."""
    ds = xr.open_dataset(filename, chunks='auto')
    return ds['temperature'].mean().compute()

files = ['file1.nc', 'file2.nc', 'file3.nc']

# Process files in parallel
with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))

print(f"Processed {len(results)} files")
```

## Computation Optimization

### Vectorized Operations

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Use vectorized operations instead of loops
result = np.sqrt(ds['temperature'] ** 2 + ds['humidity'] ** 2).compute()
```

### Avoid Unnecessary Copies

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Use in-place operations when possible
result = ds['temperature'].copy()
result[:] = result * 1.8 + 32  # Convert to Fahrenheit
```

### Use Efficient Algorithms

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Use efficient built-in methods
mean = ds['temperature'].mean().compute()
std = ds['temperature'].std().compute()

# Instead of manual calculation
# manual_mean = ds['temperature'].sum() / ds['temperature'].size
```

## Caching

### Cache Intermediate Results

```python
import xarray as xr
from joblib import Memory

# Setup cache
memory = Memory(location='./cache', verbose=0)

@memory.cache
def compute_mean(filename):
    """Compute mean with caching."""
    ds = xr.open_dataset(filename, chunks='auto')
    return ds['temperature'].mean().compute()

# First call (computes and caches)
result1 = compute_mean('large_file.nc')

# Second call (uses cache)
result2 = compute_mean('large_file.nc')
```

### Use xarray's Cache

```python
import xarray as xr

# Enable xarray's cache
xr.set_options(file_cache_maxsize=1000)

ds = xr.open_dataset('large_file.nc', chunks='auto')
result = ds['temperature'].mean().compute()
```

## Profiling

### Profile Operations

```python
import xarray as xr
import time

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Profile mean operation
start_time = time.time()
mean = ds['temperature'].mean().compute()
elapsed_time = time.time() - start_time

print(f"Mean computation: {elapsed_time:.2f} seconds")
```

### Use Dask Profiler

```python
from dask.distributed import Client, progress
import xarray as xr

client = Client(n_workers=4, threads_per_worker=1)

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Profile computation
with progress(ds['temperature'].mean().compute()):
    result = ds['temperature'].mean().compute()

client.close()
```

### Memory Profiling

```python
import xarray as xr
import psutil
import os

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Monitor memory usage
before = get_memory_usage()
result = ds['temperature'].mean().compute()
after = get_memory_usage()

print(f"Memory usage: {after - before:.2f} GB")
```

## Performance Tuning

### Tune Dask Scheduler

```python
from dask.distributed import Client
import xarray as xr

# Tune Dask scheduler
client = Client(n_workers=4,
                threads_per_worker=1,
                memory_limit='2GB',
                silence_logs=False)

# Configure scheduler
client.cluster.scale(4)  # Scale to 4 workers

ds = xr.open_dataset('large_file.nc', chunks='auto')
result = ds['temperature'].mean().compute()

client.close.close()
```

### Optimize Chunking for Operation

```python
import xarray as xr

# For time series operations
ds_time = xr.open_dataset('large_file.nc', chunks={'time': 10})

# For spatial operations
ds_space = xr.open_dataset('large_file.nc', 
                            chunks={'latitude': 90, 'longitude': 180})

# For mixed operations
ds_mixed = xr.open_dataset('large_file.nc', 
                           chunks={'time': 10, 'latitude': 90, 'longitude': 180})
```

### Use Efficient Data Types

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Convert to efficient data types
ds = ds.astype({
    'temperature': 'float32',
    'humidity': 'float32',
    'pressure': 'float32',
    'land_mask': 'int16'
})
```

## Best Practices

1. **Profile first** - Understand performance bottlenecks before optimizing
2. **Use appropriate chunking** - Choose chunk sizes based on your operation
3. **Leverage parallel processing** - Use Dask or concurrent processing
4. **Optimize I/O** - Use efficient file formats and encoding
5. **Minimize memory usage** - Use selective loading and smaller data types
6. **Use lazy evaluation** - Leverage xarray's lazy evaluation
7. **Cache intermediate results** - Avoid recomputing expensive operations
8. **Monitor performance** - Track performance metrics during optimization
9. **Test different strategies** - Experiment with different approaches
10. **Document optimizations** - Keep track of what works best
