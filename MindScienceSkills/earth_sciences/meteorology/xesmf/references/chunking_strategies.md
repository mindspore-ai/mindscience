# Chunking Strategies for xesmf

## Overview

Effective chunking is critical for performance when working with large datasets. This guide covers chunking strategies, best practices, and optimization techniques.

## Chunk Size Guidelines

### Memory-Based Chunking

Calculate chunk sizes based on available memory:

```python
import numpy as np

# Available memory in GB
available_memory_gb = 8

# Target memory per chunk (use 50-80% of available memory)
target_memory_gb = available_memory_gb * 0.7

# For float32 data (4 bytes per value)
bytes_per_value = 4

# Calculate chunk size for 3D data (time, lat, lon)
# Example: 10 time steps, 90 lat, 180 lon
chunk_size = 10 * 90 * 180 * bytes_per_value
chunk_size_gb = chunk_size / (1024**3)

print(f"Chunk size: {chunk_size_gb:.2f} GB")
```

### Rule of Thumb

- **Small datasets (< 1GB)**: No chunking needed
- **Medium datasets (1-10GB)**: Chunk along time dimension (10-50 time steps)
- **Large datasets (10-100GB)**: Multi-dimensional chunking (10 time, 90 lat, 180 lon)
- **Very large datasets (> 100GB)**: Smaller chunks (5 time, 45 lat, 90 lon)

## Chunking Strategies

### Time-Based Chunking

Best for time series analysis:

```python
import xarray as xr

# Chunk along time dimension
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Process time series
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    # Process chunk
```

### Spatial Chunking

Best for spatial analysis:

```python
import xarray as xr

# Chunk along spatial dimensions
ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Process spatial chunks
for i in range(0, ds.latitude.size, 90):
    lat_chunk = ds.isel(latitude=slice(i, i+90))
    # Process chunk
```

### Balanced Chunking

Best for general-purpose processing:

```python
import xarray as xr

# Balanced multi-dimensional chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})
```

### Automatic Chunking

Let xesmf determine optimal chunk sizes:

```python
import xarray as xr

# Automatic chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Check resulting chunks
print(ds['temperature'].chunks)
```

## Chunking for Different Operations

### Statistical Operations

```python
# For mean, std, min, max operations
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Operations are efficient with automatic chunking
mean = ds['temperature'].mean()
result = mean.compute()
```

### Time Series Analysis

```python

# For time series extraction and analysis
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')
```

### Spatial Operations

```python
# For spatial averaging and regional analysis
ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Spatial operations
spatial_mean = ds['temperature'].mean(dim=['latitude', 'longitude'])
```

## Performance Optimization

### Chunk Size Tuning

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

### Compression with Chunking

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Write with compression and chunking
encoding = {
    'temperature': {
        'zlib': True,
        'complevel': 5,
        'chunksizes': (10, 90, 180)
    }
}

ds.to_netcdf('output.nc', encoding=encoding)
```

## Common Patterns

### Pattern 1: Process by Time Period

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Process by month
for month in range(1, 13):
    month_data = ds.sel(time=ds.time.dt.month == month)
    month_mean = month_data.mean().compute()
    print(f"Month {month}: {month_mean:.2f}")
```

### Pattern 2: Process by Region

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Define regions
regions = [
    {'name': 'North America', 'lat': slice(50, 25), 'lon': slice(-125, -65)},
    {'name': 'Europe', 'lat': slice(70, 35), 'lon': slice(-10, 40)},
    {'name': 'Asia', 'lat': slice(70, 10), 'lon': slice(60, 150)}
]

for region in regions:
    region_data = ds.sel(latitude=region['lat'], longitude=region['lon'])
    region_mean = region_data.mean().compute()
    print(f"{region['name']}: {region_mean:.2f}")
```

### Pattern 3: Parallel Chunk Processing

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

## Troubleshooting

### Memory Errors

**Problem**: Out of memory errors during processing

**Solutions**:
1. Reduce chunk size
2. Process fewer chunks in parallel
3. Drop unused variables
4. Use selective loading

### Slow Performance

**Problem**: Processing is slower than expected

**Solutions**:
1. Increase chunk size
2. Use parallel processing
3. Optimize chunking strategy for your operation
4. Consider using Zarr format for better performance

### Inconsistent Results

**Problem**: Results vary between runs

**Solutions**:
1. Ensure consistent chunking strategy
2. Use deterministic operations
3. Check for floating-point precision issues
4. Validate data integrity

## Best Practices

1. **Start with automatic chunking** - Let xesmf determine optimal chunk sizes
2. **Monitor memory usage** - Track memory consumption during processing
3. **Test different strategies** - Experiment with chunk sizes for your use case
4. **Document your strategy** - Keep track of what works best
5. **Use compression** - Compress output files to save space
6. **Process in parallel** - Leverage multiple cores when possible
7. **Load only needed data** - Select regions and variables to reduce memory
8. **Handle missing data** - Use masked arrays appropriately
