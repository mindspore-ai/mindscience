---
name: xesmf
description: Comprehensive skill for working with xesmf (xarray extensions for large datasets) to efficiently process meteorological data. Use when Claude needs to: (1) Process large GRIB/NetCDF files that don't fit in memory, (1) Process large GRIB/NetCDF files that don't fit in memory, (2) Read and process datasets in chunks, (3) Use out-of-core computation for large arrays, (4) Apply operations to large datasets efficiently, (5) Handle datasets larger than available RAM, (6) Perform parallel processing on large datasets, (7) Optimize I/O for large datasets, (8) Work with time series data efficiently, (9) Process multi-dimensional arrays efficiently, (10) Use lazy evaluation for large datasets
---

# xesmf

## Overview

xesmf provides xarray extensions for handling large datasets that don't fit in memory. It enables out-of-core computation, lazy evaluation, and efficient processing of meteorological data that would otherwise exceed available RAM.

## Quick Start

**Opening a large file with chunking:**

```python
import xarray as xr
import xesmf as xm

# Open with automatic chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Operations are lazy
mean = ds['temperature'].mean()
result = mean.compute()  # Only loads data when needed
```

**Processing in chunks:**

```python
import xarray as xr
import xesmf as xm

# Open with manual chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Process in chunks
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    chunk_mean = chunk.mean().compute()
    print(f"Chunk {i//10}: {chunk_mean:.2f}")
```

## Workflow Decision Tree

1. **What type of operation do you need?**
   - **Processing large files** → Follow "Large File Processing" workflow
   - **Memory optimization** → Follow "Memory Management" workflow
   - **Parallel processing** → Follow "Parallel Processing" workflow
   - **Time series analysis** → Follow "Time Series Analysis" workflow
   - **Spatial operations** → Follow "Spatial Operations" workflow
   - **Statistical computation** → Follow "Statistical Computation" workflow
   - **Data conversion** → Follow "Data Conversion" workflow

## Large File Processing

### Automatic Chunking

Let xesmf determine optimal chunk sizes:

```python
import xarray as xr

# Automatic chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Check chunks
print(ds['temperature'].chunks)
```

### Manual Chunking

Specify chunk sizes manually:

```python
import xarray as xr

# Manual chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Check chunks
print(ds['temperature'].chunks)
```

### Dimension-Based Chunking

Chunk along specific dimensions:

```python
import xarray as xr

# Chunk along time only
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10})

# Chunk along spatial dimensions
ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Chunk all dimensions
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})
```

### Time-Based Chunking

Chunk based on time periods:

```python
import xarray as xr
import pandas as pd

# Read with time chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Process by month
for month in range(1, 13):
    month_data = ds.sel(time=ds.time.dt.month == month)
    month_mean = month_data.mean().compute()
    print(f"Month {month}: {month_mean:.2f}")
```

See `references/chunking_strategies.md` for detailed chunking strategies.

## Memory Management

### Processing in Chunks

Process large datasets in chunks to avoid memory issues:

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Process in chunks
chunk_means = []
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    chunk_mean = chunk['temperature'].mean().compute()
    chunk_means.append(chunk_mean)
    print(f"Processed chunk {i//10 + 1}: {len(chunk.time)} time steps")

# Combine results
overall_mean = np.mean(chunk_means)
print(f"Overall mean: {overall_mean:.2f}")
```

### Reducing Memory Footprint

Reduce memory usage with smaller chunks:

```python
import xarray as xr

# Smaller chunks for limited memory
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 5, 'latitude': 45, 'longitude': 90})

# Use smaller data types
ds = ds.astype({'temperature': 'float32'})
```

### Selective Loading

Load only needed data:

```python
import xarray as xr

# Load only specific variables
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10},
                      drop_variables=['variable1', 'variable2'])

# Load only specific region
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10})
ds = ds.sel(latitude=slice(50, 30), longitude=slice(-120, -90))
```

## Parallel Processing

### Dask Integration

Use Dask for parallel processing:

```python
import xarray as xr

# Open with Dask
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Parallel operations
mean_temp = ds['temperature'].mean()
std_temp = ds['temperature'].std()

# Compute in parallel
mean_result = mean_temp.compute()
std_result = std_temp.compute()

print(f"Mean: {mean_result:.2f}")
print(f"Std: {std_result:.2f}")
```

### Multi-File Parallel Processing

Process multiple files in parallel:

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_file(filename):
    """Process a single file."""
    ds = xr.open_dataset(filename, chunks='auto')
    result = ds['temperature'].mean().compute()
    return result

# Process multiple files
files = ['file1.nc', 'file2.nc', 'file3.nc']

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))

print(f"Processed {len(results)} files")
for i, result in enumerate(results):
    print(f"File {i+1}: {result:.2f}")
```

### Parallel Chunk Processing

Process chunks in parallel:

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_chunk(ds, chunk_idx):
time_range):
    """Process a single chunk."""
    chunk = ds.isel(time=slice(chunk_idx, chunk_idx + 10))
    return chunk['temperature'].mean().compute()

# Open with chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Process chunks in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    chunk_indices = range(0, len(ds.time), 10)
    results = list(executor.map(process_chunk, [ds] * len(chunk_indices), chunk_indices))

print(f"Processed {len(results)} chunks")
```

See `references/parallel_processing.md` for detailed parallel processing techniques.

## Time Series Analysis

### Efficient Time Series Processing

Process time series without loading entire dataset:

```python
import xarray as xr

# Open with time chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series for a location
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Process time series in chunks
for i in range(0, len(ts.time), 10):
    chunk = ts.isel(time=slice(i, i+10))
    chunk_mean = chunk.mean().compute()
    print(f"Time step {i}: {chunk_mean:.2f}")
```

### Time Series Statistics

Calculate time series statistics efficiently:

```python
import xarray as xr

# Open with time chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate statistics in chunks
chunk_means = []
chunk_stds = []

for i in range(0, len(ts.time), 10):
    chunk = ts.isel(time=slice(i, i+10))
    chunk_means.append(chunk.mean().compute())
    chunk_stds.append(chunk.std().compute())

# Combine results
overall_mean = np.mean(chunk_means)
overall_std = np.mean(chunk_stds)

print(f"Time series mean: {overall_mean:.2f}")
print(f"Time series std: {overall_std:.2f}")
```

### Time Series Aggregation

Aggregate time series efficiently:

```python
import xarray as xr

# Open with time chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate daily means
daily_means = []
for i in range(0, len(ts.time), 40):  # Assuming 6-hourly data
    day_chunk = ts.isel(time=slice(i, i+40))
    daily_mean = day_chunk.mean().compute()
    daily_means.append(daily_mean)

print(f"Number of days: {len(daily_means)}")
```

## Spatial Operations

### Efficient Spatial Averaging

Calculate spatial averages in chunks:

```python
import xarray as xr

# Open with spatial chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Calculate spatial means in chunks
for i in range(0, ds.latitude.size, 90):
    lat_chunk = ds.isel(latitude=slice(i, i+90))
    spatial_mean = lat_chunk.mean(dim='longitude').compute()
    print(f"Latitude chunk {i//90 + 1}: {spatial_mean:.2f}")
```

### Regional Analysis

Analyze regions efficiently:

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Select region
region = ds.sel(latitude=slice(50, 30), longitude=slice(-120, -90))

# Process region in time chunks
for i in range(0, len(region.time), 10):
    time_chunk = region.isel(time=slice(i, i+10))
    chunk_mean = time_chunk.mean().compute()
    print(f"Time chunk {i//10 + 1}: {chunk_mean:.2f}")
```

### Zonal Averaging

Calculate zonal averages efficiently:

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Calculate zonal means in time chunks
for i in range(0, len(ds.time), 10):
    time_chunk = ds.isel(time=slice(i, i+10))
    zonal_mean = time_chunk.mean(dim='longitude').compute()
    print(f"Time chunk {i//10 + 1}: {zonal_mean:.2f}")
```

## Statistical Computation

### Global Statistics

Calculate global statistics efficiently:

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate global mean
mean = ds['temperature'].mean().compute()

# Calculate global std
std = ds['temperature'].std().compute()

# Calculate global min/max
min_val = ds['temperature'].min().compute()
max_val = ds['display('temperature'].max().compute()

print(f"Mean: {mean:.2f}")
print(f"Std: {std:.2f}")
print(f"Min: {min_val:.2f}")
print(f"Max: {max_val:.2f}")
```

### Percentiles

Calculate percentiles efficiently:

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate percentiles
p25 = ds['temperature'].quantile(0.25).compute()
p50 = ds['temperature'].quantile(0.50).compute()
p75 = ds['temperature'].quantile(0.75).compute()
p90 = ds['temperature'].quantile(0.90).compute()

print(f"25th percentile: {p25:.2f}")
print(f"50th percentile: {p50:.2f}")
print(f"75th percentile: {p75:.2f}")
print(f"90th percentile: {p90:.2f}")
```

### Grouping Operations

Perform grouping operations efficiently:

```python
import xarray as xr

# Open with time chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Group by month
monthly_means = []
for month in range(1, 13):
    month_data = ds.sel(time=ds.time.dt.month == month)
    month_mean = month_data.mean().compute()
    monthly_means.append(month_mean)

print(f"Monthly means: {monthly_means}")
```

## Data Conversion

### Converting with Chunking

Convert large files with chunking:

```python
import xarray as xr

# Open with chunking
ds = xr.to_netcdf('large_file.nc', chunks='auto')

# Write with compression
encoding = {
    'temperature': {
        'zlib': True, 
        'complevel': 5,
        'chunksizes': (10, 90, 180)
    }
}

ds.to_netcdf('output.nc', encoding=encoding)
```

### Zarr Conversion

Convert to Zarr format for cloud storage:

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Write to Zarr with chunking
ds.to_zarr('output.zarr', encoding={
    'temperature': {
        'chunksizes': (10, 90, 180)
    }
})
```

### CSV Conversion

Convert to CSV with chunking:

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Convert to CSV in chunks
for i in range(0, len(ds.time), 10):
    time_chunk = ds.isel(time=slice(i, i+10))
    
    # Convert chunk to CSV
    df = time_chunk.to_dataframe()
    df.to_csv(f'chunk_{i}.csv')
    
    print(f"Converted chunk {i//10 + 1} to CSV")
```

## Resources

### scripts/
Executable Python scripts for common xesmf operations:

- **process_large_file.py** - Process large GRIB/NetCDF files in chunks
- **chunk_analysis.py** - Analyze data in chunks
- **parallel_processing.py** - Process data in parallel
- **time_series_analysis.py** - Analyze time series efficiently
- **spatial_analysis.py** - Perform spatial operations on large datasets
- **statistical_computation.py** - Calculate statistics on large datasets
- **convert_format.py** - Convert large files to other formats
- **memory_optimization.py** - Optimize memory usage for large datasets
- **chunk_statistics.py** - Calculate statistics by chunk
- **validate_chunks.py** - Validate chunked data processing

### references/
Detailed documentation and reference materials:

- **chunking_strategies.md** - Detailed chunking strategies and best practices
- **memory_management.md** - Memory management techniques for large datasets
- **parallel_processing.md** - Parallel processing methods and patterns
- **time_series_analysis.md** - Efficient time series analysis techniques
- **spatial_operations.md** - Spatial operations on large datasets
- **statistical_computation.md** - Statistical computation on large datasets
- **data_conversion.md** - Data conversion techniques for large datasets
- **performance_optimization.md** - Performance optimization and tuning
- **api_reference.md** - Complete xesmf and xarray API reference
- **common_use_cases.md** - Real-world use cases and patterns

### assets/
Example files and templates:

- **sample_large_dataset.nc** - Sample large NetCDF file for testing
- **config_template.yaml** - Configuration template for xesmf operations
- **chunk_template.py** - Template for chunked data processing
- **analysis_template.py** - Template for large dataset analysis workflows

## Best Practices

1. **Use appropriate chunking** - Choose chunk sizes based on available memory
2. **Process in chunks** - Avoid loading entire datasets into memory
3. **Use parallel processing** - Leverage multiple cores for faster processing
4. **Load only needed data** - Select regions and variables to reduce memory
5. **UseCompute() method** - Trigger computation when results are needed
6. **Handle missing data** - Use masked arrays appropriately
7. **Monitor memory usage** - Track memory consumption during processing
8. **Document chunking strategy** - Keep track of chunking decisions

## Integration with Other Tools

### xarray Integration

xesmf extends xarray with chunking capabilities:

```python
import xarray as xr
import xesmf as xm

# Open with xesmf chunking
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Use all xarray operations
mean = ds['temperature'].mean()
std = ds['temperature'].std()

# Compute when needed
result = mean.compute()
```

### Dask Integration

xesmf works seamlessly with Dask for parallel processing:

```python
import xarray as xr

# Open with Dask
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Parallel operations
mean = ds['temperature'].mean()
std = ds['temperature'].std()

# Compute in parallel
mean_result = mean.compute()
std_result = std.compute()
```

### Pandas Integration

Convert chunks to pandas DataFrames:

```python
import xarray as xr
import pandas as pd

# Open with chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Convert to pandas in chunks
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    df = chunk.to_dataframe()
    df.to_csv(f'chunk_{i}.csv')
```

## Troubleshooting

### Common Issues

**"Memory error"**
- Use smaller chunks
- Process in smaller pieces
- Reduce number of parallel workers

**"Slow performance"**
- Increase chunk size
- Use parallel processing
- Optimize chunking strategy

**"Out of memory" error**
- Reduce chunk size
- Drop unused variables
- Use selective loading

**"Chunking not working"**
- Check data dimensions
- Verify chunk sizes are valid
- Check for incompatible operations

See `references` for detailed troubleshooting guide.