# Parallel Processing with xesmf

## Overview

Parallel processing can significantly speed up large dataset operations. This guide covers parallel processing techniques and best practices.

## Dask Integration

### Basic Dask Setup

```python
import xarray as xr
import dask

# Open with Dask
ds = xr.open_dataset('large_file.nc', chunks='auto')

# Operations are lazy
mean = ds['temperature'].mean()

# Compute in parallel
result = mean.compute()
```

### Dask Client Configuration

```python
from dask.distributed import Client
import xarray as xr

# Start Dask client
client = Client(n_workers=4, 
                threads_per_worker=1,
                memory_limit='2GB')

# Process with Dask
ds = xr.open_dataset('large_file.nc', chunks='auto')
result = ds['temperature'].mean().compute()

client.close()
```

### Dask Dashboard

```python
from dask.distributed import Client

# Start client with dashboard
client = Client(n_workers=4,
                threads_per_worker=1,
                memory_limit='2GB',
                dashboard_address=':8787')

# Access dashboard at http://localhost:8787
```

## Multi-File Processing

### Sequential Processing

```python
import xarray as xr

files = ['file1.nc', 'file2.nc', 'file3.nc']
results = []

for file in files:
    ds = xr.open_dataset(file, chunks='auto')
    result = ds['temperature'].mean().compute()
    results.append(result)
    print(f"Processed {file}: {result:.2f}")
```

### Parallel File Processing

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_file(filename):
    """Process a single file."""
    ds = xr.open_dataset(filename, chunks='auto')
    result = ds['temperature'].mean().compute()
    return result

files = ['file1.nc', 'file2.nc', 'file3.nc']

# Process files in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_file, files))

print(f"Processed {len(results)} files")
```

### Async File Processing

```python
import xarray as xr
import asyncio

async def process_file(filename):
    """Process a single file asynchronously."""
    ds = xr.open_dataset(filename, chunks='auto')
    result = ds['temperature'].mean().compute()
    return result

async def process_files(files):
    """Process multiple files asynchronously."""
    tasks = [process_file(file) for file in files]
    return await asyncio.gather(*tasks)

files = ['file1.nc', 'file2.nc', 'file3.nc']
results = asyncio.run(process_files(files))
```

## Chunk Parallel Processing

### Sequential Chunk Processing

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Process chunks sequentially
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    result = chunk['temperature'].mean().compute()
    print(f"Chunk {i//10 + 1}: {result:.2f}")
```

### Parallel Chunk Processing

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_chunk(ds, chunk_idx, chunk_size=10):
    """Process a single chunk."""
    chunk = ds.isel(time=slice(chunk_idx, chunk_idx + chunk_size))
    return chunk['temperature'].mean().compute()

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Process chunks in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    chunk_indices = range(0, len(ds.time), 10)
    results = list(executor.map(process_chunk, [ds] * len(chunk_indices), chunk_indices))

print(f"Processed {len(results)} chunks")
```

### Dask Chunk Processing

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Define chunk processing function
def process_chunk(chunk):
    return chunk['temperature'].mean()

# Map over chunks
results = ds['temperature'].chunk(dict(time=10)).map_blocks(process_chunk)
final_results = results.compute()
```

## Region Parallel Processing

### Parallel Regional Analysis

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_region(ds, region):
    """Process a single region."""
    region_data = ds.sel(latitude=region['lat'], longitude=region['lon'])
    return region_data['temperature'].mean().compute()

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Define regions
regions = [
    {'name': 'North America', 'lat': slice(50, 25), 'lon': slice(-125, -65)},
    {'name': 'Europe', 'lat': slice(70, 35), 'lon': slice(-10, 40)},
    {'name': 'Asia', 'lat': slice(70, 10), 'lon': slice(60, 150)}
]

# Process regions in parallel
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(process_region, [ds] * len(regions), regions))

for region, result in zip(regions, results):
    print(f"{region['name']}: {result:.2f}")
```

## Performance Optimization

### Worker Configuration

```python
from dask.distributed import Client
import psutil

# Get number of CPU cores
n_cores = psutil.cpu_count()

# Calculate optimal workers
n_workers = max(1, n_cores // 2)
memory_per_worker = '2GB'

# Start client
client = Client(n_workers=n_workers,
                threads_per_worker=2,
                memory_limit=memory_per_worker)
```

### Task Scheduling

```python
import xarray as xr
from dask.distributed import Client

client = Client(n_workers=4, threads_per_worker=1)

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Submit multiple tasks
mean_task = client.submit(ds['temperature'].mean().compute)
std_task = client.submit(ds['temperature'].std().compute)
min_task = client.submit(ds['temperature'].min().compute)
max_task = client.submit(ds['temperature'].max().compute)

# Wait for results
mean = mean_task.result()
std = std_task.result()
min_val = min_task.result()
max_val = max_task.result()

client.close()
```

### Batch Processing

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_batch(files, batch_size=10):
    """Process files in batches."""
    results = []
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            batch_results = list(executor.map(process_file, batch))
            results.extend(batch_results)
        
        print(f"Processed batch {i//batch_size + 1}")
    
    return results
```

## Error Handling

### Robust Parallel Processing

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_file_safe(filename):
    """Process a file with error handling."""
    try:
        ds = xr.open_dataset(filename, chunks='auto')
        result = ds['temperature'].mean().compute()
        return {'file': filename, 'result': result, 'success': True}
    except Exception as e:
        return {'file': filename, 'error': str(e), 'success': False}

files = ['file1.nc', 'file2.nc', 'file3.nc']

# Process with error handling
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_file_safe, file) for file in files]
    
    for future in as_completed(futures):
        result = future.result()
        if result['success']:
            print(f"Success: {result['file']} = {result['result']:.2f}")
        else:
            print(f"Error: {result['file']} - {result['error']}")
```

### Retry Logic

```python
import xarray as xr
import time

def process_file_with_retry(filename, max_retries=3):
    """Process file with retry logic."""
    for attempt in range(max_retries):
        try:
            ds = xr.open_dataset(filename, chunks='auto')
            result = ds['temperature'].mean().compute()
            return result
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Retry {attempt + 1} for {filename}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
```

## Troubleshooting

### Performance Issues

**Problem**: Parallel processing is slower than sequential

**Solutions**:
1. Increase chunk size
2. Reduce number of workers
3. Check for I/O bottlenecks
4. Optimize chunking strategy

### Memory Issues

**Problem**: Out of memory errors in parallel processing

**Solutions**:
1. Reduce memory per worker
2. Reduce number of workers
3. Use smaller chunks
4. Process fewer files in parallel

### Worker Failures

**Problem**: Workers fail during processing

**Solutions**:
1. Implement retry logic
2. Add error handling
3. Monitor worker health
4. Check resource limits

## Best Practices

1. **Start with sequential processing** - Establish baseline performance
2. **Monitor resource usage** - Track CPU, memory, and I/O
3. **Use appropriate chunking** - Choose chunk sizes for parallel processing
4. **Handle errors gracefully** - Implement robust error handling
5. **Use Dask dashboard** - Monitor processing in real-time
6. **Optimize worker configuration** - Tune workers for your system
7. **Process in batches** - Balance parallelism with resource limits
8. **Profile performance** - Identify bottlenecks
