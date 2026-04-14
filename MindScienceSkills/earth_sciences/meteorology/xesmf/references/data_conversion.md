# Data Conversion with xesmf

## Overview

Data conversion is often necessary when working with meteorological data from different sources. This guide covers techniques for converting large datasets between different formats with xesmf.

## NetCDF Conversion

### NetCDF to NetCDF

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

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

### NetCDF to NetCDF4

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Write to NetCDF4 format
ds.to_netcdf('output.nc4', engine='netcdf4')
```

### NetCDF to NetCDF Classic

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Write to NetCDF classic format
ds.to_netcdf('output_classic.nc', format='NETCDF3_CLASSIC')
```

## Zarr Conversion

### NetCDF to Zarr

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Write to Zarr
ds.to_zarr('output.zarr', encoding={
    'temperature': {
        'chunksizes': (10, 90, 180)
    }
})
```

### Zarr to NetCDF

```python
import xarray as xr

# Open Zarr dataset
ds = xr.open_zarr('input.zarr')

# Write to NetCDF
ds.to_netcdf('output.nc')
```

### Zarr to Zarr

```python
import xarray as xr

# Open Zarr dataset
ds = xr.open_zarr('input.zarr')

# Write to Zarr with different chunking
ds.to_zarr('output.zarr', encoding={
    'temperature': {
        'chunksizes': (20, 180, 360)
    }
})
```

## GRIB Conversion

### GRIB to NetCDF

```python
import xarray as xr

# Open GRIB file
ds = xr.open_dataset('input.grib', engine='cfgrib')

# Write to NetCDF
ds.to_netcdf('output.nc')
```

### GRIB to Zarr

```python
import xarray as xr

# Open GRIB file
ds = xr.open_dataset('input.grib', engine='cfgrib')

# Write to Zarr
ds.to_zarr('output.zarr')
```

## CSV Conversion

### NetCDF to CSV

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks={'time': 10})

# Convert to CSV in chunks
for i in range(0, len(ds.time), 10):
    time_chunk = ds.isel(time=slice(i, i+10))
    
    # Convert chunk to DataFrame
    df = time_chunk.to_dataframe()
    
    # Write to CSV
    df.to_csv(f'chunk_{i}.csv')
    
    print(f"Converted chunk {i//10 + 1} to CSV")
```

### CSV to NetCDF

```python
import xarray as xr
import pandas as pd

# Read CSV in chunks
chunk_size = 10000
chunks = []

for i in range(0, len(df), chunk_size):
    chunk = pd.read_csv('input.csv', skiprows=i, nrows=chunk_size)
    chunks.append(chunk)

# Combine chunks
df = pd.concat(chunks)

# Convert to xarray
ds = xr.Dataset.from_dataframe(df)

# Write to NetCDF
ds.to_netcdf('output.nc')
```

## HDF5 Conversion

### NetCDF to HDF5

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Write to HDF5
ds.to_netcdf('output.h5', engine='h5netcdf')
```

### HDF5 to NetCDF

```python
import xarray as xr

# Open HDF5 file
ds = xr.open_dataset('input.h5', engine='h5netcdf')

# Write to NetCDF
ds.to_netcdf('output.nc')
```

## Conversion with Chunking

### Chunked NetCDF to Zarr

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks={'time': 10})

# Write to Zarr in chunks
for i in range(0, len(ds.time), 10):
    time_chunk = ds.isel(time=slice(i, i+10))
    
    # Write chunk to Zarr
    time_chunk.to_zarr(f'output.zarr', region={'time': slice(i, i+10)})
    
    print(f"Converted chunk {i//10 + 1} to Zarr")
```

### Chunked Zarr to NetCDF

```python
import xarray as xr

# Open Zarr dataset
ds = xr.open_zarr('input.zarr')

# Read in chunks and write to NetCDF
output_chunks = []
for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10)).compute()
    output_chunks.append(chunk)
    print(f"Read chunk {i//10 + 1}")

# Combine and write
combined = xr.concat(output_chunks, dim='time')
combined.to_netcdf('output.nc')
```

## Data Type Conversion

### Float64 to Float32

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Convert to float32
ds = ds.astype({'temperature': 'float32'})

# Write to NetCDF
ds.to_netcdf('output.nc')
```

### Int32 to Int16

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Convert to int16
ds = ds.astype({'land_mask': 'int16'})

# Write to NetCDF
ds.to_netcdf('output.nc')
```

## Compression

### NetCDF Compression

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Write with compression
encoding = {
    'temperature': {
        'zlib': True,
        'complevel': 5
    }
}

ds.to_netcdf('output.nc', encoding=encoding)
```

### Zarr Compression

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Write to Zarr with compression
ds.to_zarr('output.zarr', encoding={
    'temperature': {
        'compressor': {
            'id': 'zlib',
            'level': 5
        }
    }
})
```

### Blosc Compression

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Write to Zarr with Blosc compression
ds.to_zarr('output.zarr', encoding={
    'temperature': {
        'compressor': {
            'id': 'blosc',
            'cname': 'lz4',
            'clevel': 5,
            'shuffle': 1
        }
    }
})
```

## Metadata Preservation

### Preserve Attributes

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Preserve attributes during conversion
ds.to_netcdf('output.nc', encoding={
    'temperature': {
        'zlib': True,
        'complevel': 5
    }
}, unlimited_dims=['time'])
```

### Add Custom Attributes

```python
import xarray as xr

# Open with chunking
ds = xr.open_dataset('input.nc', chunks='auto')

# Add custom attributes
ds.attrs['title'] = 'Converted Dataset'
ds.attrs['history'] = 'Converted from input.nc to output.nc'
ds['temperature'].attrs['units'] = 'Kelvin'

# Write to NetCDF
ds.to_netcdf('output.nc')
```

## Performance Optimization

### Parallel Conversion

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def convert_chunk(ds, chunk_idx, chunk_size=10):
    """Convert a single chunk."""
    chunk = ds.isel(time=slice(chunk_idx, chunk_idx + chunk_size))
    chunk.to_zarr('output.zarr', region={'time': slice(chunk_idx, chunk_idx + chunk_size)})
    return chunk_idx

ds = xr.open_dataset('input.nc', chunks={'time': 10})

# Convert chunks in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    chunk_indices = range(0, len(ds.time), 10)
    results = list(executor.map(convert_chunk, [ds] * len(chunk_indices), chunk_indices))

print(f"Converted {len(results)} chunks")
```

### Memory-Efficient Conversion

```python
import xarray as xr

# Open with small chunks
ds = xr.open_dataset('input.nc', 
                      chunks={'time': 5, 'latitude': 45, 'longitude': 90})

# Process and write in chunks
for i in range(0, len(ds.time), 5):
    for j in range(0, ds.latitude.size, 45):
        for k in range(0, ds.longitude.size, 90):
            chunk = ds.isel(
                time=slice(i, i+5),
                latitude=slice(j, j+45),
                longitude=slice(k, k+90)
            )
            chunk.to_zarr('output.zarr', region={
                'time': slice(i, i+5),
                'latitude': slice(j, j+45),
                'longitude': slice(k, k+90)
            })
```

## Best Practices

1. **Use appropriate compression** - Balance compression level with performance
2. **Choose optimal chunk sizes** - Select chunk sizes based on access patterns
3. **Preserve metadata** - Keep attributes and coordinate information
4. **Use efficient formats** - Choose format based on your use case
5. **Process in chunks** - Avoid loading entire datasets into memory
6. **Use parallel processing** - Convert multiple chunks in parallel
7. **Monitor disk space** - Ensure sufficient space for output files
8. **Validate conversion** - Check converted data for consistency
