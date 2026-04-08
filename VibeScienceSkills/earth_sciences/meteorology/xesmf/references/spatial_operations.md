# Spatial Operations with xesmf

## Overview

Spatial operations are fundamental in meteorological data analysis. This guide covers efficient techniques for performing spatial operations on large datasets with xesmf.

## Spatial Averaging

### Global Spatial Mean

Calculate global spatial mean:

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Calculate global spatial mean
spatial_mean = ds['temperature'].mean(dim=['latitude', 'longitude']).compute()

print(f"Global spatial mean: {spatial_mean:.2f}")
```

### Regional Spatial Mean

Calculate spatial mean for a region:

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Select region
region = ds.sel(latitude=slice(50, 30), longitude=slice(-120, -90))

# Calculate regional spatial mean
regional_mean = region['temperature'].mean(dim=['latitude', 'longitude']).compute()

print(f"Regional spatial mean: {regional_mean:.2f}")
```

### Zonal Mean

Calculate zonal mean (average along longitude):

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Calculate zonal mean
zonal_mean = ds['temperature'].mean(dim='longitude').compute()

print(f"Zonal mean shape: {zonal_mean.shape}")
```

### Meridional Mean

Calculate meridional mean (average along latitude):

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Calculate meridional mean
meridional_mean = ds['temperature'].mean(dim='latitude').compute()

print(f"Meridional mean shape: {meridional_mean.shape}")
```

## Spatial Statistics

### Spatial Standard Deviation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Calculate spatial standard deviation
spatial_std = ds['temperature'].std(dim=['latitude', 'longitude']).compute()

print(f"Spatial std: {spatial_std:.2f}")
```

### Spatial Extremes

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Calculate spatial min and max
spatial_min = ds['temperature'].min(dim=['latitude', 'longitude']).compute()
spatial_max = ds['temperature'].max(dim=['latitude', 'longitude']).compute()

print(f"Spatial min: {spatial_min:.2f}")
print(f"Spatial max: {spatial_max:.2f}")
```

### Spatial Percentiles

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Calculate spatial percentiles
p25 = ds['temperature'].quantile(0.25, dim=['latitude', 'longitude']).compute()
p50 = ds['temperature'].quantile(0.50, dim=['latitude', 'longitude']).compute()
p75 = ds['temperature'].quantile(0.75, dim=['latitude', 'longitude']).compute()

print(f"25th percentile: {p25:.2f}")
print(f"50th percentile: {p50:.2f}")
print(f"75th percentile: {p75:.2f}")
```

## Spatial Operations in Chunks

### Process Spatial Chunks

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Process spatial chunks
for i in range(0, ds.latitude.size, 90):
    lat_chunk = ds.isel(latitude=slice(i, i+90))
    spatial_mean = lat_chunk['temperature'].mean(dim=['latitude', 'longitude']).compute()
    print(f"Latitude chunk {i//90 + 1}: {spatial_mean:.2f}")
```

### Regional Analysis in Chunks

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', 
                      chunks={'time': 10, 'latitude': 90, 'longitude': 180})

# Select region
region = ds.sel(latitude=slice(50, 30), longitude=slice(-120, -90))

# Process region in time chunks
for i in range(0, len(region.time), 10):
    time_chunk = region.isel(time=slice(i, i+10))
    chunk_mean = time_chunk['temperature'].mean(dim=['latitude', 'longitude']).compute()
    print(f"Time chunk {i//10 + 1}: {chunk_mean:.2f}")
```

## Spatial Interpolation

### Nearest Neighbor Interpolation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Interpolate to new grid
new_lat = np.linspace(30, 50, 50)
new_lon = np.linspace(-120, -90, 50)

interpolated = ds['temperature'].interp(
    latitude=new_lat,
    longitude=new_lon,
    method='nearest'
).compute()
```

### Linear Interpolation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Interpolate to new grid
new_lat = np.linspace(30, 50, 50)
new_lon = np.linspace(-120, -90, 50)

interpolated = ds['temperature'].interp(
    latitude=new_lat,
    longitude=new_lon,
    method='linear'
).compute()
```

### Cubic Interpolation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Interpolate to new grid
new_lat = np.linspace(30, 50, 50)
new_lon = np.linspace(-120, -90, 50)

interpolated = ds['temperature'].interp(
    latitude=new_lat,
    longitude=new_lon,
    method='cubic'
).compute()
```

## Spatial Regridding

### Conservative Regridding

```python
import xarray as xr
import xesmf as xe

ds = xr.open_dataset('source_file.nc', chunks='auto')

# Define target grid
target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(30, 50, 50)),
    'lon': (['lon'], np.linspace(-120, -90, 50))
})

# Create regridder
regridder = xe.Regridder(ds, target_grid, method='conservative')

# Regrid data
regridded = regridder(ds['temperature']).compute()
```

### Bilinear Regridding

```python
import xarray as xr
import xesmf as xe

ds = xr.open_dataset('source_file.nc', chunks='auto')

# Define target grid
target_grid = xr.Dataset({
    'lat': (['lat'], np.linspace(30, 50, 50)),
    'lon': (['lon'], np.linspace(-120, -90, 50))
})

# Create regridder
regridder = xe.Regridder(ds, target_grid, method='bilinear')

# Regrid data
regridded = regridder(ds['temperature']).compute()
```

## Spatial Patterns

### Spatial Correlation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate spatial correlation between two variables
correlation = xr.corr(
    ds['temperature'],
    ds['precipitation'],
    dim=['latitude', 'longitude']
).compute()

print(f"Spatial correlation: {correlation:.3f}")
```

### Spatial Gradient

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate spatial gradient
grad_lat = ds['temperature'].differentiate('latitude')
grad_lon = ds['temperature'].differentiate('longitude')

# Compute gradients
grad_lat_computed = grad_lat.compute()
grad_lon_computed = grad_lon.compute()
```

### Spatial Laplacian

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate spatial Laplacian
laplacian = (
    ds['temperature'].differentiate('latitude', 2) +
    ds['temperature'].differentiate('longitude', 2)
).compute()
```

## Regional Analysis

### Define Regions

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Define regions
regions = {
    'North America': {
        'lat': slice(50, 25),
        'lon': slice(-125, -65)
    },
    'Europe': {
        'lat': slice(70, 35),
        'lon': slice(-10, 40)
    },
    'Asia': {
        'lat': slice(70, 10),
        'lon': slice(60, 150)
    }
}

# Analyze each region
for region_name, region_bounds in regions.items():
    region_data = ds.sel(
        latitude=region_bounds['lat'],
        longitude=region_bounds['lon']
    )
    region_mean = region_data['temperature'].mean().compute()
    print(f"{region_name}: {region_mean:.2f}")
```

### Masking

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Create mask (e.g., land only)
mask = ds['land_mask'] == 1

# Apply mask
masked_data = ds['temperature'].where(mask)

# Calculate statistics
masked_mean = masked_data.mean().compute()
print(f"Masked mean: {masked_mean:.2f}")
```

## Performance Optimization

### Efficient Spatial Chunking

```python
import xarray as xr

# Open with spatial chunking
ds = xr.open_dataset('large_file.nc', 
                      chunks={'latitude': 90, 'longitude': 180})

# Process spatial chunks
chunk_means = []
for i in range(0, ds.latitude.size, 90):
    for j in range(0, ds.longitude.size, 180):
        chunk = ds.isel(
            latitude=slice(i, i+90),
            longitude=slice(j, j+180)
        )
        chunk_mean = chunk['temperature'].mean().compute()
        chunk_means.append(chunk_mean)

overall_mean = np.mean(chunk_means)
print(f"Overall mean: {overall_mean:.2f}")
```

### Parallel Spatial Processing

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_spatial_chunk(ds, lat_idx, lon_idx, chunk_size_lat=90, chunk_size_lon=180):
    """Process a spatial chunk."""
    chunk = ds.isel(
        latitude=slice(lat_idx, lat_idx + chunk_size_lat),
        longitude=slice(lon_idx, lon_idx + chunk_size_lon)
    )
    return chunk['temperature'].mean().compute()

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Process chunks in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    lat_indices = range(0, ds.latitude.size, 90)
    lon_indices = range(0, ds.longitude.size, 180)
    
    results = []
    for lat_idx in lat_indices:
        for lon_idx in lon_indices:
            future = executor.submit(
                process_spatial_chunk, ds, lat_idx, lon_idx
            )
            results.append(future)
    
    chunk_means = [future.result() for future in results]

overall_mean = np.mean(chunk_means)
print(f"Overall mean: {overall_mean:.2f}")
```

## Best Practices

1. **Use spatial chunking** - Chunk along spatial dimensions for spatial operations
2. **Process in chunks** - Avoid loading entire spatial fields into memory
3. **Use appropriate interpolation** - Choose interpolation method based on your needs
4. **Use conservative regridding** - For mass-conserving quantities
5. **Monitor memory usage** - Track memory consumption during processing
6. **Use parallel processing** - Process multiple spatial chunks in parallel
7. **Handle missing data** - Use masked arrays appropriately
8. **Validate results** - Check regridding and interpolation results
