# Spatial Analysis with cfgrib

This guide covers advanced spatial analysis techniques using cfgrib and xarray.

## Spatial Statistics

### Global Statistics

Calculate statistics over entire spatial domain:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Global mean
    global_mean = ds['t2m'].mean(dim=['latitude', 'longitude'])
    
    # Global standard deviation
    global_std = ds['t2m'].std(dim=['latitude', 'longitude'])
    
    # Global minimum and maximum
    global_min = ds['t2m'].min(dim=['latitude', 'longitude'])
    global_max = ds['t2m'].max(dim=['latitude', 'longitude'])
```

### Regional Statistics

Calculate statistics for specific regions:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select region
    region = ds['t2m'].sel(
        latitude=slice(50, 30),
        longitude=slice(-120, -90)
    )
    
    # Regional statistics
    regional_mean = region.mean(dim=['latitude', 'longitude'])
    regional_std = region.std(dim=['latitude', 'longitude'])
```

### Zonal Statistics

Calculate statistics along longitude bands:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Zonal mean (average over longitudes)
    zonal_mean = ds['t2m'].mean(dim='longitude')
    
    # Zonal standard deviation
    zonal_std = ds['t2m'].std(dim='longitude')
    
    # Plot zonal mean
    zonal_mean.plot()
```

### Meridional Statistics

Calculate statistics along latitude bands:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Meridional mean (average over latitudes)
    meridional_mean = ds['t2m'].mean(dim='latitude')
    
    # Meridional standard deviation
    meridional_std = ds['t2m'].std(dim='latitude')
    
    # Plot meridional mean
    meridional_mean.plot()
```

## Spatial Gradients

### Calculate Gradients

Compute spatial gradients:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Gradient in latitude direction
    dlat = ds['t2m'].differentiate('latitude')
    
    # Gradient in longitude direction
    dlon = ds['t2m'].differentiate('longitude')
    
    # Magnitude of gradient
    gradient_magnitude = np.sqrt(dlat**2 + dlon**2)
```

### Laplacian

Compute Laplacian (second derivative):

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Second derivatives
    d2lat = ds['t2m'].differentiate('latitude').differentiate('latitude')
    d2lon = ds['t2m'].differentiate('longitude').differentiate('longitude')
    
    # Laplacian
    laplacian = d2lat + d2lon
```

## Spatial Correlation

### Spatial Autocorrelation

Calculate spatial autocorrelation:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Calculate correlation with spatially shifted version
    shifted = ds['t2m'].roll(longitude=1)
    autocorr = xr.corr(ds['t2m'], shifted)
```

### Cross-Correlation

Calculate spatial cross-correlation between variables:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Correlation between temperature and pressure
    corr = xr.corr(ds['t2m'], ds['msl'])
```

## Spatial Interpolation

### Nearest Neighbor

Interpolate to new coordinates using nearest neighbor:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define new coordinates
    new_lats = np.array([40.5, 35.5, 30.5])
    new_lons = np.array([-100.5, -105.5, -110.5])
    
    # Interpolate using nearest neighbor
    interpolated = ds['t2m'].interp(
        latitude=new_lats,
        longitude=new_lons,
        method='nearest'
    )
```

### Linear Interpolation

Interpolate using linear interpolation:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define new coordinates
    new_lats = np.linspace(50, 30, 50)
    new_lons = np.linspace(-120, -90, 50)
    
    # Linear interpolation
    interpolated = ds['t2m'].interp(
        latitude=new_lats,
        longitude=new_lons,
        method='linear'
    )
```

### Bilinear Interpolation

Interpolate using bilinear interpolation:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define new coordinates
    new_lats = np.linspace(50, 30, 50)
    new_lons = np.linspace(-120, -90, 50)
    
    # Bilinear interpolation
    interpolated = ds['t2m'].interp(
        latitude=new_lats,
        longitude=new_lons,
        method='linear'
    )
```

### Cubic Interpolation

Interpolate using cubic interpolation:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define new coordinates
    new_lats = np.linspace(50, 30, 50)
    new_lons = np.linspace(-120, -90, 50)
    
    # Cubic interpolation
    interpolated = ds['t2m'].interp(
        latitude=new_lats,
        longitude=new_lons,
        method='cubic'
    )
```

## Spatial Regridding

### Coarse Gridding

Reduce resolution by averaging:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Coarse grid by averaging every 2x2 points
    coarse = ds['t2m'].coarsen(latitude=2, longitude=2)
    
    # Or use groupby
    coarse = ds['t2m'].coarsen({
        'latitude': 2,
        'longitude': 2
    })
```

### Fine Gridding

Increase resolution by interpolation:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define finer grid
    new_lats = np.linspace(ds.latitude[0], ds.latitude[-1], 
                          len(ds.latitude) * 2)
    new_lons = np.linspace(ds.longitude[0], ds.longitude[-1], 
                          len(ds.longitude) * 2)
    
    # Interpolate to finer grid
    fine = ds['t2m'].interp(
        latitude=new_lats,
        longitude=new_lons
    )
```

## Spatial Averaging

### Area-Weighted Average

Calculate area-weighted spatial average:

```python
import xarray as xr
import numpy as np

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Calculate cell areas (simplified)
    # In practice, use proper grid cell area calculation
    dlat = np.abs(ds.latitude[1] - ds.latitude[0])
    dlon = np.abs(ds.longitude[1] - ds.longitude[0])
    
    # Weights proportional to cos(latitude)
    weights = np.cos(np.deg2rad(ds.latitude))
    
    # Area-weighted mean
    weighted_mean = ds['t2m'].weighted(weights).mean(dim=['latitude', 'longitude'])
```

### Regional Average

Calculate average for specific region:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define region
    region = ds['t2m'].sel(
        latitude=slice(50, 30),
        longitude=slice(-120, -90)
    )
    
    # Regional average
    regional_avg = region.mean(dim=['latitude', 'longitude'])
```

## Spatial Filtering

### Smoothing

Apply spatial smoothing:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Rolling mean smoothing
    smoothed = ds['t2m'].rolling(latitude=3, longitude=3, center=True).mean()
```

### Edge Detection

Detect spatial edges:

```python
import xarray as xr

with xr.open_dataset('file.grib', as engine='cfgrib') as ds:
    # Calculate gradients
    dlat = ds['t2m'].differentiate('latitude')
    dlon = ds['t2m'].differentiate('longitude')
    
    # Edge magnitude
    edges = np.sqrt(dlat**2 + dlon**2)
    
    # Threshold edges
    strong_edges = edges.where(edges > edges.quantile(0.9))
```

## Spatial Pattern Analysis

### Find Extrema

Find locations of minima and maxima:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Find maximum
    max_val = ds['t2m'].max()
    max_location = ds['t2m'].where(ds['t2m'] == max_val, drop=True)
    
    # Find minimum
    min_val = ds['t2m'].min()
    min_location = ds['t2m'].where(ds['t2m'] == min_val, drop=True)
```

### Pattern Matching

Find spatial patterns:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define threshold
    threshold = 300  # K
    
    # Find areas above threshold
    hot_spots = ds['t2m'].where(ds['t2m'] > threshold)
    
    # Find areas below threshold
    cold_spots = ds['t2m'].where(ds['t2m'] < threshold)
```

## Distance Calculations

### Haversine Distance

Calculate great-circle distance between points:

```python
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371.0):
    """Calculate great-circle distance between two points."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return earth_radius * c
```

### Distance Field

Calculate distance field from a point:

```python
import xarray as xr
import numpy as np

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Reference point
    ref_lat = 40.0
    ref_lon = -100.0
    
    # Calculate distance for each grid point
    distances = xr.DataArray(
        np.zeros((len(ds.latitude), len(ds.longitude))),
        coords={'latitude': ds.latitude, 'longitude': ds.longitude},
        dims=['latitude', 'longitude']
    )
    
    for i, lat in enumerate(ds.latitude):
        for j, lon in enumerate(ds.longitude):
            distances[i, j] = haversine_distance(ref_lat, ref_lon, lat, lon)
```

## Best Practices

### 1. Use Vectorized Operations

```python
# Good: Vectorized
regional_mean = ds['t2m'].sel(
    latitude=slice(50, 30),
    longitude=slice(-120, -90)
).mean()

# Bad: Loop-based
total = 0
count = 0
for lat in ds.latitude:
    for lon in ds.longitude:
        if 30 <= lat <= 50 and -120 <= lon <= -90:
            total += ds['t2m'].sel(latitude=lat, longitude=lon)
            count += 1
regional_mean = total / count
```

### 2. Consider Coordinate Systems

```python
# Check coordinate ranges
if ds.latitude[0] < ds.latitude[-1]:
    print("Latitude decreases from north to south")
else:
    print("Latitude increases from south to north")

if ds.longitude[0] < ds.longitude[-1]:
    print("Longitude increases from west to east")
else:
    print("Longitude crosses 180° meridian")
```

### 3. Handle Missing Data

```python
# Handle missing values in spatial operations
mean = ds['t2m'].where(ds['t2m'] > 0).mean(dim=['latitude', 'longitude'])

# Or use skipna
mean = ds['t2m'].mean(dim=['latitude', 'longitude'], skipna=True)
```

### 4. Use Appropriate Interpolation

```python
# For categorical data
nearest = ds['t2m'].interp(latitude=new_lats, longitude=new_lons, method='nearest')

# For continuous data
linear = ds['t2m'].interp(latitude=new_lats, longitude=new_lons, method='linear')

# For smooth fields
cubic = ds['t2m'].interp(latitude=new_lats, longitude=new_lons, method='cubic')
```

## References

- xarray Documentation: https://xarray.pydata.org/
- scipy.interpolate: https://docs.scipy.org/doc/scipy/reference/interpolate.html
- numpy Documentation: https://numpy.org/doc/