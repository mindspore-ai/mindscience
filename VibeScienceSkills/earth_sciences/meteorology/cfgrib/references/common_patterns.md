# Common Use Cases and Patterns

This document provides common use cases and best practices for working with cfgrib and xarray.

## Use Case 1: Weather Forecast Analysis

Analyze weather forecast data:

```python
import xarray as xr

with xr.open_dataset('forecast.grib', engine='cfgrib') as ds:
    # Select 2m temperature
    temp = ds['t2m']
    
    # Calculate statistics
    mean_temp = temp.mean()
    max_temp = temp.max()
    min_temp = temp.min()
    
    # Spatial distribution
    spatial_mean = temp.mean(dim=['latitude', 'longitude'])
    
    # Time series at specific location
    ny_temp = temp.sel(latitude=40.7, longitude=-74.0, method='nearest')
    
    print(f"Global mean: {mean_temp:.2f} K")
    print(f"Global range: {min_temp:.2f} - {max_temp:.2f} K")
    print(f"NYC temperature: {ny_temp.mean():.2f} K")
```

## Use Case 2: Climate Data Analysis

Analyze climate data from multiple files:

```python
import xarray as xr
import glob

# Read multiple files
files = glob.glob('era5_*.grib')
ds = xr.open_mfdataset(files, engine='cfgrib', combine='by_coords')

# Calculate climatology
climatology = ds['t2m'].groupby('time.month').mean()

# Calculate anomalies
anomalies = ds['t2m'].groupby('time.month') - climatology

# Calculate trend
time_numeric = (ds.time - ds.time[0]).dt.days
trend = np.polyfit(time_numeric, ds['t2m'].mean(dim=['latitude', 'longitude']), 1)

print(f"Climatology calculated for {len(climatology.month)} months")
print(f"Temperature trend: {trend[0]:.6f} K/day")
```

## Use Case 3: Regional Analysis

Analyze data for a specific region:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Define region (e.g., Europe)
    europe = ds['t2m'].sel(
        latitude=slice(70, 35),
        longitude=slice(-10, 40)
    )
    
    # Regional statistics
    regional_mean = europe.mean()
    regional_std = europe.std()
    
    # Regional time series
    regional_ts = europe.mean(dim=['latitude', 'longitude'])
    
    # Regional extremes
    regional_max = europe.max(dim=['latitude', 'longitude'])
    regional_min = europe.min(dim=['latitude', 'longitude'])
    
    print(f"European mean temperature: {regional_mean:.2f} K")
    print(f"European temperature range: {regional_min:.2f} - {regional_max:.2f} K")
```

## Use Case 4: Vertical Profile Analysis

Analyze vertical atmospheric profiles:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select location
    profile = ds['t'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Select time
    profile_time = profile.sel(time='2024-01-15T12:00:00')
    
    # Calculate lapse rate
    temp_profile = profile_time.values
    level_profile = profile_time.level.values
    
    # Simple lapse rate calculation
    lapse_rate = np.diff(temp_profile) / np.diff(level_profile)
    
    print(f"Temperature at 500 hPa: {profile_time.sel(level=500):.2f} K")
    print(f"Mean lapse rate: {np.mean(lapse_rate):.4f} K/hPa")
```

## Use Case 5: Ensemble Forecast Analysis

Analyze ensemble forecast data:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Ensemble statistics
    ensemble_mean = ds['t2m'].mean(dim='number')
    ensemble_spread = ds['t2m'].std(dim='number')
    
    # Probability of exceeding threshold
    prob_hot = (ds['t2m'] > 300).mean(dim='number')
    prob_cold = (ds['t2m'] < 280).mean(dim='number')
    
    # Ensemble percentiles
    p10 = ds['t2m'].quantile(0.10, dim='number')
    p90 = ds['t2m'].quantile(0.90, dim='number')
    
    print(f"Ensemble mean: {ensemble_mean.mean():.2f} K")
    print(f"Ensemble spread: {ensemble_spread.mean():.2f} K")
    print(f"Probability of hot day (>300K): {prob_hot.mean():.3f}")
```

## Use Case 6: Precipitation Analysis

Analyze precipitation data:

```python
import xarray as xr

with xr.open_dataset('precip.grib', engine='cfgrib') as ds:
    # Total precipitation
    total_precip = ds['tp'].sum(dim='time')
    
    # Daily precipitation
    daily_precip = ds['tp'].resample(time='1D').sum()
    
    # Monthly precipitation
    monthly_precip = ds['tp'].resample(time='1M').sum()
    
    # Find wet days (>1 mm/day)
    wet_days = (daily_precip > 0.001).sum(dim='time')
    
    # Maximum daily precipitation
    max_daily = daily_precip.max(dim='time')
    
    print(f"Total precipitation: {total_precip.mean():.3f} mm")
    print(f"Number of wet days: {wet_days.mean():.1f}")
    print(f"Maximum daily precipitation: {max_daily.max():.3f} mm")
```

## Use Case 7: Wind Analysis

Analyze wind data:

```python
import xarray as xr
import numpy as np

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Calculate wind speed
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
    
    # Calculate wind direction
    wind_dir = np.arctan2(ds['v10'], ds['u10']) * 180 / np.pi
    wind_dir = (wind_dir + 360) % 360  # Convert to 0-360
    
    # Wind statistics
    mean_speed = wind_speed.mean()
    max_speed = wind_speed.max()
    
    # Wind rose (simplified)
    speed_by_dir = wind_speed.groupby_bins(wind_dir, bins=8).mean()
    
    print(f"Mean wind speed: {mean_speed:.2f} m/s")
    print(f"Maximum wind speed: {max_speed:.2f} m/s")
```

## Use Case 8: Time Series Comparison

Compare time series at different locations:

```python
import xarray as xr

locations = [
    ('New York', 40.7, -74.0),
    ('London', 51.5, -0.1),
    ('Tokyo', 35.7, 139.7)
]

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    for name, lat, lon in locations:
        # Extract time series
        ts = ds['t2m'].sel(latitude=lat, longitude=lon, method='nearest')
        
        # Calculate statistics
        mean = ts.mean()
        std = ts.std()
        trend = np.polyfit(range(len(ts)), ts, 1)[0]
        
        print(f"{name}:")
        print(f"  Mean: {mean:.2f} K")
        print(f"  Std: {std:.2f} K")
        print(f"  Trend: {trend:.6f} K/timestep")
```

## Use Case 9: Data Quality Control

Perform quality control on data:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Check for missing values
    missing = ds['t2m'].isnull()
    missing_count = missing.sum()
    
    # Check for outliers
    mean = ds['t2m'].mean()
    std = ds['t2m'].std()
    outliers = (ds['t2m'] < mean - 3*std) | (ds['t2m'] > mean + 3*std)
    outlier_count = outliers.sum()
    
    # Check for unrealistic values
    unrealistic = (ds['t2m'] < 150) | (ds['t2m'] > 400)
    unrealistic_count = unrealistic.sum()
    
    print(f"Missing values: {missing_count}")
    print(f"Outliers (>3σ): {outlier_count}")
    print(f"Unrealistic values: {unrealistic_count}")
    
    # Flag problematic data
    quality_flag = missing | outliers | unrealistic
```

## Use Case 10: Data Subsetting and Export

Subset data and export to different formats:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Subset region and time
    subset = ds.sel(
        latitude=slice(50, 30),
        longitude=slice(-120, -90),
        time=slice('2024-01-01', '2024-01-31')
    )
    
    # Export to NetCDF
    subset.to_netcdf('subset.nc')
    
    # Export to Zarr
    subset.to_zarr('subset.zarr')
    
    # Export to CSV (single variable)
    subset['t2m'].to_dataframe().to_csv('subset.csv')
    
    print("Data exported to multiple formats")
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

### 2. Use Coordinate-Based Selection

```python
# Good: Coordinate-based
region = ds['t2m'].sel(latitude=slice(50, 30), longitude=slice(-120, -90))

# Bad: Index-based
region = ds['t2m'].isel(latitude=slice(0, 20), longitude=slice(0, 30))
```

### 3. Handle Missing Data

```python
# Handle missing values
data_clean = ds['t2m'].dropna('time')

# Or fill missing values
data_filled = ds['t2m'].fillna(ds['t2m'].mean())
```

### 4. Use Appropriate Chunking

```python
# For large files
ds = xr.open_dataset('large_file.grib', engine='cfgrib', chunks='auto')

# For specific chunking
ds = xr.open_dataset('file.grib', engine='cfgrib', 
                      chunks={'time': 10, 'latitude': 90})
```

### 5. Validate Data

```python
# Check data ranges
if ds['t2m'].min() < 150 or ds['t2m'].max() > 400:
    print("Warning: Temperature values outside expected range")

# Check for missing values
if ds['t2m'].isnull().any():
    print("Warning: Missing values present")
```

### 6. Document Your Workflow

```python
# Add comments to explain steps
# Load GRIB file
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select region of interest (North America)
    region = ds['t2m'].sel(
        latitude=slice(70, 25),
        longitude=slice(-170, -60)
    )
    
    # Calculate regional mean
    regional_mean = region.mean()
    
    # Calculate regional standard deviation
    regional_std = region.std()
```

### 7. Use Meaningful Variable Names

```python
# Good: Descriptive names
regional_temperature = ds['t2m'].sel(latitude=slice(50, 30))
regional_mean = regional_temperature.mean()

# Bad: Generic names
data = ds['t2m'].sel(latitude=slice(50, 30))
result = data.mean()
```

### 8. Optimize for Performance

```python
# Use vectorized operations
mean = ds['t2m'].mean(dim=['latitude', 'longitude'])

# Avoid loops
# Bad:
total = 0
for lat in ds.latitude:
    for lon in ds.longitude:
        total += ds['t2m'].sel(latitude=lat, longitude=lon)
mean = total / (len(ds.latitude) * len(ds.longitude))
```

## Common Patterns

### Pattern 1: Read-Process-Write

```python
import xarray as xr

# Read
with xr.open_dataset('input.grib', engine='cfgrib') as ds:
    # Process
    result = ds['t2m'].mean(dim='time')
    
    # Write
    result.to_netcdf('output.nc')
```

### Pattern 2: Multi-File Processing

```python
import xarray as xr
import glob

files = glob.glob('data_*.grib')
for file in files:
    with xr.open_dataset(file, engine='cfgrib') as ds:
        result = ds['t2m'].mean()
        print(f"{file}: {result:.2f} K")
```

### Pattern 3: Time Series Analysis

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Extract time series
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Calculate statistics
    mean = ts.mean()
    std = ts.std()
    trend = np.polyfit(range(len(ts)), ts, 1)[0]
    
    # Calculate anomalies
    climatology = ts.groupby('time.dayofyear').mean()
    anomalies = ts.groupby('time.dayofyear') - climatology
```

### Pattern 4: Spatial Analysis

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select time
    data = ds['t2m'].isel(time=0)
    
    # Calculate spatial statistics
    global_mean = data.mean()
    zonal_mean = data.mean(dim='longitude')
    meridional_mean = data.mean(dim='latitude')
    
    # Find extremes
    max_location = data.where(data == data.max(), drop=True)
    min_location = data.where(data == data.min(), drop=True)
```

### Pattern 5: Ensemble Analysis

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Ensemble statistics
    mean = ds['t2m'].mean(dim='number')
    spread = ds['t2m'].std(dim='number')
    
    # Probabilities
    prob_above = (ds['t2m'] > 300).mean(dim='number')
    prob_below = (ds['t2m'] < 280).mean(dim='number')
    
    # Percentiles
    p10 = ds['t2m'].quantile(0.10, dim='number')
    p90 = ds['t2m'].quantile(0.90, dim='number')
```

## Troubleshooting

### Issue: Memory Error

**Solution:** Use chunking
```python
ds = xr.open_dataset('large_file.grib', engine='cfgrib', chunks='auto')
```

### Issue: Slow Performance

**Solution:** Use Dask
```python
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')
result = ds['t2m'].mean().compute()
```

### Issue: Variable Not Found

**Solution:** Check available variables
```python
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    print("Available variables:", list(ds.data_vars.keys()))
```

### Issue: Coordinate Not Found

**Solution:** Check available coordinates
```python
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    print("Available coordinates:", list(ds.coords.keys()))
```

## References

- xarray Documentation: https://xarray.pydata.org/
- cfgrib Documentation: https://github.com/ecmwf/cfgrib
- pandas Documentation: https://pandas.pydata.org/