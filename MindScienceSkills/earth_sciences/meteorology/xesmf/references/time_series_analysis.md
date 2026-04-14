# Time Series Analysis with xesmf

## Overview

Time series analysis is a common operation in meteorological data processing. This guide covers efficient techniques for analyzing time series data with xesmf.

## Extracting Time Series

### Point Time Series

Extract time series for a specific location:

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series for a point
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Process time series
for i in range(0, len(ts.time), 10):
    chunk = ts.isel(time=slice(i, i+10))
    chunk_mean = chunk.mean().compute()
    print(f"Time step {i}: {chunk_mean:.2f}")
```

### Regional Time Series

Extract time series for a region:

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Select region
region = ds.sel(latitude=slice(45, 35), longitude=slice(-105, -95))

# Calculate regional mean time series
regional_ts = region['temperature'].mean(dim=['latitude', 'longitude'])

# Process time series
for i in range(0, len(regional_ts.time), 10):
    chunk = regional_ts.isel(time=slice(i, i+10))
    chunk_mean = chunk.mean().compute()
    print(f"Time step {i}: {chunk_mean:.2f}")
```

## Time Series Statistics

### Basic Statistics

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate statistics
mean = ts.mean().compute()
std = ts.std().compute()
min_val = ts.min().compute()
max_val = ts.max().compute()

print(f"Mean: {mean:.2f}")
print(f"Std: {std:.2f}")
print(f"Min: {min_val:.2f}")
print(f"Max: {max_val:.2f}")
```

### Percentiles

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate percentiles
p25 = ts.quantile(0.25).compute()
p50 = ts.quantile(0.50).compute()
p75 = ts.quantile(0.75).compute()
p90 = ts.quantile(0.90).compute()

print(f"25th percentile: {p25:.2f}")
print(f"50th percentile: {p50:.2f}")
print(f"75th percentile: {p75:.2f}")
print(f"90th percentile: {p90:.2f}")
```

### Rolling Statistics

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate rolling mean (7-day window)
rolling_mean = ts.rolling(time=7, center=True).mean().compute()

# Calculate rolling std
rolling_std = ts.rolling(time=7, center=True).std().compute()
```

## Time Aggregation

### Daily Aggregation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate daily means (assuming 6-hourly data)
daily_means = []
for i in range(0, len(ts.time), 4):  # 4 time steps per day
    day_chunk = ts.isel(time=slice(i, i+4))
    daily_mean = day_chunk.mean().compute()
    daily_means.append(daily_mean)

print(f"Number of days: {len(daily_means)}")
```

### Monthly Aggregation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Group by month
monthly_means = []
for month in range(1, 13):
    month_data = ds.sel(time=ds.time.dt.month == month)
    month_mean = month_data['temperature'].mean().compute()
    monthly_means.append(month_mean)

print(f"Monthly means: {monthly_means}")
```

### Seasonal Aggregation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Define seasons
seasons = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11]
}

# Calculate seasonal means
seasonal_means = {}
for season_name, months in seasons.items():
    season_data = ds.sel(time=ds.time.dt.month.isin(months))
    seasonal_mean = season_data['temperature'].mean().compute()
    seasonal_means[season_name] = seasonal_mean

print(f"Seasonal means: {seasonal_means}")
```

## Time Series Operations

### Anomalies

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate climatology (monthly means)
climatology = ts.groupby('time.month').mean()

# Calculate anomalies
anomalies = ts.groupby('time.month') - climatology

# Process anomalies
for i in range(0, len(anomalies.time), 10):
    chunk = anomalies.isel(time=slice(i, i+10))
    chunk_mean = chunk.mean().compute()
    print(f"Time step {i}: {chunk_mean:.2f}")
```

### Trends

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Convert to numpy array
ts_values = ts.compute().values
time_values = np.arange(len(ts_values))

# Calculate linear trend
slope, intercept = np.polyfit(time_values, ts_values, 1)
trend = slope * time_values + intercept

print(f"Trend: {slope:.4f} per time step")
```

### Detrending

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Calculate trend
ts_values = ts.compute().values
time_values = np.arange(len(ts_values))
slope, intercept = np.polyfit(time_values, ts_values, 1)
trend = slope * time_values + intercept

# Remove trend
detrended = ts - trend
```

## Multiple Time Series

### Extract Multiple Points

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Define locations
locations = [
    {'name': 'Location 1', 'lat': 40.0, 'lon': -100.0},
    {'name': 'Location 2', 'lat': 35.0, 'lon': -95.0},
    {'name': 'Location 3', 'lat': 30.0, 'lon': -90.0}
]

# Extract time series for each location
for location in locations:
    ts = ds['temperature'].sel(
        latitude=location['lat'], 
        longitude=location['lon'], 
        method='nearest'
    )
    mean = ts.mean().compute()
    print(f"{location['name']}: {mean:.2f}")
```

### Compare Time Series

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract two time series
ts1 = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')
ts2 = ds['temperature'].sel(latitude=35.0, longitude=-95.0, method='nearest')

# Calculate correlation
correlation = xr.corr(ts1, ts2).compute()
print(f"Correlation: {correlation:.3f}")

# Calculate difference
difference = ts1 - ts2
diff_mean = difference.mean().compute()
print(f"Mean difference: {diff_mean:.2f}")
```

## Performance Optimization

### Efficient Time Series Extraction

```python
import xarray as xr

# Open with time chunking
ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Extract time series efficiently
ts = ds['temperature'].sel(latitude=40.0, longitude=-100.0, method='nearest')

# Process in chunks
chunk_means = []
for i in range(0, len(ts.time), 10):
    chunk = ts.isel(time=slice(i, i+10))
    chunk_mean = chunk.mean().compute()
    chunk_means.append(chunk_mean)

overall_mean = np.mean(chunk_means)
print(f"Overall mean: {overall_mean:.2f}")
```

### Parallel Time Series Processing

```python
import xarray as xr
from concurrent.futures import ThreadPoolExecutor

def process_location(ds, location):
    """Process time series for a location."""
    ts = ds['temperature'].sel(
        latitude=location['lat'], 
        longitude=location['lon'], 
        method='nearest'
    )
    return ts.mean().compute()

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Define locations
locations = [
    {'name': 'Location 1', 'lat': 40.0, 'lon': -100.0},
    {'name': 'Location 2', 'lat': 35.0, 'lon': -95.0},
    {'name': 'Location 3', 'lat': 30.0, 'lon': -90.0}
]

# Process in parallel
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_location, [ds] * len(locations), locations))

for location, result in zip(locations, results):
    print(f"{location['name']}: {result:.2f}")
```

## Best Practices

1. **Use time chunking** - Chunk along the time dimension for time series analysis
2. **Process in chunks** - Avoid loading entire time series into memory
3. **Use lazy evaluation** - Leverage xarray's lazy evaluation
4. **Select only needed data** - Extract only the time series you need
5. **Use appropriate aggregation** - Choose aggregation based on your analysis
6. **Monitor memory usage** - Track memory consumption during processing
7. **Use parallel processing** - Process multiple time series in parallel
8. **Handle missing data** - Use masked arrays appropriately
