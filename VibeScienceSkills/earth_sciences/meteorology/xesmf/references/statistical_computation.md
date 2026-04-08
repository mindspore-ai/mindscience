# Statistical Computation with xesmf

## Overview

Statistical computation is essential for analyzing meteorological data. This guide covers efficient techniques for calculating statistics on large datasets with xesmf.

## Basic Statistics

### Mean

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate mean
mean = ds['temperature'].mean().compute()

print(f"Mean: {mean:.2f}")
```

### Standard Deviation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate standard deviation
std = ds['temperature'].std().compute()

print(f"Standard deviation: {std:.2f}")
```

### Variance

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate variance
variance = ds['temperature'].var().compute()

print(f"Variance: {variance:.2f}")
```

### Min and Max

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate min and max
min_val = ds['temperature'].min().compute()
max_val = ds['temperature'].max().compute()

print(f"Min: {min_val:.2f}")
print(f"Max: {max_val:.2f}")
```

## Percentiles

### Single Percentile

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate 50th percentile (median)
p50 = ds['temperature'].quantile(0.50).compute()

print(f"50th percentile: {p50:.2f}")
```

### Multiple Percentiles

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate multiple percentiles
percentiles = [0.25, 0.50, 0.75, 0.90, 0.95]
results = ds['temperature'].quantile(percentiles).compute()

for p, result in zip(percentiles, results):
    print(f"{p*100}th percentile: {result:.2f}")
```

### Percentile by Dimension

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate percentiles along time dimension
p25 = ds['temperature'].quantile(0.25, dim='time').compute()
p50 = ds['temperature'].quantile(0.50, dim='time').compute()
p75 = ds['temperature'].quantile(0.75, dim='time').compute()

print(f"25th percentile shape: {p25.shape}")
print(f"50th percentile shape: {p50.shape}")
print(f"75th percentile shape: {p75.shape}")
```

## Advanced Statistics

### Correlation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate correlation between two variables
correlation = xr.corr(ds['temperature'], ds['precipitation']).compute()

print(f"Correlation: {correlation:.3f}")
```

### Covariance

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate covariance between two variables
covariance = xr.cov(ds['temperature'], ds['precipitation']).compute()

print(f"Covariance: {covariance:.3f}")
```

### Skewness

```python
import xarray as xr
import scipy.stats as stats

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate skewness
data = ds['temperature'].compute().values.flatten()
skewness = stats.skew(data)

print(f"Skewness: {skewness:.3f}")
```

### Kurtosis

```python
import xarray as xr
import scipy.stats as stats

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate kurtosis
data = ds['temperature'].compute().values.flatten()
kurtosis = stats.kurtosis(data)

print(f"Kurtosis: {kurtosis:.3f}")
```

## Grouping Operations

### Group by Time

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Group by month
monthly_means = ds.groupby('time.month').mean().compute()

print(f"Monthly means: {monthly_means}")
```

### Group by Season

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'
    elif month in [3, 4, 5]:
        return 'MAM'
    elif month in [6, 7, 8]:
        return 'JJA'
    else:
        return 'SON'

# Add season coordinate
ds = ds.assign_coords(season=('time', [get_season(m) for m in ds.time.dt.month]))

# Group by season
seasonal_means = ds.groupby('season').mean().compute()

print(f"Seasonal means: {seasonal_means}")
```

### Group by Region

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Define regions
def get_region(lat):
    if lat > 30:
        return 'Northern'
    elif lat < -30:
        return 'Southern'
    else:
        return 'Tropical'

# Add region coordinate
ds = ds.assign_coords(region=('latitude', [get_region(lat) for lat in ds.latitude]))

# Group by region
regional_means = ds.groupby('region').mean().compute()

print(f"Regional means: {regional_means}")
```

## Rolling Statistics

### Rolling Mean

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Calculate rolling mean (7-day window)
rolling_mean = ds['temperature'].rolling(time=7, center=True).mean().compute()

print(f"Rolling mean shape: {rolling_mean.shape}")
```

### Rolling Standard Deviation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Calculate rolling standard deviation
rolling_std = ds['temperature'].rolling(time=7, center=True).std().compute()

print(f"Rolling std shape: {rolling_std.shape}")
```

### Rolling Sum

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Calculate rolling sum
rolling_sum = ds['precipitation'].rolling(time=7, center=True).sum().compute()

print(f"Rolling sum shape: {rolling_sum.shape}")
```

## Weighted Statistics

### Weighted Mean

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Define weights (e.g., area weights)
weights = np.cos(np.deg2rad(ds.latitude))

# Calculate weighted mean
weighted_mean = ds['temperature'].weighted(weights).mean().compute()

print(f"Weighted mean: {weighted_mean:.2f}")
```

### Weighted Sum

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Define weights
weights = np.cos(np.radians(ds.latitude))

# Calculate weighted sum
weighted_sum = ds['temperature'].weighted(weights).sum().compute()

print(f"Weighted sum: {weighted_sum:.2f}")
```

## Statistical Tests

### T-test

```python
import xarray as xr
from scipy import stats

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Select two groups
group1 = ds['temperature'].sel(time=slice('2020-01-01', '2020-06-30')).compute().values
group2 = ds['temperature'].sel(time=slice('2020-07-01', '2020-12-31')).compute().values

# Perform t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"T-statistic: {t_stat:.3f}")
print(f"P-value: {p_value:.3f}")
```

### Mann-Whitney U Test

```python
import xarray as xr
from scipy import stats

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Select two groups
group1 = ds['temperature'].sel(time=slice('2020-01-01', '2020-06-30')).compute().values
group2 = ds['temperature'].sel(time=slice('2020-07-01', '2020-12-31')).compute().values

# Perform Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(group1, group2)

print(f"U-statistic: {u_stat:.3f}")
print(f"P-value: {p_value:.3f}")
```

### Kolmogorov-Smirnov Test

```python
import xarray as xr
from scipy import stats

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Select two groups
group1 = ds['temperature'].sel(time=slice('2020-01-01', '2020-06-30')).compute().values
group2 = ds['temperature'].sel(time=slice('2020-07-01', '2020-12-31')).compute().values

# Perform KS test
ks_stat, p_value = stats.ks_2samp(group1, group2)

print(f"KS-statistic: {ks_stat:.3f}")
print(f"P-value: {p_value:.3f}")
```

## Performance Optimization

### Efficient Statistics Calculation

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks='auto')

# Calculate multiple statistics efficiently
mean = ds['temperature'].mean()
std = ds['temperature'].std()
min_val = ds['temperature'].min()
max_val = ds['temperature'].max()

# Compute all at once
mean_result = ds['temperature'].mean().compute()
std_result = ds['temperature'].std().compute()
min_result = ds['temperature'].min().compute()
max_result = ds['temperature'].max().compute()

print(f"Mean: {mean_result:.2f}")
print(f"Std: {std_result:.2f}")
print(f"Min: {min_result:.2f}")
print(f"Max: {max_result:.2f}")
```

### Chunked Statistics

```python
import xarray as xr

ds = xr.open_dataset('large_file.nc', chunks={'time': 10})

# Calculate statistics in chunks
chunk_means = []
chunk_stds = []

for i in range(0, len(ds.time), 10):
    chunk = ds.isel(time=slice(i, i+10))
    chunk_means.append(chunk['temperature'].mean().compute())
    chunk_stds.append(chunk['temperature'].std().compute())

# Combine results
overall_mean = np.mean(chunk_means)
overall_std = np.mean(chunk_stds)

print(f"Overall mean: {overall_mean:.2f}")
print(f"Overall std: {overall_std:.2f}")
```

## Best Practices

1. **Use appropriate chunking** - Choose chunk sizes based on your statistical operation
2. **Process in chunks** - Avoid loading entire datasets into memory
3. **Use lazy evaluation** - Leverage xarray's lazy evaluation
4. **Calculate multiple statistics** - Compute multiple statistics together when possible
5. **Use grouping operations** - Group data for efficient aggregation
6. **Handle missing data** - Use masked arrays appropriately
7. **Monitor memory usage** - Track memory consumption during processing
8. **Validate results** - Check statistical results for consistency
