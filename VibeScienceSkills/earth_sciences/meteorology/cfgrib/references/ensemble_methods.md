# Ensemble Data Handling with cfgrib

This guide covers ensemble data handling and statistical methods using cfgrib and xarray.

## Reading Ensemble Data

### Basic Ensemble Reading

Read ensemble forecast data:

```python
import xarray as xr

# Read ensemble data
ds = xr.open_dataset('ensemble.grib', engine='cfgrib')

# Check for ensemble dimension
if 'number' in ds.dims:
    print(f"Ensemble size: {ds.dims['number']}")
else:
    print("No ensemble dimension found")
```

### Ensemble Metadata

Access ensemble metadata:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Ensemble dimension
    print(f"Number of members: {ds.dims['number']}")
    
    # Ensemble member information
    if 'number' in ds.coords:
        print(f"Member numbers: {ds.coords['number'].values}")
    
    # Perturbation information
    if 'perturbationNumber' in ds:
        print(f"Perturbation numbers: {ds['perturbationNumber'].values}")
```

## Ensemble Statistics

### Basic Statistics

Calculate ensemble mean and spread:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Ensemble mean
    ensemble_mean = ds['t2m'].mean(dim='number')
    
    # Ensemble spread (standard deviation)
    ensemble_spread = ds['t2m'].std(dim='number')
    
    # Ensemble minimum and maximum
    ensemble_min = ds['t2m'].min(dim='number')
    ensemble_max = ds['t2m'].max(dim='number')
    
    # Ensemble range
    ensemble_range = ensemble_max - ensemble_min
    
    print(f"Ensemble mean: {ensemble_mean.mean():.2f} K")
    print(f"Ensemble spread: {ensemble_spread.mean():.2f}2 K")
```

### Percentiles

Calculate ensemble percentiles:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # 10th and 90th percentiles
    p10 = ds['t2m'].quantile(0.10, dim='number')
    p90 = ds['t2m'].quantile(0.90, dim='number')
    
    # Interquartile range
    p25 = ds['t2m'].quantile(0.25, dim='number')
    p75 = ds['t2m'].quantile(0.75, dim='number')
    iqr = p75 - p25
    
    # Median
    median = ds['t2m'].median(dim='number')
```

### Ensemble Moments

Calculate higher-order moments:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Skewness
    ensemble_mean = ds['t2m'].mean(dim='number')
    ensemble_std = ds['t2m'].std(dim='number')
    skewness = ((ds['t2m'] - ensemble_mean)**3).mean(dim='number') / ensemble_std**3
    
    # Kurtosis
    kurtosis = ((ds['t2m'] - ensemble_mean)**4).mean(dim='number') / ensemble_std**4
```

## Probability Calculations

### Exceedance Probabilities

Calculate probability of exceeding thresholds:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Probability of exceeding 300 K
    prob_above_300 = (ds['t2m'] > 300).mean(dim='number')
    
    # Probability of exceeding 305 K
    prob_above_305 = (ds['t2m'] > 305).mean(dim='number')
    
    # Probability of being below 290 K
    prob_below_290 = (ds['t2m'] < 290).mean(dim='number')
    
    # Probability of being between 290 and 300 K
    prob_between = ((ds['t2m'] >= 290) & (ds['t2m'] <= 300)).mean(dim='number')
```

### Categorical Probabilities

Calculate probabilities for categorical events:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Probability of hot day (>300 K)
    prob_hot = (ds['t2m'] > 300).mean(dim='number')
    
    # Probability of cold day (<280 K)
    prob_cold = (ds['t2m'] < 280).mean(dim='number')
    
    # Probability of moderate day (280-300 K)
    prob_moderate = ((ds['t2m'] >= 280) & (ds['t2m'] <= 300)).mean(dim='number')
    
    print(f"P(hot): {prob_hot.mean():.3f}")
    print(f"P(cold): {prob_cold.mean():.3f}")
    print(f"P(moderate): {prob_moderate.mean():.3f}")
```

### Time-Dependent Probabilities

Calculate time-dependent probabilities:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Probability of exceeding threshold at each time step
    prob_time = (ds['t2m'] > 300).mean(dim='number')
    
    # Plot probability over time
    prob_time.mean(dim=['latitude', 'longitude']).plot()
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.title('Probability of T > 300 K')
```

## Ensemble Clustering

### Correlation Analysis

Analyze correlations between ensemble members:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Flatten spatial dimensions
    temp_flat = ds['t2m'].stack(points=('latitude', 'longitude'))
    
    # Calculate correlation matrix between members
    correlation_matrix = xr.corr(temp_flat, dim='points')
    
    # Plot correlation matrix
    import matplotlib.pyplot as plt
    plt.imshow(correlation_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.xlabel('Member')
    plt.ylabel('Member')
    plt.title('Ensemble Member Correlations')
```

### Cluster Analysis

Group similar ensemble members:

```python
import xarray as xr
from sklearn.cluster import KMeans

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Flatten spatial and temporal dimensions
    temp_flat = ds['t2m'].stack(samples=('time', 'latitude', 'longitude'))
    
    # Prepare data for clustering
    data = temp_flat.values.T  # Shape: (n_members, n_samples)
    
    # Perform k-means clustering
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    
    print(f"Cluster assignments: {clusters}")
```

## Ensemble Verification

### Rank Histograms

Create rank histograms for ensemble verification:

```python
import xarray as xr
import numpy as np

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Assume we have verification data
    # This is a simplified example
    
    # Get ensemble members
    ensemble = ds['t2m']
    
    # Sort ensemble members
    sorted_ensemble = np.sort(ensemble.values, axis=0)
    
    # For each grid point, find rank of verification
    # (This requires actual verification data)
    ranks = np.zeros_like(sorted_ensemble[0])
    
    # Create histogram
    histogram, bins = np.histogram(ranks, bins=len(ensemble)+1, range=(0, len(ensemble)+1))
    
    # Plot rank histogram
    plt.bar(bins[:-1], histogram, width=1)
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title('Rank Histogram')
```

### Spread-Skill Relationship

Analyze relationship between ensemble spread and forecast skill:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Calculate ensemble spread
    spread = ds['t2m'].std(dim='number')
    
    # Calculate ensemble mean
    mean = ds['t2m'].mean(dim='number')
    
    # (This would require verification data to calculate skill)
    # For now, just analyze spread
    
    # Spatial distribution of spread
    spread_mean = spread.mean(dim=['latitude', 'longitude'])
    spread_std = spread.std(dim=['latitude', 'longitude'])
    
    print(f"Mean spread: {spread_mean.mean():.2f} K")
    print(f"Spread variability: {spread_std.mean():.2f} K")
```

## Ensemble Post-Processing

### Ensemble Mean Bias Correction

Apply bias correction to ensemble mean:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Calculate ensemble mean
    ensemble_mean = ds['t2m'].mean(dim='number')
    
    # Apply simple bias correction (example: +1 K)
    corrected_mean = ensemble_mean + 1.0
    
    # Apply to all members
    corrected_ensemble = ds['t2m'] + 1.0
```

### Ensemble Reordering

Reorder ensemble members based on criteria:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Calculate mean temperature for each member
    member_means = ds['t2m'].mean(dim=['time', 'latitude', 'longitude'])
    
    # Sort members by mean temperature
    sorted_indices = np.argsort(member_means.values)
    
    # Reorder ensemble
    sorted_ensemble = ds['t2m'].isel(number=sorted_indices)
```

### Ensemble Subset Selection

Select subset of ensemble members:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Select first 10 members
    subset = ds.isel(number=slice(0, 10))
    
    # Select every other member
    subset = ds.isel(number=slice(0, None, 2))
    
    # Select specific members
    subset = ds.isel(number=[0, 5, 10, 15, 20])
```

## Ensemble Visualization

### Ensemble Spaghetti Plot

Create spaghetti plot of ensemble members:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Select location
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Plot each member
    for member in ts.number.values:
        member_ts = ts.sel(number=member)
        member_ts.plot(alpha=0.3, color='blue')
    
    # Plot ensemble mean
    ensemble_mean = ts.mean(dim='number')
    ensemble_mean.plot(color='red', linewidth=2, label='Ensemble Mean')
    
    plt.legend()
    plt.title('Ensemble Spaghetti Plot')
```

### Ensemble Spread Plot

Plot ensemble spread over time:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Select location
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Calculate spread
    spread = ts.std(dim='number')
    
    # Plot spread
    spread.plot()
    plt.xlabel('Time')
    plt.ylabel('Spread (K)')
    plt.title('Ensemble Spread')
```

### Probability Plots

Plot probability of exceeding threshold:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Calculate probability
    prob = (ds['t2m'] > 300).mean(dim='number')
    
    # Plot spatial probability
    prob.isel(time=0).plot(cmap='YlOrRd', levels=[i/10 for i in range(11)])
    plt.title('Probability of T > 300 K')
```

## Best Practices

### 1. Check Ensemble Dimension

```python
# Always check for ensemble dimension
if 'number' in ds.dims:
    # Process ensemble data
    ensemble_mean = ds['t2m'].mean(dim='number')
else:
    # Process deterministic data
    data = ds['t2m']
```

### 2. Use Vectorized Operations

```python
# Good: Vectorized
ensemble_mean = ds['t2m'].mean(dim='number')

# Bad: Loop-based
ensemble_mean = np.zeros_like(ds['t2m'].isel(number=0))
for i in range(len(ds.number)):
    ensemble_mean += ds['t2m'].isel(number=i)
ensemble_mean /= len(ds.number)
```

### 3. Handle Missing Members

```python
# Check for missing ensemble members
if ds.dims['number'] < expected_size:
    print(f"Warning: Expected {expected_size} members, got {ds.dims['number']}")
```

### 4. Validate Ensemble Statistics

```python
# Check that ensemble statistics are reasonable
ensemble_mean = ds['t2m'].mean(dim='number')
ensemble_min = ds['t2m'].min(dim='number')
ensemble_max = ds['t2m'].max(dim='number')

# Verify that mean is within min-max range
if not (ensemble_min <= ensemble_mean).all():
    print("Warning: Ensemble mean outside min-max range")

if not (ensemble_mean <= ensemble_max).all():
    print("Warning: Ensemble mean outside min-max range")
```

## References

- xarray Documentation: https://xarray.pydata.org/
- sklearn.cluster: https://scikit-learn.org/stable/modules/clustering.html
- numpy Documentation: https://numpy.org/doc/