# cfgrib and xarray API Reference

This document provides a complete reference of cfgrib and xarray API for working with GRIB data.

## xarray.open_dataset

Open with cfgrib engine:

```python
import xarray as xr

ds = xr.open_dataset('file.grib', engine='cfgrib')
```

**Parameters:**
- `filename_or_obj`: File path or file-like object
- `engine`: Backend engine (use 'cfgrib')
- `chunks`: Chunk sizes (dict, 'auto', or None)
- `backend_kwargs`: Backend-specific options (dict)

**Returns:**
- `xarray.Dataset`: Dataset object

**Example:**
```python
ds = xr.open_dataset('file.grib', engine='cfgrib', 
                      chunks={'time': 10},
                      backend_kwargs={'indexpath': ''})
```

## xarray.open_mfdataset

Open multiple files:

```python
import xarray as xr

ds = xr.open_mfdataset(['file1.grib', 'file2.grib'], 
                        engine='cfgrib', 
                        combine='by_coords')
```

**Parameters:**
- `filepaths`: List of file paths or glob pattern
- `engine`: Backend engine (use 'cfgrib')
- `combine`: How to combine datasets ('by_coords', 'nested', etc.)
- `concat_dim`: Dimension to concatenate along
- `chunks`: Chunk sizes

**Returns:**
- `xarray.Dataset`: Combined dataset object

**Example:**
```python
import glob
files = glob.glob('data_*.grib')
ds = xr.open_mfdataset(files, engine='cfgrib', combine='by_coords')
```

## Dataset Methods

### Dataset Variables

Access data variables:

```python
# Get list of variables
variables = ds.data_vars

# Access specific variable
temperature = ds['t2m']

# Check if variable exists
if 't2m' in ds:
    temperature = ds['t2m']
```

### Dataset Coordinates

Access coordinates:

```python
# Get list of coordinates
coords = ds.coords

# Access specific coordinate
time = ds.coords['time']
latitude = ds.coords['latitude']
longitude = ds.coords['longitude']

# Check if coordinate exists
if 'time' in ds.coords:
    time = ds.coords['time']
```

### Dataset Dimensions

Access dimensions:

```python
# Get dimensions
dims = ds.dims

# Get dimension sizes
n_time = ds.dims['time']
n_lat = ds.dims['latitude']
n_lon = ds.dims['longitude']

# Print all dimensions
print(ds.dims)
```

### Dataset Attributes

Access metadata attributes:

```python
# Get attributes
attrs = ds.attrs

# Access specific attribute
if 'title' in ds.attrs:
    title = ds.attrs['title']

# Print all attributes
print(ds.attrs)
```

## DataArray Methods

### Selection

Select data using coordinates:

```python
# Single point
point = da.sel(latitude=40.0, longitude=-100.0)

# Multiple points
points = da.sel(latitude=[40.0, 35.0], 
                longitude=[-100.0, -105.0])

# Slices
region = da.sel(latitude=slice(50, 30), 
                 longitude=slice(-120, -90))

# Time selection
time_point = da.sel(time='2024-01-15T12:00:00')
time_range = da.sel(time=slice('2024-01-01', '2024-01-31'))
```

### Index Selection

Select data using indices:

```python
# Single index
first_timestep = da.isel(time=0)

# Multiple indices
timesteps = da.isel(time=[0, 6, 12, 18])

# Slices
first_10 = da.isel(time=slice(0, 10))
```

### Nearest Neighbor

Find nearest grid point:

```python
# Nearest neighbor selection
nearest = da.sel(latitude=40.5, longitude=-100.3, method='nearest')

# Interpolation
interpolated = da.sel(latitude=40.5, longitude=-100.3, method='nearest')
```

### Interpolation

Interpolate to new coordinates:

```python
# Linear interpolation
linear = da.interp(latitude=new_lats, longitude=new_lons, method='linear')

# Cubic interpolation
cubic = da.interp(latitude=new_lats, longitude=new_lons, method='cubic')

# Nearest neighbor
nearest = da.interp(latitude=new_lats, longitude=new_lons, method='nearest')
```

## Statistical Methods

### Basic Statistics

```python
# Mean
mean = da.mean()
mean_dim = da.mean(dim='time')
mean_multi = da.mean(dim=['time', 'latitude'])

# Standard deviation
std = da.std()
std_dim = da.std(dim='time')

# Minimum and maximum
min_val = da.min()
max_val = da.max()

# Median
median = da.median()

# Percentiles
p25 = da.quantile(0.25)
p75 = da.quantile(0.75)
```

### Advanced Statistics

```python
# Sum
total = da.sum()

# Product
product = da.prod()

# Variance
variance = da.var()

# Cumulative sum
cumsum = da.cumsum()

# Cumulative product
cumprod = da.cumprod()

# Argmin and argmax
min_idx = da.argmin()
max_idx = da.argmax()
```

## Mathematical Operations

### Arithmetic Operations

```python
# Addition
result = da + 10
result = da1 + da2

# Subtraction
result = da - 10
result = da1 - da2

# Multiplication
result = da * 2
result = da1 * da2

# Division
result = da / 2
result = da1 / da2

# Power
result = da ** 2
```

### Trigonometric Functions

```python
import numpy as np

# Trigonometric
sin = np.sin(da)
cos = np.cos(da)
tan = np.tan(da)

# Inverse trigonometric
asin = np.arcsin(da)
acos = np.arccos(da)
atan = np.arctan(da)
atan2 = np.arctan2(da1, da2)

# Hyperbolic
sinh = np.sinh(da)
cosh = np.cosh(da)
tanh = np.tanh(da)
```

### Exponential and Logarithmic

```python
import numpy as np

# Exponential
exp = np.exp(da)
exp2 = np.exp2(da)
exp10 = np.exp10(da)

# Logarithmic
log = np.log(da)
log10 = np.log10(da)
log2 = np.log2(da)
```

## Grouping Operations

### Groupby

Group data for aggregation:

```python
# Group by time of day
hourly_mean = da.groupby('time.hour').mean()

# Group by month
monthly_mean = da.groupby('time.month').mean()

# Group by season
seasonal_mean = da.groupby('time.season').mean()

# Group by day of year
daily_mean = da.groupby('time.dayofyear').mean()
```

### Resample

Resample time series:

```python
# Daily means
daily = da.resample(time='1D').mean()

# Monthly means
monthly = da.resample(time='1M').mean()

# Hourly means
hourly = da.resample(time='1H').mean()

# Weekly means
weekly = da.resample(time='1W').mean()
```

### Rolling

Rolling window operations:

```python
# Rolling mean
rolling_mean = da.rolling(time=7).mean()

# Rolling sum
rolling_sum = da.rolling(time=7).sum()

# Rolling standard deviation
rolling_std = da.rolling(time=7).std()

# Rolling minimum and maximum
rolling_min = da.rolling(time=7).min()
rolling_max = da.rolling(time=7).max()
```

## Boolean Operations

### Comparison

```python
# Comparison operators
greater = da > 300
less = da < 250
equal = da == 273.15
not_equal = da != 273.15
greater_equal = da >= 300
less_equal = da <= 300

# Logical operators
and_op = (da > 250) & (da < 300)
or_op = (da < 250) | (da > 300)
not_op = ~(da > 300)
```

### Where

Conditional selection:

```python
# Where condition is true
result = da.where(da > 250)

# Where condition is false, use other value
result = da.where(da > 250, other=np.nan)

# Complex condition
result = da.where((da > 250) & (da < 300), other=np.nan)
```

## Data Manipulation

### Dropna

Remove missing values:

```python
# Drop all NaN values
clean = da.dropna()

# Drop along specific dimension
clean = da.dropna(dim='time')

# Drop where all values are NaN
clean = da.dropna(how='all')
```

### Fillna

Fill missing values:

```python
# Fill with scalar
filled = da.fillna(0)

# Fill with method
filled = da.fillna(method='ffill')  # Forward fill
filled = da.fillna(method='bfill')  # Backward fill

# Interpolate
filled = da.interpolate_na(dim='time', method='linear')
```

### Rename

Rename variables or dimensions:

```python
# Rename variable
renamed = da.rename('t2m')

# Rename dimension
renamed = da.rename({'latitude': 'lat', 'longitude': 'lon'})
```

### Squeeze

Remove size-1 dimensions:

```python
# Squeeze all size-1 dimensions
squeezed = da.squeeze()

# Squeeze specific dimension
squeezed = da.squeeze('time')
```

## Data Conversion

### to_dataframe

Convert to pandas DataFrame:

```python
# Convert entire DataArray
df = da.to_dataframe()

# Convert Dataset
df = ds.to.to_dataframe()
```

### to_series

Convert to pandas Series:

```python
# Convert to Series
series = da.to_series()
```

### to_netcdf

Write to NetCDF:

```python
# Write Dataset
ds.to_netcdf('output.nc')

# Write with encoding
encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}
ds.to_netcdf('output.nc', encoding=encoding)
```

### to_zarr

Write to Zarr:

```python
# Write Dataset
ds.to_zarr('output.zarr')

# Write with encoding
encoding = {var: {'chunksizes': (10, 90, 180)} for var in ds.data_vars}
ds.to_zarr('output.zarr', encoding=encoding)
```

## Common Patterns

### Reading and Processing

```python
import xarray as xr

# Read and process
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select data
    temp = ds['t2m'].sel(latitude=slice(50, 30))
    
    # Calculate statistics
    mean = temp.mean()
    std = temp.std()
    
    print(f"Mean: {mean:.2f}, Std: {std:.2f}")
```

### Time Series Analysis

```python
import xarray as xr

# Extract time series
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select location
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Calculate statistics
    mean = ts.mean()
    trend = np.polyfit(range(len(ts)), ts, 1)[0]
    
    print(f"Mean: {mean:.2f}, Trend: {trend:.6f}")
```

### Spatial Analysis

```python
import xarray as xr

# Spatial analysis
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select time
    temp = ds['t2m'].isel(time=0)
    
    # Calculate spatial statistics
    global_mean = temp.mean()
    zonal_mean = temp.mean(dim='longitude')
    meridional_mean = temp.mean(dim='latitude')
    
    print(f"Global mean: {global_mean:.2f}")
```

### Ensemble Analysis

```python
import xarray as xr

# Ensemble analysis
with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Calculate ensemble statistics
    ensemble_mean = ds['t2m'].mean(dim='number')
    ensemble_spread = ds['t2m'].std(dim='number')
    
    # Calculate probabilities
    prob_above_300 = (ds['t2m'] > 300).mean(dim='number')
    
    print(f"Ensemble mean: {ensemble_mean.mean():.2f}")
    print(f"Ensemble spread: {ensemble_spread.mean():.2f}")
```

## Notes

- Always use context managers for proper resource cleanup
- Use coordinate-based selection for more intuitive code
- Leverage lazy evaluation for better performance
- Use appropriate chunking for large datasets
- Check documentation for complete API reference

## References

- xarray Documentation: https://xarray.pydata.org/
- cfgrib Documentation: https://github.com/ecmwf/cfgrib
- pandas Documentation: https://pandas.pydata.org/