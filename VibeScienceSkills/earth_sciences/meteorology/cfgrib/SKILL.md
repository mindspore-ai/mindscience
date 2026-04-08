---
name: cfgrib
description: Comprehensive skill for working with cfgrib (CF-compliant GRIB reader) to access GRIB meteorological data as xarray Datasets. Use when Claude needs to: (1) Read GRIB files as xarray Datasets, (2) Work with labeled multidimensional arrays with CF-compliant metadata, (3) Perform data analysis using xarray and pandas, (4) Handle multi-message GRIB files with automatic grouping, (5) Access GRIB data with coordinate-based selection, (6) Perform time series analysis on meteorological data, (7) Work with ensemble forecast data, (8) Convert GRIB data to other formats (NetCDF, Zarr), (9) Apply xarray operations to GRIB data, (10) Visualize meteorological data with matplotlib/cartopy
---

# cfgrib

## Overview

cfgrib is a Python interface for reading GRIB files as CF-compliant xarray Datasets. It provides a bridge between the binary GRIB format and the Python data science ecosystem, allowing you to work with meteorological data using familiar tools like xarray, pandas, numpy, and matplotlib.

## Quick Start

**Reading a GRIB file as xarray Dataset:**

```python
import xarray as xr

# Read GRIB file
ds = xr.open_dataset('file.grib', engine='cfgrib')

# Inspect dataset
print(ds)
print(ds.data_vars)
print(ds.coords)
```

**Accessing data variables:**

```python
# Get a specific variable
temperature = ds['t2m']

# Select data by coordinates
temp_at_point = temperature.sel(latitude=40.0, longitude=-100.0)

# Slice data
temp_region = temperature.sel(latitude=slice(50, 30), longitude=slice(-120, -90))

# Time series
temp_timeseries = temperature.sel(latitude=40.0, longitude=-100.0, method='nearest')
```

**Basic analysis:**

```python
# Calculate statistics
mean_temp = ds['t2m'].mean()
max_temp = ds['t2m'].max()
temp_std = ds['t2m'].std()

# Spatial average
spatial_mean = ds['t2m'].mean(dim=['latitude', 'longitude'])

# Time average
time_mean = ds['t2m'].mean(dim='time')
```

## Workflow Decision Tree

1. **What type of operation do you need?**
   - **Reading GRIB data** → Follow "Reading GRIB Files" workflow
   - **Data selection and subsetting**' → Follow "Data Selection" workflow
   - **Data analysis and computation** → Follow "Data Analysis" workflow
   - **Time series analysis** → Follow "Time Series Analysis" workflow
   - **Ensemble data handling** → Follow "Ensemble Data" workflow
   - **Data conversion** → Follow "Data Conversion" workflow
   - **Visualization** → Follow "Visualization" workflow
   - **Performance optimization** → Follow "Performance" workflow

## Reading GRIB Files

### Basic Reading

Read GRIB files as xarray Datasets:

```python
import xarray as xr

# Open GRIB file
ds = xr.open_dataset('file.grib', engine='cfgrib')

# Display dataset information
print(ds)

# Close dataset
ds.close()
```

### Using Context Managers

Use context managers for automatic resource cleanup:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Work with dataset
    temperature = ds['t2m']
    print(temperature.mean())
```

### Reading Multiple Files

Read multiple GRIB files into a combined dataset:

```python
import xarray as xr

# Open multiple files
ds = xr.open_mfdataset(['file1.grib', 'file2.grib', 'file3.grib'], 
                        engine='cfgrib', combine='by_coords')

# Or use glob pattern
import glob
files = glob.glob('data_*.grib')
ds = xr.open_mfdataset(files, engine='cfgrib', combine='by_coords')
```

### Reading with Backend Options

Configure backend behavior with options:

```python
import xarray as xr

# Read with specific options
ds = xr.open_dataset('file.grib', engine='cfgrib', 
                      backend_kwargs={
                          'indexpath': '',  # Disable index file
                          'filter_by_keys': {'typeOfLevel': 'surface'},  # Filter messages
                          'encode_cf': ['time', 'geography', 'vertical']  # CF encoding
                      })
```

See `references/backend_options.md` for complete backend options reference.

## Data Selection

### Coordinate-Based Selection

Select data using coordinate labels:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select single point
    point = ds['t2m'].sel(latitude=40.0, longitude=-100.0)
    
    # Select multiple points
    points = ds['t2m'].sel(latitude=[40.0, 35.0, 30.0], 
                           longitude=[-100.0, -105.0, -110.0])
    
    # Slice data
    region = ds['t2m'].sel(latitude=slice(50, 30), 
                           longitude=slice(-120, -90))
    
    # Time selection
    time_slice = ds['t2m'].sel(time=slice('2024-01-01', '2024-01-31'))
```

### Nearest Neighbor Selection

Use nearest neighbor for coordinates not exactly on grid:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Find nearest grid point
    nearest = ds['t2m'].sel(latitude=40.5, longitude=-100.3, method='nearest')
    
    # Or use interpolate
    interpolated = ds['t2m'].interp(latitude=40.5, longitude=-100.3)
```

### Level Selection

Select specific vertical levels:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select single level
    temp_500 = ds['t'].sel(level=500)
    
    # Select multiple levels
    temp_levels = ds['t'].sel(level=[1000, 850, 700, 500, 300])
    
    # Slice levels
    temp_mid = ds['t'].sel(level=slice(850, 300))
```

### Time Selection

Select data by time:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select specific time
    specific_time = ds['t2m'].sel(time='2024-01-15T12:00:00')
    
    # Select by date range
    date_range = ds['t2m'].sel(time=slice('2024-01-01', '2024-01-31'))
    
    # Select by index
    first_timestep = ds['t2m'].isel(time=0)
    
    # Select multiple timesteps
    multiple = ds['t2m'].isel(time=[0, 6, 12, 18])
```

### Boolean Selection

Select data using boolean conditions:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Select where condition is true
    high_temp = ds['t2m'].where(ds['t2m'] > 300)
    
    # Mask data
    masked = ds['t2m'].where(ds['t2m'] > 250, other=np.nan)
    
    # Select by coordinate condition
    northern_hemisphere = ds['t2m'].where(ds['latitude'] > 0)
```

## Data Analysis

### Statistical Operations

Perform statistical calculations:

```python
import xarray as xr
import numpy as np

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Basic statistics
    mean = ds['t2m'].mean()
    std = ds['t2m'].std()
    min_val = ds['t2m'].min()
    max_val = ds['t2m'].max()
    median = ds['t2m'].median()
    
    # Percentiles
    p25 = ds['t2m'].quantile(0.25)
    p75 = ds['t2m'].quantile(0.75)
    
    # Statistics along dimensions
    spatial_mean = ds['t2m'].mean(dim=['latitude', 'longitude'])
    time_mean = ds['t2m'].mean(dim='time')
    
    # Weighted mean (e.g., by latitude)
    weights = np.cos(np.deg2rad(ds['latitude']))
    weighted_mean = ds['t2m'].weightedmean(weights)
```

### Mathematical Operations

Apply mathematical operations:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Arithmetic operations
    temp_celsius = ds['t2m'] - 273.15
    temp_kelvin = temp_celsius + 273.15
    
    # Unit conversions
    pressure_hpa = ds['sp'] / 100  # Pa to hPa
    wind_speed = np.sqrt(ds['u10']**2 + ds['v10']**2)
    
    # Trigonometric functions
    wind_dir = np.arctan2(ds['v10'], ds['u10']) * 180 / np.pi
    
    # Exponential and logarithmic
    vapor_pressure = 6.112 * np.exp(17.67 * (ds['t2m'] - 273.15) / (ds['t2m'] - 29.65))
```

### Spatial Operations

Perform spatial analysis:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Spatial average
    global_mean = ds['t2m'].mean(dim=['latitude', 'longitude'])
    
    # Regional average
    regional_mean = ds['t2m'].sel(
        latitude=slice(50, 30),
        longitude=slice(-120, -90)
    ).mean(dim=['latitude', 'longitude'])
    
    # Zonal average (average over longitudes)
    zonal_mean = ds['t2m'].mean(dim='longitude')
    
    # Meridional average (average over latitudes)
    meridional_mean = ds['t2m'].mean(dim='latitude')
    
    # Spatial gradients
    dlat = ds['t2m'].differentiate('latitude')
    dlon = ds['t2m'].differentiate('longitude')
```

See `references/spatial_analysis.md` for advanced spatial operations.

### Temporal Operations

Perform time-based analysis:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Time average
    time_mean = ds['t2m'].mean(dim='time')
    
    # Time series at a point
    point_timeseries = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Daily means
    daily_mean = ds['t2m'].resample(time='1D').mean()
    
    # Monthly means
    monthly_mean = ds['t2m'].resample(time='1M').mean()
    
    # Rolling statistics
    rolling_mean = ds['t2m'].rolling(time=7).mean()
    
    # Time differences
    temp_change = ds['t2m'].diff('time')
    
    # Cumulative sum (e.g., for precipitation)
    accum_precip = ds['tp'].cumsum(dim='time')
```

### Grouping Operations

Group data for analysis:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Group by time of day
    hourly_mean = ds['t2m'].groupby('time.hour').mean()
    
    # Group by month
    monthly_climatology = ds['t2m'].groupby('time.month').mean()
    
    # Group by season
    seasonal_mean = ds['t2m'].groupby('time.season').mean()
    
    # Group by level
    level_mean = ds['t'].groupby('level').mean()
```

## Time Series Analysis

### Extracting Time Series

Extract time series for specific locations:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Time series at a point
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Convert to pandas Series
    ts_series = ts.to_series()
    
    # Plot time series
    ts.plot()
```

### Time Series Statistics

Calculate time series statistics:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Trend (linear regression)
    time_numeric = (ts.time - ts.time[0]).dt.days
    trend = np.polyfit(time_numeric, ts, 1)
    
    # Anomalies
    climatology = ts.groupby('time.dayofyear').mean()
    anomalies = ts.groupby('time.dayofyear') - climatology
    
    # Seasonal decomposition
    seasonal = ts.groupby('time.month').mean()
    detrended = ts - ts.mean()
```

### Periodic Analysis

Analyze periodic patterns:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    
    # Diurnal cycle
    diurnal = ts.groupby('time.hour').mean()
    
    # Annual cycle
    annual = ts.groupby('time.dayofyear').mean()
    
    # Fourier analysis
    from scipy.fft import fft, fftfreq
    fft_values = fft(ts.values)
    frequencies = fftfreq(len(ts), d=6)  # 6-hourly data
```

## Ensemble Data

### Reading Ensemble Data

Read ensemble forecast data:

```python
import xarray as xr

# Read ensemble data
ds = xr.open_dataset('ensemble.grib', engine='cfgrib')

# Ensemble dimension
print(ds.dims)  # Should include 'number' dimension

# Access ensemble members
member_0 = ds.isel(number=0)
member_1 = ds.isel(number=1)
```

### Ensemble Statistics

Calculate ensemble statistics:

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
    
    # Ensemble percentiles
    p10 = ds['t2m'].quantile(0.10, dim='number')
    p90 = ds['t2m'].quantile(0.90, dim='number')
    
    # Probability of exceeding threshold
    prob_above_300 = (ds['t2m'] > 300).mean(dim='number')
```

### Ensemble Clustering

Cluster ensemble members:

```python
import xarray as xr

with xr.open_dataset('ensemble.grib', engine='cfgrib') as ds:
    # Flatten spatial dimensions for clustering
    temp_flat = ds['t2m'].stack(points=('latitude', 'longitude'))
    
    # Calculate correlation between members
    correlation = xr.corr(temp_flat, dim='points')
    
    # Find similar members
    similar = correlation.where(correlation > 0.8)
```

## Data Conversion

### Converting to NetCDF

Convert GRIB to NetCDF format:

```python
import xarray as xr

# Read GRIB
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Write to NetCDF
    ds.to_netcdf('output.nc')
    
    # With compression
    ds.to_netcdf('output_compressed.nc', 
                 encoding={var: {'zlib': True, 'complevel': 5} 
                            for var in ds.data_vars})
```

### Converting to Zarr

Convert to Zarr format for cloud storage:

```python
import xarray as xr

# Read GRIB
with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Write to Zarr
    ds.to_zarr('output.zarr')
    
    # With chunking
    ds.to_zarr('output_chunked.zarr', 
               encoding={var: {'chunksizes': (10, 181, 360)} 
                          for var in ds.data_vars})
```

### Converting to Pandas

Convert to pandas DataFrames:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Convert to DataFrame (flattens dimensions)
    df = ds.to_dataframe()
    
    # Convert specific variable
    temp_df = ds['t2m'].to_dataframe()
    
    # Convert time series
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    ts_series = ts.to_series()
```

### Converting to NumPy

Convert to NumPy arrays:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Convert to numpy array
    temp_array = ds['t2m'].values
    
    # Keep coordinates
    temp_dataarray = ds['t2m']
    
    # Convert specific slice
    temp_slice = ds['t2m'].isel(time=0).values
```

Use `scripts/convert_format.py` for format conversion utilities.

## Visualization

### Basic Plotting

Create basic plots:

```python
import xarray as xr
import matplotlib.pyplot as plt

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Plot time series
    ts = ds['t2m'].sel(latitude=40.0, longitude=-100.0, method='nearest')
    ts.plot()
    plt.show()
    
    # Plot spatial field
    ds['t2m'].isel(time=0).plot()
    plt.show()
    
    # Plot zonal mean
    zonal_mean = ds['t2m'].mean(dim='longitude')
    zonal_mean.plot()
    plt.show()
```

### Custom Plots

Create custom visualizations:

```python
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Create map plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    
    # Plot data
    ds['t2m'].isel(time=0).plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        cbar_kwargs={'label': 'Temperature (K)'}
    )
    
    # Add coastlines
    ax.coastlines()
    ax.gridlines()
    
    plt.show()
```

See `references/visualization.md` for advanced visualization techniques.

## Performance

### Lazy Evaluation

cfgrib uses lazy evaluation for efficient data access:

```python
import xarray as xr

# Data is not loaded until needed
ds = xr.open_dataset('large_file.grib', engine='cfgrib')

# Only load when you access values
mean_temp = ds['t2m'].mean()  # This triggers loading

# Use .load() to explicitly load data
ds_loaded = ds.load()
```

### Chunking

Control data chunking for performance:

```python
import xarray as xr

# Read with chunking
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks={'time': 10})

# Rechunk
ds_chunked = ds.chunk({'time': 10, 'latitude': 90, 'longitude': 180})
```

### Using Dask

Use Dask for parallel processing:

```python
import xarray as xr

# Read with Dask
ds = xr.open_dataset('file.grib', engine='cfgrib', chunks='auto')

# Parallel computation
result = ds['t2m'].mean(dim='time').compute()
```

See `references/performance.md` for performance optimization techniques.

## Error Handling

### Common Errors

**"Backend cfgrib cannot be opened"**
- Install cfgrib: `pip install cfgrib`
- Check file is valid GRIB format

**"Variable not found"**
- Check variable name in dataset: `print(ds.data_vars)`
- Use correct variable name

**"Coordinate not found"**
- Check coordinate names: `print(ds.coords)`
- Use correct coordinate names

**"Memory error"**
- Use chunking: `chunks={'time': 10}`
- Process data in smaller pieces
- Use Dask for out-of-core computation

### Validation

Validate GRIB file structure:

```python
# Use scripts/validate_grib.py
python scripts/validate_grib.py file.grib
```

## Resources

### scripts/
Executable Python scripts for common cfgrib operations:

- **read_grib.py** - Read and display GRIB file contents as xarray Dataset
- **extract_timeseries.py** - Extract time series for specific locations
- **calculate_statistics.py** - Calculate statistical summaries of GRIB data
- **spatial_subset.py** - Extract spatial subsets from GRIB files
- **temporal_subset.py** - Extract temporal subsets from GRIB files
- **ensemble_analysis.py** - Analyze ensemble forecast data
- **convert_format.py** - Convert GRIB data to other formats (NetCDF, Zarr, CSV)
- **plot_data.py** - Create visualizations of GRIB data
- **compare_datasets.py** - Compare two GRIB datasets
- **validate_grib.py** - Validate GRIB file structure and contents

### references/
Detailed documentation and reference materials:

- **backend_options.md** - Complete reference of cfgrib backend options and configuration
- **data_model.md** - Detailed explanation of cfgrib data model and coordinate systems
- **spatial_analysis.md** - Advanced spatial analysis techniques and operations
- **time_series_analysis.md** - Time series analysis methods and patterns
- **ensemble_methods.md** - Ensemble data handling and statistical methods
- **visualization.md** - Visualization techniques and plotting examples
- **performance.md** - Performance optimization and memory management
- **api_reference.md** - Complete cfgrib and xarray API reference
- **common_patterns.md** - Common use cases and best practices

### assets/
Example files and templates:

- **sample_grib.grib** - Sample GRIB file for testing and demonstration
- **config_template.yaml** - Configuration template for cfgrib operations
- **plot_template.py** - Template for creating custom plots
- **analysis_template.py** - Template for data analysis workflows

## Best Practices

1. **Use context managers** - Ensure proper resource cleanup
2. **Leverage lazy evaluation** - Only load data when needed
3. **Use appropriate chunking** - Optimize performance for your data size
4. **Validate data** - Check data ranges and quality before analysis
5. **Document your workflow** - Keep track of data processing steps
6. **Use coordinate-based selection** - More intuitive than index-based
7. **Handle missing data** - Use `.where()` and `.fillna()` appropriately
8. **Close datasets** - Explicitly close or use context managers

## Integration with Other Tools

### xarray Operations

cfgrib returns xarray Datasets, enabling all xarray operations:

```python
import xarray as xr

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # All xarray operations available
    result = ds['t2m'].groupby('time.month').mean().rolling(time=3).mean()
```

### pandas Integration

Convert to pandas for tabular analysis:

```python
import xarray as xr
import pandas as pd

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Convert to DataFrame
    df = ds.to_dataframe()
    
    # Use pandas operations
    result = df.groupby('time.month').mean()
```

### NumPy Integration

Access underlying NumPy arrays:

```python
import xarray as xr
import numpy as np

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Access numpy array
    temp_array = ds['t2m'].values
    
    # Use NumPy operations
    result = np.mean(temp_array)
```

### Scientific Stack Integration

Works with scipy, sklearn, and other scientific libraries:

```python
import xarray as xr
from scipy import stats

with xr.open_dataset('file.grib', engine='cfgrib') as ds:
    # Use scipy
    correlation = stats.pearsonr(ds['t2m'].values, ds['msl'].values)
```

## Troubleshooting

### Common Issues

**"ImportError: No module named 'cfgrib'"**
- Install cfgrib: `pip install cfgrib`

**"File not found" error**
- Check file path is correct
- Use absolute paths

**"Memory exhausted"**
- Use chunking: `chunks={'time': 10}`
- Process in smaller pieces
- Use Dask

**"Slow performance"**
- Use appropriate chunking
- Enable caching
- Use Dask for parallel processing