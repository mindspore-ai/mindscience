# Reading Satellite Data in Satpy

Complete guide to reading various satellite file formats.

## Supported File Formats

### NetCDF Files

**Reading NetCDF:**
```python
from satpy import Scene

# Load NetCDF file
scn = Scene('satellite_file.nc')

# Convert to xarray
ds = scn.to_xarray()

# List available datasets
print("Available datasets:")
for name in ds:
    print(f"  {name}")
```

**Accessing datasets:**
```python
# Access specific dataset
dataset = ds['dataset_name']

# Get metadata
print(f"Dataset shape: {dataset.shape}")
print(f"Dataset attributes: {dataset.attrs}")
```

### HDF4/5 Files

**Reading HDF4:**
```python
from satpy import Scene

# Load HDF4 file
scn = Scene('satellite_file.h5')

# Convert to xarray
ds = scn.to_xarray()
```

### GeoTIFF Files

**Reading GeoTIFF:**
```python
from satpy import Scene

# Load GeoTIFF file
scn = Scene('satellite_file.tif')

# Convert to xarray
ds = scn.to_xarray()
```

### CF Files

**Reading CF:**
```python
from satpy import Scene

# Load CF NetCDF file
scn = Scene('satellite_file.nc')

# Convert to xarray
ds = scn.to_xarray()
```

### MODIS Files

**Reading MODIS:**
```python
from satpy import Scene

# Load MODIS file
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Convert to xarray
ds = scn.to_xarray()
```

### VIIRS Files

**Reading VIIRS:**
```python
from satpy import Scene

# Load VIIRS file
scn = Scene('VIIRS_file.nc')

# Convert to xarray
ds = scn.to_xarray()
```

### Landsat Files

**Reading Landsat:**
```python
from satpy import Scene

# Load Landsat file
scn = Scene('Landsat_file.nc')

# Convert to xarray
ds = scn.to_xarray()
```

### GOES Files

**Reading GOES:**
```python
from satpy import Scene

# Load GOES file
scn = Scene('GOES_file.nc')

# Convert to xarray
ds = scn.to_xarray()
```

### SEVIRI Files

**Reading SEVIRI:**
```python
from satpy import Scene

# Load SEVIRI file
scn = Scene('SEVIRI_file.nc')

# Convert to xarray
ds = scn.to_xarray()
```

## Data Access

### Dataset Information

```python
# List all datasets
print("Available datasets:")
for name in ds:
    dataset = ds[name]
    print(f"  {name}: {dataset.shape}")
    print(f"    Attributes: {list(dataset.attrs.keys())}")
```

### Metadata

```python
# Access global metadata
print(f"Scene metadata: {scn.attrs}")

# Access dataset metadata
dataset = ds['dataset_name']
print(f"Dataset metadata: {dataset.attrs}")
```

### Coordinate Variables

```python
# Common coordinate variables
lon = ds['longitude']
lat = ds['latitude']
time = ds['time']

print(f"Longitude shape: {lon.shape}")
print(f"Latitude shape: {lat.shape}")
print(f"Time shape: {time.shape}")
```

## Loading Specific Datasets

### MODIS Datasets

```python
from satpy import Scene

# Load MODIS file
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Load temperature dataset
scn.load(['temperature'])

# Access temperature
temperature = scn['temperature']
print(f"Temperature shape: {temperature.shape}")
```

### VIIRS Datasets

```python
# Load VIIRS file
scn = Scene('VIIRS_file.nc')

# Load atmospheric profiles
scn.load(['temperature', 'humidity', 'cloud_top_pressure'])

# Access data
temperature = scn['temperature']
humidity = scn['humidity']
cloud_top_pressure = scn['cloud_top_pressure']
```

### Landsat Datasets

```python
# Load Landsat file
scn = Scene('Landsat_file.nc')

# Load surface temperature
scn.load(['surface_temperature'])

# Access surface temperature
surface_temp = scn['surface_temperature']
print(f"Surface temperature shape: {surface_temp.shape}")
```

## Common Applications

### MODIS Processing

```python
from satpy import Scene

# Load MODIS file
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Load datasets
scn.load(['temperature', 'cloud_top_pressure'])

# Access data
temperature = scn['temperature']
cloud_top_pressure = scn['cloud_top_pressure']

# Calculate cloud top temperature
ctt = temperature - 273.15  # Convert to Celsius
print(f"Cloud top temperature range: [{ctt.min():.1f}, {ctt.max():.1f}]°C")
```

### VIIRS Analysis

```python
# Load VIIRS file
scn = Scene('VIIRS_file.nc')

# Load atmospheric profiles
scn.load(['temperature', 'humidity', 'cloud_top_pressure'])

# Calculate derived parameters
# See references/thermodynamics.md for details
```

### Landsat Processing

```python
# Load Landsat file
scn = Scene('Landsat_file.nc')

# Load surface temperature
scn.load(['surface_temperature'])

# Access surface temperature
surface_temp = scn['surface_temperature']
print(f"Surface temperature range: [{surface_temp.min():.1f}, {surface_temp.max():.1f}] K")
```

## Numerical Considerations

### Memory Management

**Large files:**
```python
# Use xarray for lazy loading
import xarray as xr

ds = xr.open_dataset('satellite_file.nc')
# Data not loaded until accessed
```

**Chunk processing:**
```python
# Process in time chunks
for time_idx in range(0, len(time), 10):
    chunk = scn.isel(time=slice(time_idx, time_idx+10))
    # Process chunk
```

### Data Validation

**Check for missing values:**
```python
# Check for missing values
import numpy as np

missing_count = np.sum(np.isnan(temperature))
print(f"Missing values: {missing_count}")

# Check for out-of-range values
invalid_count = np.sum((temperature < 150) | (temperature > 350))
print(f"Invalid temperature values: {invalid_count}")
```

**Check coordinate ranges:**
```python
# Validate coordinate ranges
assert -90 <= lat.min() and lat.max() <= 90, "Latitude out of range"
assert -180 <= lon.min() and lon.max() <= 180, "Longitude out of range"
```

## Troubleshooting

### File Not Supported

**Issue:** File format not supported

**Solutions:**
- Check file extension
- Verify Satpy version
- Check for reader plugin
- Consider converting format

### Dataset Not Found

**Issue:** Dataset not found in file

**Solutions:**
- Check dataset name spelling
- Verify file format
- List available datasets
- Check file structure

### Incorrect Array Shape

**Issue:** Unexpected array dimensions

**Solutions:**
- Check coordinate dimensions
- Verify file structure
- Check dataset metadata
- Review satellite documentation

### Memory Errors

**Issue:** Out of memory

**Solutions:**
- Use xarray for large files
- Load only needed datasets
- Process in chunks
- Reduce spatial resolution

## Advanced Topics

### Custom Readers

```python
# Define custom reader
# See Satpy documentation for details
```

### Multiple File Reading

```python
# Read multiple files
files = ['file1.nc', 'file2.nc', 'file3.nc']

for file in files:
    scn = Scene(file)
    ds = scn.to_xarray()
    # Process file
```

### Time Series Processing

```python
# Process time series
for time_idx in range(len(time)):
    time_slice = scn.isel(time=time_idx)
    # Process time slice
```

## Resources

- Satpy reading documentation: https://satpy.readthedocs.io/en/reading.html
- Reader table: https://satpy.readthedocs.io/en/reading.html#reader-table
- Available readers: https://satpy.readthedocs.io/en/reading.html#available-readers
- MODIS reader: https://satpy.readthedocs.io/en/readers/formats.html#modis
- VIIRS reader: https://satpy.readthedocs.io/en/readers/formats.html#virs
