---
name: satpy
description: Python library for reading, manipulating, and writing data from remote-sensing earth-observing satellite instruments. Use when working with satellite data for:(1) Reading various satellite file formats (HDF5, NetCDF, GeoTIFF, etc.), (2) Creating RGB images from multiple bands, (3) Resampling satellite data to different grids, (4) Applying atmospheric corrections (cloud detection, etc.), (5) Creating geographic projections, (6) Writing to multiple output formats (PNG, GeoTIFF, NetCDF), or (7) Processing data from specific satellites (MODIS, VIIRS, SEVIRI, Landsat, GOES, etc.)
license: BSD-3-Clause
metadata:
    skill-author: K-Dense Inc.
---

# Satpy

Python library for reading, manipulating, and writing data from remote-sensing earth-observing satellite instruments.

## Overview

Satpy provides readers that convert geophysical parameters from various file formats to common xarray DataArray and Dataset classes for easier interoperability with other scientific Python libraries. Satpy also provides interfaces for creating RGB (Red/Green/Blue) images and other composite types by combining data from multiple instrument bands or products. Various atmospheric corrections and visual enhancements are provided for improving usefulness and quality of output images.

## Quick Start

**Installation:**
```bash
conda install -c conda-forge satpy
# or
pip install satpy
```

**Basic file reading:**
```python
from satpy import Scene

# Load satellite file
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Get dataset
ds = scn.to_xarray()

# List available datasets
print("Available datasets:")
for name in ds:
    print(f"  {name}")
```

**Creating RGB image:**
```python
from satpy import Scene

# Load scene
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Load RGB bands
scn.load(['MYD06', 'MYD07', 'MYD08'])

# Create RGB composite
rgb = scn['RGB']

# Save to file
rgb.save('output.png')
```

## Core Workflow

### 1. Reading Satellite Data

**Supported file formats:**
- NetCDF3/4
- HDF4/5
- GeoTIFF
- CF
- SAFE
- AAPP
- And many more

**Basic reading:**
```python
from satpy import Scene

# Load file
scn = Scene('satellite_file.nc')

# Convert to xarray
ds = scn.to_xarray()

# Access dataset
dataset = ds['dataset_name']
print(f"Dataset shape: {dataset.shape}")
```

**Reading specific datasets:**
```python
# Load specific datasets
scn.load(['dataset1', 'dataset2'])

# Access loaded data
data1 = scn['dataset1']
data2 = scn['dataset2']
```

### 2. Data Selection

**Filter by region:**
```python
from satpy import Scene

scn = Scene('satellite_file.nc')

# Load data
scn.load(['temperature', 'humidity'])

# Filter by region
swath = scn.swath_spatial_rectangle_ll(
    name='region_name',
    llcrn=(lon_min, lat_min, lon_max, lat_max)
)

# Apply filter
swath_scn = scn.where(swath)
```

**Filter by time:**
```python
# Filter by time range
scn = scn.between_time(
    start_time='2020-01-01T00:00:00',
    end_time='2020-01-01T23:59:59'
)
```

### 3. Creating Composites

**RGB composite:**
```python
from satpy import Scene

scn = Scene('satellite_file.nc')

# Load RGB bands
scn.load(['red_band', 'green_band', 'blue_band'])

# Create RGB composite
rgb = scn['RGB']

# Save to file
rgb.save('rgb_image.png')
```

**True color composite:**
```python
# Load true color bands
scn.load(['true_color_band'])

# Create true color composite
true_color = scn['true_color']
```

**Custom composite:**
```python
# See references/compositing.md for details
# Define custom composite configuration
```

### 4. Resampling

**Geographic resampling:**
```python
from satpy import Scene

scn = Scene('satellite_file.nc')

# Resample to specific grid
swath = scn.resample('my_area', 5000)  # 5000m resolution

# Apply resampling
resampled_scn = scn.where(swath)
```

**Nearest neighbor resampling:**
```python
# Resample using nearest neighbor
swath = scn.resample('my_area', 5000, resampler='nn')
```

### 5. Enhancements

**Cloud detection:**
```python
# See references/enhancements.md for details
# Apply cloud detection
```

**Atmospheric corrections:**
```python
# Apply atmospheric corrections
# See references/enhancements.md for details
```

**Sun glint correction:**
```python
# Apply sun glint correction
# See references/enhancements.md for details
```

### 6. Writing Data

**Saving to PNG:**
```python
# Save RGB image
rgb.save('output.png')
```

**Saving to GeoTIFF:**
```python
# Save to GeoTIFF format
dataset.save('output.tif')
```

**Saving to NetCDF:**
```
# Save to NetCDF format
dataset.save('output.nc')
```

## Common Applications

### MODIS Data Processing

```python
from satpy import Scene

# Load MODIS file
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Load temperature and cloud top pressure
scn.load(['T', 'PS'])

# Create composite
# See references/compositing.md for details
```

### VIIRS Processing

```python
# Load VIIRS data
scn = Scene('VIIRS_file.nc')

# Process VIIRS data
# See references/reading.md for details
```

### Landsat Processing

```python
# Load Landsat data
scn = Scene('Landsat_file.nc')

# Process Landsat data
# See references/reading.md for details
```

### GOES Processing

```python
# Load GOES data
scn = Scene('GOES_file.nc')

# Process GOES data
# See references/reading.md for details
```

## Best Practices

### 1. File Format Selection
- Use appropriate reader for file format
- Check file format compatibility
- Handle missing datasets gracefully
- Validate file structure

### 2. Memory Management
- Use xarray for large files
- Load only needed datasets
- Use chunking for very large files
- Close files after reading

### 3. Data Selection
- Use appropriate spatial/temporal filters
- Filter by region of interest
- Use time slicing for large datasets
- Apply quality filters (cloud detection, etc.)

### 4. Composite Creation
- Use appropriate band combinations
- Apply atmospheric corrections
- Use proper enhancement sequence
- Validate output quality

### 5. Resampling Strategy
- Choose appropriate resampling method
- Consider output requirements
- Balance quality vs. performance
- Use appropriate resolution

### 6. Output Format
- Use appropriate format for application
- Include metadata in output files
- Use compression for large files
- Validate output files

## Resources

### Scripts

**`scripts/template_reader.py`**
Basic satellite data reading template.

**`scripts/template_composite.py`**
RGB composite creation template.

**`scripts/template_resample.py`**
Data resampling template.

**`scripts/template_writer.py`**
Data writing template.

**`scripts/template_rgb.py`**
RGB image creation template.

### References

- **`references/reading.md`** - Reading various satellite file formats
- **`references/compositing.md`** - Creating composites from multiple bands
- **`references/resampling.md`** - Resampling to different grids
- **`references/enhancements.md`** - Atmospheric corrections and enhancements
- **`references/writing.md`** - Writing to various output formats
- **`references/coordinate_systems.md`** - Satellite coordinate systems
- **`references/advanced_usage.md`** - Advanced features and performance tips

## Common File Formats

### Common Satellite File Formats

**MODIS:**
- NASA's MODIS instrument
- Temperature, humidity, cloud properties
- See references/reading.md for details

**VIIRS:**
- NOAA's VIIRS instrument
- Atmospheric profiles
- See references/reading.md for details

**Landsat:**
- EUMETSAT Landsat data
- Land surface temperature
- See references/reading.md for details

**GOES:**
- GOES instrument data
- Various geophysical parameters
- See references/reading.md for details

**SEVIRI:**
- SEVIRI instrument data
- Atmospheric profiles
- See references/reading.md for details

## Common Issues

**File not supported:**
- Check file format
- Verify Satpy version
- Check for reader plugin
- Consider converting format

**Memory issues:**
- Use xarray for large files
- Load only needed datasets
- Process in chunks
- Close files after reading

**Dataset not found:**
- Check dataset name spelling
- Verify file format
- Check file structure
- List available datasets

**Incorrect array shape:**
- Check coordinate dimensions
- Verify dataset structure
- Check file metadata
- Validate spatial/temporal dimensions

**Output format issues:**
- Check output format support
- Verify file permissions
- Validate output data
- Check disk space

## Additional Resources

- Official documentation: https://satpy.readthedocs.io/
- GitHub repository: https://github.com/pytroll/satpy
- PyTroll group: http://pytroll.github.io/
- Reader table: https://satpy.readthedocs.io/en/reading.html#reader-table
