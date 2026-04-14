# Writing Satellite Data in Satpy

Complete guide to writing satellite data to various output formats.

## Output Formats

### PNG Files

**Saving RGB image:**
```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Create RGB composite
scn.load(['red_band', 'green_band', 'blue_band'])
rgb = scn['RGB']

# Save to PNG
rgb.save('output.png')
```

**Saving with custom settings:**
```python
# Save with quality settings
rgb.save('output.png', 
           fill_value=0,
           dpi=300,
           compression=9)
```

### GeoTIFF Files

**Saving dataset:**
```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Get dataset
dataset = scn['dataset_name']

# Save to GeoTIFF
dataset.save('output.tif')
```

### NetCDF Files

**Saving dataset:**
```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Get dataset
dataset = scn['dataset_name']

# Save to NetCDF
dataset.save('output.nc')
```

### CF Files

**Saving dataset:**
```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Get dataset
dataset = scn['dataset_name']

# Save to CF NetCDF
dataset.save('output.cf')
```

## Multi-Scene Writing

### Saving Multiple Scenes

```python
from satpy import Scene

# Load multiple scenes
scn1 = Scene('file1.nc')
scn2 = Scene('file2.nc')

# Create composites
rgb1 = scn1['RGB']
rgb2 = scn2['RGB']

# Save scenes
rgb1.save('output1.png')
rgb2.save('output2.png')
```

### Blending Scenes

```python
# Create composite from multiple scenes
# See references/compositing.md for details
```

## Writing Metadata

### Adding Global Metadata

```python
# Add global metadata
scn.attrs['title'] = 'Satellite Data'
scn.attrs['institution'] = 'My Institution'
scn.attrs['contact'] = 'email@example.com'
```

### Adding Dataset Metadata

```python
# Add dataset metadata
dataset = scn['dataset_name']
dataset.attrs['description'] = 'Dataset description'
dataset.attrs['units'] = 'K'
dataset.attrs['long_name'] = 'Temperature'
dataset.attrs['standard_name'] = 'Air Temperature'
```

### Adding Coordinate Metadata

```python
# Add coordinate metadata
lon = scn['longitude']
lat = scn['latitude']

lon.attrs['long_name'] = 'Longitude'
lon.attrs['units'] = 'degrees_east'
lat.attrs['lat_name'] = 'Latitude'
lat.attrs['units'] = 'degrees_north'
```

## Common Applications

### MODIS Output

```python
from satpy import Scene

# Load MODIS scene
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Load temperature
scn.load(['temperature'])

# Save temperature to NetCDF
temperature = scn['temperature']
temperature.save('MODIS_temperature.nc')
```

### VIIRS Output

```python
from satpy import Scene

# Load VIIRS scene
scn = Scene('VIIRS_file.nc')

# Load atmospheric profiles
scn.load(['temperature', 'humidity', 'cloud_top_pressure'])

# Save profiles to NetCDF
temperature.save('VIIRS_temperature.nc')
humidity.save('VIIRS_humidity.nc')
cloud_top_pressure.save('VIIRS_cloud_top_pressure.nc')
```

### Landsat Output

```python
from satpy import Scene

# Load Landsat scene
scn = Scene('Landsat_file.nc')

# Load surface temperature
scn.load(['surface_temperature'])

# Save to NetCDF
surface_temperature = scn['surface_temperature']
surface_temperature.save('Landsat_surface_temperature.nc')
```

### GOES Output

```python
from satpy import Scene

# Load GOES scene
scn = Scene('GOES_file.nc')

# Load geophysical parameters
scn.load(['surface_temperature', 'surface_pressure'])

# Save to NetCDF
surface_temperature = scn['surface_temperature']
surface_pressure = scn['surface_pressure']
surface_temperature.save('GOES_surface_temperature.nc')
surface_pressure.save('GOES_surface_pressure.nc')
```

## Best Practices

### 1. Output Format Selection
- **PNG**: For visualization and quick viewing
- **GeoTIFF**: For geospatial analysis software
- **NetCDF**: For data sharing and archival
- **CF**: For compatibility with CF conventions

### 2. Image Quality
- Use appropriate DPI for resolution
- Apply appropriate compression
- Use appropriate fill values
- Consider bit depth for color depth

### 3. Metadata
- Include descriptive metadata
- Add contact information
- Document processing steps
- Include coordinate system information

### 4. Coordinate Systems
- Use appropriate projection for region
- Set correct coordinate units
- Document coordinate transformations
- Validate coordinate ranges

### 5. Performance
- Process in chunks for large datasets
- Use appropriate data types
- Close files after writing
- Use vectorized operations

## Troubleshooting

### File Writing Errors

**Issue:** Cannot write to file

**Solutions:**
- Check file permissions
- Check disk space
- Verify file path
- Check output format support

### Incorrect Output

**Issue:** Output has incorrect data

**Solutions:**
- Verify data ranges
- Check coordinate systems
- Validate metadata
- Check enhancement sequence

### Memory Errors

**Issue:** Out of memory

**Solutions:**
- Process in smaller chunks
- Reduce spatial resolution
- Use appropriate data types
- Close files after writing

### Format Errors

**Issue:** Output format not supported

**Solutions:**
- Check format support
- Verify file extension
- Check for required libraries
- Consult documentation

## Advanced Topics

### Custom Writers

```python
# Define custom writer
# See Satpy documentation for details
```

### Multi-File Output

```python
# Write to multiple formats at once
# See Satpy documentation for details
```

### Batch Processing

```python
# Process multiple files in batch
# See references/advanced_usage.md for details
```

## Resources

- Satpy writing: https://satpy.readthedocs.io/en/writing.html
- Satpy API: https://satpy.readthedocs.io/en/api/modules.html
- File formats: https://satpy.readthedocs.io/en/writing.html#file-formats
