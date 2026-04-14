# Creating Composites in Satpy

Complete guide to creating composites from multiple satellite bands.

## RGB Composites

### Basic RGB Creation

```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Load RGB bands
scn.load(['red_band', 'green_band', 'blue_band'])

# Create RGB composite
rgb = scn['RGB']

# Save to file
rgb.save('rgb_image.png')
```

### True Color Composite

```python
# Load true color bands
scn.load(['true_color_band'])

# Create true color composite
true_color = scn['true_color']
```

## Custom Composites

### Defining Custom Composite

```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Load bands
scn.load(['band1', 'band2', 'band3'])

# Create custom composite
# See references/compositing.md for details
```

### Multi-Band Composites

```python
# Load multiple bands
bands = ['band1', 'band2', 'band3', 'band4', 'band5']
scn.load(bands)

# Create composite
composite = scn['custom_composite_name']
```

## Common Applications

### Natural Color Composites

```python
# MODIS natural color
scn = Scene('MODIS_A20190106_060328_20190106_060328.nc')

# Load natural color bands
scn.load(['natural_color'])

# Create composite
natural_color = scn['natural_color']
```

### Enhanced Infrared Composites

```python
# Enhanced infrared composite
# See references/enhancements.md for details
```

### Vegetation Indices

```python
# Calculate vegetation indices
# See references/enhancements.md for details
```

## Best Practices

### Band Selection

**Guidelines:**
- Choose appropriate bands for application
- Check band availability
- Consider spatial resolution
- Check temporal resolution

### Composite Configuration

**Guidelines:**
- Define composite clearly
- Include enhancement steps
- Validate output quality
- Document composite purpose

### Output Format

**Guidelines:**
- Choose appropriate format (PNG, GeoTIFF, NetCDF)
- Include metadata in output
- Use appropriate compression
- Validate output file

## Troubleshooting

### Band Not Found

**Issue:** Band not available in file

**Solutions:**
- Check band name spelling
- Verify file format
- Check satellite documentation
- List available bands

### Composite Creation Failed

**Issue:** Composite creation fails

**Solutions:**
- Check band compatibility
- Verify enhancement steps
- Check output permissions
- Review Satpy documentation

### Output Issues

**Issue:** Output file not created

**Solutions:**
- Check write permissions
- Verify disk space
- Check output format support
- Validate output data

## Advanced Topics

### Dynamic Composites

```python
# Create composites dynamically
# See Satpy documentation for details
```

### Multi-Scene Composites

```python
# Combine data from multiple scenes
# See Satpy documentation for details
```

### Custom Enhancements

```python
# Define custom enhancements
# See Satpy documentation for details
```

## Resources

- Satpy composites: https://satpy.readthedocs.io/en/composites.html
- Built-in composites: https://satpy.readthedocs.io/en/composites.html#built-in-composites
- Satpy API: https://satpy.readthedocs.io/en/api/modules.html
