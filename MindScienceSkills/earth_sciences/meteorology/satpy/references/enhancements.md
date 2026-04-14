# Atmospheric Enhancements in Satpy

Complete guide to atmospheric corrections and visual enhancements.

## Cloud Detection

### Basic Cloud Detection

```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Apply cloud detection
scn.clouds.clear()
scn.clouds['mask'] = 'cf_mask'
```

### Cloud Mask Types

```python
# Available cloud masks
# See Satpy documentation for complete list

# CF mask (Cloud from reflectivity)
scn.clouds['mask'] = 'cf_mask'

# Custom threshold
scn.clouds['mask'] = 'cf_mask'
scn.clouds['cf_threshold'] = 0.05  # 5% threshold
```

## Sun Glint Correction

### Basic Sun Glint

```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Apply sun glint correction
scn.sunz.clear()
scn.sunz['mask'] = 'sunz_mask'
```

### Sun Glint Parameters

```python
# Sun zenith angle
scn.sunz['sunz'] = 45.0  # degrees

# Solar constant
scn.sunz['solar_constant'] = 1350.0

# Day of year
scn.sunz['day_of_year'] = 180
```

## Atmospheric Corrections

### MODIS Correction

```python
from satpy import Scene

# Load MODIS scene
scn = Scene('MODIS_file.nc')

# Apply MODIS correction
scn.corrections.clear()
scn.corrections['mask'] = 'modis_mask'
```

### VIIRS Correction

```python
# Load VIIRS scene
scn = Scene('VIIRS_file.nc')

# Apply VIIRS correction
scn.corrections.clear()
scn.corrections['mask'] = 'viris_mask'
```

## Image Enhancements

### Contrast Stretching

```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Apply contrast stretching
scn.enhancements.clear()
scn.enhancements['contrast_stretch'] = {
    'min_value': 0.0,
    'max_value': 255.0,
    'name': 'linear'
}
```

### Gamma Correction

```python
# Apply gamma correction
scn.enhancements['gamma'] = 2.2
```

### Histogram Equalization

```python
# Apply histogram equalization
scn.enhancements['histogram_equalization'] = True
```

## Writing Enhancement Functions

### Custom Enhancement

```python
# Define custom enhancement function
def custom_enhancement(scn):
    # Apply custom processing
    pass

# Apply enhancement
scn.enhancements['custom'] = custom_enhancement
```

### Writing Enhancement

```python
# Write enhancement to file
# See Satpy documentation for details
```

## Enhancement Order

### Recommended Order

```python
# Apply enhancements in recommended order
scn.clouds.clear()
scn.clouds['mask'] = 'cf_mask'

scn.sunz.clear()
scn.sunz['mask'] = 'sunz_mask'

scn.corrections.clear()
scn.corrections['mask'] = 'modis_mask'

scn.enhancements.clear()
scn.enhancements['contrast_stretch'] = {
    'min_value': 0.0,
    'max_value': 255.0,
    'name': 'linear'
}
```

## Common Applications

### MODIS Processing

```python
from satpy import Scene

# Load MODIS file
scn = Scene('MODIS_file.nc')

# Apply MODIS correction
scn.corrections.clear()
scn.corrections['mask'] = 'modis_mask'

# Load datasets
scn.load(['temperature', 'humidity', 'cloud_top_pressure'])

# Process data
# Apply enhancements as needed
```

### VIIRS Processing

```python
# Load VIIRS file
scn = Scene('VIIRS_file.nc')

# Apply VIIRS correction
scn.corrections.clear()
scn.corrections['mask'] = 'viris_mask'

# Load atmospheric profiles
scn.load(['temperature', 'humidity', 'cloud_top_pressure'])

# Process data
# Apply enhancements as needed
```

### True Color Images

```python
# Load scene
scn = Scene('satellite_file.nc')

# Load RGB bands
scn.load(['red_band', 'green_band', 'blue_band'])

# Apply enhancements
scn.clouds['mask'] = 'cf_mask'
scn.sunz['mask'] = 'sunz_mask'

# Create true color composite
true_color = scn['true_color']
true_color.save('true_color.png')
```

## Best Practices

### Enhancement Selection

**Guidelines:**
- Apply cloud detection for visible imagery
- Use sun glint correction for accurate geometry
- Apply MODIS correction for MODIS data
- Use VIIRS correction for VIIRS data
- Use contrast stretching for better visualization

### Enhancement Order

**Recommended order:**
1. Cloud detection
2. Sun glint correction
3. Atmospheric corrections
4. Image enhancements

### Parameter Tuning

**Cloud detection:**
- Use appropriate threshold for application
- Consider cloud type (stratus vs. cumulus)
- Check for false positives

**Sun glint:**
- Use appropriate solar constant
- Consider satellite and scene geometry
- Check for limb effects

**Atmospheric corrections:**
- Use appropriate correction for data type
- Consider time of day
- Check for atmospheric conditions

### Performance

**Memory management:**
- Process in chunks for large scenes
- Use appropriate data types
- Close files after processing

**Processing speed:**
- Use vectorized operations when possible
- Consider parallel processing for large datasets
- Use appropriate enhancement order

## Troubleshooting

### Enhancement Fails

**Issue:** Enhancement fails to apply

**Solutions:**
- Check enhancement availability
- Verify input data requirements
- Check parameter values
- Consult Satpy documentation

### Incorrect Results

**Issue:** Enhancement produces unexpected results

**Solutions:**
- Check enhancement parameters
- Verify input data quality
- Try different enhancement settings
- Consult Satpy documentation

### Performance Issues

**Issue:** Slow processing

**Solutions:**
- Process in chunks
- Use appropriate data types
- Reduce spatial resolution
- Use fewer enhancements

### Memory Issues

**Issue:** Out of memory

**Solutions:**
- Process in chunks
- Use appropriate data types
- Reduce spatial resolution
- Close files after processing
- Use lazy loading with xarray

## Advanced Topics

### Custom Enhancements

```python
# Define custom enhancement
# See Satpy documentation for details
```

### Enhancement Functions

```python
# Define custom enhancement function
# See Satpy documentation for details
```

### Writing Custom Enhancements

```python
# Write custom enhancement
# See Satpy documentation for details
```

## Resources

- Satpy enhancements: https://satpy.readthedocs.io/en/enhancements.html
- Satpy API: https://satpy.readthedocs.io/en/api/modules.html
