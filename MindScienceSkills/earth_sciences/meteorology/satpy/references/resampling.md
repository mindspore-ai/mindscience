# Resampling in Satpy

Complete guide to resampling satellite data to different grids.

## Geographic Resampling

### Nearest Neighbor

```python
from satpy import Scene

# Load scene
scn = Scene('satellite_file.nc')

# Resample to specific grid resolution
swath = scn.resample('my_area', 5000)  # 5000m resolution

# Apply resampling
resampled_scn = scn.where(swath)
```

### Nearest Neighbor with Quality

```python
# Resample with quality settings
swath = scn.resample('my_area', 5000, 
                           resampler='nn',
                           radius_of_influence=25000)
```

### Bilinear Interpolation

```python
# Bilinear interpolation
swath = scn.resample('my_area', 5000,
                           resampler='bil')
```

### Geographic Projection

```python
# Resample to specific projection
swath = scn.resample('my_area', 5000,
                           projection='geostationary')
```

## Temporal Resampling

### Time Slice

```python
# Resample to specific time range
swath = scn.isel_time('2020-01-01T00:00:00',
                        '2020-01-01T23:59:59')
```

### Time Averaging

```python
# Average over time range
swath = scn.average_time('2020-01-01T00:00:00',
                            '2020-01-01T23:59:59')
```

### Nearest Time

```python
# Find nearest time
swath = scn.nearest_time('2020-01-01T12:00:00')
```

## Combined Resampling

### Spatial and Temporal

```python
# Combine spatial and temporal resampling
swath = scn.resample('my_area', 5000)
swath = swath.isel_time('2020-01-01T00:00:00',
                          '2020-01-01T23:59:59')
```

### Multiple Areas

```python
# Resample multiple areas
areas = ['area1', 'area2', 'area3']
for area in areas:
    swath = scn.resample(area, 5000)
    # Process area
```

## Area Definitions

### Bounding Box

```python
# Define bounding box
swath = scn.swath_corners_llur(
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    name='my_area'
)
```

### Shapefile

```python
# Load shapefile
from satpy import Scene

scn = Scene('satellite_file.nc')
swath = scn.swath_from_shapefile('area_shapefile.shp')
```

### Grid Definition

```python
# Define grid with specific projection
swath = scn.swath_corners_llur(
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    projection='geostationary',
    name='my_area'
)
```

## Advanced Resampling

### Dynamic Resolution

```python
# Dynamic resolution based on data extent
swath = scn.resample('my_area', 5000,
                           dynamic_area=True)
```

### Custom Projection

```python
# Custom projection parameters
swath = scn.swath_corners_llur(
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    projection='geostationary',
    proj_info={'proj': 'geos'},
    name='my_area'
)
```

### Satellite Projection

```python
# Use satellite's native projection
swath = scn.swath_corners_llur(
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    llcrn=(-180.0, -90.0, 0.0, 90.0),
    urcrn=(-180.0, 0.0, 0.0, 90.0),
    projection='satellite',
    name='my_area'
)
```

## Common Applications

### Regional Analysis

```python
# Resample to specific region
swath = scn.resample('my_region', 5000)

# Process regional data
# Save regional composite
```

### Global Mosaic

```python
# Create global mosaic at specific resolution
swath = scn.resample('global', 20000)  # 20km resolution

# Save global composite
```

### Time Series

```python
# Extract time series at specific location
swath = scn.resample('point_of_interest', 5000)
time_series = swath.isel_time('2020-01-01T00:00:00',
                               '2020-12-31T23:59:59')
```

### Daily Composites

```python
# Create daily composites
for day in range(1, 32):
    swath = scn.isel_time(f'2020-{month:02d}-02T00:00:00',
                           f'2020-{month:02d}-02T23:59:59')
    # Save daily composite
```

## Best Practices

### Resolution Selection

**Guidelines:**
- Choose appropriate resolution for application
- Higher resolution for detailed analysis
- Lower resolution for regional/global views
- Consider output file size

### Projection Selection

**Guidelines:**
- Use geostationary for global/regional views
- Use satellite projection for native views
- Use custom projection for specific requirements
- Be aware of projection distortions

### Memory Management

**Guidelines:**
- Process large areas in chunks
- Use appropriate data types
- Close files after processing
- Use lazy loading when possible

### Coordinate Handling

**Guidelines:**
- Be aware of coordinate systems
- Check for wraparound issues
- Validate coordinate ranges
- Handle missing data appropriately

## Troubleshooting

### Resampling Errors

**Issue:** Resampling fails

**Solutions:**
- Check area definition
- Verify coordinate ranges
- Check projection parameters
- Validate input data

### Projection Errors

**Issue:** Projection fails

**Solutions:**
- Check projection name
- Verify projection parameters
- Check coordinate system
- Try different projection

### Memory Issues

**Issue:** Out of memory

**Solutions:**
- Process in smaller chunks
- Reduce spatial resolution
- Use appropriate data types
- Close files after processing

## Advanced Topics

### Custom Resamplers

```python
# Define custom resampler
# See Satpy documentation for details
```

### Dynamic Area Definitions

```python
# Create dynamic area definitions
# Based on data extent or other criteria
```

### Multi-Projection Mosaics

```python
# Create mosaic with multiple projections
# See references/advanced_usage.md for details
```

## Resources

- Satpy resampling: https://satpy.readthedocs.io/en/resample.html
- Coordinate systems: https://satpy.readthedocs.io/en/coordinates.html
- Satpy API: https://satpy.readthedocs.io/en/api/modules.html
