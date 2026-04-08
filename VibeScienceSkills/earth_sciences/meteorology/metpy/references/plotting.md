# Plotting

MetPy provides comprehensive plotting capabilities for meteorological data.

## Declarative Plotting

### Basic Plotting

```python
from metpy.plots import declarative
from metpy.io import read_grib

# Read data
data = read_grib('data.grib2')

# Create declarative plot
declarative.plot(data, 'temperature')
```

### Custom Contour Levels

```python
from metpy.plots import declarative
import numpy as np

# Create plot with custom contour levels
declarative.plot(data, 'temperature', clevs=np.arange(250, 310, 5))
```

### Map Projections

```python
from metpy.plots import declarative

# Create plot with map projection
declarative.plot(data, 'temperature', projection='lcc')
```

## Cross Sections

### Basic Cross Section

```python
from metpy.plots import cross_section

# Create cross section plot
cross_section.plot(data, 'temperature', lat=40, lon=-100)
```

### Multiple Cross Sections

```python
from metpy.plots import cross_section

# Create multiple cross sections
cross_section.plot(data, 'temperature', lat=[30, 40, 50], lon=-100)
```

### Vertical Cross Section

```python
from metpy.plots import cross_section

# Create vertical cross section
cross_section.plot(data, 'temperature', lat=40, lon=-100, vertical=True)
```

## Hodographs

### Basic Hodograph

```python
from metpy.plots import hodograph

# Create hodograph
hodograph.plot(u, v)
```

### Colored Hodograph

```python
from metpy.plots import hodograph

# Create colored hodograph
hodograph.plot(u, v, color=temperature)
```

## Skew-T Plots

### Basic Skew-T Plot

```python
from metpy.plots import skewt

# Create skew-T plot
skewt.plot(temperature, pressure)
```

### Custom Skew-T Plot

```python
from metpy.plots import skewt

# Create custom skew-T plot
skewt.plot(temperature, pressure, p_levels=[850, 700, 500])
```

## Wind Barbs

### Basic Wind Barb

```python
from metpy.plots import wind_barb

# Create wind barb plot
wind_barb.plot(u, v)
```

### Subsampled Wind Barbs

```python
from metpy.plots import wind_barb

# Create subsampled wind barb plot
wind_barb.plot(u, v, subsample=5)
```

## Streamlines

### Basic Streamlines

```python
from metpy.plots import streamlines

# Create streamlines
streamlines.plot(u, v)
```

### Colored Streamlines

```python
from metpy.plots import streamlines

# Create colored streamlines
streamlines.plot(u, v, color=speed)
```

## Map Overlays

### Add Map Background

```python
from metpy.plots import declarative

# Create plot with map background
declarative.plot(data, 'temperature', map_background=True)
```

### Add Coastlines

```python
from metpy.plots import declarative

# Create plot with coastlines
declarative.plot(data, 'temperature', coastlines=True)
```

### Add Borders

```python
from metpy.plots import declarative

# Create plot with country borders
declarative.plot(data, 'temperature', borders=True)
```

## Plot Customization

### Color Maps

```python
from metpy.plots import declarative

# Create plot with custom colormap
declarative.plot(data, 'temperature', cmap='viridis')
```

### Titles and Labels

```python
from metpy.plots import declarative

# Create plot with custom title
declarative.plot(data, 'temperature', title='Temperature (K)')
```

### Color Bars

```python
from metpy.plots import declarative

# Create plot with color bar
declarative.plot(data, 'temperature', colorbar=True)
```

## Common Issues and Solutions

### Plotting Failures

**Problem**: Cannot create plot

**Solutions**:
- Check data validity
- Verify data dimensions
- Check for missing values
- Try different plot type

### Coordinate Issues

**Problem**: Coordinates not displayed correctly

**Solutions**:
- Check coordinate system
- Verify projection settings
- Check coordinate units
- Manually specify coordinates

### Memory Issues

**Problem**: Out of memory with large datasets

**Solutions**:
- Use subsampling
- Reduce data resolution
- Process in chunks
- Use appropriate data types

## Best Practices

1. **Validate data**: Check data quality before plotting
2. **Use appropriate projections**: Match projection to data
3. **Add map features**: For context and readability
4. **Customize plots**: For clarity and presentation
5. **Handle errors**: Catch and handle plotting exceptions
6. **Document plot settings**: Keep track of plot parameters
