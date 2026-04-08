# Data I/O

MetPy provides comprehensive data I/O capabilities for meteorological data formats.

## Reading GRIB Files

### Basic GRIB Reading

```python
from metpy.io import read_grib

# Read GRIB2 file
data = read_grib('data.grib2')

# Access data as xarray
temperature = data['temperature']
print(temperature.dims)
print(temperature.coords)
```

### Reading Specific Variables

```python
from metpy.io import read_grib

# Read specific variable
temperature = read_grib('data.grib2', var_name='temperature')

# Read multiple variables
data = read_grib('data.grib2', var_names=['temperature', 'pressure', 'u', 'v'])
```

### GRIB Message Selection

```python
from metpy.io import read_grib

# Read specific message
data = read_grib('data.grib2', grib_msg=1)

# Read multiple messages
data = read_grib('data.grib2', grib_msg=[1, 2, 3])
```

### GRIB Metadata

```python
from metpy.io import read_grib

# Read data
data = read_grib('data.grib2')

# Access metadata
print(data.attrs)  # Global attributes
print(temperature.attrs)  # Variable attributes
```

## Reading NetCDF Files

### Basic NetCDF Reading

```python
from metpy.io import read_netcdf

# Read NetCDF file
data = read_netcdf('data.nc')

# Access data as xarray
temperature = data['temperature']
```

### Reading Specific Variables

```python
from metpy.io import read_netcdf

# Read specific variable
temperature = read_netcdf('data.nc', var_name='temperature')

# Read multiple variables
data = read_netcdf('data.nc', var_names=['temperature', 'pressure'])
```

### NetCDF Groups

```python
from metpy.io import read_netcdf

# Read from specific group
data = read_netcdf('data.nc', group='model')

# Read multiple groups
data = read_netcdf('data.nc', group=['model', 'observations'])
```

## Writing GRIB Files

### Basic GRIB Writing

```python
from metpy.io import write_grib

# Write to GRIB2
write_grib(data, 'output.grib2')

# Write specific variable
write_grib(data['temperature'], 'output.grib2')
```

### GRIB Metadata

```python
from metpy.io import write_grib

# Write with metadata
data.attrs['model'] = 'GFS'
data.attrs['date'] = '2024-01-01'
write_grib(data, 'output.grib2')
```

### GRIB Templates

```python
from metpy.io import write_grib

# Write using template
write_grib(data, 'output.grib2', template='template.grib2')
```

## Writing NetCDF Files

### Basic NetCDF Writing

```python
from metpy.io import write_netcdf

# Write to NetCDF
write_netcdf(data, 'output.nc')

# Write specific variable
write_netcdf(data['temperature'], 'output.nc')
```

### NetCDF Compression

```python
from metpy.io import write_netcdf

# Write with compression
write_netcdf(data, 'output.nc', compression=True)
```

## File Format Selection

| Format | Extension | Use Case | Features |
|--------|-----------|---------|---------|
| GRIB2 | .grib2 | Model output | Widespread standard |
| GRIB1 | .grib, .grb | Legacy data | Backward compatibility |
| NetCDF3 | .nc | Observations | Self-describing |
| NetCDF4 | .nc4 | Large datasets | Compression |

## Data Format Conversion

```python
from metpy.io import read_grib, write_netcdf

# Read GRIB2
data = read_grib('input.grib2')

# Write to NetCDF
write_netcdf(data, 'output.nc')
```

## Common Issues and Solutions

### GRIB Reading Failures

**Problem**: Cannot read GRIB file

**Solutions**:
- Check file format (GRIB1 vs GRIB2)
- Verify message number
- Check variable name spelling
- Try different reader options

### NetCDF Reading Failures

**Problem**: Cannot read NetCDF file

**Solutions**:
- Check file format (NetCDF3 vs NetCDF4)
- Verify group structure
- Check variable name spelling
- Try different reader options

### Memory Issues

**Problem**: Out of memory with large files

**Solutions**:
- Read specific variables only
- Use chunked reading
- Read specific time steps
- Use appropriate data types

### Coordinate Issues

**Problem**: Coordinates not recognized correctly

**Solutions**:
- Check coordinate variable names
- Verify coordinate order
- Check coordinate units
- Manually specify coordinates

## Best Practices

1. **Use appropriate format**: Match format to application
2. **Read only needed data**: Minimize memory usage
3. **Check metadata**: Verify data quality
4. **Handle errors**: Catch I/O exceptions
5. **Document format choices**: Keep track of format decisions

