# cfgrib Data Model

This document explains the cfgrib data model and coordinate systems.

## Overview

cfgrib converts GRIB files into xarray Datasets with CF-compliant coordinates. Understanding this data model is essential for effective data manipulation.

## Dataset Structure

A cfgribrib Dataset consists of:

- **Data variables**: Meteorological fields (temperature, wind, etc.)
- **Coordinates**: Dimension labels (time, latitude, longitude, level)
- **Attributes**: Metadata (units, long names, GRIB keys)

## Coordinate Systems

### Geographic Coordinates

**Latitude (`latitude`)**
- Units: degrees north
- Range: -90 to 90
- Direction: Decreasing from north to south

**Longitude (`longitude`)**
- Units: degrees east
- Range: 0 to 360 or -180 to 180
- Direction: Increasing from west to east

### Temporal Coordinates

**Time (`time`)**
- Units: Typically datetime64[ns]
- Format: ISO 8601 (YYYY-MM-DDTHH:MM:SS)
- Reference: Analysis or forecast reference time

**Step (`step`)**
- Units: hours
- Type: Forecast step from reference time

### Vertical Coordinates

**Pressure Levels (`level`)**
- Units: hPa
- Type: isobaricInhPa
- Common values: 1000, 850, 700, 500, 300, 200, 100

**Height Levels (`heightAboveGround`)**
- Units: meters
- Type: heightAboveGround
- Common values: 2, 10, 50, 100

**Model Levels (`hybrid`)**
- Units: dimensionless
- Type: hybrid
- Range: 0 to 1

## Dimensionality

### 2D Fields
Surface fields with geographic coordinates:
```python
Dimensions: (latitude, longitude)
Example: t2m, sp, msl
```

### 3D Fields
Atmospheric fields with vertical dimension:
```python
Dimensions: (level, latitude, longitude)
Example: t, u, v, gh
```

### 4D Fields
Time-varying atmospheric fields:
```python
Dimensions: (time, level, latitude, longitude)
Example: t, u, v, gh
```

### Ensemble Fields
Ensemble forecast data:
```python
Dimensions: (number, time, level, latitude, longitude)
Example: ensemble forecasts
```

## CF Encoding

cfgrib applies CF (Climate and Forecast) conventions:

### Variable Attributes
- `long_name`: Descriptive variable name
- `units`: Variable units
- `standard_name`: CF standard name (when available)
- `valid_min`: Minimum valid value
- `valid_max`: Maximum valid value

### Coordinate Attributes
- `units`: Coordinate units
- `standard_name`: CF standard coordinate name
- `axis`: Axis type (X, Y, Z, T)

## GRIB to CF Mapping

### Variable Names
- GRIB `shortName` → CF variable name
- GRIB `name` → CF `long_name`
- GRIB `units` → CF `units`

### Coordinate Mapping
- GRIB `latitudeOfFirstGridPoint` → CF `latitude`
- GRIB `longitudeOfFirstGridPoint` → CF `longitude`
- GRIB `dataDate`/`dataTime` → CF `time`
- GRIB `level` → CF `level` or `height`

### Grid Types
- `regular_ll` → Regular latitude-longitude grid
- `gaussian` → Gaussian grid
- `reduced_gg` → Reduced Gaussian grid

## Data Access Patterns

### Spatial Access
```python
# Point selection
point = ds['t2m'].sel(latitude=40.0, longitude=-100.0)

# Regional selection
region = ds['t2m'].sel(
    latitude=slice(50, 30),
    longitude=slice(-120, -90)
)
```

### Temporal Access
```python
# Time selection
time_point = ds['t2m'].sel(time='2024-01-15T12:00:00')

# Time range
time_range = ds['t2m'].sel(time=slice('2024-01-01', '2024-01-31'))
```

### Vertical Access
```python
# Level selection
level_point = ds['t'].sel(level=500)

# Multiple levels
levels = ds['t'].sel(level=[1000, 850, 700, 500])
```

## Ensemble Dimension

Ensemble data includes a `number` dimension:
```python
# Access specific member
member_0 = ds.isel(number=0)

# Ensemble statistics
mean = ds.mean(dim='number')
spread = ds.std(dim='number')
```

## Best Practices

1. **Use coordinate-based selection** - More intuitive than index-based
2. **Understand dimension order** - Affects performance and memory layout
3. **Check coordinate ranges** - Ensure selections are valid
4. **Use CF metadata** - Provides valuable information about variables
5. **Be aware of grid types** - Different grids have different coordinate systems

## References

- CF Conventions: http://cfconventions.org/
- xarray Documentation: https://xarray.pydata.org/
- cfgrib Documentation: https://github.com/ecmwf/cfgrib