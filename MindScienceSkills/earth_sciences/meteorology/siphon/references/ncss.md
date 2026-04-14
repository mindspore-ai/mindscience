# NCSS Guide

Complete guide for querying NetCDF Subset Service (NCSS) with Siphon.

## Overview

NetCDF Subset Service (NCSS) allows server-side subsetting of gridded meteorological data. Siphon's `siphon.ncss` module provides query building and data retrieval for NCSS endpoints.

## Basic NCSS Access

### Initialize NCSS Client
```python
from siphon.ncss import NCSS

# Connect to NCSS endpoint
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

# View available variables
print(ncss.variables)

# Access metadata
print(ncss.metadata)
```

### Metadata Exploration

```python
# Time coverage
print(ncss.metadata.time_coverage)

# Spatial coverage
print(ncss.metadata.lat_lon_box)

# Variable information
for var_name in ncss.variables:
    var = ncss.metadata.variables[var_name]
    print(f"{var_name}: {var.description}")
```

## Query Building

### Create Query

```python
# Create empty query
query = ncss.query()

# Query methods can be chained
query = (ncss.query()
    .time(datetime(2024, 1, 1, 0))
    .lonlat_box(-125, -65, 25, 50)
    .variables('Temperature'))
```

### Temporal Queries

#### Specific Time

```python
from datetime import datetime

query = ncss.query().time(datetime(2024, 1, 1, 0))
```

#### Time Range

```python
start = datetime(2024, 1, 1, 0)
end = datetime(2024, 1, 2, 0)
query = ncss.query().time_range(start, end)
```

#### All Times

```python
query = ncss.query().all_times()
```

### Spatial Queries

#### Geographic Bounding Box

```python
# west, east, south, north
query = ncss.query().lonlat_box(-125, -65, 25, 50)
```

#### Geographic Point

```python
# longitude, latitude
query = ncss.query().lonlat_point(-105.0, 40.0)
```

#### Projected Bounding Box

```python
# min_x, min_y, max_x, max_y (native coordinates)
query = ncss.query().projection_box(0, 0, 1000, 1000)
```

### Variable Selection

#### Single Variable

```python
query = ncss.query().variables('Temperature')
```

#### Multiple Variables

```python
query = ncss.query().variables('Temperature', 'Relative_humidity', 'u-component_of_wind_height_above_ground')
```

#### All Variables

```python
query = ncss.query().variables('all')
```

### Vertical Level Selection

```python
# Specific pressure level (hPa)
query = ncss.query().vertical_level(500)

# Specific height level (meters)
query = ncss.query().vertical_level(10000)
```

### Strides

Reduce data resolution by skipping points.

```python
# Every 2nd time step, every 3rd spatial point
query = ncss.query().strides(time=2, spatial=3)

# Only time stride
query = ncss.query().strides(time=2)

# Only spatial stride
query = ncss.query().strides(spatial=3)
```

### Output Format

```python
# netCDF format (default)
query = ncss.query().accept('netcdf')

# XML format
query = ncss.query().accept('xml')

# CSV format
query = ncss.query().accept('csv')
```

### Add Latitude/Longitude

Make returned data CF-compliant by adding lat/lon coordinates.

```python
query = ncss.query().add_lonlat(True)
```

## Executing Queries

### Get Parsed Data

```python
# Execute query and get parsed data
data = ncss.get_data(query)

# For netCDF output, returns xarray Dataset
print(data)
print(data.variables)
```

### Get Raw Data

```python
# Get raw bytes
raw_data = ncss.get_data_raw(query)

# Save to file
with open('output.nc', 'wb') as f:
    f.write(raw_data)
```

### Validate Query

```python
# Check if query is valid
is_valid = ncss.validate_query(query)
if not is_valid:
    print("Query is invalid")
```

## Complete Examples

### Simple Spatial Subset

```python
from siphon.ncss import NCSS
from datetime import datetime

ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

query = (ncss.query()
    .time(datetime(2024, 1, 1, 0))
    .lonlat_box(-125, -65, 25, 50)
    .variables('Temperature'))

data = ncss.get_data(query)
temp = data['Temperature']
```

### Time Series at Point

```python
from datetime import datetime, timedelta

ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

start = datetime(2024, 1, 1, 0)
end = datetime(2024, 1, 2, 0)

query = (ncss.query()
    .time_range(start, end)
    .lonlat_point(-105.0, 40.0)
    .variables('Temperature', 'Relative_humidity'))

data = ncss.get_data(query)
```

### Vertical Profile

```python
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

query = (ncss.query()
    .time(datetime(2024, 1, 1, 0))
    .lonlat_point(-105.0, 40.0)
    .vertical_level(500)
    .variables('Temperature', 'Geopotential_height'))

data = ncss.get_data(query)
```

### Multiple Time Steps with Strides

```python
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

query = (ncss.query()
    .time_range(datetime(2024, 1, 1, 0), datetime(2024, 1, 7, 0))
    .lonlat_box(-125, -65, 25, 50)
    .variables('Temperature')
    .strides(time=6, spatial=2))  # Every 6 hours, every 2nd grid point

data = ncss.get_data(query)
```

### CSV Output for Point Data

```python
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

query = (ncss.query()
    .time(datetime(2024, 1, 1, 0))
    .lonlat_point(-105.0, 40.0)
    .variables('Temperature')
    .accept('csv'))

data = ncss.get_data(query)
# Returns pandas DataFrame or similar structure
```

## Working with Returned Data

### xarray Dataset (netCDF output)

```python
data = ncss.get_data(query)

# Access variable
temp = data['Temperature']

# Get coordinate values
lats = data['lat'].values
lons = data['lon'].values
times = data['time'].values

# Get attributes
print(temp.attrs)

# Convert to numpy array
temp_array = temp.values
```

### XML Output

```python
query = ncss.query().accept('xml')
data = ncss.get_data(query)

# Data structure depends on query type
# Point data: list of dictionaries
# Grid data: nested structure
```

### CSV Output

```python
query = ncss.query().accept('csv')
data = ncss.get_data(query)

# Returns pandas DataFrame or similar
print(data.head())
print(data.columns)
```

## Common NCSS Endpoints

### GFS Model Data

```python
# Global 0.25 degree
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

# Global 0.5 degree
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p5deg/catalog.xml')
```

### NAM Model Data

```python
# CONUS 12km
ncss = NCSS('http://thredds.ucar.edu/thrdds/ncss/nam/NCEP/NAM/CONUS_12km/catalog.xml')

# CONUS 12km nested
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/nam/NCEP/NAM/CONUS_12km/nest/catalog.xml')
```

### RAP Model Data

```python
# CONUS 13km
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/rap/NCEP/RAP/CONUS_13km/catalog.xml')
```

### HRRR Model Data

```python
# CONUS 3km
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/hrrr/NCEP/HRRR/CONUS_3km/catalog.xml')
```

ari = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')

# RTOFS
ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/rtofs/NCEP/RTOFS/Global_RTOFS/catalog.xml')
```

## Error Handling

### Invalid Query

```python
from siphon.http_util import BadQueryError

try:
    data = ncss.get_data(query)
except BadQueryError as e:
    print(f"Query error: {e}")
```

### Network Error

```python
from requests.exceptions import RequestException

try:
    data = ncss.get_data(query)
except RequestException as e:
    print(f"Network error: {e}")
```

### Variable Not Found

```python
if 'Temperature' not in ncss.variables:
    print("Temperature variable not available")
    print(f"Available variables: {sorted(ncss.variables)}")
```

### Time Out of Range

```python
# Check time coverage
time_coverage = ncss.metadata.time_coverage
requested_time = datetime(2024, 1, 1, 0)

if requested_time < time_coverage.start or requested_time > time_coverage.end:
    print("Requested time outside available range")
```

## Performance Optimization

### Reduce Data Transfer

```python
# Select only needed variables
query = ncss.query().variables('Temperature')

# Use spatial subsetting
query = ncss.query().lonlat_box(-125, -65, 25, 50)

# Use temporal subsetting
query = ncss.query().time_range(start, end)

# Use strides for large areas
query = ncss.query().strides(spatial=2)
```

### Choose Appropriate Format

```python
# Small point data: CSV or XML
query = ncss.query().accept('csv')

# Large gridded data: netCDF
query = ncss.query().accept('netcdf')
```

### Server-Side Processing

```python
# Let server handle subsetting
query = (ncss.query()
    .lonlat_box(-125, -65, 25, 50)  # Server clips to this box
    .vertical_level(500))  # Server extracts this level
```

## Best Practices

1. **Validate queries before execution** - Catch errors early
2. **Use appropriate output formats** - netCDF for large data, CSV/XML for small
3. **Subset aggressively** - Minimize data transfer
4. **Check metadata** - Verify variables and time coverage before querying
5. **Handle errors gracefully** - Network and query errors are common
6. **Use strides for exploration** - Reduce resolution when testing queries
