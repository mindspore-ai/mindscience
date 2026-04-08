# Radar Server Guide

Complete guide for accessing radar data through THREDDS radar query services with Siphon.

## Overview

THREDDS radar servers provide access to NEXRAD (Next Generation Weather Radar) Level 2 and Level 3 data. Siphon's `siphon.radarserver` module simplifies querying these servers and retrieving radar data catalogs.

## Basic Radar Server Access

### Initialize Radar Server

```python
from siphon.radarserver import RadarServer

# Connect to radar server
rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')

# View available stations
print(rs.stations)

# View available variables
print(rs.variables)
```

### Station Information

```python
# Get station list
for station_id, station in rs.stations.items():
    print(f"{station_id}: {station.name}")
    print(f"  Location: {station.latitude}, {station.longitude}")
    print(f"  Elevation: {station.elevation}m")
```

### Metadata Exploration

```python
# Time coverage
print(rs.metadata.time_coverage)

# Spatial coverage
print(rs.metadata.lat_lon_box)

# Variable information
for var_name in rs.variables:
    var = rs.metadata.variables[var_name]
    print(f"{var_name}: {var.description}")
```

## Query Building

### Create Query

```python
# Create empty query
query = rs.query()

# Query methods can be chained
query = (rs.query()
    .time(datetime(2024, 1, 1, 12, 30))
    .stations('KOUN')
    .variables('Reflectivity'))
```

### Temporal Queries

#### Specific Time

```python
from datetime import datetime

query = rs.query().time(datetime(2024, 1, 1, 12, 30))
```

#### Time Range

```python
start = datetime(2024, 1, 1, 0, 0)
end = datetime(2024, 1, 2, 0, 0)
query = rs.query().time_range(start, end)
```

#### All Times

```python
query = rs.query().all_times()
```

### Station Queries

#### Single Station

```python
query = rs.query().stations('KOUN')
```

#### Multiple Stations

```python
query = rs.query().stations('KOUN', 'KTLX', 'KFDR')
```

### Spatial Queries

#### Geographic Bounding Box

```python
# west, east, south, north
query = rs.query().lonlat_box(-100, -90, 35, 40)
```

#### Geographic Point

```python
# longitude, latitude
query = rs.query().lonlat_point(-97.5, 35.5)
```

### Variable Selection

#### Single Variable

```python
query = rs.query().variables('Reflectivity')
```

#### Multiple Variables

```python
query = rs.query().variables('Reflectivity', 'Velocity', 'SpectrumWidth')
```

#### All Variables

```python
query = rs.query().variables('all')
```

## Executing Queries

### Get Catalog

```python
# Execute query and get parsed catalog
catalog = rs.get_catalog(query)

# Access datasets in catalog
for name, dataset in catalog.datasets.items():
    print(f"{name}: {dataset.url_path}")
```

### Get Raw Catalog XML

```python
# Get raw XML catalog
xml_data = rs.get_catalog_raw(query)

# Save to file
with open('radar_catalog.xml', 'wb') as f:
    f.write(xml_data)
```

### Validate Query

```python
# Check if query is valid
is_valid = rs.validate_query(query)
if not is_valid:
    print("Query is invalid")
```

## Complete Examples

### Query Single Station at Specific Time

```python
from siphon.radarserver import RadarServer
from datetime import datetime

rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')

query = (rs.query()
    .time(datetime(2024, 1, 1, 12, 30))
    .stations('KOUN')
    .variables('Reflectivity'))

catalog = rs.get_catalog(query)
```

### Query Multiple Stations in Time Range

```python
from datetime import datetime, timedelta

rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')

start = datetime(2024, 1, 1, 12, 0)
end = datetime(2024, 1, 1, 13, 0)

query = (rs.query()
    .time_range(start, end)
    .stations('KOUN', 'KTLX')
    .variables('Reflectivity', 'Velocity'))

catalog = rs.get_catalog(query)
```

### Spatial Query

```python
rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')

query = (rs.query()
    .time(datetime(2024, 1, 1, 12, 30))
    .lonlat_box(-100, -90, 35, 40)
    .variables('Reflectivity'))

catalog = rs.get_catalog(query)
```

### Find Nearest Station

```python
import math

def find_nearest_station(radar_server, target_lat, target_lon):
    """Find nearest radar station to target location."""
    nearest_station = None
    min_distance = float('inf')

    for station_id, station in radar_server.stations.items():
        distance = math.sqrt(
            (station.latitude - target_lat)**2 +
            (station.longitude - target_lon)**2
        )
        if distance < min_distance:
            min_distance = distance
            nearest_station = station_id

    return nearest_station, min_distance

rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')
nearest_id, distance = find_nearest_station(rs, 35.5, -97.5)
print(f"Nearest station: {nearest_id}, distance: {distance:.2f} degrees")
```

### Get Available Times for Station

```python
from siphon.radarserver import get_radarserver_datasets

# Get top-level catalog
server_url = 'http://thredds.ucar.edu/thredds/'
datasets = get_radarserver_datasets(server_url)

# Find Level 2 catalog
level2_ref = datasets.get('NEXRAD Level II')
if level2_ref:
    catalog = level2_ref.follow()
    print(f"Available datasets: {list(catalog.datasets.keys())}")
```

## Working with Returned Catalogs

### Access Datasets from Catalog

```python
catalog = rs.get_catalog(query)

# List datasets
for name in catalog.datasets:
    print(name)

# Access specific dataset
dataset = catalog.datasets['KOUN20240101_123031_V06']
```

### Access Dataset Data

```python
dataset = catalog.datasets['KOUN20240101_123031_V06']

# Get access URL
opendap_url = dataset.access_urls['OPENDAP']

# Remote access
ds = dataset.remote_access(service='OPENDAP')

# Access data
reflectivity = ds.variables['Reflectivity'][:]

# Close connection
ds.close()
```

### Download Dataset

```python
dataset = catalog.datasets['KOUN20240101_123031_V06']

# Download to file
dataset.download('radar_data.nc')
```

## Common Radar Servers

### NEXRAD Level 2

```python
# All Level 2 stations
rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')

# Specific station (e.g., KOUN)
rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/KOUN/')
```

### NEXRAD Level 3

```python
# All Level 3 products
rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level3/IDD/')

# Specific station
rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level3/KOUN/')
```

### TDWR (Terminal Doppler Weather Radar)

```python
rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/tdwr/level2/IDD/')
```

## Common Variables

### Level 2 Variables

```python
# Reflectivity (dBZ)
query = rs.query().variables('Reflectivity')

# Velocity (m/s)
query = rs.query().variables('Velocity')

# Spectrum Width (m/s)
query = rs.query().variables('SpectrumWidth')

# Differential Reflectivity (dB)
query = rs.query().variables('DifferentialReflectivity')

# Differential Phase (degrees)
query = rs.query().variables('DifferentialPhase')

# Correlation Coefficient
query = rs.query().variables('CorrelationCoefficient')

# Specific Differential Phase (degrees/km)
query = rs.query().variables('SpecificDifferentialPhase')
```

### Level 3 Products

```python
# Base Reflectivity (0.5°)
query = rs.query().variables('BaseReflectivity')

# Base Velocity (0.5°)
query = rs.query().variables('BaseVelocity')

# Composite Reflectivity
query = rs.query().variables('CompositeReflectivity')

# Composite Velocity
query = rs.query().variables('CompositeVelocity')

# One-Hour Precipitation
query = rs.query().variables('OneHourPrecipitation')

# Storm Total Precipitation
query = rs.query().variables('StormTotalPrecipitation')

# Echo Tops
query = rs.query().variables('EchoTops')

# Vertically Integrated Liquid
query = rs.query().variables('VIL')
```

## Error Handling

### Invalid Query

```python
from siphon.http_util import BadQueryError

try:
    catalog = rs.get_catalog(query)
except BadQueryError as e:
    print(f"Query error: {e}")
```

### Network Error

```python
from requests.exceptions import RequestException

try:
    catalog = rs.get_catalog(query)
except RequestException as e:
    print(f"Network error: {e}")
```

### Station Not Found

```python
station_id = 'INVALID'
if station_id not in rs.stations:
    print(f"Station {station_id} not found")
    print(f"Available stations: {list(rs.stations.keys())}")
```

### No Data Available

```python
catalog = rs.get_catalog(query)

if not catalog.datasets:
    print("No data available for query")
```

## Performance Tips

1. **Use station-specific servers** - Reduces catalog size
2. **Limit time ranges** - Reduces number of returned datasets
3. **Select specific variables** - Some servers filter by variable
4. **Cache station information** - Station metadata rarely changes
5. **Use spatial queries sparingly** - May return many stations

## Best Practices

1. **Validate station IDs** - Check station exists before querying
2. **Use specific times** - Time range queries can return many datasets
3. **Handle empty results** - No data may be available for some queries
4. **Close connections** - Always close remote access connections
5. **Check metadata** - Verify variables and time coverage before querying
