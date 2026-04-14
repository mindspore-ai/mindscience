---
name: siphon
description: Comprehensive Python library for accessing remote meteorological and atmospheric data through THREDDS Data Servers and various web services. Use when Claude needs to:(1) Access THREDDS catalog data and datasets, (2) Query NetCDF Subset Service (NCSS) for gridded data subsetting, (3) Retrieve radar data from THREDDS radar servers, (4) Download upper air sounding data from Wyoming/Iowa State/IGRA2 archives, (5) Get buoy data from NDBC, (6) Access climate data from ACIS web services, or (7) Work with remote meteorological datasets in netCDF format
---

# Siphon

Siphon provides Python utilities for downloading meteorological and atmospheric data from remote data services, primarily THREDDS Data Servers.

## Quick Start

### THREDDS Catalog Access

```python
from siphon.catalog import TDSCatalog

cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog.xml')
print(cat.datasets)
```

### NCSS Data Query

```python
from siphon.ncss import NCSS
from datetime import datetime

ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')
query = ncss.query().time(datetime(2024, 1, 1, 0)).lonlat_box(-125, -65, 25, 50).variables('Temperature')
data = ncss.get_data(query)
```

### Upper Air Data

```python
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir

df = WyomingUpperAir.request_data(datetime(2024, 1, 1, 0), 'DEN')
```

## Workflow Decision Tree

Determine the data source and access method:

**THREDDS Catalog?**
- Browse datasets → See [THREDDS Catalog Guide](references/catalog.md)
- Access remote datasets → See [THREDDS Catalog Guide](references/catalog.md)
- Find latest dataset → See [THREDDS Catalog Guide](references/catalog.md)

**Gridded Model Data (GFS, NAM, etc.)?**
- Subset spatial/temporal data → See [NCSS Guide](references/ncss.md)
- Request specific variables → See [NCSS Guide](references/ncss.md)

**Radar Data?**
- NEXRAD Level 2/3 data → See [Radar Server Guide](references/radarserver.md)
- Query radar stations → See [Radar Server Guide](references/radarserver.md)

**Upper Air Soundings?**
- Wyoming archive → See [Simple Web Services Guide](references/simplewebservice.md#upper-air)
- Iowa State archive → See [Simple Web Services Guide](references/simplewebservice.md#upper-air)
- IGRA2 archive → See [Simple Web Services Guide](references/simplewebservice.md#upper-air)

**Buoy/Marine Data?**
- NDBC observations → See [Simple Web Services Guide](references/simplewebservice.md#ndbc)

**Climate/Station Data?**
- ACIS climate data → See [Simple Web Services Guide](references/simplewebservice.md#acis)

## Core Modules

### THREDDS Catalog (siphon.catalog)

Access and navigate THREDDS Data Server catalogs to discover and access datasets.

**Key capabilities:**
- Parse THREDDS XML catalogs
- Navigate catalog hierarchies
- Find latest datasets
- Access datasets via OPENDAP, HTTPServer, NetcdfSubset
- Filter datasets by time

See [THREDDS Catalog Guide](references/catalog.md) for complete usage.

### NCSS (siphon.ncss)

Query NetCDF Subset Service for spatial and temporal subsetting of gridded data.

**Key capabilities:**
- Spatial subsetting (bounding boxes, points)
- Temporal subsetting (time ranges, specific times)
- Variable selection
- Multiple output formats (netCDF, XML, CSV)
- Vertical level selection

See [NCSS Guide](references/ncss.md) for complete usage.

### Radar Server (siphon.radarserver)

Access radar data through THREDDS radar query services.

**Key capabilities:**
- Query NEXRAD Level 2/3 data
- Station-based queries
- Spatial and temporal filtering
- Variable selection

See [Radar Server Guide](references/radarserver.md) for complete usage.

### Simple Web Services (siphon.simplewebservice)

Access various meteorological data sources through simple web service APIs.

**Supported services:**
- **Upper Air:** Wyoming, Iowa State, IGRA2 archives
- **Marine:** NDBC buoy data
- **Climate:** ACIS web services

See [Simple Web Services Guide](references/simplewebservice.md) for complete usage.

## Common Patterns

### Access Latest Dataset from Catalog

```python
from siphon.catalog import TDSCatalog, get_latest_access_url

cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/nexrad/nexrad.xml')
latest_url = get_latest_access_url(cat.catalog_url, 'OPENDAP')
```

### Time-Based Dataset Filtering

```python
from datetime import datetime
from siphon.catalog import TDSCatalog

cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.xml')
nearest = cat.datasets.filter_time_nearest(datetime(2024, 1, 1, 12))
```

### NCSS Spatial Query with Multiple Variables

```python
from siphon.ncss import NCSS
from datetime import datetime

ncss = NCSS('http://thredds.ucar.edu/thredds/ncss/gfs/NCEP/GFS/Global_0p25deg/catalog.xml')
query = (ncss.query()
    .time(datetime(2024, 1, 1, 0))
    .lonlat_box(-125, -65, 25, 50)
    .variables('Temperature', 'Relative_humidity', 'u-component_of_wind_height_above_ground'))
data = ncss.get_data(query)
```

### Upper Air Data with Units

```python
from datetime import datetime
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir

df = WyomingUpperAir.request_data(datetime(2024, 1, 1, 0), 'DEN')
pressure = df['pressure'].values * units(df.units['pressure'])
temperature = df['temperature'].values * units(df.units['temperature'])
```

### NDBC Buoy Data

```python
from siphon.simplewebservice.ndbc import NDBC

# Get latest observations from all buoys
df = NDBC.latest_observations()

# Get realtime data from specific buoy
df = NDBC.realtime_observations('41001')
```

## Data Access Patterns

### Remote Access Methods

When accessing datasets from THREDDS catalogs, choose the appropriate service:

**OPENDAP** - NetCDF4-like remote access
```python
dataset = cat.datasets['model_data.nc'].remote_access(service='OPENDAP')
temp = dataset.variables['Temperature'][:]
```

**HTTPServer** - Download entire file
```python
dataset = cat.datasets['model_data.nc'].download('local_file.nc')
```

**NetcdfSubset** - Server-side subsetting via NCSS
```python
ncss_client = cat.datasets['model_data.nc'].subset(service='NetcdfSubset')
```

### Output Format Selection

For NCSS queries, choose appropriate format:

**netCDF** - Full structured data (default)
```python
query = ncss.query().variables('Temperature')
data = ncss.get_data(query)  # Returns xarray Dataset
```

**XML** - Point data or small subsets
```python
query = ncss.query().accept('xml')
data = ncss.get_data(query)
```

**CSV** - Tabular data
```python
query = ncss.query().accept('csv')
data = ncss.get_data(query)
```

## Error Handling

### Network Errors

```python
from requests.exceptions import RequestException

try:
    cat = TDSCatalog('http://example.com/catalog.xml')
except RequestException as e:
    print(f"Network error: {e}")
```

### Query Validation

```python
from siphon.http_util import BadQueryError

try:
    data = ncss.get_data(query)
except BadQueryError as e:
    print(f"Invalid query: {e}")
```

### Missing Data

```python
from datetime import datetime
from siphon.simplewebservice.wyoming import WyomingUpperAir

try:
    df = WyomingUpperAir.request_data(datetime(2024, 1, 1, 0), 'INVALID')
except ValueError as e:
    print(f"No data available: {e}")
```

## Performance Tips

1. **Use NCSS for subsetting** - Download only needed data
2. **Leverage server-side operations** - Reduce data transfer
3. **Cache catalog metadata** - Avoid repeated catalog parsing
4. **Use appropriate output formats** - netCDF for large data, CSV/XML for small
5. **Batch requests** - Use time ranges instead of individual times

## Resources

### references/

Detailed documentation for each Siphon module:

- **[THREDDS Catalog Guide](references/catalog.md)** - Complete TDSCatalog usage, dataset access, time filtering
- **[NCSS Guide](references/ncss.md)** - NetCDF Subset Service queries, spatial/temporal subsetting
- **[Radar Server Guide](references/radarserver.md)** - Radar data access and station queries
- **[Simple Web Services Guide](references/simplewebservice.md)** - Upper air, buoy, and climate data access
- **[Common Patterns](references/common_patterns.md)** - Detailed examples and best practices

### scripts/

Executable utilities for common Siphon workflows:

- **[download_latest.py](scripts/download_latest.py)** - Download latest dataset from THREDDS catalog
- **query_ncss.py](scripts/query_ncss.py)** - NCSS query builder and executor
- **get_upper_air.py](scripts/get_upper_air.py)** - Upper air data retrieval utility
