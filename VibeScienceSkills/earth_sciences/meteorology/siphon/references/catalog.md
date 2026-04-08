# THREDDS Catalog Guide

Complete guide for accessing and navigating THREDDS Data Server catalogs with Siphon.

## Overview

THREDDS (Thematic Real-time Environmental Distributed Data Services) catalogs provide hierarchical organization of meteorological datasets. Siphon's `siphon.catalog` module parses these catalogs and provides tools for dataset discovery and access.

## Basic Catalog Access

### Initialize Catalog

```python
from siphon.catalog import TDSCatalog

# Access top-level catalog
cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog.xml')

# Access specific dataset catalog
cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/nexrad/nexrad.xml')
```

### Catalog Properties

```python
# Catalog URL
print(cat.catalog_url)

# Base TDS server URL
print(cat.base_tds_url)

# Available datasets
print(list(cat.datasets.keys()))

# Available services
print([s.name for s in cat.services])

# Catalog references (nested catalogs)
print(list(cat.catalog_refs.keys()))
```

## Navigating Catalog Hierarchies

### Follow Catalog References

```python
# List catalog references
for name, ref in cat.catalog_refs.items():
    print(f"{name}: {ref.title}")

# Follow a catalog reference
ref = cat.catalog_refs['NEXRAD Level II']
subcat = ref.follow()
```

### Recursive Catalog Navigation

```python
def explore_catalog(catalog, depth=0):
    indent = "  " * depth
    print(f"{indent}Catalog: {catalog.catalog_url}")

    # Print datasets
    for name in catalog.datasets:
        print(f"{indent}  Dataset: {name}")

    # Recurse into sub-catalogs
    for name, ref in catalog.catalog_refs.items():
        print(f"{indent}  Reference: {name}")
        subcat = ref.follow()
        explore_catalog(subcat, depth + 1)
```

## Dataset Access

### Dataset Information

```python
# Access specific dataset
dataset = cat.datasets['model_data.nc']

# Dataset name
print(dataset.name)

# URL path
print(dataset.url_path)

# Available access methods
print(dataset.access_urls.keys())
# Output: dict_keys(['OPENDAP', 'HTTPServer', 'NetcdfSubset', 'WMS'])
```

### Access via OPENDAP

OPENDAP provides netCDF4-like remote access to datasets.

```python
# Remote access (default service is CdmRemote, falls back to OPENDAP)
ds = dataset.remote_access()

# Explicit OPENDAP access
ds = dataset.remote_access(service='OPENDAP')

# Access variables
temp = ds.variables['Temperature'][:]
lat = ds.variables['lat'][:]
lon = ds.variables['lon'][:]

# Close connection
ds.close()
```

### Access via HTTPServer

Download entire dataset file.

```python
# Download to current directory
dataset.download()

# Download to specific location
dataset.download('/path/to/local_file.nc')
```

### Access via NetcdfSubset

Use NCSS for server-side subsetting.

```python
# Get NCSS client
ncss = dataset.subset(service='NetcdfSubset')

# Build query
query = ncss.query().time_range(start, end).variables('Temperature')

# Get data
data = ncss.get_data(query)
```

### Get Access URL

```python
# Get URL for specific access method
opendap_url = dataset.access_urls['OPENDAP']
http_url = dataset.access_urls['HTTPServer']
```

## Finding Latest Datasets

### Using get_latest_access_url

```python
from siphon.catalog import get_latest_access_url

# Get latest OPENDAP URL
latest_url = get_latest_access_url(
    'http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.xml',
    'OPENDAP'
)
```

### Using catalog.latest property

```python
cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/latest.xml')

# Access latest dataset
latest_dataset = cat.latest
print(latest_dataset.name)
```

## Time-Based Filtering

### Filter by Nearest Time

Find dataset closest to specified time.

```python
from datetime import datetime

cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml')

# Find dataset nearest to specific time
target_time = datetime(2024, 1, 1, 12, 0)
nearest_ds = cat.datasets.filter_time_nearest(target_time)

# Custom regex for time patterns
nearest_ds = cat.datasets.filter_time_nearest(
    target_time,
    regex=r'(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})_(?P<hour>\d{2})(?P<minute>\d{2})'
)
```

### Filter by Time Range

Find all datasets within time range.

```python
# Define time range
start = datetime(2024, 1, 1, 0, 0)
end = datetime(2024, 1, 2, 0, 0)

# Get datasets in range
datasets_in_range = cat.datasets.filter_time_range(start, end)

# Iterate through results
for ds in datasets_in_range:
    print(f"{ds.name}: {ds.url_path}")
```

### Custom Time Patterns

For datasets with non-standard time naming.

```python
# GOES-16 data with julian day
datasets = cat.datasets.filter_time_range(
    start,
    end,
    regex=r'_s(?P<strptime>\d{13})',
    strptime='%Y%j%H%M%S'
)
```

## Services

### Service Information

```python
# List all services
for service in cat.services:
    print(f"Service: {service.name}")
    print(f"  Type: {service.service_type}")
    print(f"  Access URLs: {list(service.access_urls.keys())}")
```

### Compound Services

Some services are compound (contain multiple sub-services).

```python
for service in cat.services:
    if service.service_type == 'COMPOUND':
        print(f"Compound service: {service.name}")
        for sub_service in service.services:
            print(f"  - {sub_service.name}: {sub_service.service_type}")
```

### Resolver Services

Check if service is a resolver (resolves to actual data URL).

```python
for service in cat.services:
    if service.is_resolver():
        print(f"Resolver service: {service.name}")
```

## Common THREDDS Servers

### Unidata THREDDS Server

```python
# Main catalog
cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog.xml')

# NEXRAD radar data
cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/nexrad/nexrad.xml')

# GFS model data
cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p25deg/catalog.xml')

# NAM model data
cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/NAM/CONUS_12km/catalog.xml')
```

### NOMADS

```python
# NCEP model data
cat = TDSCatalog('https://nomads.ncep.noaa.gov/cgi-bin/thredds/catalog.xml')
```

### Other Servers

```python
# ESRL
cat = TDSCatalog('https://www.esrl.noaa.gov/psd/thredds/catalog.xml')

# RDA
cat = TDSCatalog('https://rda.ucar.edu/thredds/catalog.xml')
```

## Error Handling

### Invalid Catalog URL

```python
from requests.exceptions import RequestException

try:
    cat = TDSCatalog('http://invalid-url/catalog.xml')
except RequestException as e:
    print(f"Failed to access catalog: {e}")
```

### Dataset Not Found

```python
try:
    dataset = cat.datasets['nonexistent.nc']
except KeyError:
    print("Dataset not found in catalog")
```

### Access Method Not Available

```python
dataset = cat.datasets['model_data.nc']

if 'OPENDAP' not in dataset.access_urls:
    print("OPENDEND access not available")
    print(f"Available methods: {list(dataset.access_urls.keys())}")
```

## Best Practices

1. **Cache catalog objects** - Reuse TDSCatalog instances when possible
2. **Use appropriate access methods** - OPENDAP for subsetting, HTTPServer for full downloads
3. **Handle missing data gracefully** - Check for dataset existence before access
4. **Use time filtering** - Reduce catalog traversal with time-based filters
5. **Close connections** - Always close OPENDAP connections to free resources
