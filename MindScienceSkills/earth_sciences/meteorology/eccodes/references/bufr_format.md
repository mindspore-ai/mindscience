# BUFR Format Reference

## Overview

BUFR (Binary Universal Form for the Representation of meteorological data) is a binary data format designed for observational data. It is highly flexible and can represent a wide variety of meteorological and oceanographic observations.

## BUFR Structure

A BUFR file consists of one or more BUFR messages. Each message contains:

### Sections

**Section 0: Indicator Section**
- Identifies the message as BUFR
- Contains message length
- BUFR edition number

**Section 1: Identification Section**
- Master table
- Originating center
- Originating subcenter
- Update sequence number
- Data category
- Data subcategory
- Master table version
- Local table version

**Section 2: Data Description Section (Optional)**
- Descriptors for data template
- Data descriptors

**Section 3: Data Section**
- Packed data values
- Operated on by data description

**[Section 4: Data Description Section (Optional)]**
- Alternative data description

**Section 5: Data Section (Optional)**
- Alternative data values

## Data Categories

BUFR messages are organized by data categories:

- 0: Surface data (land and marine)
- 1: Upper air data (vertical soundings)
- 2: Sounding data (satellite)
- 3: Wind profiler data
- 4: Radar data
- 5: Physical oceanographic data
- 6: Chemical constituents
- 7: Radar wind profiles
- 8: Oceanographic vertical profiles
- 9: Moving ship data
- 10: Radar data (satellite)
- 11: Synographic cloud data
- 12: Synographic radiation data
- 13: Radiance data
- 14: GPS radio occultation soundings
- 15: Satellite precipitation
- 16: Aerosol optical depth
- 17: Surface data (land)
- 18: Surface data (marine)
- 19: High resolution surface data
- 20: Derived parameters
- 21: Satellite sounding data
- 22: Satellite wind data
- 23: Satellite surface data
- 24: Satellite radiation data
- 25: Satellite cloud data
- 254: Satellite data

## Common BUFR Descriptors

BUFR uses descriptors to define data structure:

### Table A (Category Descriptors)
- F: Table identifier (0)
- X: Data category
- Y: Data subcategory

### Table B (Element Descriptors)
- F: Table identifier (0)
- X: Parameter number
- Y: Scale factor and units

### Table C (Sequence Descriptors)
- F: Table identifier (3)
- X: Sequence number
- Y: Number of elements in sequence

### Table D (Replication Descriptors)
- F: Table identifier (1)
- X: Number of repetitions
- Y: Associated descriptor

## Common Observation Types

### Surface Observations
- Temperature
- Humidity
- Pressure
- Wind speed and direction
- Precipitation
- Visibility
- Cloud cover

### Upper Air Soundings
- Pressure, temperature, humidity profiles
- Wind profiles
- Geopotential height
- Derived parameters

### Satellite Data
- Radiances
- Brightness temperatures
- Cloud properties
- Atmospheric motion vectors
- Precipitation estimates

### Oceanographic Data
- Sea surface temperature
- Salinity
- Current profiles
- Wave data
- Water level

## Reading BUFR Messages

### Basic Reading Pattern

```python
import eccodes

with open('observations.bufr', 'rb') as f:
    while True:
        msg_id = eccodes.codes_bufr_new_from_file(f)
        if msg_id is None:
            break
        
        # Process message
        eccodes.codes_release(msg_id)
```

### Extracting Data Values

BUFR messages must be "unpacked" before accessing data values:

```python
import eccodes

msg_id = eccodes.codes_bufr_new_from_file(f)

# Unpack the message
eccodes.codes_set(msg_id, 'unpack', 1)

# Now you can access values
value = eccodes.codes_get(msg_id, 'temperature')

eccodes.codes_release(msg_id)
```

### Iterating Through Replicated Data

For observations with multiple levels or time steps:

```python
import eccodes

msg_id = eccodes.codes_bufr_new_from_file(f)
eccodes.codes_set(msg_id, 'unpack', 1)

# Get number of subsets
num_subsets = eccodes.codes_get(msg_id, 'numberOfSubsets')

for i in range(num_subsets):
    # Select subset
    eccodes.codes_set(msg_id, 'subsetNumber', i + 1)
    
    # Access values for this subset
    pressure = eccodes.codes_get(msg_id, 'pressure')
    temperature = eccodes.codes_get(msg_id, 'temperature')
    print(f"Level {i+1}: P={pressure}, T={temperature}")

eccodes.codes_release(msg_id)
```

## Common BUFR Keys

### Identification Keys
- `bufrHeaderCentre`: Originating center
- `bufrHeaderSubCentre`: Originating subcenter
- `masterTableVersion`: Master table version
- `localTableVersion`: Local table version
- `dataCategory`: Data category
- `dataSubCategory`: Data subcategory

### Data Keys
- `numberOfSubsets`: Number of data subsets
- `numberOfObservations`: Number of observations
- `compressedData`: Whether data is compressed

### Observation Keys
- `latitude`: Latitude of observation
- `longitude`: Longitude of observation
- `heightOfStation`: Station height
- `timeSignificance`: Time significance
- `year`, `month`, `day`, `hour`, `minute`: Observation time

## Working with Different Data Types

### Scalar Values
Single values per observation:
```python
temperature = eccodes.codes_get(msg_id, 'temperature')
```

### Arrays
Multiple values per observation:
```python
windComponents = eccodes.codes_get(msg_id, 'windComponents')
# Returns array of values
```

### Character Data
Text data:
```python
stationName = eccodes.codes_get(msg_id, 'stationName')
```

### Quality Control Flags
BUFR often includes quality control information:
```python
qcFlags = eccodes.codes_get(msg_id, 'qualityControlFlags')
```

## Data Quality

### Quality Control
BUFR messages typically include:
- Original data values
- Quality control flags
- Quality control information

### Missing Data
Missing data in BUFR is represented by:
- Special values (e.g., -9999)
- Missing value indicators
- Bit flags

## Best Practices

1. **Always unpack messages** - Before accessing data values
2. **Handle subsets** - Many observations have multiple subsets
3. **Check data categories** - To understand observation type
4. **Validate descriptors** - Ensure data structure is correct
5. **Handle missing data** - Check for missing value indicators
6. **Use appropriate tables** - Verify table versions

## Common Use Cases

### Processing Surface Observations
```python
import eccodes

with open('surface.bufr', 'rb') as f:
    while True:
        msg_id = eccodes.codes_bufr_new_from_file(f)
        if msg_id is None:
            break
        
        eccodes.codes_set(msg_id, 'unpack', 1)
        
        lat = eccodes.codes_get(msg_id, 'latitude')
        lon = eccodes.codes_get(msg_id, 'longitude')
        temp = eccodes.codes_get(msg_id, 'airTemperature')
        
        print(f"Location: ({lat}, {lon}), Temperature: {temp}")
        
        eccodes.codes_release(msg_id)
```

### Processing Upper Air Soundings
```python
import eccodes

with open('sounding.bufr', 'rb') as f:
    while True:
        msg_id = eccodes.codes_bufr_new_from_file(f)
        if msg_id is None:
            break
        
        eccodes.codes_set(msg_id, 'unpack', 1)
        
        num_subsets = eccodes.codes_get(msg_id, 'numberOfSubsets')
        
        for i in range(num_subsets):
            eccodes.codes_set(msg_id, 'subsetNumber', i + 1)
            
            pressure = eccodes.codes_get(msg_id, 'pressure')
            temperature = eccodes.codes_get(msg_id, 'temperature')
            height = eccodes.codes_get(msg_id, 'geopotentialHeight')
            
            print(f"Level: P={pressure}, T={temperature}, Z={height}")
        
        eccodes.codes_release(msg_id)
```

## Troubleshooting

### Common Issues

**"Cannot read value"**
- Ensure message is unpacked
- Check if key exists for this data category
- Verify table versions

**"Invalid BUFR message"**
- Check file integrity
- Verify BUFR edition
- Check section lengths

**"Subset errors"**
- Verify number of subsets
- Check subset numbering (1-based)
- Ensure proper subset selection

**"Missing data"**
- Check for missing value indicators
- Verify data quality flags
- Handle special values appropriately

## References

- WMO BUFR Documentation: https://www.wmo.int/pages/prog/www/WMOCodes.html
- ECMWF BUFR API: https://confluence.ecmwf.int/display/UDOC/BUFR+API+documentation
- NCEP BUFR Tables: https://www.nco.ncep.noaa.gov/pmb/docs/bufr/