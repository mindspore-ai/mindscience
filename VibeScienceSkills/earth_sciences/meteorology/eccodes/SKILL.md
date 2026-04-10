---
name: eccodes
description: Comprehensive skill for working with ECMWF ecCodes library to handle GRIB and BUFR meteorological data files. Use when Claude needs to:(1) Read, decode, and analyze GRIB/BUFR files, (2) Extract and manipulate meteorological data fields, (3) Convert between different meteorological data formats, (4) Inspect metadata and keys from GRIB/BUFR messages, (5) Write or modify GRIB/BUFR files, (6) Work with ECMWF forecast model outputs, (7) Process satellite or reanalysis data in GRIB format, (8) Handle meteorological data from various weather services
---

# ecCodes

## Overview

ecCodes is the ECMWF library for decoding and encoding messages in the GRIB (General Regularly-distributed Information in Binary) and BUFR (Binary Universal Form for the Representation of meteorological data) formats. These are the primary formats used by meteorological services worldwide for storing and exchanging weather and climate data.

## Quick Start

**Reading a GRIB file:**
```python
import eccodes

with open('file.grib', 'rb') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)
    print(eccodes.codes_get(msg_id, 'shortName'))
    eccodes.codes_release(msg_id)
```

**Iterating through all messages:**
```python
import eccodes

with open('file.grib', 'rb') as f:
    while True:
        msg_id = eccodes.codes_grib_new_from_file(f)
        if msg_id is None:
            break
        # Process message
        eccodes.codes_release(msg_id)
```

**Using Python bindings for data extraction:**
```python
import eccodes
import numpy as np

with open('file.grib', 'rb') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)
    values = eccodes.codes_get_values(msg_id)
    print(f"Shape: {values.shape}")
    print(f"Min: {np.min(values)}, Max: {np.max(values)}")
    eccodes.codes_release(msg_id)
```

## Workflow Decision Tree

1. **What type of operation do you need?**
   - **Reading/analyzing data** → Follow "Reading GRIB/BUFR Files" workflow
   - **Extracting specific fields** → Follow "Data Extraction" workflow
   - **Converting formats** → Follow "Format Conversion" workflow
   - **Writing/creating files** → Follow "Writing GRIB/BUFR Files" workflow
   - **Inspecting metadata** → Follow "Metadata Inspection" workflow

## Reading GRIB/BUFR Files

### Basic Reading Pattern

For reading GRIB files, use `scripts/read_grib.py` for comprehensive file analysis:

```python
# See scripts/read_grib.py for complete implementation
python scripts/read_grib.py input.grib --summary
```

Key concepts:
- GRIB files contain one or more messages
- Each message represents a meteorological field (temperature, pressure, etc.)
- Messages must be explicitly released after use
- Use context managers or try-finally for proper resource cleanup

### Reading Multiple Messages

When working with multi-message GRIB files:

```python
import eccodes

def read_all_messages(filepath):
    messages = []
    with open(filepath, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            messages.append(msg_id)
    return messages

# Don't forget to release!
for msg_id in messages:
    eccodes.codes_release(msg_id)
```

### BUFR File Reading

For BUFR files (observational data, satellite data):

```python
import eccodes

with open('observations.bufr', 'rb') as f:
    while True:
        msg_id = eccodes.codes_bufr_new_from_file(f)
        if msg_id is None:
            break
        # Process BUFR message
        eccodes.codes_release(msg_id)
```

See `references/bufr_format.md` for BUFR-specific handling details.

## Data Extraction

### Extracting Values

Get data values as numpy arrays:

```python
import eccodes
import numpy as np

with open('file.grib', 'rb') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)
    
    # Get all values
    values = eccodes.codes_get_values(msg_id)
    
    # Get specific subset
    values_subset = eccodes.codes_get_values(msg_id, start=0, count=100)
    
    # Get as numpy array directly
    arr = np.array(values)
    
    eccodes.codes_release(msg_id)
```

### Extracting Metadata

Common metadata keys for GRIB messages:

```python
# Identification
shortName = eccodes.codes_get(msg_id, 'shortName')        # Variable name
name = eccodes.codes_get(msg_id, 'name')                  # Full variable name
units = eccodes.codes_get(msg_id, 'units')                # Units

# Temporal information
dataDate = eccodes.codes_get(msg_id, 'dataDate')         # YYYYMMDD
dataTime = eccodes.codes_get(msg_id, 'dataTime')         # HHMM
step = eccodes.codes_get(msg_id, 'step')                  # Forecast step (hours)
stepType = eccodes.codes_get(msg_id, 'stepType')         # Step type indicator

# Spatial information
Ni = eccodes.codes_get(msg_id, 'Ni')                      # Number of points in i-direction
Nj = eccodes.codes_get(msg_id, 'Nj')                      # Number of points in j-direction
latitudeOfFirstGridPoint = eccodes.codes_get(msg_id, 'latitudeOfFirstGridPoint')
longitudeOfFirstGridPoint = eccodes.codes_get(msg_id, 'longitudeOfFirstGridPoint')

# Model information
centre = eccodes.codes_get(msg_id, 'centre')              # Originating centre
generatingProcessIdentifier = eccodes.codes_get(msg_id, 'generatingProcessIdentifier')
```

See `references/grib_keys.md` for comprehensive key reference.

### Selecting Specific Messages

Use selection keys to find specific fields:

```python
import eccodes

def find_message_by_shortname(filepath, shortname):
    with open(filepath, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            if eccodes.codes_get(msg_id, 'shortName') == shortname:
                return msg_id
            
            eccodes.codes_release(msg_id)
    return None
```

Use `scripts/extract_field.py` for advanced field selection and extraction.

## Metadata Inspection

### Inspecting Message Structure

To understand the structure of a GRIB/BUFR message:

```python
import eccodes

def inspect_message(msg_id):
    # Get namespace
    namespace = eccodes.codes_get_namespace(msg_id)
    print(f"Namespace: {namespace}")
    
    # Get number of keys
    num_keys = eccodes.codes_get_count(msg_id)
    print(f"Number of keys: {num_keys}")
    
    # Get all keys
    keys = eccodes.codes_get_keys(msg_id)
    print(f"Keys: {keys[:10]}...")  # Show first 10
```

### Detailed Key Information

For detailed information about available keys and their types:

```python
import eccodes

def get_key_info(msg_id, key):
    try:
        value = eccodes.codes_get(msg_id, key)
        key_type = eccodes.codes_get_native_type(msg_id, key)
        size = eccodes.codes_get_size(msg_id, key)
        
        return {
            'value': value,
            'type': key_type,
            'size': size
        }
    except eccodes.EcCodesError:
        return None
```

Use `scripts/inspect_grib.py` for comprehensive message inspection.

## Writing GRIB/BUFR Files

### Creating New GRIB Messages

Create a new GRIB message from scratch:

```python
import eccodes

# Create a new GRIB2 message
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')

# Set required keys
eccodes.codes_set(msg_id, 'Ni', 360)  # Longitude points
eccodes.codes_set(msg_id, 'Nj', 181)  # Latitude points
eccodes.codes_set(msg_id, 'dataDate', 20240330)
eccodes.codes_set(msg_id, 'dataTime', 1200)
eccodes.codes_set(msg_id, 'shortName', 't')
eccodes.codes_set(msg_id, 'units', 'K')

# Set values
values = [273.15] * (360 * 181)  # Example data
eccodes.codes_set_values(msg_id, values)

# Write to file
with open('output.grib', 'wb') as f:
    eccodes.codes_write(msg_id, f)

eccodes.codes_release(msg_id)
```

### Modifying Existing Messages

Modify an existing GRIB message:

```python
import eccodes

with open('input.grib', 'rb') as f:
    msg_id = eccodes.codes_grib_new_from_file(f)
    
    # Modify values
    values = eccodes.codes_get_values(msg_id)
    values = values * 1.5  # Scale by 1.5
    eccodes.codes_set_values(msg_id, values)
    
    # Modify metadata
    eccodes.codes_set(msg_id, 'shortName', 't2m')
    
    # Write modified message
    with open('output.grib', 'wb') as out_f:
        eccodes.codes_write(msg_id, out_f)
    
    eccodes.codes_release(msg_id)
```

See `references/writing_guide.md` for detailed writing patterns and best practices.

## Format Conversion

### GRIB to NetCDF

Convert GRIB files to NetCDF format:

```python
# Use scripts/grib_to_netcdf.py
python scripts/grib_to_netcdf.py input.grib output.nc
```

### GRIB to CSV

Extract data to CSV format:

```python
# Use scripts/grib_to_csv.py
python scripts/grib_to_csv.py input.grib output.csv --field t2m
```

### BUFR to JSON

Convert BUFR observational data to JSON:

```python
# Use scripts/bufr_to_json.py
python scripts/bufr_to_json.py observations.bufr output.json
```

## Common Operations

### Spatial Subsetting

Extract a spatial subset from a GRIB file:

```python
# Use scripts/extract_subset.py
python scripts/extract_subset.py input.grib output.grib \
    --lat-north 60 --lat-south 30 \
    --lon-west -120 --lon-east -90
```

See `references/spatial_operations.md` for spatial processing details.

### Temporal Subsetting

Extract specific time steps:

```python
# Use scripts:extract_timesteps.py
python scripts/extract_timesteps.py input.grib output.grib \
    --date 20240330 --time 1200 --step 0
```

### Field Combination

Combine multiple fields into a single file:

```python
# Use scripts/combine_fields.py
python scripts/combine_fields.py output.grib \
    --input1 temp.grib --field1 t2m \
    --input2 pressure.grib --field2 sp
```

## Error Handling

### Common Error Patterns

```python
import eccodes

try:
    msg_id = eccodes.codes_grib_new_from_file(f)
    if msg_id is None:
        raise ValueError("No valid GRIB message found")
    
    value = eccodes.codes_get(msg_id, 'nonexistent_key')
except eccodes.EcCodesError as e:
    print(f"ecCodes error: {e}")
    print(f"Error code: {e.code}")
    print(f"Error message: {e.message}")
except Exception as e:
    print(f"General error: {e}")
finally:
    if 'msg_id' in locals() and msg_id is not None:
        eccodes.codes_release(msg_id)
```

### Validation

Validate GRIB file integrity:

```python
# Use scripts/validate_grib.py
python scripts/validate_grib.py input.grib
```

## Performance Optimization

### Efficient Reading

For large files, use memory-efficient patterns:

```python
import eccodes

def process_large_file(filepath, process_func):
    """Process large GRIB files without loading all messages into memory"""
    with open(filepath, 'rb') as f:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f)
            if msg_id is None:
                break
            
            try:
                process_func(msg_id)
            finally:
                eccodes.codes_release(msg_id)
```

### Indexing

Use indexing for fast message access:

```python
import eccodes

# Create index
index_id = eccodes.codes_index_new_from_file('file.grib', 'shortName,step')

# Select messages by keys
eccodes.codes_index_select(index_id, 'shortName', 't2m')
eccodes.codes_index_select(index_id, 'step', 0)

# Get selected messages
msg_id = eccodes.codes_new_from_index(index_id)

# Clean up
eccodes.codes_release(msg_id)
eccodes.codes_index_release(index_id)
```

See `references/performance_guide.md` for optimization techniques.

## Resources

### scripts/
Executable Python scripts for common ecCodes operations:

- **read_grib.py** - Read and display GRIB file contents with options for summary, detailed view, or specific field extraction
- **extract_field.py** - Extract specific fields from GRIB files based on shortName, parameter name, or other criteria
- **inspect_grib.py** - Comprehensive inspection of GRIB message structure, keys, and metadata
- **grib_to_netcdf.py** - Convert GRIB files to NetCDF format using cfgrib or xarray
- **grib_to_csv.py** - Export GRIB data to CSV format with configurable output options
- **bufr_to_json.py** - Convert BUFR observational data to JSON format
- **extract_subset.py** - Extract spatial subsets from GRIB files
- **extract_timesteps.py** - Extract specific time steps from GRIB files
- **combine_fields.py** - Combine multiple fields from different files into a single GRIB file
- **validate_grib.py** - Validate GRIB file integrity and structure
- **modify_grib.py** - Modify GRIB file metadata or values

### references/
Detailed documentation and reference materials:

- **grib_format.md** - Comprehensive guide to GRIB format structure, editions (GRIB1, GRIB2), and message organization
- **bufr_format.md** - BUFR format details, message structure, and handling observational data
- **grib_keys.md** - Complete reference of GRIB keys by category (identification, temporal, spatial, model information)
- **writing_guide.md** - Detailed patterns for creating and writing GRIB/BUFR files
- **spatial_operations.md** - Spatial processing techniques including subsetting, regridding, and coordinate transformations
- **performance_guide.md** - Performance optimization techniques for large file processing and efficient memory usage
- **api_reference.md** - Complete ecCodes Python API reference with function signatures and usage examples
- **common_use_cases.md** - Real-world use cases and patterns for common meteorological data processing tasks

### assets/
Example files and templates:

- **sample_grib2.grib** - Sample GRIB2 file for testing and demonstration
- **sample_bufr.bufr** - Sample BUFR file with observational data
- **grib_template.json** - JSON template for GRIB message structure
- **key_mapping.yaml** - YAML file mapping common parameter names to GRIB keys
- **config_template.ini** - Configuration template for ecCodes operations

## Best Practices

1. **Always release messages** - Use try-finally or context managers to ensure proper cleanup
2. **Use indexing for large files** - Significantly improves performance for repeated access
3. **Validate input files** - Check file integrity before processing
4. **Handle errors gracefully** - Catch and report ecCodes-specific errors
5. **Use appropriate data types** - ecCodes provides native type information for each key
6. **Document custom keys** - When using local or experimental keys, document their meaning
7. **Test with sample files** - Use provided sample files to verify operations
8. **Consider memory usage** - Process large files in chunks rather than loading entirely into memory

## Troubleshooting

### Common Issues

**"No valid GRIB message found"**
- Check file format and integrity
- Verify file is not corrupted
- Ensure correct file type (GRIB vs BUFR)

**"Key not found" errors**
- Verify key name is correct for the message edition
- Check if key is optional for the specific message type
- Use `codes_get_keys()` to list available keys

**Memory issues with large files**
- Process messages one at a time
- Use indexing for selective access
- Avoid loading entire file into memory

**Performance issues**
- Use indexing for repeated access
- Process files sequentially rather than in parallel
- Consider using C API for critical performance sections

See `references/troubleshooting.md` for detailed troubleshooting guide.