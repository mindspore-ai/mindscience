# Writing GRIB/BUFR Files Guide

This guide provides detailed patterns and best practices for creating and writing GRIB and BUFR files with ecCodes.

## Creating GRIB Files

### Basic GRIB2 Message Creation

The simplest way to create a GRIB message is to start from a sample:

```python
import eccodes
import numpy as np

# Create a new GRIB2 message from a sample
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')

# Set required keys
eccodes.codes_set(msg_id, 'Ni', 360)  # Longitude points
eccodes.codes_set(msg_id, 'Nj', 181)  # Latitude points
eccodes.codes_set(msg_id, 'dataDateDate', 20240330)
eccodes.codes_set(msg_id, 'dataTime', 1200)
eccodes.codes_set(msg_id, 'shortName', 't')
eccodes.codes_set(msg_id, 'units', 'K')

# Create and set values
values = np.random.rand(360 * 181) * 50 + 250  # Random temperatures 250-300K
eccodes.codes_set_values(msg_id, values)

# Write to file
with open('output.grib', 'wb') as f:
    eccodes.codes_write(msg_id, f)

eccodes.codes_release(msg_id)
```

### Using Different Samples

ecCodes provides sample templates for common GRIB messages:

```python
# Regular lat-lon grid
msg_id = eccodes.codes_grib_new_from_samples('regular_ll_sfc_grib2')

# Gaussian grid
msg_id = eccodes.codes_grib_new_from_samples('gg_sfc_grib2')

# Reduced Gaussian grid
msg_id = eccodes.codes_grib_new_from_samples('reduced_gg_sfc_grib2')

# Lambert conformal
msg_id = eccodes.codes_grib_new_from_samples('lambert_sfc_grib2')

# Polar stereographic
msg_id = eccodes.codes_grib_new_from_samples('polar_stereographic_sfc_grib2')
```

### Setting Grid Definition

#### Regular Latitude-Longitude Grid

```python
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')

# Grid type
eccodes.codes_set(msg_id, 'gridType', 0)  # 0 = regular_ll

# Grid dimensions
eccodes.codes_set(msg_id, 'Ni', 360)
eccodes.codes_set(msg_id, 'Nj', 181)

# Grid extent
eccodes.codes_set(msg_id, 'latitudeOfFirstGridPoint', 90.0)
eccodes.codes_set(msg_id, 'longitudeOfFirstGridPoint', 0.0)
eccodes.codes_set(msg_id, 'latitudeOfLastGridPoint', -90.0)
eccodes.codes_set(msg_id, 'longitudeOfLastGridPoint', 359.0)

# Grid resolution
eccodes.codes_set(msg_id, 'iDirectionIncrement', 1.0)
eccodes.codes_set(msg_id, 'jDirectionIncrement', 1.0)
```

#### Gaussian Grid

```python
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')

# Grid type
eccodes.codes_set(msg_id, 'gridType', 4)  # 4 = gaussian

# Grid dimensions
eccodes.codes_set(msg_id, 'N', 32)  # N32 Gaussian grid
eccodes.codes_set(msg_id, 'Ni', 96)  # 96 longitude points
eccodes.codes_set(msg_id, 'Nj', 64)  # 64 latitude points (2*N)
```

#### Lambert Conformal Conic

```python
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')

# Grid type
eccodes.codes_set(msg_id, 'gridType', 3)  # 3 = lambert

# Projection parameters
eccodes.codes_set(msg_id, 'LaD', 40.0)  # Latitude of projection center
eccodes.codes_set(msg_id, 'LoD', -100.0)  # Longitude of projection center
eccodes.codes_set(msg_id, 'Latin1', 30.0)  # First standard parallel
eccodes.codes_set(msg_id, 'Latin2', 50.0)  # Second standard parallel

# Grid spacing
eccodes.codes_set(msg_id, 'DxInMetres', 12000)  # 12 km
eccodes.codes_set(msg_id, 'DyInMetres', 12000)

# Grid dimensions
eccodes.codes_set(msg_id, 'Nx', 100)
eccodes.codes_set(msg_id, 'Ny', 100)
```

### Setting Product Definition

#### Temperature at 2m

```python
# Parameter identification
eccodes.codes_set(msg_id, 'discipline', 0)  # Meteorology
eccodes.codes_set(msg_id, 'parameterCategory', 0)  # Temperature
eccodes.codes_set(msg_id, 'parameterNumber', 0)  # Temperature
eccodes.codes_set(msg_id, 'typeOfLevel', 'heightAboveGround')
eccodes.codes_set(msg_id, 'level', 2)

' Alternative: use shortName
eccodes.codes_set(msg_id, 'shortName', 't2m')
eccodes.codes_set(msg_id, 'units', 'K')
```

#### Wind Components

```python
# U-component of wind at 10m
eccodes.codes_set(msg_id, 'shortName', 'u10')
eccodes.codes_set(msg_id, 'units', 'm s-1')

# V-component of wind at 10m
eccodes.codes_set(msg_id, 'shortName', 'v10')
eccodes.codes_set(msg_id, 'units', 'm s-1')
```

#### Pressure Levels

```python
# Temperature at 500 hPa
eccodes.codes_set(msg_id, 'shortName', 't')
eccodes.codes_set(msg_id, 'typeOfLevel', 'isobaricInhPa')
eccodes.codes_set(msg_id, 'level', 500)
eccodes.codes_set(msg_id, 'units', 'K')
```

### Setting Temporal Information

#### Analysis Data

```python
# Reference time
eccodes.codes_set(msg_id, 'dataDate', 20240330)  # YYYYMMDD
eccodes.codes_set(msg_id, 'dataTime', 1200)  # HHMM

# Analysis (no forecast step)
eccodes.codes_set(msg_id, 'step', 0)
eccodes.codes_set(msg_id, 'stepType', 'instant')
```

#### Forecast Data

```python
# Reference time
eccodes.codes_set(msg_id, 'dataDate', 20240330)
eccodes.codes_set(msg_id, 'dataTime', 1200)

# Forecast step
eccodes.codes_set(msg_id, 'step', 6)  # 6-hour forecast
eccodes.codes_set(msg_id, 'stepType', 'fc')
```

#### Accumulated Fields

```python
# 6-hour precipitation accumulation
eccodes.codes_set(msg_id, 'shortName', 'tp')
eccodes.codes_set(msg_id, 'units', 'kg m-2')

# Accumulation period
eccodes.codes_set(msg_id, 'stepType', 'accum')
eccodes.codes_set(msg_id, 'startStep', 0)
eccodes.codes_set(msg_id, 'endStep', 6)
```

### Setting Data Values

#### Simple Values

```python
import numpy as np

# Create values
Ni = 360
Nj = 181
values = np.random.rand(Ni * Nj) * 50 + 250

# Set values
eccodes.codes_set_values(msg_id, values)
```

#### Missing Data

```python
import numpy as np

# Create values with missing data
values = np.random.rand(360 * 181) * 50 + 250

# Set some values to missing
values[100:200] = np.nan

# Set missing value indicator
eccodes.codes_set(msg_id, 'missingValue', -1e+100)

# Set values
eccodes.codes_set_values(msg_id, values)
```

#### Bitmap for Missing Data

```python
import numpy as np

# Create values
values = np.random.rand(360 * 181) * 50 + 250

# Create bitmap
bitmap = np.ones(360 * 181, dtype=np.int32)
bitmap[100:200] = 0  # Mark as missing

# Set bitmap
eccodes.codes_set(msg_id, 'bitmapPresent', 1)
eccodes.codes_set(msg_id, 'bitmap', bitmap)

# Set values
eccodes.codes_set_values(msg_id, values)
```

### Writing Multiple Messages

```python
import eccodes
import numpy as np

with open('output.grib', 'wb') as f:
    # Write temperature at multiple levels
    levels = [1000, 850, 700, 500, 300, 200]
    
    for level in levels:
        msg_id = eccodes.codes_grib_new_from_samples('GRIB2')
        
        # Set grid
        eccodes.codes_set(msg_id, 'Ni', 360)
        eccodes.codes_set(msg_id, 'Nj', 181)
        
        # Set parameter
        eccodes.codes_set(msg_id, 'shortName', 't')
        eccodes.codes_set(msg_id, 'typeOfLevel', 'isobaricInhPa')
        eccodes.codes_set(msg_id, 'level', level)
        
        # Set time
        eccodes.codes_set(msg_id, 'dataDate', 20240330)
        eccodes.codes_set(msg_id, 'dataTime', 1200)
        eccodes.codes_set(msg_id, 'step', 0)
        
        # Set values
        values = np.random.rand(360 * 181) * 50 + 250
        eccodes.codes_set_values(msg_id, values)
        
        # Write message
        eccodes.codes_write(msg_id, f)
        
        # Release message
        eccodes.codes_release(msg_id)
```

## Modifying GRIB Files

### Modifying Values

```python
import eccodes
import numpy as np

with open('input.grib', 'rb') as f_in:
    with open('output.grib', 'wb') as f_out:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f_in)
            if msg_id is None:
                break
            
            # Get values
            values = eccodes.codes_get_values(msg_id)
            
            # Modify values (e.g., convert K to C)
            values = values - 273.15
            
            # Set modified values
            eccodes.codes_set_values(msg_id, values)
            
            # Update units
            eccodes.codes_set(msg_id, 'units', 'C')
            
            # Write modified message
            eccodes.codes_write(msg_id, f_out)
            
            eccodes.codes_release(msg_id)
```

### Modifying Metadata

```python
import eccodes

with open('input.grib', 'rb') as f_in:
    with open('output.grib', 'wb') as f_out:
        while True:
            msg_id = eccodes.codes_grib_new_from_file(f_in)
            if msg_id is None:
                break
            
            # Modify metadata
            eccodes.codes_set(msg_id, 'generatingProcessIdentifier', 255)
            eccodes.codes_set(msg_id, 'productionStatusOfProcessedData', 1)  # Test data
            
            # Write modified message
            eccodes.codes_write(msg_id, f_out)
            
            eccodes.codes_release(msg_id)
```

### Cloning Messages

```python
import eccodes

with open('input.grib', 'rb') as f_in:
    msg_id = eccodes.codes_grib_new_from_file(f_in)
    
    # Clone message
    clone_id = eccodes.codes_clone(msg_id)
    
    # Modify clone
    eccodes.codes_set(clone_id, 'shortName', 'modified_field')
    
    # Write clone
    with open('output.grib', 'wb') as f_out:
        eccodes.codes_write(clone_id, f_out)
    
    eccodes.codes_release(msg_id)
    eccodes.codes_release(clone_id)
```

## Creating BUFR Files

### Basic BUFR Message Creation

```python
import eccodes

# Create a new BUFR message from a sample
msg_id = eccodes.codes_bufr_new_from_samples('BUFR4')

# Set identification
eccodes.codes_set(msg_id, 'bufrHeaderCentre', 98)  # ECMWF
eccodes.codes_set(msg_id, 'bufrHeaderSubCentre', 0)
eccodes.codes_set(msg_id, 'masterTableVersion', 13)
eccodes.codes_set(msg_id, 'localTableVersion', 0)

# Set data category
eccodes.codes_set(msg_id, 'dataCategory', 0)  # Surface data
eccodes.codes_set(msg_id, 'dataSubCategory', 0)

# Set observation
eccodes.codes_set(msg_id, 'unpack', 1)  # Unpack to set values
eccodes.codes_set(msg_id, 'latitude', 40.0)
eccodes.codes_set(msg_id, 'longitude', -100.0)
eccodes.codes_set(msg_id, 'airTemperature', 293.15)

# Write to file
with open('output.bufr', 'wb') as f:
    eccodes.codes_write(msg_id, f)

eccodes.codes_release(msg_id)
```

### BUFR with Multiple Subsets

```python
import eccodes

msg_id = eccodes.codes_bufr_new_from_samples('BUFR4')

# Set identification
eccodes.codes_set(msg_id, 'bufrHeaderCentre', 98)
eccodes.codes_set(msg_id, 'dataCategory', 1)  # Upper air data

# Set number of subsets
eccodes.codes_set(msg_id, 'numberOfSubsets', 3)
eccodes.codes_set(msg_id, 'compressedData', 0)

# Unpack
eccodes.codes_set(msg_id, 'unpack', 1)

# Set values for each subset
for i in range(3):
    eccodes.codes_set(msg_id, 'subsetNumber', i + 1)
    eccodes.codes_set(msg_id, 'pressure', 1000 - i * 100)
    eccodes.codes_set(msg_id, 'airTemperature', 290 - i * 5)

# Write to file
with open('output.bufr', 'wb') as f:
    eccodes.codes_write(msg_id, f)

eccodes.codes_release(msg_id)
```

## Best Practices

### 1. Always Release Messages

```python
msg_id = eccodes.codes_grib_new_from_samples('GRIB2')
try:
    # Work with message
    eccodes.codes_set(msg_id, 'Ni', 360)
    # ...
finally:
    eccodes.codes_release(msg_id)
```

### 2. Validate Before Writing

```python
# Check required keys are set
required_keys = ['Ni', 'Nj', 'shortname', 'dataDate', 'dataTime']

for key in required_keys:
    try:
        value = eccodes.codes_get(msg_id, key)
    except eccodes.EcCodesError:
        print(f"Error: Required key '{key}' not set")
```

### 3. Use Appropriate Samples

Choose samples that match your needs:
- `regular_ll_sfc_grib2` for surface data on regular grid
- `gg_sfc_grib2` for surface data on Gaussian grid
- `regular_ll_pl_grib2` for pressure level data
- `regular_ll_ml_grib2` for model level data

### 4. Set Grid Type Explicitly

```python
# Always set grid type
eccodes.codes_set(msg_id, 'gridType', 0)  # regular_ll
```

### 5. Handle Missing Data Properly

```python
# Use bitmap for irregular missing data
eccodes.codes_set(msg_id, 'bitmapPresent', 1)
eccodes.codes_set(msg_id, 'bitmap', bitmap)
```

### 6. Use Appropriate Data Types

```python
# Use appropriate numpy data types
values = np.array(data, dtype=np.float32)  # or np.float64
eccodes.codes_set_values(msg_id, values)
```

### 7. Document Custom Parameters

If using custom parameters or local tables:
```python
# Set local table version
eccodes.codes_set(msg_id, 'localTablesVersion', 1)

# Document in metadata
eccodes.codes_set(msg_id, 'localDefinition', 1)
```

## Common Patterns

### Creating Analysis File

```python
def create_analysis_file(output_file, fields, grid_params, date, time):
    """Create an analysis GRIB file with multiple fields."""
    
    with open(output_file, 'wb') as f:
        for field in fields:
            msg_id = eccodes.codes_grib_new_from_samples('GRIB2')
            
            # Set grid
            eccodes.codes_set(msg_id, 'gridType', grid_params['type'])
            eccodes.codes_set(msg_id, 'Ni', grid_params['Ni'])
            eccodes.codes_set(msg_id, 'Nj', grid_params['Nj'])
            
            # Set parameter
            eccodes.codes_set(msg_id, 'shortName', field['shortName'])
            eccodes.codes_set(msg_id, 'typeOfLevel', field['level_type'])
            eccodes.codes_set(msg_id, 'level', field['level'])
            
            # Set time (analysis)
            eccodes.codes_set(msg_id, 'dataDate', date)
            eccodes.codes_set(msg_id, 'dataTime', time)
            eccodes.codes_set(msg_id, 'step', 0)
            
            # Set values
            eccodes.codes_set_values(msg_id, field['values'])
            
            # Write
            eccodes.codes_write(msg_id, f)
            eccodes.codes_release(msg_id)
```

### Creating Forecast File

```python
def create_forecast_file(output_file, field, grid_params, ref_date, ref_time, steps):
    """Create a forecast GRIB file with multiple time steps."""
    
    with open(output_file, 'wb') as f:
        for step in' steps:
            msg_id = eccodes.codes_grib_new_from_samples('GRIB2')
            
            # Set grid
            eccodes.codes_set(msg_id, 'gridType', grid_params['type'])
            eccodes.codes_set(msg_id, 'Ni', grid_params['Ni'])
            eccodes.codes_set(msg_id, 'Nj', grid_params['Nj'])
            
            # Set parameter
            eccodes.codes_set(msg_id, 'shortName', field['shortName'])
            eccodes.codes_set(msg_id, 'typeOfLevel', field['level_type'])
            eccodes.codes_set(msg_id, 'level', field['level'])
            
            # Set time (forecast)
            eccodes.codes_set(msg_id, 'dataDate', ref_date)
            eccodes.codes_set(msg_id, 'dataTime', ref_time)
            eccodes.codes_set(msg_id, 'step', step)
            
            # Set values
            eccodes.codes_set_values(msg_id, field['values'])
            
            # Write
            eccodes.codes_write(msg_id, f)
            eccodes.codes_release(msg_id)
```

## Troubleshooting

### Common Issues

**"Required key not set"**
- Ensure all required keys are set before writing
- Check grid definition is complete
- Verify temporal information is set

**"Invalid grid definition"**
- Check grid type is set
- Verify grid dimensions are consistent
- Ensure grid increments are set

**"Data packing errors"**
- Check values array length matches grid size
- Verify data type is appropriate
- Ensure no invalid values (NaN, Inf)

**"Write failed"**
- Check file is open in binary mode
- Verify message is complete
- Ensure sufficient disk space

## References

- ECMWF GRIB API: https://confluence.ecmwf.int/display/UDOC/GRIB+API+documentation
- WMO GRIB Specification: https://www.wmo.int/pages/prog/www/WMOCodes.html