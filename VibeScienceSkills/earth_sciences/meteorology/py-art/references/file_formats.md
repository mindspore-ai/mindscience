# Py-ART File Formats Reference

This document provides a comprehensive reference of radar data formats supported by Py-ART.

## Overview

Py-ART supports multiple radar data formats commonly used in atmospheric research and operational meteorology.

## ARM netCDF Format

### Description

ARM (Atmospheric Radiation Measurement) program uses netCDF format for radar data. This is the primary format used by ARM research facilities.

### Reading ARM netCDF

```python
import pyart

# Read ARM netCDF file
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Display radar information
print(radar)
print(f"Radar type: {radar.radar_type}")
print(f"Number of sweeps: {radar.nsweeps}")
print(f"Number of gates: {radar.ngates}")
```

### ARM netCDF Structure

**Global Attributes:**
- `radar_type`: Type of radar (e.g., 'xsapr_saprc', 'kasacri')
- `site_name`: ARM site name
- `instrument_name`: Radar instrument name
- `latitude`: Radar latitude
- `longitude`: Radar longitude
- `altitude`: Radar altitude (meters)
- `frequency`: Radar frequency (Hz)
- `wavelength`: Radar wavelength (meters)
- `beam_width`: Beam width (degrees)
- `prf`: Pulse repetition frequency (Hz)
- `nyquist_velocity`: Nyquist velocity (m/s)

**Field Attributes:**
- `reflectivity`: Radar reflectivity (dBZ)
- `velocity`: Doppler velocity (m/s)
- `spectrum_width`: Spectrum width (m/s)
- `differential_reflectivity`: Differential reflectivity (dB)
- `differential_phase`: Differential phase (degrees)
- `cross_correlation_ratio`: Cross-correlation coefficient (0-1)

**Coordinate Variables:**
- `time`: Time coordinate (datetime)
- `range`: Range coordinate (meters)
- `azimuth`: Azimuth coordinate (degrees)
- `elevation`: Elevation coordinate (degrees)

### Common ARM Radars

**X-SAPR:**
- X-band scanning precipitation radar
- Frequency: ~9.4 GHz
- Polarimetric: Yes

**KASACR:**
- Ka-band scanning cloud radar
- Frequency: ~35 GHz
- Polarimetric: No

**XSACR:**
- X-band scanning cloud radar
- Frequency: ~9.4 GHz
- Polarimetric: Yes

## MDV Format

### Description

SIGMET MDV (Meteorological Data Volume) format used by many radar systems.

### Reading MDV

```python
import pyart

# Read MDV file
radar = pyart.io.read_mdv('radar_file.mdv')

# Display radar information
print(radar)
```

### MDV Structure

**Header Information:**
- Radar location and configuration
- Volume scanning parameters
- Data encoding information

**Data Fields:**
- Reflectivity
- Velocity
- Spectrum width
- (Optional) Polarimetric variables

## SIGMET Format

### Description

SIGMET format used by various radar systems for data exchange.

### Reading SIGMET

```python
import pyart

# Read SIGMET file
radar = pyart.io.read_sigmet('radar_file.sigmet')

# Display radar information
print(radar)
```

### SIGMET Structure

**Header Information:**
- Radar identification
- Scan configuration
- Calibration parameters

**Data Fields:**
- Reflectivity
- Velocity
- Polarimetric variables (if available)

## NEXRAD Level II Format

### Description

NEXRAD Level II format used by NWS (National Weather Service) radars.

### Reading NEXRAD Level II

```python
import pyart

# Read NEXRAD Level II file
radar = pyart.io.read_nexrad_level2('radar_file')

# Display radar information
print(radar)
```

### NEXRAD Level II Structure

**Message Types:**
- Volume scan
- Elevation scan
- Azimuth scan

**Data Fields:**
- Reflectivity
- Velocity
- Spectrum width
- (Optional) Polarimetric variables

## ODIM_H5 Format

### Description

ODIM_H5 format used by ODIM (Operational Doppler Imager) radar systems.

### Reading ODIM_H5

```python
import pyart

# Read ODIM_H5 file
radar = pyart.io.read_odim_h5('radar_file.h5')

# Display radar information
print(radar)
```

### ODIM_H5 Structure

**HDF5 Structure:**
- Grouped data organization
- Metadata attributes
- Multiple sweeps in single file

## Field Conventions

### Reflectivity

**Units:** dBZ (decibels relative to Z)

**Typical Range:**
- -30 to 70 dBZ

**Missing Data:** Often represented as -9999 or masked values

### Velocity

**Units:** m/s (meters per second)

**Typical Range:**
- -30 to 30 m/s

**Nyquist Velocity:** Depends on PRF and wavelength

### Spectrum Width

**Units:** m/s

**Typical Range:**
- 0 to 10 m/s

**Indicator:** Higher values indicate non-meteorological targets

### Differential Reflectivity (ZDR)

**Units:** dB

**Typical Range:**
- -5 to 5 dB

**Hydrometeor Dependence:**
- Rain: 0.5 to 2 dB
- Snow: 0.2 to 0.8 dB
- Hail: 2 to 6 dB

### Differential Phase (PhiDP)

**Units:** Degrees

**Typical Range:**
- 0 to 360 degrees

**Cyclic:** Requires unfolding for processing

### Cross-Correlation Ratio (RHOHV)

**Units:** Dimensionless (0-1)

**Typical Range:**
- 0.5 to 1.0

**Quality Indicator:** Lower values indicate noise or mixed hydrometeors

## Coordinate Systems

### Polar Coordinates

**Azimuth:**
- Units: Degrees
- Range: 0 to 360
- Direction: Clockwise from north

**Elevation:**
- Units: Degrees
- Range: 0 to 90
- Direction: Above horizontal

**Range:**
- Units: Meters
- Range: 0 to max_range
- Resolution: Range gate spacing

### Cartesian Coordinates

**X, Y:**
- Units: Meters
- Origin: Radar location
- X: East-west direction
- Y: North-south direction

**Z:**
- Units: Meters
- Height above radar altitude

## Data Quality

### Missing Data

**Representation:**
- Masked arrays (numpy.ma)
- Special values (-9999, NaN)
- Quality flags

**Detection:**
```python
import numpy as np

# Check for missing data
if hasattr(field_data, 'mask'):
    missing_count = field_data.mask.sum()
else:
    missing_count = np.sum(np.isnan(field_data))
```

### Data Range Validation

**Reflectivity:**
```python
# Valid range
valid_range = (field_data >= -30) & (field_data <= 70)
```

**Velocity:**
```python
# Check against Nyquist
valid_velocity = np.abs(field_data) <= nyquist_velocity
```

## Best Practices

### 1. Always Check Radar Type

```python
# Check radar type
print(f"Radar type: {radar.radar_type}")

# Adjust processing based on radar type
if radar.radar_type == 'xsapr_saprc':
    # Polarimetric processing
    pass
```

### 2. Validate Data Ranges

```python
# Validate reflectivity
if 'reflectivity' in radar.fields:
    refl = radar.fields['reflectivity']['data']
    if np.any(refl < -30) or np.any(refl > 70):
        print("Warning: Reflectivity values outside expected range")
```

### 3. Handle Missing Data

```python
# Handle missing data
field_data = radar.fields['reflectivity']['data']

if hasattr(field_data, 'mask'):
    # Use masked array operations
    mean = np.mean(field_data)
else:
    # Regular array
    mean = np.mean(field_data)
```

### 4. Check Coordinate Systems

```python
# Check coordinate system
if hasattr(radar, 'azimuth'):
    print("Azimuth scan")
elif hasattr(radar, 'elevation'):
    print("Elevation scan")
```

### 5. Verify Units

```python
# Check field units
for field_name, field_dict in radar.fields.items():
    if 'metadata' in field_dict:
        if 'units' in field_dict['metadata']:
            print(f"{field_name}: {field_dict['metadata']['units']}")
```

## Format Conversion

### Converting Between Formats

Py-ART can convert between formats:

```python
import pyart

# Read in one format
radar = pyart.io.read_arm_netcdf('radar.nc')

# Write in another format
pyart.io.write_arm_netcdf(radar, 'output.nc')
```

### Format-Specific Processing

```python
import pyart

# Detect format
filename = 'radar_file.nc'

if filename.endswith('.nc'):
    radar = pyart.io.read_arm_netcdf(filename)
    # ARM-specific processing
elif filename.endswith('.mdv'):
    radar = pyart.io.read_mdv(filename)
    # MDV-specific processing
```

## Troubleshooting

### Common Issues

**"File not found" error**
- Check file path is correct
- Verify file format is supported

**"Field not found" error**
- Check available fields: `list(radar.fields.keys())`
- Verify field name is correct

**"Invalid data range" warning**
- Check radar type and calibration
- Verify data quality flags

**"Memory error"**
- Process radar data in smaller pieces
- Use appropriate data types

## References

- ARM Data Guide: https://www.arm.gov/data
- SIGMET Documentation: http://www.sigmetsystems.com/
- NEXRAD Documentation: https://www.nws.noaa.gov/nexrad/
- Py-ART Documentation: https://arm-doe.github.io/Py-ART/