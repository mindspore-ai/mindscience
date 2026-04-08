# Retrieval Algorithms with Py-ART

This guide covers retrieval algorithms for precipitation and hydrometeors.

## Overview

Retrieval algorithms convert radar measurements into meteorological quantities like rain rate, snow rate, and hydrometeor type.

## Rain Rate Retrieval

### Z-R Relationship

Estimate rain rate using Z-R relationship:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

# Estimate rain rate using Z-R
radar_rain = pyart.retrieve.est_rain_rate_zr(radar)

# Write result
pyart.io.write_arm_netcdf(radar_rain, 'radar_rain_zr.nc')

print("Rain rate (Z-R) retrieved successfully")
```

### KDP-Based Rain Rate

Estimate rain rate using KDP:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for KDP field
if 'specific_differential_phase' not in radar.fields:
    print("Warning: KDP field not found, computing...")
    radar = pyart.retrieve.compute_kdp(radar)

# Estimate rain rate using KDP
radar_rain = pyart.retrieve.est_rain_rate_kdp(radar)

# Write result
pyart.io.write_arm_netcdf(radar_rain, 'radar_rain_kdp.nc')

print("Rain rate (KDP) retrieved successfully")
```

### Hybrid Z-R/KDP

Estimate rain rate using hybrid method:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

# Estimate rain rate using hybrid method
radar_rain = pyart.retrieve.est_rain_rate_hybrid(
    radar, field_name='reflectivity')

# Write result
pyart.io.write_arm_netcdf(radar_rain, 'radar_rain_hybrid.nc')

print("Rain rate (hybrid) retrieved successfully")
```

## Snow Rate Retrieval

### Z-S Relationship

Estimate snow rate using Z-S relationship:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

# Estimate snow rate using Z-S
radar_snow = pyart.retrieve.est_snow_rate_zs(radar)

# Write result
pyart.io.write_arm_netcdf(radar_snow, 'radar_snow_zs.nc')

print("Snow rate (Z-S) retrieved successfully")
```

### KDP-Based Snow Rate

Estimate snow rate using KDP:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for KDP field
if 'specific_differential_phase' not in radar.fields:
    print("Warning: KDP field not found, computing...")
    radar = pyart.retrieve.compute_kdp(radar)

# Estimate snow rate using KDP
radar_snow = pyart.retrieve.est_snow_rate_kdp(radar)

# Write result
pyart.io.write_arm_netcdf(radar_snow, 'radar_snow_kdp.nc')

print("Snow rate (KDP) retrieved successfully")
```

## Liquid-Ice Content

### Estimating Liquid-Ice Content

Estimate liquid-ice content:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

# Estimate liquid-ice content
radar_lic = pyart.retrieve.est_liquid_ice_content(
    radar, field_name='reflectivity')

# Write result
pyart.io.write_arm_netcdf(radar_lic, 'radar_lic.nc')

print("Liquid-ice content estimated successfully")
```

## Hail Detection

### Detecting Hail

Detect hail using radar data:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

# Detect hail
radar_hail = pyart.retrieve.detect_hail(radar)

# Write result
pyart.io.write_arm_netcdf(radar_hail, 'radar_hail.nc')

print("Hail detection completed successfully")
```

### Hail Probability

Calculate hail probability:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Detect hail
radar_hail = pyart.retrieve.detect_hail(radar)

# Get hail probability field
if 'hail_probability' in radar_hail.fields:
    hail_prob = radar_hail.fields['hail_probability']['data']
    
    # Calculate statistics
    hail_mean = np.nanmean(hail_prob)
    hail_max = np.nanmax(hail_prob)
    
    print(f"Hail probability mean: {hail_mean:.3f}")
    print(f"Hail probability max: {hail_max:.3f}")
```

## Best Practices

### 1. Check Required Fields

```python
# Always check for required fields
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

if 'differential_phase' in radar.fields:
    # Can use KDP-based methods
    pass
```

### 2. Validate Input Data

```python
# Validate reflectivity range
refl = radar.fields['reflectivity']['data']
if np.any(refl < -30) or np.any(refl > 70):
    print("Warning: Reflectivity values outside expected range")
```

### 3. Handle Missing Data

```python
# Handle missing data in retrievals
refl = radar.fields['reflectivity']['data']
refl_clean = np.where(np.isnan(refl), 0, refl)

# Estimate rain rate on clean data
radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
```

### 4. Use Appropriate Methods

```python
# Choose method based on radar type and data availability
if 'specific_differential_phase' in radar.fields:
    # Use KDP-based method
    radar_rain = pyart.retrieve.est_rain_rate_kdp(radar)
elif 'reflectivity' in radar.fields:
    # Use Z-R method
    radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
```

### 5. Document Retrieval Steps

```python
# Keep track of retrieval steps
retrieval_steps = []

# Apply retrievals
if 'reflectivity' in radar.fields:
    radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
    retrieval_steps.append('Rain rate (Z-R)')

if 'specific_differential_phase' in radar.fields:
    radar_kdp = pyart.retrieve.compute_kdp(radar)
    retrieval_steps.append('KDP calculation')

print(f"Applied retrievals: {', '.join(retrieval_steps)}")
```

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf
- Radar Meteorology: https://www.wmo.int/pages/prog/www/WMOCodes.html