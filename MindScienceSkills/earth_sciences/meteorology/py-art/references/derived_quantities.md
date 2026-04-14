# Derived Quantities with Py-ART

This guide covers derived quantities that can be computed from radar data.

## Overview

Derived quantities are computed from basic radar measurements (reflectivity, Doppler velocity, etc.) to provide meteorological information.

## Reflectivity at Horizontal Polarization

### Computing ZH

Compute horizontal polarization reflectivity:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Compute ZH
radar_zh = pyart.retrieve.compute_zh(radar)

# Write result
pyart.io.write_arm_netcdf(radar_zh, 'radar_zh.nc')

print("ZH computed successfully")
```

## Differential Reflectivity

### Computing ZDR

Compute differential reflectivity (ZDR):

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for polarimetric data
if 'differential_reflectivity' not in radar.fields:
    print("Warning: ZDR field not found")
    exit(1)

# Compute ZDR
radar_zdr = pyart.retrieve.compute_zdr(radar)

# Write result
pyart.io.write_arm_netcdf(radar_zdr, 'radar_zdr.nc')

print("ZDR computed successfully")
```

### ZDR Quality Check

Check ZDR quality:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Get ZDR field
zdr = radar.fields['differential_reflectivity']['data']

# Check ZDR range
zdr_min = np.nanmin(zdr)
zdr_max = np.nanmax(zdr)
zdr_mean = np.nanmean(zdr)

print(f"ZDR range: {zdr_min:.2f} to {zdr_max:.2f} dB")
print(f"ZDR mean: {zdr_mean:.2f} dB")

# Check for outliers
outliers = (np.abs(zdr - zdr_mean) > 3 * np.nanstd(zdr))
outlier_count = np.sum(outliers)

print(f"Outliers: {outlier_count} ({outlier_count/len(zdr)*100:.1f}%)")
```

## Specific Differential Phase

### Computing KDP

Compute specific differential phase (KDP):

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for PhiDP field
if 'differential_phase' not in radar.fields:
    print("Warning: PhiDP field not found")
    exit(1)

# Compute KDP
radar_kdp = pyart.retrieve.compute_kdp(radar)

# Write result
pyart.io.write_arm_netcdf(radar_kdp, 'radar_kdp.nc')

print("KDP computed successfully")
```

### KDP Quality Check

Check KDP quality:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_kdp.nc')

# Get KDP field
kdp = radar.fields['specific_differential_phase']['data']

# Check KDP range
kdp_min = np.nanmin(kdp)
kdp_max = np.nanmax(kdp)
kdp_mean = np.nanmean(kdp)

print(f"KDP range: {kdp_min:.2f} to {kdp_max:.2f} deg/km")
print(f"KDP mean: {kdp_mean:.2f} deg/km")

# Check for missing data
missing_count = np.sum(np.isnan(kdp))
print(f"Missing data: {missing_count} ({missing_count/len(kdp)*100:.1f}%)")
```

## Rain Rate Estimation

### Z-R Relationship

Estimate rain rate using Z-R relationship:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Warning: Reflectivity field not found")
    exit(1)

# Estimate rain rate using Z-R
radar_rain = pyart.retrieve.est_rain_rate_zr(radar)

# Write result
pyart.io.write_arm_netcdf(radar_rain, 'radar_rain_zr.nc')

print("Rain rate (Z-R) estimated successfully")
```

### KDP-Based Rain Rate

Estimate rain rate using KDP:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for KDP field
if 'specific_differential_phase' not in radar.fields:
    print("Warning: KDP field not found")
    exit(1)

# Estimate rain rate using KDP
radar_rain = pyart.retrieve.est_rain_rate_kdp(radar)

# Write result
pyart.io.write_arm_netcdf(radar_rain, 'radar_rain_kdp.nc')

print("Rain rate (KDP) estimated successfully")
```

### Hybrid Z-R/KDP

Estimate rain rate using hybrid method:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for required fields
if 'reflectivity' not in radar.fields:
    print("Warning: Reflectivity field not found")
    exit:1)

# Estimate rain rate using hybrid method
radar_rain = pyart.retrieve.est_rain_rate_hybrid(
    radar, field_name='reflectivity')

# Write result
pyart.io.write_arm_netcdf(radar_rain, 'radar_rain_hybrid.nc')

print("Rain rate (hybrid) estimated successfully")
```

## Snow Rate Estimation

### Z-S Relationship

Estimate snow rate using Z-S relationship:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Warning: Reflectivity field not found")
    exit(1)

# Estimate snow rate using Z-S
radar_snow = pyart.retrieve.est_snow_rate_zs(radar)

# Write result
pyart.io.write_arm_netcdf(radar_snow, 'radar_snow_zs.nc')

print("Snow rate (Z-S) estimated successfully")
```

### KDP-Based Snow Rate

Estimate snow rate using KDP:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for KDP field
if 'specific_differential_phase' not in radar.fields:
    print("Warning: KDP field not found")
    exit(1)

# Estimate snow rate using KDP
radar_snow = pyart.retrieve.est_snow_rate_kdp(radar)

# Write result
pyart.io.write_arm_netcdf(radar_snow, 'radar_snow_kdp.nc')

print("Snow rate (KDP) estimated successfully")
```

## Hail Detection

### Detecting Hail

Detect hail using radar data:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for required fields
if 'reflectivity' not in radar.fields:
    print("Warning: Reflectivity field not found")
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

# Get hail field
hail_field = radar_hail.fields['hail_probability']['data']

# Calculate statistics
hail_mean = np.nanmean(hail_field)
hail_max = np.nanmax(hail_field)

print(f"Hail probability mean: {hail_mean:.3f}")
print(f"Hail probability max: {hail_max:.3f}")
```

## Liquid-Ice Content

### Estimating Liquid-Ice Content

Estimate liquid-ice content:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for required fields
if 'reflectivity' not in radar.fields:
    print("Warning: Reflectivity field not found")
    exit(1)

# Estimate liquid-ice content
radar_lic = pyart.retrieve.est_liquid_ice_content(
    radar, field_name='reflectivity')

# Write result
pyart.io.write_arm_netcdf(radar_lic, 'radar_lic.nc')

print("Liquid-ice content estimated successfully")
```

## Best Practices

### 1. Check Required Fields

```python
# Always check for required fields
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

if 'differential_reflectivity' not in radar.fields:
    print("Warning: ZDR field not found, skipping ZDR computations")
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
# Handle missing data in computations
import numpy as np

refl = radar.fields['reflectivity']['data']
refl_clean = np.where(np.isnan(refl), 0, refl)

# Compute on clean data
rain_rate = pyart.retrieve.est_rain_rate_zr(radar)
```

### 4. Use Appropriate Relationships

```python
# Choose appropriate Z-R relationship
if radar.frequency > 5e9:  # C-band or higher
    # Use standard Z-R
    radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
else:  # S-band
    # Use adjusted Z-R
    radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
```

### 5. Document Computations

```python
# Keep track of applied computations
computations = []

# Compute ZH
radar = pyart.retrieve.compute_zh(radar)
computations.append('ZH')

# Compute ZDR
if 'differential_reflectivity' in radar.fields:
    radar = pyart.retrieve.compute_zdr(radar)
    computations.append('ZDR')

print(f"Applied computations: {', '.join(computations)}")
```

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf
- Radar Meteorology: https://www.wmo.int/pages/prog/www/WMOCodes.html