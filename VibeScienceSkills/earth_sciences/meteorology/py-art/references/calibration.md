# Calibration and Correction with Py-ART

This guide covers radar calibration and correction procedures.

## Overview

Calibration ensures radar measurements are accurate and consistent. Py-ART provides tools for common calibration procedures.

## ZDR Calibration

### Differential Reflectivity Calibration

Calibrate differential reflectivity (ZDR):

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for ZDR field
if 'differential_reflectivity' not in radar.fields:
    print("Warning: ZDR field not found")
    exit(1)

# Calibrate ZDR
radar_cal = pyart.correct.correct_zdr(radar)

# Write calibrated data
pyart.io.write_arm_netcdf(radar_cal, 'radar_calibrated.nc')

print("ZDR calibration completed")
```

### ZDR Calibration Parameters

Common ZDR calibration parameters:
- `zdr_offset`: ZDR offset (dB)
- `zdr_slope`: ZDR slope (dimensionless)
- `reference_value`: Reference ZDR value

## RHOHV Correction

### Correlation Coefficient Correction

Correct cross-correlation coefficient (RHOHV):

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for RHOHV field
if 'cross_correlation_ratio' not in radar.fields:
    print("Warning: RHOHV field not found")
    exit(1)

# Correct RHOHV
radar_cal = pyart.correct.correct_rhohv(radar)

# Write corrected data
pyart.io.write_arm_netcdf(radar_cal, 'radar_corrected.nc')

print("RHOHV correction completed")
```

### RHOHV Quality Check

Check RHOHV quality:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Get RHOHV field
rhohv = radar.fields['cross_correlation_ratio']['data']

# Check RHOHV values
good_rhohv = rhohv > 0.9
good_fraction = np.sum(good_rhohv) / rhohv.size

print(f"Good RHOHV fraction: {good_fraction:.3f}")

if good_fraction < 0.8:
    print("Warning: Low RHOHV quality")
```

## PhiDP Correction

### Differential Phase Correction

Correct differential phase (PhiDP):

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for PhiDP field
if 'differential_phase' not in radar.fields:
    print("Warning: PhiDP field not found")
    exit(1)

# Correct PhiDP
radar_cal = pyart.correct.correct_phidp(radar)

# Write corrected data
pyart.io.write_arm_netcdf(radar_cal, 'radar_corrected.nc')

print("PhiDP correction completed")
```

### Phase Unfolding

Unfold differential phase:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Unfold phase
radar_uf = pyart.correct.phase.unfold_phase(
    radar, phase_field='differential_phase')

# Write unfolded data
pyart.io.write_arm_netcdf(radar_uf, 'radar_unfolded.nc')

print("Phase unfolding completed")
```

## KDP Calculation

### Specific Differential Phase

Calculate specific differential phase (KDP):

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for PhiDP field
if 'differential_phase' not in radar.fields:
    print("Warning: PhiDP field not found")
    exit(1)

# Calculate KDP
radar_kdp = pyart.retrieve.compute_kdp(radar)

# Write KDP field
pyart.io.write_arm_netcdf(radar_kdp, 'radar_kdp.nc')

print("KDP calculation completed")
```

### KDP Quality Check

Check KDP quality:

```python
import pyart
import numpy as np

# Read radar data with KDP
radar = pyart.io.read_arm_netcdf('radar_kdp.nc')

# Get KDP field
kdp = radar.fields['specific_differential_phase']['data']

# Check KDP values
good_kdp = np.abs(kdp) < 5.0  # Typical range: -5 to 5 deg/km
good_fraction = np.sum(good_kdp) / kdp.size

print(f"Good KDP fraction: {good_fraction:.3f}")

if good_fraction < 0.8:
    print("Warning: Low KDP quality")
```

## Attenuation Correction

### Reflectivity Attenuation Correction

Correct reflectivity for attenuation:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Warning: Reflectivity field not found")
    exit(1)

# Correct for attenuation
radar_atten = pyart.correct.correct_attenuation_hb(
    radar, field_name='reflectivity')

# Write corrected data
pyart.io.write_arm_netcdf(radar_atten, 'radar_corrected.nc')

print("Attenuation correction completed")
```

### Self-Consistent Attenuation Correction

Apply self-consistent attenuation correction:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Apply self-consistent correction
radar_sc = pyart.correct.correct_attenuation_sc(
    radar, field_name='reflectivity')

# Write corrected data
pyart.io.write_arm_netcdf(radar_sc, 'radar_corrected.nc')

print("Self-consistent attenuation correction completed")
```

## Calibration Workflow

### Complete Calibration Pipeline

Apply complete calibration pipeline:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Step 1: ZDR calibration
if 'differential_reflectivity' in radar.fields:
    radar = pyart.correct.correct_zdr(radar)
    print("Step 1: ZDR calibration completed")

# Step 2: RHOHV correction
if 'cross_correlation_ratio' in radar.fields:
    radar = pyart.correct.correct_rhohv(radar)
    print("Step 2: RHOHV correction completed")

# Step 3: PhiDP correction
if 'differential_phase' in radar.fields:
    radar = pyart.correct.correct_phidp(radar)
    print("Step 3: PhiDP correction completed")

# Step 4: Attenuation correction
if 'reflectivity' in radar.fields:
    radar = pyart.correct.correct_attenuation_hb(
        radar, field_name='reflectivity')
    print("Step 4: Attenuation correction completed")

# Write calibrated data
pyart.io.write_arm_netcdf(radar, 'radar_calibrated.nc')

print("Calibration pipeline completed")
```

## Best Practices

### 1. Check Required Fields

```python
# Always check for required fields
if 'differential_reflectivity' not in radar.fields:
    print("Warning: ZDR field not found, skipping ZDR calibration")
else:
    radar = pyart.correct.correct_zdr(radar)
```

### 2. Validate Calibration Results

```python
# Validate calibration results
radar_cal = pyart.correct.correct_zdr(radar)

# Check ZDR values
zdr = radar_cal.fields['differential_reflectivity']['data']
if np.any(np.abs(zdr) > 10):
    print("Warning: ZDR values outside expected range")
```

### 3. Document Calibration Steps

```python
# Keep track of calibration steps
calibration_steps = []

# Apply calibrations
if 'differential_reflectivity' in radar.fields:
    radar = pyart.correct.correct_zdr(radar)
    calibration_steps.append('ZDR calibration')

if 'cross_correlation_ratio' in radar.fields:
    radar = pyart.correct.correct_rhohv(radar)
    calibration_steps.append('RHOHV correction')

print(f"Applied calibrations: {', '.join(calibration_steps)}")
```

### 4. Use Appropriate Correction Method

```python
# Choose correction method based on radar type
if radar.radar_type == 'xsapr_saprc':
    # Use self-consistent attenuation correction
    radar = pyart.correct.correct_attenuation_sc(
        radar, field_name='reflectivity')
else:
    # Use HB attenuation correction
    radar = pyart.correct.correct_attenuation_hb(
        radar, field_name='reflectivity')
```

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf
- Radar Calibration Guide: https://www.wmo.int/pages/prog/www/WMOCodes.html