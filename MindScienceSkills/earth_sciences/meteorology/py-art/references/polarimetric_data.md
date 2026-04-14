# Polarimetric Data Processing with Py-ART

This guide covers polarimetric data processing and analysis.

## Overview

Polarimetric radars measure both horizontal and vertical polarization, providing additional information about hydrometeors.

## Polarimetric Variables

### Differential Reflectivity (ZDR)

Access and analyze ZDR:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for ZDR field
if 'differential_reflectivity' in radar.fields:
    zdr = radar.fields['differential_reflectivity']['data']
    
    print(f"ZDR shape: {zdr.shape}")
    print(f"ZDR mean: {np.nanmean(zdr):.2f} dB")
    print(f"ZDR std: {np.nanstd(zdr):.2f} dB")
    
    # ZDR range
    zdr_min = np.nanmin(zdr)
    zdr_max = np.nanmax(zdr)
    print(f"ZDR range: {zdr_min:.2f} to {zdr_max:.2f} dB")
```

### Differential Phase (PhiDP)

Access and analyze PhiDP:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for PhiDP field
if 'differential_phase' in radar.fields:
    phidp = radar.fields['differential_phase']['data']
    
    print(f"PhiDP shape: {phidp.shape}")
    print(f"PhiDP mean: {np.nanmean(phidp):.2f} degrees")
    
    # PhiDP range
    phidp_min = np.nanmin(phidp)
    phidp_max = np.nanmax(phidp)
    print(f"PhiDP range: {phidp_min:.2f} to {phidp_max:.2f} degrees")
```

### Cross-Correlation Ratio (RHOHV)

Access and analyze RHOHV:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for RHOHV field
if 'cross_correlation_ratio' in radar.fields:
    rhohv = radar.fields['cross_correlation_ratio']['data']
    
    print(f"RHOHV shape: {rhohv.shape}")
    print(f"RHOHV mean: {np.nanmean(rhohv):.3f}")
    print(f"RHOHV std: {np.nanstd(rhohv):.3f}")
    
    # RHOHV range
    rhohv_min = np.nanmin(rhohv)
    rhohv_max = np.nanmax(rhohv)
    print(f"RHOHV range: {rhohv_min:.3f} to {rhohv_max:.3f}")
    
    # Quality check
    good_rhohv = rhohv > 0.9
    good_fraction = np.sum(good_rhohv) / rhohv.size
    print(f"Good RHOHV fraction: {good_fraction:.3f}")
```

## Hydrometeor Classification

### H-S Classification

Classify hydrometeors using H-S method:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for required fields
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

# Classify hydrometeors
radar_hydro = pyart.retrieve.hydroclass_hs(
    radar, 
    field_name='reflectivity',
    hydro_class='HS'
)

# Get classification field
if 'hydro_class' in radar_hydro.fields:
    hydro_class = radar_hydro.fields['hydro_class']['data']
    
    print(f"Hydrometeor classification shape: {hydro_class.shape}")
    print(f"Unique classes: {np.unique(hydro_class)}")
```

### Custom Classification

Create custom hydrometeor classification:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Get reflectivity and ZDR
refl = radar.fields['reflectivity']['data']

if 'differential_reflectivity' in radar.fields:
    zdr = radar.fields['differential_reflectivity']['data']
    
    # Simple classification
    hydro_class = np.zeros_like(refl)
    
    # Rain
    rain_mask = (refl > 5) & (refl < 50) & (zdr > 0.2) & (zdr < 2.0)
    hydro_class[rain_mask] = 1
    
    # Snow
    snow_mask = (refl > 5) & (refl < 40) & (zdr > 0.1) & (zdr < 0.8)
    hydro_class[snow_mask] = 2
    
    # Hail
    hail_mask = (refl > 45) & (zdr > 2.0)
    hydro_class[hail_mask] = 3
    
    # Add classification to radar
    radar.fields['hydro_class'] = {
        'data': hydro_class,
        'metadata': {
            'long_name': 'Hydrometeor classification',
            'units': '1'
        }
    }
    
    print(f"Custom classification completed")
    print(f"Rain pixels: {np.sum(hydro_class == 1)}")
    print(f"Snow pixels: {np.sum(hydro_class == 2)}")
    print(f"Hail pixels: {np.sum(hydro_class == 3)}")
```

## Melting Layer Detection

### Detect Melting Layer

Detect melting layer using reflectivity:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for reflectivity field
if 'reflectivity' not in radar.fields:
    print("Error: Reflectivity field not found")
    exit(1)

# Detect melting layer
radar_melt = pyart.retrieve.melting_layer(
    radar, 
    field_name='reflectivity',
    threshold=-5.0,
    min_beam_width=2.0
)

# Get melting layer field
if 'melting_layer' in radar_melt.fields:
    melting_layer = radar_melt.fields['melting_layer']['data']
    
    print(f"Melting layer detection completed")
    print(f"Melting layer pixels: {np.sum(melting_layer > 0)}")
```

### Bright Band Detection

Detect bright band in melting layer:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Detect melting layer
radar_melt = pyart.retrieve.melting_layer(
    radar, 
    field_name='reflectivity',
    threshold=-5.0,
    min_beam_width=2.0
)

# Get bright band
if 'bright_band' in radar_melt.fields:
    bright_band = radar_melt.fields['bright_band']['data']
    
    print(f"Bright band detection completed")
    print(f"Bright band pixels: {np.sum(bright_band > 0)}")
```

## Polarimetric Quality Control

### ZDR Quality Check

Check ZDR quality:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for ZDR field
if 'differential_reflectivity' in radar.fields:
    zdr = radar.fields['differential_reflectivity']['data']
    
    # Check ZDR range
    zdr_min = np.nanmin(zdr)
    zdr_max = np.nanmax(zdr)
    
    if zdr_min < -5.0 or zdr_max > 5.0:
        print("Warning: ZDR values outside expected range")
    
    # Check for missing data
    missing_count = np.sum(np.isnan(zdr))
    if missing_count > 0:
        print(f"Warning: {missing_count} missing ZDR values")
    
    # Check for outliers
    zdr_mean = np.nanmean(zdr)
    zdr_std = np.nanstd(zdr)
    outliers = np.abs(zdr - zdr_mean) > 3 * zdr_std
    outlier_count = np.sum(outliers)
    
    if outlier_count > 0:
        print(f"Warning: {outlier_count} ZDR outliers detected")
```

### RHOHV Quality Check

Check RHOHV quality:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for RHOHV field
if 'cross_correlation_ratio' in radar.fields:
    rhohv = radar.fields['cross_correlation_ratio']['data']
    
    # Check RHOHV range
    rhohv_min = np.nanmin(rhohv)
    rhohv_max = np.nanmax(rhohv)
    
    if rhohv_min < 0.0 or rhohv_max > 1.0:
        print("Warning: RHOHV values outside expected range")
    
    # Check for low quality
    low_quality = rhohv < 0.7
    low_quality_count = np.sum(low_quality)
    
    if low_quality_count > 0:
        print(f"Warning: {low_quality_count} low quality RHOHV values")
    
    # Check for missing data
    missing_count = np.sum(np.isnan(rhohv))
    if missing_count > 0:
        print(f"Warning: {missing_count} missing RHOHV values")
```

## Polarimetric Visualization

### ZDR Visualization

Visualize ZDR field:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Check for ZDR field
if 'differential_reflectivity' in radar.fields:
    # Create display
    display = pyart.graph.RadarDisplay(radar)
    
    # Plot ZDR
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    display.plot('differential_reflectivity', ax=ax, 
              vmin=-5, vmax=5, cmap='coolwarm')
    
    ax.set_title('Differential Reflectivity (ZDR)')
    plt.colorbar(ax.collections[0], ax=ax, label='ZDR (dB)')
    
    plt.tight_layout()
    plt.savefig('zdr_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### RHOHV Visualization

Visualize RHOHV field:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read.read_arm_netcdf('radar_file.nc')

# Check for RHOHV field
if 'cross_correlation_ratio' in radar.fields:
    # Create display
    display = pyart.graph.RadarDisplay(radar)
    
    # Plot RHOHV
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    display.plot('cross_correlation_ratio', ax=ax, 
              vmin=0.5, vmax=1.0, cmap='viridis')
    
    ax.set_title('Cross-Correlation Ratio (RHOHV)')
    plt.colorbar(ax.collections[0], ax=ax, label='RHOHV')
    
    plt.tight_layout()
    plt.savefig('rhohv_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
```

### Hydrometeor Classification Visualization

Visualize hydrometeor classification:

```python
import pyart
import matplotlib.pyplot as plt
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Classify hydrometeors
radar_hydro = pyart.retrieve.hydroclass_hs(
    radar, 
    field_name='reflectivity',
    hydro_class='HS'
)

# Get classification field
if 'hydro_class' in radar_hydro.fields:
    hydro_class = radar_hydro.fields['hydro_class']['data']
    
    # Create display
    display = pyart.graph.RadarDisplay(radar_hydro)
    
    # Plot classification
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    
    display.plot('hydro_class', ax=ax, 
              vmin=0, vmax=10, cmap='tab10')
    
    ax.set_title('Hydrometeor Classification')
    plt.colorbar(ax.collections[0], ax=ax, label='Class')
    
    plt.tight_layout()
    plt.savefig('hydro_class_plot.png', dpi=300, bbox_inches='tight')
    plt.show()
```

## Best Practices

### 1. Check Polarimetric Capability

```python
# Check if radar is polarimetric
if 'differential_reflectivity' in radar.fields:
    print("Radar is polarimetric")
else:
    print("Radar is single-polarization")
```

### 2. Validate Polarimetric Data

```python
# Validate ZDR range
if 'differential_reflectivity' in radar.fields:
    zdr = radar.fields['differential_reflectivity']['data']
    
    if np.any(zdr < -5.0) or np.any(zdr > 5.0):
        print("Warning: ZDR values outside expected range")
```

### 3. Handle Missing Data

```python
# Handle missing polarimetric data
if 'differential_reflectivity' in radar.fields:
    zdr = radar.fields['differential_reflectivity']['data']
    
    # Replace missing values
    zdr_clean = np.where(np.isnan(zdr), 0, zdr)
```

### 4. Use Appropriate Classification

```python
# Choose classification method based on radar type
if radar.radar_type == 'kasacri':
    # Use H-S classification
    radar_hydro = pyart.retrieve.hydroclass_hs(
        radar, field_name='reflectivity', hydro_class='HS')
elif radar.radar_type == 'xsapr_saprc':
    # Use polarimetric classification
    radar_hydro = pyart.retrieve.hydroclass_hs(
        radar, field_name='reflectivity', hydro_class='HS')
```

### 5. Document Processing Steps

```python
# Keep track of processing steps
processing_steps = []

# Apply corrections
if 'differential_phase' in radar.fields:
    radar = pyart.correct.correct_phidp(radar)
    processing_steps.append('PhiDP correction')

# Classify hydrometeors
radar_hydro = pyart.retrieve.hydroclass_hs(
    radar, field_name='reflectivity', hydro_class='HS')
processing_steps.append('Hydrometeor classification')

print(f"Applied processing: {', '.join(processing_steps)}")
```

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf
- Polarimetric Radar Meteorology: https://www.wmo.int/pages/prog/www/WMOCodes.html