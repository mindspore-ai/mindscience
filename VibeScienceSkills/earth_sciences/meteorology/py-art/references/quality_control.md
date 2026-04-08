# Quality Control with Py-ART

This guide covers quality control methods and techniques for radar data.

## Overview

Quality control is essential for radar data analysis to ensure data quality and remove artifacts.

## Gate Filtering

### Simple Gate Filtering

Filter gates based on value thresholds:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Filter reflectivity gates
radar_qc = pyart.correct.GateFilter(radar, 
                                     field_name='reflectivity',
                                     min_value=-30,
                                     max_value=70)

# Filter velocity gates
radar_qc = pyart.correct.GateFilter(radar_qc,
                                     field_name='velocity',
                                     min_value=-30,
                                     max_value=30)

# Filter spectrum width gates
radar_qc = pyart.correct.GateFilter(radar_qc,
                                     field_name='spectrum_width',
                                     min_value=0,
                                     max_value=10)
```

### Field-Specific Filtering

Different fields have different valid ranges:

```python
import pyart

# Reflectivity filtering
radar_qc = pyart.correct.GateFilter(radar, 
                                     field_name='reflectivity',
                                     min_value=-30,
                                     max_value=70)

# Velocity filtering
radar_qc = pyart.correct.GateFilter(radar_qc,
                                     field_name='velocity',
                                     min_value=-30,
                                     max_value=30)

# Spectrum width filtering
radar_qc = pyart.correct.GateFilter(radar_qc,
                                     field_name='spectrum_width',
                                     min_value=0,
                                     max_value=10)

# RHOHV filtering
radar_qc = pyart.correct.GateFilter(radar_qc,
                                     field_name='cross_correlation_ratio',
                                     min_value=0.5,
                                     max_value=1.0)
```

## Moment and Gate Filtering

### Combined Filtering

Apply moment-based and gate filtering:

```python
import pyart

# Apply moment and gate filtering
radar_qc = pyart.correct.moment_and_gatefilter.GateFilter(
    radar, field_name='reflectivity')
```

### Custom Moment Thresholds

Specify custom moment thresholds:

```python
import pyart

# Apply with custom thresholds
radar_qc = pyart.correct.moment_and_gatefilter.GateFilter(
    radar, 
    field_name='reflectivity',
    min_value=-30,
    max_value=70,
    min_z=-30,
    max_z=70,
    min_v=-30,
    max_v=30,
    min_w=0,
    max_w=10
)
```

## Despeckle

### Removing Speckles

Remove isolated speckles from radar data:

```python
import pyart

# Despeckle reflectivity field
radar_ds = pyart.correct.despeckle.despeckle_field(
    radar, field_name='reflectivity', 
    threshold=3, 
    gatefilter=False, 
    fsize=5)
```

### Despeckle Parameters

- `threshold`: Number of standard deviations for threshold
- `gatefilter`: Whether to apply gate filtering
- `fsize`: Filter size for speckle detection

### Multi-Field Despeckle

Apply despeckle to multiple fields:

```python
import pyart

# Despeckle reflectivity
radar_ds = pyart.correct.despeckle.despeckle_field(
    radar, field_name='reflectivity', threshold=3)

# Despeckle velocity
radar_ds = pyart.correct.despeckle.despeckle_field(
    radar_ds, field_name='velocity', threshold=2)
```

## Phase Processing

### Differential Phase Processing

Process differential phase for polarimetric radars:

```python
import pyart

# Process differential phase
radar_pp = pyart.correct.phase.ProcessPhase(
    radar, phase_field='differential_phase',
    temp_field='temperature')
```

### Phase Unfolding

Unfold phase for continuous values:

```python
import pyart

# Unfold phase
radar_uf = pyart.correct.phase.unfold_phase(
    radar, phase_field='differential_phase')
```

## Texture Calculation

### Field Texture

Calculate texture for radar fields:

```python
import pyart

# Calculate texture
radar_tex = pyart.correct.calculate_texture(
    radar, field_name='reflectivity', 
    num_points=4)
```

### Texture Parameters

- `num_points`: Number of points for texture calculation
- Texture helps identify meteorological vs non-meteorological echoes

## Signal-to-Noise Ratio

### SNR Calculation

Calculate signal-to-noise ratio:

```python
import pyart

# Calculate SNR
radar_snr = pyart.correct.calculate_snr(
    radar, field_name='reflectivity')
```

### SNR-Based Filtering

Filter based on SNR:

```python
import pyart

# Calculate and filter by SNR
radar_snr = pyart.correct.filter_by_snr(
    radar, field_name='reflectivity',
    min_snr=1.0,
    max_snr=100.0)
```

## Clutter Removal

### Ground Clutter Removal

Remove ground clutter echoes:

```python
import pyart

# Remove ground clutter
radar_gc = pyart.correct.remove_ground_clutter(
    radar, field_name='reflectivity')
```

### Sea Clutter Removal

Remove sea clutter echoes:

```python
import pyart

# Remove sea clutter
radar_sc = pyart.correct.remove_sea_clutter(
    radar, field_name='reflectivity')
```

### Anomalous Propagation Removal

Remove anomalous propagation echoes:

```python
import pyart

# Remove anomalous propagation
radar_ap = pyart.correct.remove_anomalous_propagation(
    radar, field_name='reflectivity')
```

## Quality Control Workflow

### Complete QC Pipeline

Apply complete quality control pipeline:

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Step 1: Gate filtering
radar_qc = pyart.correct.GateFilter(radar, 
                                     field_name='reflectivity',
                                     min_value=-30,
                                     max_value=70)

# Step 2: Moment and gate filtering
radar_qc = pyart.correct.moment_and_gatefilter.GateFilter(
    radar_qc, field_name='reflectivity')

# Step 3: Despeckle
radar_qc = pyart.correct.despeckle.despeckle_field(
    radar_qc, field_name='reflectivity', threshold=3)

# Step 4: Remove clutter
radar_qc = pyart.correct.remove_ground_clutter(
    radar_qc, field_name='reflectivity')

# Write quality-controlled data
pyart.io.write_arm_netcdf(radar_qc, 'radar_qc.nc')
```

## Best Practices

### 1. Apply QC Early

Apply quality control as first step in processing:

```python
# Good: Apply QC first
radar_qc = pyart.correct.GateFilter(radar, field_name='reflectivity')
radar_cal = pyart.correct.correct_zrdr(radar_qc)

# Bad: Calibrate before QC
radar_cal = pyart.correct.correct_zrdr(radar)
radar_qc = pyart.correct.GateFilter(radar_cal, field_name='reflectivity')
```

### 2. Use Appropriate Thresholds

Choose thresholds based on radar type and frequency:

```python
# X-band radar
if radar.frequency > 8e9 and radar.frequency < 12e9:
    min_refl, max_refl = -30, 70

# C-band radar
elif radar.frequency > 4e9 and radar.frequency < 8e9:
    min_refl, max_refl = -30, 60

# S-band radar
elif radar.frequency > 2e9 and radar.frequency < 4e9:
    min_refl, max_refl = -30, 50

radar_qc = pyart.correct.GateFilter(radar, 
                                     field_name='reflectivity',
                                     min_value=min_refl,
                                     max_value=max_refl)
```

### 3. Document QC Results

Keep track of QC results:

```python
# Apply QC and track results
original_count = np.sum(~radar.fields['reflectivity']['data'].mask)
radar_qc = pyart.correct.GateFilter(radar, field_name='reflectivity', min_value=-30, max_value=70)
qc_count = np.sum(~radar_qc.fields['reflectivity']['data'].mask)

print(f"Original gates: {original_count}")
print(f"After QC: {qc_count}")
print(f"Removed: {original_count - qc_count}")
```

### 4. Validate QC Results

Check QC results:

```python
# Validate reflectivity after QC
refl = radar_qc.fields['reflectivity']['data']

if np.any(refl < -30) or np.any(refl > 70):
    print("Warning: Reflectivity values outside expected range after QC")
```

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf