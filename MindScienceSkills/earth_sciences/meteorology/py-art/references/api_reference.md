# Py-ART API Reference

This document provides a complete reference of Py-ART API.

## I/O Functions

### Reading Radar Data

**read_arm_netcdf**
```python
radar = pyart.io.read_arm_netcdf('filename.nc')
```

**read_mdv**
```python
radar = pyart.io.read_mdv('filename.mdv')
```

**read_sigmet**
```python
radar = pyart.io.read_sigmet('filename.sigmet')
```

**read_nexrad_level2**
```python
radar = pyart.io.read_nexrad_level2('filename')
```

**read_odim_h5**
```python
radar = pyart.io.read_odim_h5('filename.h5')
```

### Writing Radar Data

**write_arm_netcdf**
```python
pyart.io.write_arm_netcdf(radar, 'output.nc')
```

**to_xarray**
```python
ds = pyart.io.to_xarray(radar)
```

## Quality Control Functions

### Gate Filtering

**GateFilter**
```python
radar_qc = pyart.correct.GateFilter(radar, 
                                     field_name='reflectivity',
                                     min_value=-30,
                                     max_value=70)
```

### Despeckle

**despeckle_field**
```python
radar_ds = pyart.correct.despeckle.despeckle_field(
    radar, field_name='reflectivity', 
    threshold=3, 
    gatefilter=False, 
    fsize=5)
```

### Moment and Gate Filtering

**GateFilter**
```python
radar_qc = pyart.correct.moment_and_gatefilter.GateFilter(
    radar, field_name='reflectivity')
```

## Calibration Functions

### ZDR Calibration

**correct_zdr**
```python
radar_cal = pyart.correct.correct_zdr(radar)
```

### RHOHV Correction

**correct_rhohv**
```python
radar_cal = pyart.correct.correct_rhohv(radar)
```

### PhiDP Correction

**correct_phidp**
```python
radar_cal = pyart.correct.correct_phidp(radar)
```

### Attenuation Correction

**correct_attenuation_hb**
```python
radar_cal = pyart.correct.correct_attenuation_hb(
    radar, field_name='reflectivity')
```

**correct_attenuation_sc**
```python
radar_cal = pyart.correct.correct_attenuation_sc(
    radar, field_name='reflectivity')
```

## Retrieval Functions

### Reflectivity

**compute_zh**
```python
radar_zh = pyart.retrieve.compute_zh(radar)
```

**compute_zdr**
```python
radar_zdr = pyart.retrieve.compute_zdr(radar)
```

### Differential Phase

**compute_kdp**
```python
radar_kdp = pyart.retrieve.compute_kdp(radar)
```

### Precipitation

**est_rain_rate_zr**
```python
radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
```

**est_rain_rate_kdp**
```python
radar_rain = pyart.retrieve.est_rain_rate_kdp(radar)
```

**est_rain_rate_hybrid**
```python
radar_rain = pyart.retrieve.est_rain_rate_hybrid(
    radar, field_name='reflectivity')
```

### Snow Rate

**est_snow_rate_zs**
```python
radar_snow = pyart.retrieve.est_snow_rate_zs(radar)
```

**est_snow_rate_kdp**
```python
radar_snow = pyart.retrieve.est_snow_rate_kdp(radar)
```

### Other Retrievals

**est_liquid_ice_content**
```python
radar_lic = pyart.retrieve.est_liquid_ice_content(
    radar, field_name='reflectivity')
```

**detect_hail**
```python
radar_hail = pyart.retrieve.detect_hail(radar)
```

## Hydrometeor Classification

**hydroclass_hs**
```python
radar_hydro = pyart.retrieve.hydroclass_hs(
    radar, 
    field_name='reflectivity',
    hydro_class='HS'
)
```

## Feature Detection

**melting_layer**
```python
radar_melt = pyart.retrieve.melting_layer(
    radar, 
    field_name='reflectivity',
    threshold=-5.0,
    min_beam_width=2.0
)
```

## Mapping Functions

### Grid Conversion

**grid_constant_azimuth_range**
```python
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity', 'velocity'],
    edge_factor=0.0
)
```

**grid_from_radars**
```python
grid = pyart.map.grid_from_radars(
    (radar,), 
    grid_shape=(500, 500),
    grid_limits=((0, 50000), (0, 50000)),
    grid_origin='lower_left',
    fields=['reflectivity'],
    weighting_function='Bilinear',
    roi_func='dist_from_radar_center',
    min_dist=0.0,
    max_dist=50000.0
)
```

**grid_from_sector**
```python
grid = pyart.map.grid_from_sector(
    radar, 
    center_angle=90.0, 
    width_angle=45.0,
    range_1_km=0.0,
    range_2_km=50.0,
    grid_shape=(101, 101),
    fields=['reflectivity']
)
```

## Graph/Visualization

### RadarDisplay

**RadarDisplay**
```python
display = pyart.graph.RadarDisplay(radar)
```

**Plot Methods**

```python
# PPI plot
display.plot('reflectivity', ax=ax, vmin=-30, vmax=70)

# RHI plot
display.plot_rhi('reflectivity', sweep_num, ax=ax, vmin=-30, vmax=70)

# CAPPI plot
display.plot_ppi('reflectivity', altitude, ax=ax, vmin=-30, vmax=70)
```

## Common Patterns

### Reading and Processing

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Apply quality control
radar_qc = pyart.correct.GateFilter(radar, 
                                     field_name='reflectivity',
                                     min_value=-30,
                                     max_value=70)

# Compute derived quantities
radar_zh = pyart.retrieve.compute_zh(radar_qc)
radar_zdr = pyart.retrieve.compute_zdr(radar_zh)

# Write output
pyart.io.write_arm_netcdf(radar_zdr, 'output.nc')
```

### Coordinate Conversion

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert to Cartesian grid
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity'],
    edge_factor=0.0
)

# Write grid
pyart.io.write_arm_netcdf(grid, 'grid.nc')
```

### Retrieval

```python
import pyart

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Estimate rain rate
radar_rain = pyart.retrieve.est_rain_rate_zr(radar)

# Write rain rate
pyart.io.write_arm_netcdf(radar_rain, 'rain_rate.nc')
```

## Notes

- Always check for required fields before operations
- Apply quality control before analysis
- Use appropriate calibration for radar type
- Handle missing data with masked arrays
- Document processing steps for reproducibility

## References

- Py-ART Documentation: https://arm-doe.github.io/Py-ART/
- ARM Radar Handbook: https://www.arm.gov/publications/handbooks/radar_handbook.pdf