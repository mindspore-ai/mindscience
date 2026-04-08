---
name: py-art
description: Comprehensive skill for working with Py-ART (Python ARM Radar Toolkit) to process and analyze meteorological radar data. Use when Claude needs to: (1) Read and process ARM radar data files, (2) Analyze radar reflectivity and Doppler velocity, (3) Apply radar quality control and filtering, (4) Perform radar calibration and corrections, (5) Extract radar moments and derived quantities, (6) Process polarimetric radar data, (7) Convert radar data to Cartesian coordinates, (8) Apply radar retrieval algorithms, (9) Work with ARM netCDF radar data formats, (10) Visualize radar data fields and products
---

# Py-ART

## Overview

Py-ART (Python ARM Radar Toolkit) is a Python package for processing and analyzing radar data from the Atmospheric Radiation Measurement (ARM) program. It provides tools for reading, quality controlling, calibrating, and analyzing radar reflectivity, Doppler velocity, and polarimetric data.

## Quick Start

**Reading ARM radar data:**

```python
import pyart

# Read ARM netCDF radar file
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Display radar information
print(radar)
print(f"Radar type: {radar.radar_type}")
print(f"Number of sweeps: {radar.nsweeps}")
print(f"Number of gates: {radar.ngates}")
```

**Accessing radar fields:**

```python
# Get reflectivity field
reflectivity = radar.fields['reflectivity']['data']

# Get Doppler velocity field
velocity = radar.fields['velocity']['data']

# Get spectral width field
width = radar.fields['spectrum_width']['data']
```

**Basic quality control:**

```python
# Apply basic quality control
radar_qc = pyart.correct.GateFilter(radar, 
                                     field_name='reflectivity',
                                     min_value=-30,
                                     max_value=70)

# Apply moment-based filtering
radar_qc = pyart.correct.moment_and_gagefilter.GateFilter(
    radar_qc, field_name='reflectivity')
```

## Workflow Decision Tree

1. **What type of operation do you need?**
   - **Reading radar data** → Follow "Reading Radar Data" workflow
   - **Quality control** → Follow "Quality Control" workflow
   - **Calibration and correction** → Follow "Calibration" workflow
   - **Data conversion** → Follow "Data Conversion" workflow
   - **Derived quantities** → Follow "Derived Quantities" workflow
   - **Polarimetric processing** → Follow "Polarimetric Data" workflow
   - **Coordinate conversion** → Follow "Coordinate Conversion" workflow
   - **Retrieval algorithms** → Follow "Retrieval Algorithms" workflow
   - **Visualization** → Follow "Visualization" workflow

## Reading Radar Data

### ARM netCDF Format

Read ARM radar data in netCDF format:

```python
import pyart

# Read ARM netCDF file
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Display radar information
print(radar)

# Check available fields
print("Available fields:", list(radar.fields.keys()))
```

### MDV Format

Read SIGMET MDV format:

```python
import pyart

# Read MDV file
radar = pyart.io.read_mdv('radar_file.mdv')

# Display radar information
print(radar)
```

### SIGMET Format

Read SIGMET format:

```python
import pyart

# Read SIGMET file
radar = pyart.io.read_sigmet('radar_file.sigmet')

# Display radar information
print(radar)
```

### NEXRAD Level II Format

Read NEXRAD Level II format:

```python
import pyart

# Read NEXRAD Level II file
radar = pyart.io.read_nexrad_level2('radar_file')

# Display radar information
print(radar)
```

### ODIM_H5 Format

Read ODIM_H5 format:

```python
import pyart

# Read ODIM_H5 file
radar = pyart.io.read_odim_h5('radar_file.h5')

# Display radar information
print(radar)
```

See `references/file_formats.md` for complete format reference.

## Quality Control

### Gate Filtering

Filter gates based on value thresholds:

```python
import pyart

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
```

### Moment and Gate Filtering

Apply moment-based quality control:

```python
import pyart

# Apply moment and gate filtering
radar_qc = pyart.correct.moment_and_gatefilter.GateFilter(
    radar, field_name='reflectivity')
```

### Despeckle

Remove isolated speckles:

```python
import pyart

# Despeckle reflectivity field
radar_ds = pyart.correct.despeckle.despeckle_field(
    radar, field_name='reflectivity', 
    threshold=3, 
    gatefilter=False, 
    fsize=5)
```

### Phase Processing

Process phase data for polarimetric radars:

```python
import pyart

# Process differential phase
radar_pp = pyart.correct.phase.ProcessPhase(
    radar, phase_field='differential_phase',
    temp_field='temperature')
```

### Texture Calculation

Calculate texture for field:

```python
import pyart

# Calculate texture
radar_tex = pyart.correct.calculate_texture(
    radar, field_name='reflectivity', 
    num_points=4)
```

See `references/quality_control.md` for detailed quality control methods.

## Calibration and Correction

### ZDR Calibration

Calibrate differential reflectivity:

```python
import pyart

# Calibrate ZDR
radar_zdr = pyart.correct.correct_zdr(radar)
```

### RHOHV Correction

Correct correlation coefficient:

```python
import pyart

# Correct RHOHV
radar_rhohh = pyart.correct.correct_rhohv(radar)
```

### PhiDP Correction

Correct differential phase:

```python
import pyart

# Correct PhiDP
radar_phidp = pyart.correct.correct_phidp(radar)
```

### KDP Calculation

Calculate specific differential phase:

```python
import pyart

# Calculate KDP
radar_kdp = pyart.retrieve.kdp_maes(radar)
```

### Attenuation Correction

Correct for attenuation:

```python
import pyart

# Correct reflectivity for attenuation
radar_atten = pyart.correct.correct_attenuation_hb(
    radar, field_name='reflectivity')
```

See `references/calibration.md` for detailed calibration methods.

## Data Conversion

### Converting to Cartesian

Convert polar radar data to Cartesian coordinates:

```python
import pyart

# Convert to Cartesian grid
grid = pyart.map.grid_constant_azimuth_range(
    radar, 
    range_1_km=0.0, 
    range_2_km=50.0, 
    grid_shape=(101, 101),
    fields=['reflectivity', 'velocity'],
    edge_factor=0.0)

# Display grid information
print(grid)
print(f"Grid shape: {grid.fields['reflectivity']['data'].shape}")
```

### Converting to Grid

Convert to grid with specified resolution:

```python
import pyart

# Convert to grid
grid = pyart.map.grid_from_radars(
    (radar,), 
    grid_shape=(500, 500),
    grid_limits=((0, 50000), (0, 50000)),
    grid_origin='lower_left',
    fields=['reflectivity'],
    weighting_function='Bilinear',
    roi_func='dist_from_radar_center',
    min_dist=0.0,
    max_dist=50000.0)
```

### Sector Conversion

Convert sector to Cartesian:

```python
import pyart

# Convert sector
grid = pyart.map.grid_from_sector(
    radar, 
    center_angle=90.0, 
    width_angle=45.0,
    range_1_km=0.0,
    range_2_km=50.0,
    grid_shape=(101, 101),
    fields=['reflectivity'])
```

See `references/coordinate_conversion.md` for detailed conversion methods.

## Derived Quantities

### Reflectivity at Horizontal Polarization

Calculate ZH:

```python
import pyart

# Calculate ZH
radar_zh = pyart.retrieve.compute_zh(radar)
```

### Differential Reflectivity

Calculate ZDR:

```python
import pyart

# Calculate ZDR
radar_zdr = pyart.retrieve.compute_zdr(radar)
```

### Specific Differential Phase

Calculate KDP:

```python
import pyart

# Calculate KDP
radar_kdp = pyart.retrieve.compute_kdp(radar)
```

### Rain Rate Estimation

Estimate rain rate from reflectivity:

```python
import pyart

# Estimate rain rate using Z-R relationship
radar_rain = pyart.retrieve.est_rain_rate_zr(radar)
```

### Rain Rate from KDP

Estimate rain rate from KDP:

```python
import pyart

# Estimate rain rate from KDP
radar_rain_kdp = pyart.retrieve.est_rain_rate_kdp(radar)
```

### Snow Rate Estimation

Estimate snow rate:

```python
import pyart

# Estimate snow rate
radar_snow = pyart.retrieve.est_snow_rate_zs(radar)
```

### Hail Detection

Detect hail:

```python
import pyart

# Detect hail
radar_hail = pyart.retrieve.detect_hail(radar)
```

See `references/derived_quantities.md` for complete derived quantities reference.

## Polarimetric Data

### Polarimetric Variables

Access polarimetric variables:

```python
import pyart

# Check for polarimetric data
if 'differential_reflectivity' in radar.fields:
    zdr = radar.fields['differential_reflectivity']['data']
    print(f"ZDR shape: {zdr.shape}")

if 'differential_phase' in radar.fields:
    phidp = radar.fields['differential_phase']['data']
    print(f"PhiDP shape: {phidp.shape}")

if 'cross_correlation_ratio' in radar.fields:
    rhohv = radar.fields['cross_correlation_ratio']['data']
    print(f"RHOHV shape: {rhohv.shape}")
```

### Hydrometeor Classification

Classify hydrometeors:

```python
import pyart

# Classify hydrometeors
radar_hydro = pyart.retrieve.hydroclass_hs(
    radar, 
    field_name='reflectivity',
    hydro_class='HS')
```

### Melting Layer Detection

Detect melting layer:

```python
import pyart

# Detect melting layer
radar_melt = pyart.retrieve.melting_layer(
    radar, 
    field_name='reflectivity',
    thresh=-5.0,
    min_beam_width=2.0)
```

See `references/polarimetric_data.md` for detailed polarimetric processing.

## Retrieval Algorithms

### Rain Rate Retrieval

Retrieve rain rate using various methods:

```python
import pyart

# Z-R relationship
radar_zr = pyart.retrieve.est_rain_rate_zr(radar)

# KDP-based
radar_kdp = pyart.retrieve.est_rain_rate_kdp(radar)

# Hybrid Z-R/KDP
radar_hybrid = pyart.retrieve.est_rain_rate_hybrid(
    radar, field_name='reflectivity')
```

### Snow Rate Retrieval

Retrieve snow rate:

```python
import pyart

# Z-S relationship
radar_zs = pyart.retrieve.est_snow_rate_zs(radar)

# KDP-based snow rate
radar_snow_kdp = pyart.retrieve.est_snow_rate_kdp(radar)
```

### Liquid-Ice Content

Estimate liquid-ice content:

```python
import pyart

# Estimate liquid-ice content
radar_lic = pyart.retrieve.est_liquid_ice_content(
    radar, field_name='reflectivity')
```

### Attenuation Correction

Correct for attenuation:

```python
import pyart

# Correct for attenuation
radar_atten = pyart.correct.correct_attenuation_hb(
    radar, field_name='reflectivity')
```

See `references/retrieval_algorithms.md` for complete retrieval reference.

## Visualization

### PPI Plot

Create Plan Position Indicator (PPI) plot:

```python
import pyart
import matplotlib.pyplot as plt

# Create PPI display
display = pyart.graph.RadarDisplay(radar)

# Plot reflectivity
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
display.plot('reflectivity', ax=ax, vmin=-30, vmax=70)
display.set_limits(xlim=(-50, 50), ylim=(-50, 50))
plt.show()
```

### CAPPI Plot

Create Constant Altitude PPI plot:

```python
import pyart
import matplotlib.pyplot as plt

# Create CAPPI display
display = pyart.graph.RadarDisplay(radar)

# Plot at specific altitude
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
display.plot_ppi('reflectivity', 2000, ax=ax, vmin=-30, vmax=70)
plt.show()
```

### RHI Plot

Create Range-Height Indicator (RHI) plot:

```python
import pyart
import matplotlib.pyplot as plt

# Create RHI display
display = pyart.graph.RadarDisplay(radar)

# Plot RHI
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
display.plot_rhi('reflectivity', 0, ax=ax, vmin=-30, vmax=70)
plt.show()
```

### Field Comparison

Compare multiple fields:

```python
import pyart
import matplotlib.pyplot as plt

# Create display
display = pyart.graph.RadarDisplay(radar)

# Plot multiple fields
fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(131)
display.plot('reflectivity', ax=ax1, vmin=-30, vmax=70)
ax1.set_title('Reflectivity')

ax2 = fig.add_subplot(132)
display.plot('velocity', ax=ax2, vmin=-30, vmax=30)
ax2.set_title('Velocity')

ax3 = fig.add_subplot(133)
display.plot('spectrum_width', ax=ax3, vmin=0, vmax=10)
ax3.set_title('Spectrum Width')

plt.tight_layout()
plt.show()
```

See `references/visualization.md` for detailed visualization techniques.

## Resources

### scripts/
Executable Python scripts for common Py-ART operations:

- **read_radar.py** - Read and display ARM radar data files
- **quality_control.py** - Apply quality control to radar data
- **calibrate_radar.py** - Calibrate and correct radar data
- **convert_coordinates.py** - Convert radar data to Cartesian coordinates
- **compute_derived.py** - Compute derived quantities from radar data
- **retrieve_rain.py** - Retrieve rain rate from radar data
- **polarimetric_analysis.py** - Analyze polarimetric radar data
- **detect_features.py** - Detect radar features (hail, melting layer, etc.)
- **visualize_radar.py** - Create visualizations of radar data
- **export_data.py** - Export radar data to various formats

### references/
Detailed documentation and reference materials:

- **file_formats.md** - Complete reference of supported radar data formats
- **quality_control.md** - Detailed quality control methods and techniques
- **calibration.md** - Radar calibration and correction procedures
- **coordinate_conversion.md** - Polar to Cartesian coordinate conversion methods
- **derived_quantities.md** - Complete reference of derived quantities
- **polarimetric_data.md** - Polarimetric data processing and analysis
- **retrieval_algorithms.md** - Retrieval algorithms for precipitation and hydrometeors
- **visualization.md** - Visualization techniques and plotting examples
- **api_reference.md** - Complete Py-ART API reference
- **common_use_cases.md** - Real-world use cases and patterns

### assets/
Example files and templates:

- **sample_radar.nc** - Sample ARM radar netCDF file for testing
- **config_template.yaml** - Configuration template for Py-ART operations
- **plot_template.py** - Template for creating custom radar plots
- **analysis_template.py** - Template for radar data analysis workflows

## Best Practices

1. **Always check radar type** - Different radar types have different capabilities
2. **Apply quality control** - Always apply QC before analysis
3. **Use appropriate corrections** - Apply calibration corrections as needed
4. **Validate data ranges** - Check that values are within expected ranges
5. **Handle missing data** - Use masked arrays for missing/invalid data
6. **Document processing steps** - Keep track of applied corrections
7. **Use appropriate visualization** - Choose correct plot type for data
8. **Consider memory usage** - Large radar files can consume significant memory

## Integration with Other Tools

### xarray Integration

Convert Py-ART radar objects to xarray:

```python
import pyart
import xarray as xr

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Convert to xarray Dataset
ds = pyart.io.to_xarray(radar)

# Use xarray operations
mean_reflectivity = ds['reflectivity'].mean()
```

### NumPy Integration

Access underlying NumPy arrays:

```python
import pyart
import numpy as np

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Access numpy array
reflectivity = radar.fields['reflectivity']['data']

# Use NumPy operations
mean = np.nanmean(reflectivity)
std = np.nanstd(reflectivity)
```

### Matplotlib Integration

Use Py-ART with matplotlib:

```python
import pyart
import matplotlib.pyplot as plt

# Read radar data
radar = pyart.io.read_arm_netcdf('radar_file.nc')

# Create display
display = pyart.graph.RadarDisplay(radar)

# Plot with matplotlib
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
display.plot('reflectivity', ax=ax)
plt.show()
```

## Troubleshooting

### Common Issues

**"File not found" error**
- Check file path is correct
- Verify file format is supported

**"Field not found" error**
- Check available fields: `list(radar.fields.keys())`
- Verify field name is correct

**"Memory error"**
- Process radar data in smaller pieces
- Use appropriate data types
- Consider coordinate conversion before analysis

**"Calibration failed" error**
- Check required fields are present
- Verify radar type supports calibration
- Check data quality

See `references/troubleshooting.md` for detailed troubleshooting guide.