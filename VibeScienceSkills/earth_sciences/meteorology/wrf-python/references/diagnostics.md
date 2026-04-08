# Diagnostics in WRF-Python

Complete guide to available diagnostic calculations.

## Available Diagnostics

### Thermodynamic Diagnostics

**CAPE (Convective Available Potential Energy):**
```python
from wrf import getcape

# Calculate CAPE (J/kg)
cape = getcape(ncfile, True)  # True = use unpivoted data
print(f"CAPE shape: {cape.shape}")  # (time, level, lat, lon)
```

**Potential Temperature:**
```python
from wrf import getvar

# Get potential temperature (K)
t_p = getvar(ncfile, 'T', True)
```

**Equivalent Potential Vorticity:**
```python
from wrf import getethet

# Get equivalent potential vorticity (1/s)
et_p = getethet(ncfile, True)
```

### Dynamic Diagnostics

**Storm Relative Helicity:**
```python
from wrf import srhel

# Calculate storm relative helicity
srh = srhel(ncfile, True)
print(f"Storm relative helicity shape: {srh.shape}")
```

**Brunt-Väisälä Frequency:**
```python
from wrf import getbrunt

# Calculate Brunt-Väisälä frequency (1/s)
brunt = getbrunt(ncfile, True)
```

**Bulk Richardson Number:**
```python
from wrf import getrichardson

# Calculate bulk Richardson number
ri = getrichardson(ncfile, True)
```

### Moisture Diagnostics

**Moisture Flux Convergence:**
```python
from wrf import getmflcon

# Get moisture flux convergence
mflcon = getmflcon(ncfile, True)
```

**Precipitable Water:**
```python
from wrf import getpw

# Get precipitable water (kg/m²)
pw = getpw(ncfile, True)
```

**Integrated Vapor Transport:**
```python
from wrf import getit

# Get integrated vapor transport
it = getit(ncfile, True)
```

### Wind Diagnostics

**Wind Components:**
```python
from wrf import getvar

# Get U, V, W wind components (m/s)
u = getvar(ncfile, 'U', True)
v = getvar(ncfile, 'V', True)
w = getvar(ncfile, 'W', True)
```

**Wind Speed:**
```python
from wrf import uvmet

# Calculate wind speed (m/s)
spd = uvmet(ncfile, 'uvmet', True)
print(f"Wind speed shape: {spd.shape}")
```

**10-Meter Wind Speed:**
```python
from wrf import uvmet10

# Calculate 10-meter wind speed (m/s)
spd10 = uvmet10(ncfile, 'uvmet10', True)
```

**Wind Direction:**
```python
from wrf import uvmetdir

# Calculate wind direction (degrees)
dir = uvmetdir(ncfile, 'uvmetdir', True)
```

### Thermodynamic and Dynamic Combined

**Equivalent Potential Temperature:**
```python
from wrf import getethe

# Get equivalent potential temperature (K)
t_e = getethe(ncfile, True)
```

**Bulk Shear:**
```python
from wrf import getbshear

# Calculate bulk shear (1/s)
bshear = getbshear(ncfile, True)
```

### Stability Diagnostics

**Divergence:**
```python
from wrf import getdiv

# Calculate divergence (1/s)
div = getdiv(ncfile, True)
```

**Vorticity:**
```python
from wrf import getvort

# Calculate vorticity (1/s)
vort = getvort(ncfile, True)
```

## Multiple Diagnostics

**Calculate several diagnostics:**
```python
from wrf import getcape, srhel, uvmet

# Calculate multiple diagnostics
cape = getcape(ncfile, True)
srh = srhel(ncfile, True)
spd = uvmet(ncfile, 'uvmet', True)

print(f"CAPE range: [{cape.min():.1f}, {cape.max():.1f}] J/kg")
print(f"SRH range: [{srh.min():.3f}, {srh.max():.3f}]")
print(f"Speed range: [{spd.min():.2f}, {spd.max():.2f}] m/s")
```

## Base State Diagnostics

**Get base state (unperturbed):**
```python
from wrf import getgetvar_base

# Get base state
t_base = getgetvar_base(ncfile, 'T', True)
```

**Perturbation diagnostics:**
```python
from wrf import getgetvar_pert

# Get perturbation
t_pert = getgetvar_pert(ncfile, 'T', True)
```

## Common Applications

### Severe Weather Analysis

```python
# Identify high CAPE regions
cape = getcape(ncfile, True)

# High CAPE threshold (J/kg)
high_cape = cape > 2000

print(f"High CAPE grid points: {np.sum(high_cape)}")
print(f"Max CAPE: {cape.max():.1f} J/kg")
```

### Storm Identification

```python
# Identify storm regions
srh = srhel(ncfile, True)

# Storm threshold
storm = srh > 1.0

print(f"Storm grid points: {np.sum(storm)}")
print(f"Max SRH: {srh.max():.3f}")
```

### Wind Analysis

```python
# Analyze wind field
spd = uvmet(ncfile, 'uvmet', True)
dir = uvmetdir(ncfile, 'uvmetdir', True)

# Find strong winds
strong_wind = spd > 20.0  # m/s

print(f"Strong wind grid points: {np.sum(strong_wind)}")
print(f"Max wind speed: {spd.max():.2f} m/s")
```

### Stability Assessment

```python
# Assess atmospheric stability
ri = getrichardson(ncfile, True)

# Unstable regions
unstable = ri > 2.0

print(f"Unstable grid points: {np.sum(unstable)}")
print(f"Max Richardson number: {ri.max():.3f}")
```

## Numerical Considerations

### Memory Management

**Large files:**
```python
# Use xarray for large netCDF files
import xarray as xr

ds = xr.open_dataset('wrfout_d01.nc')
t = ds['T']  # Lazy loading
```

**Chunk processing:**
```python
# Process in time chunks
for time_idx in range(0, len(time), 10):
    cape = getcape(ncfile, True, timeidx=time_idx)
    # Process chunk
```

### Coordinate Handling

**Grid staggering:**
```python
# WRF uses staggered grids
# Be aware of mass points vs. velocity points
# See references/coordinate_systems.md for details
```

**Domain boundaries:**
```python
# Check boundary conditions
# Handle cyclic boundaries
# Check for missing values
```

## Troubleshooting

### Variable Not Found

**Issue:** Variable not in netCDF file

**Solutions:**
- Check variable name (case-sensitive)
- Verify WRF model configuration
- Check netCDF file structure
- List all variables in file

### Incorrect Array Shape

**Issue:** Unexpected array dimensions

**Solutions:**
- Check WRF model configuration
- Verify coordinate dimensions
- Check time/level dimensions
- Review WRF documentation

### Missing Values

**Issue:** Array contains missing values

**Solutions:**
- Check WRF model output
- Verify domain boundaries
- Handle missing values appropriately
- Use masking if needed

### Memory Errors

**Issue:** Out of memory

**Solutions:**
- Use xarray for large files
- Process in smaller chunks
- Close files after reading
- Reduce spatial resolution

## Advanced Topics

### Custom Diagnostics

```python
# Define custom diagnostic calculations
# See WRF-Python documentation for details
```

### Time Series Diagnostics

```python
# Calculate diagnostics over time
# Analyze temporal evolution
```

### Spatial Averaging

```python
# Average diagnostics over space
# Create time series of spatial averages
```

## Resources

- WRF-Python diagnostics: https://wrf-python.read.readocs.io/en/latest/diagnostics.html
- WRF-ARW model: https://www2.mmm.ucar.edu/wrf/users/
- WRF-Python GitHub: https://github.com/NCAR/wrf-python
