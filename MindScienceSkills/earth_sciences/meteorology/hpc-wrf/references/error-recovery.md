# WRF Error Recovery

## Common Errors

### CFL Error

**Error Message**: `CFL ERROR: time step too large`

**Cause**: Time step exceeds stability criterion

**Solution**:
```
&domains
 time_step = 60,    ! Reduce from 72
/
```

Or use adaptive time stepping:
```
&domains
 use_adaptive_time_step = .true.,
/
```

### Segmentation Fault

**Error**: `Segmentation fault` in rsl.error files

**Causes and Solutions**:

1. **Memory limit**
```bash
ulimit -s unlimited
ulimit -v unlimited
```

2. **Domain too large for memory**
- Reduce domain size
- Increase number of MPI tasks

3. **Vertical levels issue**
```
&domains
 e_vert = 40,    ! Reduce vertical levels
/
```

### Missing met_em Files

**Error**: `Error opening met_em.d01.*`

**Causes**:
- ungrib failed
- metgrid failed
- Wrong file path

**Solution**:
```bash
# Check ungrib output
ls -la GFS:*

# Check metgrid output
ls -la met_em.d01.*

# Verify namelist.wps dates match data availability
```

### rsl Files Show Errors

**Check rsl files**:
```bash
grep -i error rsl.out.*
grep -i error rsl.error.*
```

**Common issues**:
- `ERROR: Error in metgrid`: Check met_em files
- `ERROR: Error reading wrfinput`: Check real.exe output
- `ERROR: Error writing wrfout`: Check disk space

### Slow Performance

| Symptom | Cause | Solution |
|---------|-------|----------|
| Slow I/O | Single-threaded I/O | Enable quilting |
| Poor scaling | Load imbalance | Adjust decomposition |
| Memory errors | Insufficient memory | Increase tasks or reduce domain |

### Restart Issues

**Error**: `Error reading restart file`

**Solution**:
```
&time_control
 restart = .true.,
 io_form_restart = 2,
/
```

Ensure restart file exists:
```bash
ls -la wrfrst_d01_*
```

## Debugging Tips

1. **Enable debug output**
```
&time_control
 debug_level = 100,
/
```

2. **Check model status**
```bash
# Look for SUCCESS messages
grep SUCCESS rsl.out.0000

# Check for errors
grep -i error rsl.error.*
```

3. **Verify input data**
```bash
# Check NetCDF files
ncdump -h wrfinput_d01
ncdump -h wrfbdy_d01
```
