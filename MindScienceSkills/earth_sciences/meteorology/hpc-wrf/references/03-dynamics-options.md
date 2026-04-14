# WRF Dynamics Options

## ARW Solver

The Advanced Research WRF (ARW) solver uses:
- Fully compressible, non-hydrostatic equations
- Arakawa C-grid staggering
- Terrain-following hydrostatic pressure coordinate

## Key namelist.input Parameters

### Time Step
```
&domains
 time_step = 72,           ! 6 * dx (km) for stability
 time_step_fract_num = 0,
 time_step_fract_den = 1,
/
```

**Stability criterion**: time_step <= 6 * dx (km)
- dx = 12 km: time_step <= 72 s
- dx = 4 km: time_step <= 24 s
- dx = 1 km: time_step <= 6 s

### Vertical Levels
```
&domains
 e_vert = 45,              ! Number of vertical levels
 p_top_requested = 5000,   ! Top pressure (Pa)
/
```

Recommended vertical levels:
- 30-40: General applications
- 40-50: PBL studies
- 50-60: High-resolution simulations

### Diffusion Options
```
&dynamics
 diff_opt = 1,             ! 1=simple, 2=full
 km_opt = 4,               ! 4=2nd order horizontal
 diff_6th_opt = 0,         ! 6th order diffusion
 diff_6th_factor = 0.12,
/
```

### Damping Options
```
&dynamics
 w_damping = 0,            ! Vertical velocity damping
 damp_opt = 0,             ! 0=none, 1=Rayleigh, 2=diffusive
 zdamp = 5000.0,           ! Damping depth (m)
 dampcoef = 0.2,           ! Damping coefficient
/
```

## Non-hydrostatic vs Hydrostatic

```
&dynamics
 non_hydrostatic = .true.,  ! .true. for non-hydrostatic
/
```

- Non-hydrostatic: Required for dx < 10 km
- Hydrostatic: Optional for large-scale simulations

## Advection Options

```
&dynamics
 moist_adv_opt = 1,        ! Moisture advection
 scalar_adv_opt = 1,       ! Scalar advection
/
```

Options:
- 1: 2nd order upstream
- 2: 4th order
- 3: Positive-definite
- 4: Monotonic
- 5: 5th order WENO
