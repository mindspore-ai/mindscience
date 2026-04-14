---
name: hpc-wrf
description: WRF (Weather Research and Forecasting) model for mesoscale numerical weather prediction. Supports atmospheric simulation, weather forecasting, climate research, and regional climate modeling. Use for atmospheric science, weather prediction, and climate studies.
---

# HPC-WRF

WRF (Weather Research and Forecasting) is a state-of-the-art mesoscale numerical weather prediction system designed for both atmospheric research and operational forecasting applications.

## Scientific Applications

| Application | Use Case |
|------------|----------|
| **Weather Forecasting** | Operational NWP, convection-permitting forecasts |
| **Climate Research** | Regional climate projections, climate change studies |
| **Atmospheric Science** | Severe weather, monsoon systems, cyclones |
| **Air Quality** | Tracer transport, atmospheric chemistry |
| **Hydrology** | Precipitation forecasting, flood prediction |
| **Renewable Energy** | Wind and solar resource assessment |

## Key Concepts

### Preprocessing System (WPS)
| Component | Purpose |
|-----------|---------|
| `geogrid` | Define domain, interpolate static terrain/land use data |
| `ungrib` | Extract meteorological fields from GRIB data |
| `metgrid` | Interpolate met data to model grid |

### WRF Model Components
| Component | Purpose |
|-----------|---------|
| `real.exe` | Real-data initialization |
| `wrf.exe` | Model integration |
| `ndown.exe` | One-way nesting |
| `tc.exe` | Tropical cyclone tracker |

### Map Projections
| Projection | Use Case |
|------------|----------|
| `lambert` | Mid-latitudes (most common) |
| `polar` | High latitudes |
| `mercator` | Tropics |
| `lat-lon` | Global simulations |

## Workflow

```
1. WPS workflow → [references/01-wps-workflow.md]
2. Physics options → [references/02-physics-options.md]
3. Dynamics options → [references/03-dynamics-options.md]
4. Parallelization → [references/04-parallelization.md]
5. Error diagnosis → [references/error-recovery.md]
```

### Standard Execution Flow
```bash
# WPS
./geogrid.exe      # Domain setup
./ungrib.exe       # GRIB data extraction
./metgrid.exe      # Interpolation

# WRF
./real.exe         # Initialization
mpirun -np 128 ./wrf.exe  # Forecast
```

## WPS Workflow

See [references/01-wps-workflow.md](references/01-wps-workflow.md) for:
- geogrid domain configuration
- ungrib GRIB data sources (GFS, ERA5, FNL, NAM)
- metgrid interpolation
- Map projections

## Physics Options

See [references/02-physics-options.md](references/02-physics-options.md) for:
- Microphysics schemes (mp_physics)
- Cumulus parameterization (cu_physics)
- PBL schemes (bl_pbl_physics)
- Land surface models (sf_surface_physics)
- Radiation (ra_lw_physics, ra_sw_physics)
- Pre-configured physics suites (CONUS, tropical, marine)

## Dynamics Options

See [references/03-dynamics-options.md](references/03-dynamics-options.md) for:
- Time step and stability (6*dx criterion)
- Vertical levels configuration
- Non-hydrostatic vs hydrostatic
- Diffusion and damping options
- Advection schemes

## Parallelization

See [references/04-parallelization.md](references/04-parallelization.md) for:
- MPI distributed memory
- OpenMP shared memory
- Hybrid parallelization
- Domain decomposition
- I/O quilting optimization

## Error Recovery

See [references/error-recovery.md](references/error-recovery.md) for diagnosis of:
- CFL errors and time step instability
- Segmentation faults
- Missing met_em files
- Restart issues
- Performance bottlenecks

## Templates

Template files in [assets/templates/](assets/templates/) serve as starting points:

| Template | Purpose |
|----------|---------|
| [`namelist.wps`](assets/templates/namelist.wps) | WPS domain configuration |
| [`namelist.input`](assets/templates/namelist.input) | WRF model configuration |
| [`run_wrf.sh`](assets/templates/run_wrf.sh) | Complete WPS+WRF workflow script |
| [`wrf_slurm.sh`](assets/templates/wrf_slurm.sh) | SLURM submission for WRF |
| [`wrf_submit_pbs.sh`](assets/templates/wrf_submit_pbs.sh) | PBS submission for WRF |

## Skill Decision Map

```
User Requirements
├─ WPS Setup
│  ├─ Domain → geogrid (dx, dy, map_proj)
│  ├─ Boundary data → ungrib (GFS, ERA5, FNL)
│  └─ Interpolation → metgrid
├─ Physics Selection
│  ├─ Microphysics → mp_physics (Thompson for convection)
│  ├─ Cumulus → cu_physics (0 for dx < 4km)
│  ├─ PBL → bl_pbl_physics (YSU, MYNN)
│  └─ Radiation → ra_lw/sw_physics (RRTMG)
├─ Dynamics
│  ├─ Time step → 6*dx (km) stability criterion
│  ├─ Vertical levels → 40+ for PBL studies
│  └─ Non-hydrostatic → .true. for dx < 10km
└─ Parallelization
   ├─ MPI → mpirun -np N
   ├─ OpenMP → OMP_NUM_THREADS
   └─ I/O → quilting (nio_tasks_per_group)
```

## Guardrails

### Domain Configuration
- `e_we` and `e_sn`: Must be odd numbers for polar stereographic
- Time step: `time_step <= 6 * dx` (km) for stability
- Vertical levels: 40+ recommended for PBL representation

### Data Requirements
- Geogrid: Requires GEOGRID static data
- Ungrib: GRIB data covering entire simulation period
- Metgrid: Output must cover all domains

### Resource Planning
- Memory: ~1 GB per 100x100 grid points per variable
- Storage: ~100 MB per output timestep
- Runtime: ~1 hour per 24-hour forecast on 128 cores

## Required Output

Always report:
- Domain configuration (grid, resolution, projection)
- Physics suite and parameterization schemes
- Simulation period and output frequency
- Output files (wrfout, wrfrst, auxhist)
- Model performance metrics
