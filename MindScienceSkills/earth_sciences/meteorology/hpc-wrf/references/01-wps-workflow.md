# WPS (WRF Preprocessing System) Workflow

## Overview

WPS prepares input data for WRF model:
1. **geogrid**: Define model domain and interpolate static data
2. **ungrib**: Extract meteorological fields from GRIB files
3. **metgrid**: Horizontally interpolate meteorological data to model grid

## geogrid

### Purpose
- Define model domain(s)
- Interpolate static geographical data (terrain, land use, soil type)

### Key namelist.wps Parameters
```
&geogrid
 parent_id         = 1, 1,           ! Parent domain ID
 parent_grid_ratio = 1, 3,           ! Nesting ratio
 i_parent_start    = 1, 30,          ! Nest start i-index
 j_parent_start    = 1, 30,          ! Nest start j-index
 e_we              = 151, 151,       ! Grid points in x
 e_sn              = 151, 151,       ! Grid points in y
 dx = 12000,                        ! Grid spacing (m)
 dy = 12000,
 map_proj = 'lambert',               ! Map projection
 ref_lat = 35.0,                     ! Reference latitude
 ref_lon = 120.0,                    ! Reference longitude
/
```

### Map Projections
- `lambert`: Lambert Conformal (mid-latitudes)
- `polar`: Polar Stereographic (high latitudes)
- `mercator`: Mercator (tropics)
- `lat-lon`: Latitude-Longitude (global)

## ungrib

### Purpose
- Extract meteorological fields from GRIB data
- Convert to intermediate format

### Data Sources
| Source | Vtable | Typical Resolution |
|--------|--------|-------------------|
| GFS | Vtable.GFS | 0.25-1.0 degree |
| ERA5 | Vtable.ECMWF | 0.25 degree |
| FNL | Vtable.GFS | 1.0 degree |
| NAM | Vtable.NAM | 12 km |

### Running ungrib
```bash
# Link Vtable
ln -sf ungrib/Variable_Tables/Vtable.GFS Vtable

# Link GRIB files
./link_grib.csh /path/to/gfs.*

# Run ungrib
./ungrib.exe
```

## metgrid

### Purpose
- Horizontally interpolate meteorological data to model grid
- Create met_em* files for real.exe

### Running metgrid
```bash
./metgrid.exe
```

### Output Files
- `met_em.d01.YYYY-MM-DD_HH:00:00.nc` for each time period
