# WRF Physics Options

## Microphysics (mp_physics)

| ID | Scheme | Description | Recommended For |
|----|--------|-------------|-----------------|
| 1 | Kessler | Warm rain | Tropical |
| 2 | Lin | Ice | Research |
| 3 | WSM3 | 3-class | Operational |
| 4 | WSM5 | 5-class | General |
| 6 | WSM6 | 6-class | High-res |
| 8 | Thompson | Aerosol-aware | Convection |
| 10 | Morrison | Double-moment | Research |

## Cumulus (cu_physics)

| ID | Scheme | Description | Resolution |
|----|--------|-------------|------------|
| 0 | None | Explicit | < 4 km |
| 1 | Kain-Fritsch | Deep convection | 4-10 km |
| 2 | Betts-Miller-Janjic | Adjustment | 10-30 km |
| 3 | Grell-Freitas | Ensemble | 3-30 km |
| 5 | Grell 3D | Ensemble | 10-30 km |

**Important**: Set cu_physics=0 for dx < 4 km (convection-resolving)

## PBL (bl_pbl_physics)

| ID | Scheme | Description |
|----|--------|-------------|
| 1 | YSU | Non-local, strong PBL |
| 2 | MYJ | Local, stable PBL |
| 5 | MYNN2 | TKE-based |
| 6 | MYNN3 | Higher-order |
| 7 | ACM2 | Asymmetric |

## Land Surface (sf_surface_physics)

| ID | Scheme | Description | Soil Layers |
|----|--------|-------------|-------------|
| 1 | 5-layer thermal | Simple | 5 |
| 2 | Noah | Multi-layer | 4 |
| 3 | RUC | Rapid update | 6 |
| 4 | Noah-MP | Multi-physics | 4 |

## Radiation

### Longwave (ra_lw_physics)
| ID | Scheme |
|----|--------|
| 1 | RRTM |
| 4 | RRTMG |

### Shortwave (ra_sw_physics)
| ID | Scheme |
|----|--------|
| 1 | Dudhia |
| 4 | RRTMG |

## Physics Suites

Pre-configured combinations:
- `CONUS`: Continental US (mp=8, cu=1, bl=1, sf=2)
- `tropical`: Tropical regions (mp=6, cu=3, bl=1, sf=2)
- `marine`: Marine stratocumulus (mp=8, cu=0, bl=5, sf=3)
