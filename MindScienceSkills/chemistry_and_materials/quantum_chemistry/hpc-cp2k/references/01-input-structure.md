# CP2K Input File Structure

## Basic Structure

CP2K input files use a hierarchical structure, mainly containing the following sections:

```
&GLOBAL          # Global settings
&END GLOBAL

&FORCE_EVAL      # Force calculation settings
  &DFT           # DFT calculation parameters
  &END DFT
  &SUBSYS        # Subsystem (coordinates, basis sets, etc.)
  &END SUBSYS
&END FORCE_EVAL

&MOTION          # Motion related (optimization, MD)
&END MOTION
```

## GLOBAL Section

| Parameter | Description | Example |
|------|------|------|
| PROJECT | Project name | PROJECT my_calc |
| RUN_TYPE | Run type | ENERGY/GEO_OPT/MD |
| PRINT_LEVEL | Output verbosity | LOW/MEDIUM/HIGH |

## FORCE_EVAL Section

### DFT Settings

```
&DFT
  &QS
    EPS_DEFAULT 1.0E-10
  &END QS
  
  &SCF
    SCF_GUESS ATOMIC
    EPS_SCF 1.0E-6
    MAX_SCF 100
  &END SCF
  
  &MGRID
    CUTOFF 300
    REL_CUTOFF 30
  &END MGRID
  
  &XC
    &XC_FUNCTIONAL PBE
    &END XC_FUNCTIONAL
  &END XC
&END DFT
```

### SUBSYS Settings

```
&SUBSYS
  &COORD
    C  0.0  0.0  0.0
    H  0.0  0.0  1.09
  &END COORD
  
  &KIND C
    BASIS_SET DZVP-MOLOPT-SR-GTH
    POTENTIAL GTH-PBE
  &END KIND
&END SUBSYS
```

## Common RUN_TYPE

| Type | Description |
|------|------|
| ENERGY | Single point energy calculation |
| GEO_OPT | Geometry optimization |
| MD | Molecular dynamics |
| CELL_OPT | Cell optimization |
| BAND | Transition state search |