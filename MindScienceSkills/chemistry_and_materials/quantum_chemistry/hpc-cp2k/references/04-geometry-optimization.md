# CP2K Geometry Optimization

## Basic Settings

```
&MOTION
  &GEO_OPT
    OPTIMIZER BFGS
    MAX_ITER 200
    RMS_FORCE 1.0E-4
    RMS_STEP 1.0E-3
  &END GEO_OPT
&MOTION
```

## Optimizer Selection

| Optimizer | Characteristics | Use Cases |
|--------|------|----------|
| BFGS | Quasi-Newton method | Small molecules |
| CG | Conjugate gradient | Large molecules |
| LBFGS | Limited-memory BFGS | Large systems |

## Convergence Criteria

| Parameter | Default Value | Description |
|------|--------|------|
| RMS_FORCE | 3.0E-4 | RMS of forces |
| MAX_FORCE | 4.5E-4 | Maximum force |
| RMS_STEP | 1.0E-3 | RMS of step |
| MAX_STEP | 1.0E-3 | Maximum step |

## Transition State Search

### NEB Method
```
&MOTION
  &BAND
    N_REPLICAS 7
    K_SPRING 0.1
    OPTIMIZE_BAND .TRUE.
  &END BAND
&MOTION
```

### DIMER Method
```
&MOTION
  &GEO_OPT
    OPTIMIZER BFGS
    TYPE TRANSITION_STATE
  &END GEO_OPT
&MOTION
```

## Constrained Optimization

```
&CONSTRAINT
  &FIXED_ATOMS
    LIST 1 2 3
  &END FIXED_ATOMS
&END CONSTRAINT
```