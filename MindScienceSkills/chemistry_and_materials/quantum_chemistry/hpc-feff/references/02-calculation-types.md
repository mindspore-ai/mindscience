# FEFF Calculation Types

## XANES Calculation

```
EDGE K
TARGET 1
RMAX 6.0
SCF 7.0
```

Output files:
- `xmu.dat`: Absorption coefficient
- `chi.dat`: Normalized spectrum

## EXAFS Calculation

```
EDGE K
TARGET 1
RMAX 8.0
NLEG 4
DEBYE 300 315
```

Output files:
- `paths.dat`: Scattering paths
- `feffNNNN.dat`: Path amplitude/phase

## DOS Calculation

```
LDOS -10 20 0.1
```

Parameter description:
- Energy range (eV)
- Energy step (eV)

Output files:
- `ldos.dat`: Local density of states

## Electronic Structure

```
RHOSEP 0.0
```

Output:
- Charge density
- Charge transfer

## Multiple Scattering

```
NLEG 6
```

- NLEG: Maximum number of scatterings
- Default: 6
