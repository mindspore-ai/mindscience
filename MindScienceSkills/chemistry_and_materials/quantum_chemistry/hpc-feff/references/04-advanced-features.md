# FEFF Advanced Features

## Temperature Effects

```
DEBYE temperature Debye_temperature
```

Example:
```
DEBYE 300 400
```

## Spin Polarization

```
SPIN 1
```

## Exchange Potential

```
EXCHANGE 0 0.0 0.0
```

Options:
- 0: Hedin-Lundqvist (default)
- 1: Dirac-Hara
- 2: Overhauser

## Screening Effects

```
CORRECTIONS Vr Vi
```

- Vr: Real part correction (eV)
- Vi: Imaginary part correction (eV)

## Path Filtering

```
CRITERIA 0.0 0.0
```

## Output Control

```
PRINT 1 0 0 0 0 3
```

## Parallel Computing

```bash
mpirun -np 8 feff_mpi
```
