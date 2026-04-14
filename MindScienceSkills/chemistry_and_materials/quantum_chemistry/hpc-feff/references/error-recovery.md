# FEFF Error Recovery

## Cluster Too Small

### Symptoms
- Poor spectral convergence
- EXAFS amplitude too low

### Solution

Increase cluster radius:
```
RMAX 8.0
```

## SCF Not Converging

### Symptoms
```
SCF did not converge
```

### Solution

1. Adjust SCF parameter:
```
SCF 10.0
```

2. Adjust convergence criteria:
```
TOLERANCE 1e-6
```

## Insufficient Memory

### Solution

1. Reduce cluster size:
```
RMAX 5.0
```

2. Reduce number of paths:
```
NLEG 3
```

## Path Errors

### Symptoms
- EXAFS fitting failed
- Missing path files

### Solution

1. Check structure file
2. Adjust path filtering:
```
CRITERIA 0.0 0.0
```

## Output File Issues

### Common Output Files

| File | Description |
|------|------|
| xmu.dat | Absorption coefficient |
| chi.dat | EXAFS signal |
| paths.dat | Scattering paths |
| feffNNNN.dat | Path data |
| ldos.dat | Density of states |

### Check Output

```bash
# View absorption spectrum
head xmu.dat
```