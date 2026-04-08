# XTB Error Recovery

## SCF Not Converged

### Symptoms
```
SCF not converged
```

### Solutions

1. Increase iteration count
```bash
xtb molecule.xyz --sp --iterations 500
```

2. Adjust temperature parameter
```bash
xtb molecule.xyz --sp --etemp 500
```

3. Use different initial guess
```bash
xtb molecule.xyz --sp --guess sad
```

## Geometry Optimization Failed

### Symptoms
```
Optimization failed
```

### Solutions

1. Relax convergence criteria
```bash
xtb molecule.xyz --opt --etol 1e-5 --gtol 1e-3
```

2. Increase maximum steps
```bash
xtb molecule.xyz --opt --maxopt 500
```

3. Check initial structure

## Insufficient Memory

### Solutions

1. Reduce parallel processes
```bash
xtb molecule.xyz --sp -P 4
```

2. Use GFN1-xTB
```bash
xtb molecule.xyz --sp --gfn 1
```

## Solvent Error

### Symptoms
```
Solvent not found
```

### Solutions

Check solvent name spelling, supported solvents:
- water, acetone, acetonitrile
- benzene, ch2cl2, chloroform
- dmf, dmso, ethanol
- methanol, thf, toluene

## Charge/Spin Error

### Solutions

Set charge and spin correctly:
```bash
# Anion
xtb molecule.xyz --sp --chrg -1

# Diradical
xtb molecule.xyz --sp --uhf 2
```