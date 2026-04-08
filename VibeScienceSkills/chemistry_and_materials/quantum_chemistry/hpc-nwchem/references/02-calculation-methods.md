# NWChem Calculation Methods

## SCF Method

```
scf
 rhf
 maxiter 100
 convergence energy 1e-8
end

task scf energy
```

## DFT Method

```
dft
 xc b3lyp
 mult 1
 convergence energy 1e-8
end

task dft energy
```

### Common Functionals

| Functional | Description |
|------------|-------------|
| b3lyp | Hybrid GGA |
| pbe0 | Hybrid GGA |
| pbe | GGA |
| m06-2x | meta-GGA |

## MP2

```
scf
 rhf
end

task scf energy
task mp2 energy
```

## CCSD

```
task scf energy
task ccsd energy
```

## Geometry Optimization

```
dft
 xc b3lyp
end

task dft optimize
```

## Frequency Calculation

```
task dft frequencies
```