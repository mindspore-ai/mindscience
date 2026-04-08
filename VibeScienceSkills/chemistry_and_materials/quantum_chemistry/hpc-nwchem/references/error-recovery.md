# NWChem Error Recovery

## SCF Not Converging

### Solution

```
scf
 maxiter 200
 convergence energy 1e-6
 level shift 0.5
end
```

## Insufficient Memory

### Solution

1. Increase memory
```
memory total 32 gb
```

2. Use direct integrals
```
scf
 direct
end
```

## Geometry Optimization Failed

### Solution

```
driver
 maxiter 200
 trust 0.1
end
```

## Parallel Errors

### Solution

1. Check MPI environment
2. Confirm process count matches
3. Check network configuration

## Basis Set Errors

### Solution

1. Check basis set name
2. Use standard basis sets
3. Check element support