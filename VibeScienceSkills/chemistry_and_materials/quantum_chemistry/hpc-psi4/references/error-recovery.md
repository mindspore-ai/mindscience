# PSI4 Error Recovery

## SCF Convergence Failure

### Symptoms
```
SCF failed to converge
```

### Solutions

1. Increase maximum iterations
```python
psi4.set_options({'maxiter': 200})
```

2. Use different initial guess
```python
psi4.set_options({'guess': 'sad'})  # or 'gwh', 'core'
```

3. Use level shift
```python
psi4.set_options({'level_shift': 0.1})
```

4. Adjust DIIS
```python
psi4.set_options({'diis': True, 'diis_max_vecs': 8})
```

## Insufficient Memory

### Symptoms
```
Memory allocation failed
```

### Solutions

1. Increase memory allocation
```python
psi4.set_memory('16 GB')
```

2. Use density fitting
```python
psi4.set_options({'scf_type': 'df'})
```

## Basis Set Error

### Symptoms
```
Basis set not found
```

### Solutions

1. Check basis set name spelling
2. Use a supported basis set

## Geometry Optimization Failure

### Solutions

1. Relax convergence criteria
```python
psi4.set_options({'g_convergence': 'gau_loose'})
```

2. Use different optimizer
```python
psi4.set_options({'optimizer': 'rf'})  # or 'nr', 'ms'
```

## CCSD Convergence Failure

### Solutions

1. Increase maximum iterations
```python
psi4.set_options({'cc_maxiter': 100})
```

2. Adjust convergence criteria
```python
psi4.set_options({'r_convergence': 1e-6})
```