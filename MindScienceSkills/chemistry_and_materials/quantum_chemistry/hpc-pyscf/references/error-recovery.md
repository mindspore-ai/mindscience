# PySCF Error Recovery

## SCF Not Converging

### Symptoms
```
SCF not converged
```

### Solutions

1. Increase DIIS space
```python
mf.diis = True
mf.diis_space = 12
```

2. Use level shift
```python
mf.level_shift = 0.1
```

3. Adjust damping
```python
mf.damp = 0.2
```

4. Use initial guess
```python
mf.init_guess = 'minao'  # or 'atom', 'huckel'
```

5. Increase maximum iterations
```python
mf.max_cycle = 200
```

## Insufficient Memory

### Solutions

1. Use density fitting
```python
mf = mf.density_fit()
```

2. Reduce memory usage
```python
mf.max_memory = 4000  # MB
```

3. Use smaller basis set

## Integral Error

### Symptoms
```
Integral error
```

### Solutions

1. Adjust integral precision
```python
mol.intor_threshold = 1e-12
```

2. Check molecular geometry

## Symmetry Error

### Solutions

Disable symmetry
```python
mol = gto.M(atom='...', symmetry=False)
```

## CCSD Not Converging

### Solutions

1. Adjust convergence criteria
```python
ccsd.conv_tol = 1e-6
ccsd.max_cycle = 100
```

2. Use DIIS
```python
ccsd.diis = True
```

3. Start from MP2
```python
ccsd.init_guess = 'mp2'
```