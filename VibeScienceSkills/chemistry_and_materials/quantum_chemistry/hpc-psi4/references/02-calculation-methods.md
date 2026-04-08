# PSI4 Calculation Methods

## SCF Methods

```python
# RHF
energy = psi4.energy('scf')

# UHF
psi4.set_options({'reference': 'uhf'})
energy = psi4.energy('scf')

# ROHF
psi4.set_options({'reference': 'rohf'})
energy = psi4.energy('scf')
```

## DFT Methods

```python
# B3LYP
energy = psi4.energy('b3lyp')

# PBE0
energy = psi4.energy('pbe0')

# M06-2X
energy = psi4.energy('m06-2x')

# wB97X-D
energy = psi4.energy('wb97x-d')
```

### Common Functionals

| Functional | Type | Use Case |
|------------|------|----------|
| B3LYP | Hybrid GGA | Organic molecules |
| PBE0 | Hybrid GGA | General purpose |
| M06-2X | meta-GGA | Thermochemistry |
| wB97X-D | Range-separated | Weak interactions |
| SCAN | meta-GGA | Solids |

## Post-HF Methods

### MP2
```python
energy = psi4.energy('mp2')
```

### CCSD
```python
energy = psi4.energy('ccsd')
```

### CCSD(T)
```python
energy = psi4.energy('ccsd(t)')
```

## Geometry Optimization

```python
# Optimization
opt_energy = psi4.optimize('b3lyp')

# Set convergence criteria
psi4.set_options({
    'g_convergence': 'gau_tight'
})
```

## Frequency Calculation

```python
energy, wfn = psi4.frequency('b3lyp', return_wfn=True)
```

## Excited States

```python
# TD-DFT
psi4.set_options({'tdscf_states': 10})
energy = psi4.energy('td-b3lyp')
```