# PSI4 Advanced Features

## Symmetry

```python
# Use symmetry
mol = psi4.geometry('''
0 1
O
H 1 0.96
H 1 0.96 2 104.5
symmetry d2h
''')

# Disable symmetry
mol = psi4.geometry('''
0 1
O  0.0  0.0  0.0
H  0.0  0.0  0.96
H  0.0  0.96  0.0
symmetry c1
''')
```

## Solvation

```python
# PCM solvation model
psi4.set_options({
    'pcm': True,
    'pcm_scf_type': 'total'
})

# Set solvent
psi4.pcm_helper('''
Medium {
  Solvent = Water
}
Cavity {
  Type = Geometric
  Scaling = False
  Area = 0.3
}
''')
```

## Open-Shell Calculations

```python
# UHF
energy = psi4.energy('uhf')

# UKS
energy = psi4.energy('uks-B3LYP')

# Set spin
mol = psi4.geometry('''
0 2  # Doublet radical
O  0.0  0.0  0.0
''')
```

## Multi-Reference Methods

```python
# CASSCF
psi4.set_options({
    'basis': 'cc-pvdz',
    'scf_type': 'df',
    'reference': 'rohf'
})

# Set active space
psi4.set_options({
    'casscf_nact_orb': 6,
    'casscf_nact_el': 6
})

energy = psi4.energy('casscf')
```

## Parallel Computing

```bash
# MPI parallel
psi4 -n 8 input.py

# Set number of threads
export OMP_NUM_THREADS=8
```

## Output Control

```python
# Set output verbosity
psi4.set_options({
    'print': 2  # 0-5, higher is more verbose
})

# Output to file
psi4.core.set_output_file('output.dat', False)
```