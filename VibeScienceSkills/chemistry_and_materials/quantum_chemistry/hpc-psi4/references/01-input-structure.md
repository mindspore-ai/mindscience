# PSI4 Input File Structure

## Python API

```python
import psi4

# Set memory
psi4.set_memory('4 GB')

# Set output file
psi4.core.set_output_file('output.dat', False)

# Molecule definition
mol = psi4.geometry('''
0 1
O  0.0  0.0  0.0
H  0.0  0.0  0.96
H  0.0  0.96  0.0
symmetry c1
''')

# Set options
psi4.set_options({
    'basis': 'cc-pvdz',
    'scf_type': 'df'
})

# Run calculation
energy = psi4.energy('scf')
```

## Molecule Definition Format

```
Charge SpinMultiplicity
Element X Y Z
Element X Y Z
...
symmetry SymmetryGroup
```

### Example

```python
mol = psi4.geometry('''
-1 1
O  0.0  0.0  0.0
H  0.0  0.0  0.96
symmetry c1
''')
```

## Common Options

| Option | Description | Default |
|--------|-------------|---------|
| basis | Basis set | Required |
| scf_type | SCF type | df |
| e_convergence | Energy convergence | 1e-6 |
| d_convergence | Density convergence | 1e-6 |
| maxiter | Maximum iterations | 50 |
| guess | Initial guess | sad |
| reference | Reference state | rhf |

## SCF Types

| Type | Description |
|------|-------------|
| pk | Exact integration |
| df | Density fitting |
| cd | Cholesky decomposition |
| direct | Direct SCF |

## Symmetry

| Symmetry | Description |
|----------|-------------|
| c1 | No symmetry |
| cs | Cs point group |
| c2v | C2v point group |
| d2h | D2h point group |