# PySCF Molecule Definition

## Basic Syntax

```python
from pyscf import gto

mol = gto.M(
    atom='''
    O  0.0  0.0  0.0
    H  0.0  0.0  0.96
    H  0.0  0.96  0.0
    ''',
    basis='cc-pvdz',
    charge=0,
    spin=1
)
```

## Parameter Description

| Parameter | Description | Default |
|------|------|--------|
| atom | Atomic coordinates | Required |
| basis | Basis set | Required |
| charge | Molecular charge | 0 |
| spin | Spin multiplicity (2S+1) | 1 |
| unit | Coordinate unit | Angstrom |
| symmetry | Whether to use symmetry | True |
| verbose | Output verbosity level | 3 |

## Coordinate Formats

### Inline Format
```python
atom='O 0 0 0; H 0 0 0.96; H 0 0.96 0'
```

### Multiline Format
```python
atom='''
O  0.0  0.0  0.0
H  0.0  0.0  0.96
H  0.0  0.96  0.0
'''
```

### Read from File
```python
mol = gto.M(atom='molecule.xyz', basis='cc-pvdz')
```

## Common Basis Sets

| Basis Set | Description |
|------|------|
| sto-3g | Minimal basis set |
| 6-31g | Pople double-zeta |
| 6-31g* | With polarization functions |
| 6-311g** | Triple-zeta + polarization |
| cc-pvdz | Dunning correlation-consistent |
| cc-pvtz | Triple-zeta |
| def2-svp | Ahlrichs basis set |
| def2-tzvp | Triple-zeta |
| aug-cc-pvdz | With diffuse functions |

## Spin Settings

| System | spin Value |
|------|---------|
| Closed-shell molecules | 1 |
| Diradicals | 3 |
| Singlet excited states | 1 |
| Triplets | 3 |