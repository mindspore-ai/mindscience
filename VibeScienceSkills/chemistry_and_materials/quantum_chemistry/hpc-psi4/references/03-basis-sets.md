# PSI4 Basis Set Selection

## Pople Basis Sets

| Basis Set | Description |
|-----------|-------------|
| sto-3g | Minimal basis |
| 3-21g | Double-zeta |
| 6-31g | Double-zeta |
| 6-31g* | With polarization |
| 6-31g** | With polarization and diffuse |
| 6-311g** | Triple-zeta |

## Dunning Basis Sets

| Basis Set | Description |
|-----------|-------------|
| cc-pvdz | Double-zeta |
| cc-pvtz | Triple-zeta |
| cc-pvqz | Quadruple-zeta |
| aug-cc-pvdz | With diffuse |
| aug-cc-pvtz | With diffuse |

## Ahlrichs Basis Sets

| Basis Set | Description |
|-----------|-------------|
| def2-svp | Double-zeta |
| def2-tzvp | Triple-zeta |
| def2-qzvp | Quadruple-zeta |

## Usage

```python
# Single basis set
psi4.set_options({'basis': 'cc-pvdz'})

# Mixed basis sets
mol.set_basis_by_symbol('C', 'cc-pvtz')
mol.set_basis_by_symbol('H', 'cc-pvdz')
```

## Basis Set Recommendations

| System Type | Recommended Basis Set |
|-------------|----------------------|
| Quick preview | 6-31g* |
| Routine calculations | def2-tzvp |
| High accuracy | cc-pvtz |
| Anions | aug-cc-pvdz |
| Weak interactions | aug-cc-pvtz |