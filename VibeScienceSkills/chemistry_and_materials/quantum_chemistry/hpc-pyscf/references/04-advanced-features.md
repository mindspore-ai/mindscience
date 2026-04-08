# PySCF Advanced Features

## Solvation Models

### PCM
```python
from pyscf import gto, scf, solvent

mol = gto.M(atom='...', basis='cc-pvdz')
mf = scf.RHF(mol)
mf = solvent.PCM(mf, eps=78.4)  # Dielectric constant of water
mf.kernel()
```

### COSMO
```python
mf = solvent.COSMO(mf, eps=78.4)
```

## Periodic Systems

```python
from pyscf import gto, scf

mol = gto.M(
    atom='...',
    basis='cc-pvdz',
    a=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # Lattice vectors
    dimension=3  # 3D periodic
)

mf = scf.KRHF(mol, kpts=mol.make_kpts([2,2,2]))
mf.kernel()
```

## Relativistic Effects

### X2C
```python
from pyscf import gto, scf, x2c

mol = gto.M(atom='...', basis='cc-pvdz')
mf = x2c.RHF(mol)
mf.kernel()
```

### DKH
```python
from pyscf import gto, scf, dkh

mol = gto.M(atom='...', basis='cc-pvdz')
mf = dkh.RHF(mol)
mf.kernel()
```

## Density Fitting

```python
mf = scf.RHF(mol).density_fit()
mf.kernel()
```

## Geometry Optimization

```python
from pyscf import gto, dft
from pyscf.geomopt.geometric_solver import optimize

mol = gto.M(atom='...', basis='def2-tzvp')
mf = dft.RKS(mol)
mf.xc = 'B3LYP'

# Optimize
mol_eq = optimize(mf)
print(mol_eq.atom_coords())
```

## Parallel Computing

```python
import os
os.environ['OMP_NUM_THREADS'] = '8'

from pyscf import gto, scf
mol = gto.M(atom='...', basis='cc-pvdz')
mf = scf.RHF(mol)
mf.max_memory = 8000  # MB
mf.kernel()
```