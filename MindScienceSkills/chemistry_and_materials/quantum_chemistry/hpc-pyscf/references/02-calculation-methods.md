# PySCF Calculation Methods

## Hartree-Fock Methods

```python
from pyscf import gto, scf

mol = gto.M(atom='...', basis='cc-pvdz')

# RHF (closed-shell)
mf = scf.RHF(mol)

# UHF (open-shell)
mf = scf.UHF(mol)

# ROHF (restricted open-shell)
mf = scf.ROHF(mol)

energy = mf.kernel()
```

## DFT Methods

```python
from pyscf import dft

# RKS (closed-shell)
mf = dft.RKS(mol)
mf.xc = 'B3LYP'

# UKS (open-shell)
mf = dft.UKS(mol)
mf.xc = 'PBE0'

energy = mf.kernel()
```

### Common Functionals

| Functional | Type | Application |
|------|------|----------|
| PBE | GGA | Solids, surfaces |
| B3LYP | Hybrid GGA | Organic molecules |
| PBE0 | Hybrid GGA | Accurate band gaps |
| M06-2X | meta-GGA | Thermochemistry |
| wB97X-D | Range-separated + dispersion | Weak interactions |
| SCAN | meta-GGA | General purpose |

## Post-HF Methods

### MP2
```python
from pyscf import mp
mf = scf.RHF(mol).run()
mp2 = mp.MP2(mf)
mp2_energy = mp2.kernel()[0]
```

### CCSD/CCSD(T)
```python
from pyscf import cc
mf = scf.RHF(mol).run()
ccsd = cc.CCSD(mf)
ccsd_energy = ccsd.kernel()[0]
ccsd_t = ccsd.ccsd_t()
```

## Multireference Methods

### CASSCF
```python
from pyscf import mcscf
mf = scf.RHF(mol).run()
mc = mcscf.CASSCF(mf, ncas=6, nelecas=6)
casscf_energy = mc.kernel()[0]
```

### CASPT2
```python
from pyscf import mrpt
caspt2 = mrpt.CASPT2(mc)
caspt2_energy = caspt2.kernel()
```