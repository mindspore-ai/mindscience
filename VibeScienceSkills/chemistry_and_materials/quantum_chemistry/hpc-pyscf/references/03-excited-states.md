# PySCF Excited State Calculations

## TDDFT

```python
from pyscf import gto, dft, tddft

mol = gto.M(atom='...', basis='def2-tzvp')
mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

# TDDFT calculation
td = tddft.TDDFT(mf)
td.nstates = 10  # Number of excited states
td.singlet = True  # Singlet excitation

e, v = td.kernel()

# Output results
for i, exc_energy in enumerate(e):
    ev = exc_energy * 27.2114
    nm = 1239.8 / ev
    f = td.oscillator_strength(gauge='length')[i]
    print(f"State {i+1}: {ev:.4f} eV, {nm:.2f} nm, f={f:.6f}")
```

## EOM-CCSD

```python
from pyscf import gto, scf, cc

mol = gto.M(atom='...', basis='cc-pvdz')
mf = scf.RHF(mol).run()
ccsd = cc.CCSD(mf)
ccsd.kernel()

# EOM-CCSD excited states
eom = ccsd.EOMCCSD()
e, v = eom.ipccsd(nroots=5)  # Ionized states
e, v = eom.eeccsd(nroots=5)  # Excited states
```

## CIS

```python
from pyscf import gto, scf, cis

mol = gto.M(atom='...', basis='cc-pvdz')
mf = scf.RHF(mol).run()

cis_calc = cis.CIS(mf)
cis_calc.nstates = 10
e, v = cis_calc.kernel()
```

## Vibrational Spectroscopy

```python
from pyscf import gto, dft, hessian

mol = gto.M(atom='...', basis='def2-tzvp')
mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

# Calculate Hessian
h = hessian.RKS(mf)
h_mat = h.kernel()

# Frequency analysis
from pyscf.hessian import thermo
freq_info = thermo.harmonic_analysis(mol, h_mat)
print(freq_info['freq_wavenumber'])
```