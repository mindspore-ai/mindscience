#!/usr/bin/env python3
"""PySCF TDDFT excited state calculation template"""
from pyscf import gto, dft, tddft

# Molecule definition (benzene)
mol = gto.M(
    atom='''
C  0.0000  0.0000  0.0000
C  1.3960  0.0000  0.0000
C  2.0940  1.2090  0.0000
C  1.3960  2.4180  0.0000
C  0.0000  2.4180  0.0000
C -0.6980  1.2090  0.0000
H -0.5430 -0.9360  0.0000
H  1.9390 -0.9360  0.0000
H  3.1770  1.2090  0.0000
H  1.9390  3.3540  0.0000
H -0.5430  3.3540  0.0000
H -1.7810  1.2090  0.0000
    ''',
    basis='def2-tzvp',
    charge=0,
    spin=1,
    verbose=4
)

# DFT calculation
mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.kernel()

# TDDFT calculation
td = tddft.TDDFT(mf)
td.nstates = 10
td.singlet = True

# Run TDDFT
e, v = td.kernel()

print("TDDFT Excitation Energies:")
for i, exc_energy in enumerate(e):
    ev = exc_energy * 27.2114
    nm = 1239.8 / ev
    f = td.oscillator_strength(gauge='length')[i]
    print(f"State {i+1}: {ev:.4f} eV, {nm:.2f} nm, f={f:.6f}")
