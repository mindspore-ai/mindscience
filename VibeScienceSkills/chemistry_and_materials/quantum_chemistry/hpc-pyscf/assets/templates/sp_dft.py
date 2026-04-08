#!/usr/bin/env python3
"""PySCF Hartree-Fock single point energy template"""
from pyscf import gto, scf

# Molecule definition
mol = gto.M(
    atom='''
C  0.0000  0.0000  0.0000
H  0.0000  0.0000  1.0890
H  1.0267  0.0000 -0.3630
H -0.5134 -0.8892 -0.3630
H -0.5134  0.8892 -0.3630
    ''',
    basis='cc-pvdz',
    charge=0,
    spin=1,
    verbose=4
)

# RHF calculation
mf = scf.RHF(mol)
mf.diis = True
mf.diis_space = 8
mf.conv_tol = 1e-10
mf.max_cycle = 100

# Run calculation
energy = mf.kernel()

print(f"HF Energy: {energy:.10f} Ha")
print(f"Number of basis functions: {mol.nao_nr()}")
