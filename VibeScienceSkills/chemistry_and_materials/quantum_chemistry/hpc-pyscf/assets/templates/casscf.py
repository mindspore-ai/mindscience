#!/usr/bin/env python3
"""PySCF CASSCF multi-reference calculation template"""
from pyscf import gto, scf, mcscf

# Molecule definition (example: N2 molecule)
mol = gto.M(
    atom='''
N  0.0000  0.0000  0.0000
N  0.0000  0.0000  1.0977
    ''',
    basis='cc-pvdz',
    charge=0,
    spin=1,
    verbose=4
)

# Perform HF calculation first
mf = scf.RHF(mol)
hf_energy = mf.kernel()
print(f"HF Energy: {hf_energy:.10f} Ha")

# CASSCF calculation
ncas = 6  # Number of active space orbitals
nelecas = 6  # Number of active space electrons

mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.conv_tol = 1e-8
mc.max_cycle = 100

# Run CASSCF
casscf_energy = mc.kernel()[0]
print(f"CASSCF Energy: {casscf_energy:.10f} Ha")
print(f"Active Space: ({nelecas}e, {ncas}o)")
