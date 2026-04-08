#!/usr/bin/env python3
"""PySCF post-HF correlation calculation template (MP2, CCSD, CCSD(T))"""
from pyscf import gto, scf, mp, cc

# Molecule definition
mol = gto.M(
    atom='''
N  0.0000  0.0000  0.0000
H  0.0000  0.0000  1.0080
H  1.0080  0.0000 -0.3360
H -0.5040  0.8730 -0.3360
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

# MP2 calculation
mp2 = mp.MP2(mf)
mp2.conv_tol = 1e-8
mp2_energy = mp2.kernel()[0]
print(f"MP2 Energy: {mp2_energy:.10f} Ha")

# CCSD calculation
ccsd = cc.CCSD(mf)
ccsd.conv_tol = 1e-8
ccsd_energy = ccsd.kernel()[0]
print(f"CCSD Energy: {ccsd_energy:.10f} Ha")

# CCSD(T) calculation
ccsd_t = ccsd.ccsd_t()
print(f"CCSD(T) Energy: {ccsd_energy + ccsd_t:.10f} Ha")
