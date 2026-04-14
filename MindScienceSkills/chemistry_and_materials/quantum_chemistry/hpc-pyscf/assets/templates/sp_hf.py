#!/usr/bin/env python3
"""PySCF DFT single point energy calculation template"""
from pyscf import gto, dft

# Molecule definition
mol = gto.M(
    atom='''
O  0.0000  0.0000  0.0000
H  0.7586  0.0000  0.5042
H  0.7586  0.0000 -0.5042
    ''',
    basis='def2-tzvp',
    charge=0,
    spin=1,
    verbose=4
)

# DFT calculation
mf = dft.RKS(mol)
mf.xc = 'B3LYP'
mf.conv_tol = 1e-8
mf.max_cycle = 100

# Run calculation
energy = mf.kernel()

print(f"DFT Energy ({mf.xc}): {energy:.10f} Ha")

# Output orbital energies
mo_energy = mf.mo_energy
nocc = mol.nelec[0]
print(f"HOMO energy: {mo_energy[nocc-1]:.6f} Ha")
print(f"LUMO energy: {mo_energy[nocc]:.6f} Ha")
