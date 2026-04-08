#!/usr/bin/env python3
"""PSI4 post-HF correlation calculation template"""
import psi4

psi4.set_memory('8 GB')
psi4.core.set_output_file('output.dat', False)

# Molecule definition
mol = psi4.geometry('''
0 1
N  0.0000  0.0000  0.0000
H  0.0000  0.0000  1.0080
H  1.0080  0.0000 -0.3360
H -0.5040  0.8730 -0.3360
symmetry c1
''')

# Set calculation options
psi4.set_options({
    'basis': 'cc-pvdz',
    'scf_type': 'df',
    'mp2_type': 'df',
    'cc_type': 'df',
    'e_convergence': 1e-8,
    'd_convergence': 1e-6
})

# HF calculation
hf_energy = psi4.energy('scf')
print(f"HF Energy: {hf_energy:.10f} Ha")

# MP2 calculation
mp2_energy = psi4.energy('mp2')
print(f"MP2 Energy: {mp2_energy:.10f} Ha")

# CCSD calculation
ccsd_energy = psi4.energy('ccsd')
print(f"CCSD Energy: {ccsd_energy:.10f} Ha")

# CCSD(T) calculation
ccsd_t_energy = psi4.energy('ccsd(t)')
print(f"CCSD(T) Energy: {ccsd_t_energy:.10f} Ha")

print(f"\nEnergy Summary:")
print(f"HF:      {hf_energy:.10f} Ha")
print(f"MP2:     {mp2_energy:.10f} Ha")
print(f"CCSD:    {ccsd_energy:.10f} Ha")
print(f"CCSD(T): {ccsd_t_energy:.10f} Ha")
