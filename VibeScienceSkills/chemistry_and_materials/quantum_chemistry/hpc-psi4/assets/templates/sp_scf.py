#!/usr/bin/env python3
"""PSI4 Hartree-Fock single point energy template"""
import psi4

# Set memory and output
psi4.set_memory('4 GB')
psi4.core.set_output_file('output.dat', False)

# Molecule definition
mol = psi4.geometry('''
0 1
C  0.0000  0.0000  0.0000
H  0.0000  0.0000  1.0890
H  1.0267  0.0000 -0.3630
H -0.5134 -0.8892 -0.3630
H -0.5134  0.8892 -0.3630
symmetry c1
''')

# Set calculation options
psi4.set_options({
    'basis': 'cc-pvdz',
    'scf_type': 'df',
    'e_convergence': 1e-10,
    'd_convergence': 1e-8,
    'maxiter': 100,
    'guess': 'sad'
})

# Run SCF calculation
energy = psi4.energy('scf')

print(f"SCF Energy: {energy:.10f} Ha")
print(f"Number of basis functions: {mol.nbasis()}")
