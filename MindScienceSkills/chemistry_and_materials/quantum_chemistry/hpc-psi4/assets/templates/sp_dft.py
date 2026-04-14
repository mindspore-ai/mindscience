#!/usr/bin/env python3
"""PSI4 DFT single point energy calculation template"""
import psi4

# Set memory and output
psi4.set_memory('4 GB')
psi4.core.set_output_file('output.dat', False)

# Molecule definition
mol = psi4.geometry('''
0 1
O  0.0000  0.0000  0.0000
H  0.7586  0.0000  0.5042
H  0.7586  0.0000 -0.5042
symmetry c1
''')

# Set calculation options
psi4.set_options({
    'basis': 'def2-tzvp',
    'scf_type': 'df',
    'e_convergence': 1e-8,
    'd_convergence': 1e-6
})

# Run DFT calculation
# Common functionals: B3LYP, PBE0, M06-2X, wB97X-D
energy = psi4.energy('B3LYP')

print(f"DFT Energy (B3LYP): {energy:.10f} Ha")

# Get orbital energies
wfn = psi4.core.get_variable('SCF TOTAL ENERGY')
print(f"Wavefunction energy: {wfn:.10f} Ha")
