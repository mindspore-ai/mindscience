#!/usr/bin/env python3
"""PSI4 geometry optimization and frequency calculation template"""
import psi4

psi4.set_memory('8 GB')
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
    'd_convergence': 1e-6,
    'g_convergence': 'gau_tight'
})

# Geometry optimization
print("Starting geometry optimization...")
opt_energy = psi4.optimize('B3LYP')
print(f"Optimized Energy: {opt_energy:.10f} Ha")

# Frequency calculation
print("\nStarting frequency calculation...")
freq_energy, wfn = psi4.frequency('B3LYP', return_wfn=True)

# Output vibrational frequencies
vib_freqs = wfn.frequency_analysis
print(f"\nVibrational frequencies (cm-1):")
for i, freq in enumerate(vib_freqs['frequency']):
    print(f"  {i+1}: {freq:.2f}")
