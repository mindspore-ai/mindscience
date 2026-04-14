#!/bin/bash
# XTB single point energy calculation script

# Set parallel threads
export OMP_NUM_THREADS=4

# Input file
INPUT="molecule.xyz"

# Run single point energy calculation
xtb $INPUT --sp --chrg 0 --uhf 0 -P 4

echo "Calculation completed"
