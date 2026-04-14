#!/bin/bash
# XTB geometry optimization script

export OMP_NUM_THREADS=4

INPUT="molecule.xyz"

# Geometry optimization
xtb $INPUT --opt --chrg 0 --uhf 0 -P 4

# Optimized structure saved to xtbopt.xyz
echo "Optimized structure saved to xtbopt.xyz"
