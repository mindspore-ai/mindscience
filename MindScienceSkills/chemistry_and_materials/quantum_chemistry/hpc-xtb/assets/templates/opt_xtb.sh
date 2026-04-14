#!/bin/bash
# XTB frequency calculation script

export OMP_NUM_THREADS=4

INPUT="molecule.xyz"

# Frequency calculation
xtb $INPUT --hess --chrg 0 --uhf 0 -P 4

# Frequency results saved to xtbhess.dat
echo "Frequency calculation completed"
