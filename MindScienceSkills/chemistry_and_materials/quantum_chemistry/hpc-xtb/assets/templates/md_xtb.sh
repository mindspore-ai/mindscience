#!/bin/bash
# XTB molecular dynamics script

export OMP_NUM_THREADS=4

INPUT="molecule.xyz"

# MD simulation (1000 steps, 300K)
xtb $INPUT --md --temp 300 --time 1000 --chrg 0 -P 4

# Trajectory saved to xtbmd.xyz
echo "MD simulation completed"
