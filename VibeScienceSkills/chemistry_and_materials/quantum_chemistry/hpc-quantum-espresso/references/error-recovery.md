---
name: hpc-quantum-espresso-error-recovery
description: Error recovery guide covering input parsing failures, SCF convergence failures, pseudopotential mismatches, and downstream workflow failures
type: reference
---

# Quantum ESPRESSO Error Recovery

## Contents

- input parsing failures
- SCF convergence failures
- pseudopotential and species mismatches
- downstream workflow failures

## Input parsing failures

If parsing fails:

1. inspect namelist syntax
2. inspect section order and required cards
3. inspect species labels and counts

Do not treat QE input as forgiving free-form syntax.

## SCF convergence failures

If SCF stalls or oscillates:

1. inspect occupations and smearing
2. inspect k-point density
3. inspect cutoffs
4. inspect structural reasonableness
5. then inspect electronic mixing and thresholds

## Pseudopotential and species mismatches

Typical failure class:

- species labels do not match pseudopotential files
- cutoff assumptions are incompatible with the pseudo set
- pseudo paths are wrong

Fix pseudo coherence before tuning convergence knobs.

## Downstream workflow failures

If relax, vc-relax, or bands stages fail:

1. confirm SCF setup was valid
2. confirm prefixes and directories are consistent
3. confirm the stage-specific namelists are present
