---
name: hpc-quantum-espresso-cutoff-mixing-matrix
description: Wavefunction and charge-density cutoff guidance, SCF mixing parameters, and first-response troubleshooting matrix for Quantum ESPRESSO
type: reference
---

# Quantum ESPRESSO Cutoff And Mixing Matrix

## Contents

- cutoff matrix
- SCF mixing matrix
- first-response troubleshooting matrix

## Cutoff matrix

| Concern | Primary variable |
| --- | --- |
| wavefunction basis cutoff | `ecutwfc` |
| charge-density cutoff | `ecutrho` |

The official troubleshooting guide explicitly notes that for some pseudopotential families, raising `ecutrho` can help SCF convergence.

If convergence is poor and the pseudo family suggests a denser charge representation, inspect `ecutrho` before overfitting electronic mixing.

## SCF mixing matrix

| Concern | Primary variable |
| --- | --- |
| SCF threshold | `conv_thr` |
| mixing aggressiveness | `mixing_beta` |
| mixing style | `mixing_mode` |
| max SCF iterations | `electron_maxstep` |

The official troubleshooting guide suggests reducing `mixing_beta` toward roughly `0.3` to `0.1` or smaller when charge sloshing or instability appears.

For slab or elongated systems, the official guide notes that `mixing_mode='local-TF'` can help.

## First-response troubleshooting matrix

| Symptom | First things to inspect |
| --- | --- |
| SCF oscillation | metallic behavior, smearing, `mixing_beta`, `mixing_mode` |
| very slow convergence | structure sanity, k-points, cutoffs, `electron_maxstep` |
| pseudo-related instability | pseudo coherence, `ecutwfc`, `ecutrho` |

Do not tune all electronic parameters at once. Change the smallest set that matches the observed failure mode.
