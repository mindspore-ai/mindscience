---
name: hpc-quantum-espresso-scf-relax-bands-convergence
description: Stage-selection map, SCF convergence controls, relaxation workflow, and bands workflow guidance for Quantum ESPRESSO
type: reference
---

# Quantum ESPRESSO SCF, Relax, Bands, And Convergence

## Contents

- Stage-selection map
- SCF convergence controls
- Relaxation workflow
- Bands workflow notes

## Stage-selection map

Use this quick map:

| Goal | Typical stage |
| --- | --- |
| ground-state density and total energy | SCF |
| optimize atomic coordinates | relax |
| optimize coordinates and cell | vc-relax |
| band structure path | bands-related workflow after SCF or NSCF setup |

## SCF convergence controls

High-value controls typically include:

- `conv_thr`
- mixing strategy
- diagonalization settings

If SCF is unstable:

1. inspect cutoff and k-points
2. inspect occupations and smearing
3. inspect structural sanity
4. only then retune electronic controls

## Relaxation workflow

Use relaxation only after:

- pseudopotentials are chosen
- cutoff and k-point strategy are reasonable
- the starting structure is physically meaningful

Do not optimize geometry on an obviously unconverged electronic setup.

## Bands workflow notes

Bands are a downstream workflow.

Practical rule:

- keep the prefix and directory consistent with the underlying SCF state
- do not treat bands input as a substitute for proper SCF groundwork
