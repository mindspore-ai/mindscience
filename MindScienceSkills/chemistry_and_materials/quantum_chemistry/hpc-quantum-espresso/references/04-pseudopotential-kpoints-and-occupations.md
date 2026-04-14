---
name: hpc-quantum-espresso-pseudo-kpoints-occupations
description: Pseudopotential coherence rules, k-point modes, occupation and smearing settings, and metallic versus insulating logic for Quantum ESPRESSO
type: reference
---

# Quantum ESPRESSO Pseudopotential, K-Point, And Occupations Manual

## Contents

- pseudopotential coherence
- k-point modes
- occupations and smearing
- metallic versus insulating logic

## Pseudopotential coherence

Treat pseudopotentials as a coordinated set, not isolated filenames.

Practical rules:

- species labels in `ATOMIC_SPECIES` must match the rest of the input
- pseudo filenames must exist under `pseudo_dir`
- cutoff expectations are tied to the pseudo family

If multiple pseudo families are mixed casually, expect convergence or consistency problems.

## K-point modes

The official `INPUT_PW` docs describe:

- `K_POINTS automatic`
- `gamma`
- `tpiba`
- `crystal`
- `*_b` band-structure path modes

Use:

- `automatic` for Monkhorst-Pack style uniform grids
- `gamma` for Gamma-only runs where appropriate
- `crystal_b` or `tpiba_b` for band-structure paths

Do not use a bands-style path as if it were a uniform SCF sampling grid.

## Occupations and smearing

The official docs distinguish:

- `occupations='fixed'` for gapped systems
- `occupations='smearing'` for metals
- tetrahedra modes for DOS-oriented post-SCF workflows with uniform grids

Smearing settings involve:

- `smearing`
- `degauss`

Do not apply metallic smearing blindly to a clearly insulating workflow unless there is a reason.

## Metallic versus insulating logic

If SCF convergence is erratic because frontier states keep swapping:

- inspect whether the system behaves metallically
- add suitable smearing and possibly extra empty bands

This is explicitly highlighted in the QE troubleshooting guidance.
