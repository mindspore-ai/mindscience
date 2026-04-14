---
name: hpc-quantum-espresso-pwx-workflow
description: Core workflow stages, input-file structure, execution patterns, and downstream stage logic for Quantum ESPRESSO pw.x
type: reference
---

# Quantum ESPRESSO `pw.x` Workflow Manual

## Contents

- Core workflow stages
- Input-file structure
- Execution pattern
- Downstream stage logic

## Core workflow stages

Typical sequence:

1. SCF
2. optional relaxation or variable-cell relaxation
3. optional NSCF or bands-related stage
4. downstream post-processing as needed

Do not jump to bands or post-processing before the ground-state setup is trustworthy.

## Input-file structure

High-value sections:

- `&CONTROL`
- `&SYSTEM`
- `&ELECTRONS`
- optional ions or cell namelists for structural optimization
- `ATOMIC_SPECIES`
- `ATOMIC_POSITIONS`
- `K_POINTS`
- `CELL_PARAMETERS` when needed

Treat the file as a structured contract, not free-form text.

## Execution pattern

The main execution path is `pw.x` with redirected input.

Practical rules:

- keep stage-specific inputs separate
- keep pseudo directories explicit
- keep output directories and prefixes stable across related stages

## Downstream stage logic

SCF establishes the charge density and wavefunction base for later stages.

If SCF is not converged or the structure is not settled, later stages are usually wasted effort.
