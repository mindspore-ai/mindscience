---
name: hpc-quantum-espresso-convergence-thresholds
description: SCF convergence thresholds, ionic/dynamics controls, and numerical stability settings for Quantum ESPRESSO pw.x
type: reference
---

# Quantum ESPRESSO Convergence Thresholds And Stability

## Contents

- SCF convergence controls
- Ionic and cell dynamics controls
- Numerical stability guidelines

## SCF convergence controls

The primary SCF convergence knob is `conv_thr` in `&ELECTRONS`, which sets the total energy threshold per electron iteration.

| Parameter | Section | Role |
| --- | --- | --- |
| `conv_thr` | `&ELECTRONS` | SCF total-energy convergence threshold (Ry). Typical: `1.0d-6` to `1.0d-8` |
| `electron_maxstep` | `&ELECTRONS` | Maximum SCF iterations per loop. Default: 100 |
| `mixing_beta` | `&ELECTRONS` | Mixing step size. Reduce toward `0.1` when charge sloshing occurs |
| `mixing_mode` | `&ELECTRONS` | Mixing scheme. `plain`, `TF` (Thomas-Fermi), or `local-TF`. Use `local-TF` for slab/chain systems |
| `diagonalization` | `&ELECTRONS` | `david` (default, Davidson) or `cg` (conjugate gradient) |
| `diago_full_acc` | `&ELECTRONS` | Force full accuracy on all eigenvalues. `.true.` helps near degenerate states |

For production ground-state runs, `conv_thr = 1.0d-8` or tighter is standard. For quick sanity checks, `1.0d-5` may suffice.

## Ionic and cell dynamics controls

| Parameter | Section | Role |
| --- | --- | --- |
| `ion_dynamics` | `&IONS` | `'bfgs'` (default for relax), `'sd'` (steepest descent), `'cg'`, `'verlet'` |
| `cell_dynamics` | `&CELL` | `'bfgs'`, `'prism'`, `'steepest'` for variable-cell runs |
| `nstep` | `&CONTROL` | Maximum ionic steps in relaxation or dynamics |
| `tprnfor` | `&CONTROL` | Print forces to output. `.true.` for relaxation |
| `tstress` | `&CONTROL` | Print stress tensor. `.true.` for vc-relax |
| `press` | `&CELL` | Target pressure for pressure-controlled vc-relax (kbar) |
| `w_1`, `w_2`, `w_3` | `&CELL` | Historic weights for cell optimization in bfgs (defaults usually acceptable) |

For relaxation (`calculation='relax'`):
- `ion_dynamics='bfgs'` is the default and generally preferred
- `nstep` controls maximum ionic steps (default 100)
- Force convergence is internal to `conv_thr` through the electronic minimizer

For variable-cell relaxation (`calculation='vc-relax'`):
- Both `&IONS` and `&CELL` sections are required
- Use `tstress=.true.` to track stress
- Ensure `press` is set appropriately if running isobaric vc-relax

## Numerical stability guidelines

When tightening convergence or switching relaxation mode:

1. Never tighten `conv_thr` by more than one order of magnitude at a time when debugging
2. For metallic systems, reduce `mixing_beta` before tightening `conv_thr`
3. For variable-cell runs, confirm stress is converged alongside energy
4. If SCF oscillates in relax/vc-relax, reduce `mixing_beta` in `&ELECTRONS` to `0.1`–`0.3`

Do not change both `conv_thr` and `mixing_beta` simultaneously when debugging — isolate the cause.
