# VASP Restarts, Spin, And Wavefunction Files

## Contents

- restart files
- ISTART and ICHARG logic
- spin and magnetization setup
- practical continuation rules

## Restart files

High-value VASP artifacts include:

- `WAVECAR`
- `CHGCAR`
- `CONTCAR`
- `OUTCAR`

Treat them as stage handoff artifacts, not generic clutter.

## ISTART and ICHARG logic

The VASP wiki documents:

- `ISTART` for wavefunction restart behavior
- `ICHARG` for charge-density initialization behavior

Use them deliberately:

- fresh run -> initialize cleanly
- continuation or follow-on static stage -> reuse consistent prior state when appropriate

Do not carry restart logic blindly into a stage where structure, k-points, or intended physics changed substantially.

## Spin and magnetization setup

High-value tags:

- `ISPIN`
- `MAGMOM`

If the material is magnetic or may break spin symmetry, the initial magnetic setup can determine whether SCF converges to a physically meaningful state.

## Practical continuation rules

Use this pattern:

1. relaxation writes a final structure
2. later static or DOS stage reuses the trustworthy structure
3. restart files are reused only if the stage compatibility is clear

If a continuation behaves strangely, inspect `ISTART`, `ICHARG`, and whether the prior files really match the new stage.
