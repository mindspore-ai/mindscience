---
name: hpc-quantum-espresso-restarts-postprocessing
description: Clean restart rules, prefix and outdir discipline, NSCF and bands handoff, and post-processing workflow for Quantum ESPRESSO
type: reference
---

# Quantum ESPRESSO Restarts And Post-Processing Manual

## Contents

- clean restart rules
- `prefix` and `outdir`
- NSCF and bands handoff
- post-processing workflow notes

## Clean restart rules

The official PW user guide states that arbitrary mid-code restart is no longer generally supported.

Practical implication:

- the code must terminate properly for restart to be reliable

Official clean-stop mechanisms include:

- reaching `max_seconds`
- creating `$prefix.EXIT`
- supported signal trapping builds

To restart after a proper stop, use `restart_mode='restart'`.

## `prefix` and `outdir`

Treat `prefix` and `outdir` as workflow identity keys.

Keep them consistent across related SCF, NSCF, relax, and bands stages when those stages depend on shared data.

If they change unintentionally, later stages may silently look for the wrong state.

## NSCF and bands handoff

The official PW guide recommends:

- SCF first
- then NSCF for DOS-like post-processing on a uniform grid
- `calculation='bands'` for band paths

For bands and NSCF stages:

- keep `prefix` and `outdir` aligned with the source SCF
- use the appropriate k-point mode for the downstream task

## Post-processing workflow notes

If the user asks for DOS or bands:

1. verify SCF is converged
2. choose NSCF or bands appropriately
3. ensure k-point mode matches the downstream quantity

Do not treat post-processing as a substitute for a poor SCF foundation.
