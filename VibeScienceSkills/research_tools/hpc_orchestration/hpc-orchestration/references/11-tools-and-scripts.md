# HPC Tools And Scripts

## Contents

- Multi-scheduler submitter
- Queue monitor
- Runtime log tracker
- Slurm deploy helper

## Multi-scheduler submitter

`scripts/hpc_job_submitter.py` generates and optionally submits scripts for Slurm, PBS/Torque, or LSF.

Use it when:

- scheduler family is known or can be detected
- a solver command should be wrapped in a batch script

## Queue monitor

`scripts/hpc_job_monitor.py` normalizes scheduler-specific states into a smaller monitoring vocabulary.

Use it when:

- the workflow needs to wait for `RUNNING`
- queue state changes drive downstream decisions

## Runtime log tracker

`scripts/hpc_log_tracker.py` tails logs, extracts metrics, and can kill jobs on explicit divergence signatures.

Use it when:

- log-based self-healing is desired
- repeated polling of output files would be less reliable than streaming the log

## Slurm deploy helper

`scripts/hpc_slurm_deploy.py` is a thinner Slurm-only helper.

Use it when:

- the environment is known to be Slurm-only
- a simpler wrapper is preferable to the multi-scheduler path
