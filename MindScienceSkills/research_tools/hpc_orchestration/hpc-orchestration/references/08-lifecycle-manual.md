## HPC Lifecycle Manual

## Contents

- Lifecycle phases
- Shared execution layer
- Self-healing loop
- Cross-skill coordination

## Lifecycle phases

Repository-wide HPC workflows should explicitly cover:

1. preprocessing
2. configuration generation
3. execution and deployment
4. post-processing and reporting

If a solver skill only covers input generation, pair it with `hpc-orchestration` for actual cluster execution.

## Shared execution layer

Use the orchestration scripts in this sequence when needed:

- generate a batch script
- submit the job
- monitor queue state
- track runtime logs
- patch and resubmit if the solver-specific skill identifies a known failure mode

## Self-healing loop

Use this loop:

1. submit
2. monitor
3. tail logs
4. classify failure
5. repair solver input
6. resubmit

Do not improvise repairs before checking the solver-specific error references.

## Cross-skill coordination

Use solver-specific skills for:

- input files
- physics and numerics
- domain-specific post-processing

Use `hpc-orchestration` for:

- scheduler choice
- cluster-safe execution
- queue monitoring
- log-driven control flow
