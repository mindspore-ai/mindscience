---
name: hpc-paraview
description: Build, review, debug, and automate ParaView post-processing and visualization workflows. Use when working with ParaView readers, filters, color maps, screenshots, state files, pvpython or pvbatch scripts, remote pvserver sessions, or cluster-side batch visualization.
---

# HPC ParaView

Treat ParaView as a pipeline-driven post-processing and visualization stack with both GUI and scripted execution paths.

## Quick Start

| Step | Task | Reference |
|------|------|-----------|
| 1 | Understand pipeline mindset and execution modes | [01-workflow-manual](references/01-workflow-manual.md) |
| 2 | Map datasets to readers, filters, and pipeline | [02-readers-filters-and-pipeline](references/02-readers-filters-and-pipeline.md) |
| 3 | State files, animation, and data extracts | [03-state-files-animation-and-extracts](references/03-state-files-animation-and-extracts.md) |
| 4 | Tune representations, coloring, views, screenshots | [04-color-maps-layout-and-views](references/04-color-maps-layout-and-views.md) |
| 5 | pvserver, reverse connections, distributed rendering | [05-remote-and-parallel-visualization](references/05-remote-and-parallel-visualization.md) |
| 6 | Choose between GUI tracing, pvpython, pvbatch | [06-pvpython-pvbatch-and-traces](references/06-pvpython-pvbatch-and-traces.md) |
| 7 | Scheduler-backed cluster execution | [07-cluster-execution](references/07-cluster-execution.md) |
| - | Diagnose failures in readers, filters, scripts | [error-recovery](references/error-recovery.md) |

## Skill Map

```
                    PARAVIEW WORKFLOW
    ┌──────────────────────────────────────────┐
    │  Workflow-Manual (Execution Modes)    │
    │  ──────────────────────────────────    │
    │  Readers-Filters-Pipeline │ Color-Maps │
    │  ────────────────────────────│─────────── │
    │  PVPython-PVBatch-Traces │ State-Files │
    │  ────────────────────────────│─────────── │
    │  Remote-Parallel-Viz  │ Cluster-Exec │
    │  ────────────────────────────│─────────── │
    │  Error-Recovery                      │
    └──────────────────────────────────────────┘
```

## Key Decision Points

| Question | Guide | Summary |
|----------|-------|---------|
| Execution mode? | `01-workflow-manual` | GUI, pvpython, pvbatch, or pvserver |
| Reader selection? | `02-readers-filters-and-pipeline` | Match reader to actual data format, not just extension |
| Scripting approach? | `06-pvpython-pvbatch-and-traces` | pvpython for interactive, pvbatch for batch/MPI |
| Remote visualization? | `05-remote-and-parallel-visualization` | pvserver for heavy data or distributed rendering |
| State persistence? | `03-state-files-animation-and-extracts` | .pvsm for robustness, Python state for editing |
| Cluster execution? | `07-cluster-execution` | Pair with hpc-orchestration for scheduler workflows |

## Guardrails

- Do not guess reader or filter semantics from file extensions alone when metadata is available.
- Do not use `pvbatch` and remote `Connect()` logic in the same script.
- Do not save brittle state files with absolute data paths unless that is intentional.
- Do not run heavy rendering or batch extraction on login nodes.

## Outputs

Summarize:

- input datasets and readers
- key filters and representations
- chosen execution path such as GUI, `pvpython`, `pvbatch`, or `pvserver`
- expected screenshots, extracts, animations, or saved data products

## Assets

**When to include**: When the skill needs files that will be used in the final output.

**Use cases**: Templates, boilerplate code, batch scripts that get copied or modified.

| File | Use Case |
|------|----------|
| `assets/templates/pvpython_screenshot.py` | pvpython screenshot automation |
| `assets/templates/pvbatch_slice_extract.py` | Batch slice extraction with pvbatch |
| `assets/templates/paraview-pvbatch-slurm.sh` | SLURM script for pvbatch execution |
| `assets/templates/paraview-pvserver-slurm.sh` | SLURM script for pvserver mode |

## Reference Summary

All references are used in this skill:

| Document | Topic |
|----------|-------|
| [01-workflow-manual](references/01-workflow-manual.md) | Pipeline mindset and execution modes |
| [02-readers-filters-and-pipeline](references/02-readers-filters-and-pipeline.md) | Readers, filters, and pipeline objects |
| [03-state-files-animation-and-extracts](references/03-state-files-animation-and-extracts.md) | .pvsm, Python state, screenshots, data products |
| [04-color-maps-layout-and-views](references/04-color-maps-layout-and-views.md) | Representations, coloring, views, screenshots |
| [05-remote-and-parallel-visualization](references/05-remote-and-parallel-visualization.md) | pvserver, reverse connections, distributed rendering |
| [06-pvpython-pvbatch-and-traces](references/06-pvpython-pvbatch-and-traces.md) | GUI tracing, pvpython, and pvbatch |
| [07-cluster-execution](references/07-cluster-execution.md) | Scheduler-backed cluster execution |
| [error-recovery](references/error-recovery.md) | Readers, filters, scripts, remote connections |

## Error Recovery

Consult `references/error-recovery.md` for structured diagnosis of:

- data-path and pipeline failures
- scripting and execution model mismatches
- remote connection and tunnel issues
- state-file and rendering errors
