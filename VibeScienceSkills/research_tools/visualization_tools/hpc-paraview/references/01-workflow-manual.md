---
name: hpc-paraview-workflow
description: >
  Orient ParaView workflows across GUI, script, and cluster execution paths.
  Use as the primary entry point before diving into readers, filters, state files,
  remote sessions, or batch rendering.
---

# ParaView Workflow Manual

## Purpose

Use this as the primary entry point for ParaView work. It orients the workflow before choosing GUI, script, or cluster execution paths.

## Core execution modes

ParaView commonly appears in four modes:

- interactive GUI with `paraview`
- interactive scripting with `pvpython`
- batch scripting with `pvbatch`
- remote visualization with `pvserver`

Choose the lightest mode that solves the task.

## Pipeline mindset

ParaView workflows are built as pipelines:

1. open data with a reader
2. apply filters
3. configure representations and coloring
4. render or export results

Keep this separation clear. Reader, filter, and output problems should not be debugged as if they were the same issue.

## Common outputs

Typical outputs include:

- screenshots
- animations
- extracted datasets
- saved state files
- trace-generated Python scripts

## Cluster pairing

For scheduler-backed batch rendering or remote visualization, pair this skill with `hpc-orchestration`.

- ParaView-specific pipeline and scripting logic lives here
- scheduler, transfer, and remote session orchestration lives there
