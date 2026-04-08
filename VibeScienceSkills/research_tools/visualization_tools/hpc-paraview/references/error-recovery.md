# ParaView Error Recovery

## Purpose

Use this reference when readers, filters, scripts, state files, or remote ParaView sessions fail.

## Recovery sequence

1. classify the failure as data-path, pipeline, scripting, rendering, or remote-connection related
2. reduce the workflow to the smallest reproducible case
3. repair one layer at a time
4. rerun the script or state on a small dataset before returning to full-scale post-processing

## Data and pipeline failures

Typical causes:

- stale file paths
- wrong reader
- incompatible filter chain

Repair:

- validate the dataset load first
- simplify the pipeline to the minimum failing chain

## Scripting failures

Typical causes:

- active-object assumptions
- mixed GUI and batch assumptions
- `pvbatch` used with remote-connect logic

Repair:

- make readers, views, and outputs explicit
- choose one execution model

## Remote failures

Typical causes:

- tunnel mismatch
- firewall or reverse-connection mismatch
- `pvserver` started outside the intended allocation

Repair:

- restate the end-to-end connection path explicitly
- reduce the problem to one known-good port and one known-good host path
