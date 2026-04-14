# ParaView Remote And Parallel Visualization

## Purpose

Use this reference when the workflow needs `pvserver`, reverse connections, SSH tunnels, or distributed rendering.

## Remote model

Typical remote pattern:

1. start `pvserver` on the remote resource
2. connect from the ParaView client or `pvpython`
3. build the visualization pipeline against the remote server

For some firewall or scheduler environments, reverse connections are preferable.

## Security and network rules

High-value guardrails:

- do not expose `pvserver` directly on untrusted networks
- prefer SSH tunneling or equivalent secured transport
- keep ports and tunnel commands explicit

## Parallel processing rules

Remote parallel visualization commonly uses `mpirun -np <n> pvserver`.

`pvbatch` is the major exception because it can run as its own batch-processing server when built with MPI support.

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| client cannot connect | port, tunnel, or firewall mismatch | rebuild the connection path explicitly |
| data is huge and GUI is sluggish | wrong execution mode | move the heavy work to `pvserver` or `pvbatch` |
| parallel script uses `Connect()` under `pvbatch` | mixed execution models | switch to `pvpython` plus `pvserver`, or keep `pvbatch` self-contained |
