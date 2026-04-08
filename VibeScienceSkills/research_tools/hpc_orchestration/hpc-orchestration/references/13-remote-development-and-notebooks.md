## Remote Development And Notebooks

## Purpose

Use this reference when the cluster workflow needs VS Code Remote SSH, Jupyter, port forwarding, or browser-based inspection.

## Core rule

Remote development tools are still cluster workloads. They must respect the same scheduler and resource boundaries as solver jobs.

## SSH structure

Typical portable topology:

1. local machine
2. login or bastion node
3. allocated compute node when interactive resources are needed

Prefer:

- explicit SSH config entries
- `ProxyJump` or equivalent bastion rules
- one documented forwarding command per workflow

## VS Code Remote SSH

Recommended usage:

- connect first to the login node for light editing and repo navigation
- move heavy terminals, notebook kernels, or compute-bound processes into an allocation
- if the site supports compute-node attachment, route through the login node explicitly

Guardrails:

- do not run full simulations from an editor terminal on the login node
- document where language servers, caches, and large extensions will write data
- keep remote extension caches away from tiny quota-constrained home directories if needed

## Jupyter on HPC

Portable pattern:

1. obtain an interactive allocation
2. start Jupyter on the compute node with `--no-browser`
3. bind to `127.0.0.1`
4. forward the port back through the login node

Example launch on the allocated node:

```bash
jupyter lab --no-browser --ip=127.0.0.1 --port=8888
```

Forwarding pattern from the local machine:

```bash
ssh -L 8888:localhost:8888 user@login
```

If the notebook is running on a compute node behind the login node, adapt the SSH path with `ProxyJump` or a site-supported equivalent.

## Port forwarding hygiene

Recommended habits:

- keep the forwarded port local-only
- document the exact tunnel command with the notebook URL
- terminate stale notebook servers after use
- use random or user-selected ports if the default is occupied

## Browser and file workflows

Good fits:

- lightweight plotting
- inspecting post-processed results
- notebook-driven analysis in an interactive allocation

Poor fits:

- long production kernels on login nodes
- massive file browsing through notebook UIs
- treating the notebook server as a permanent daemon

## Security and access

Portable defaults:

- keep notebook tokens enabled unless the site provides another secure method
- do not expose notebook ports to all interfaces unless the site requires it
- do not publish private cluster hostnames or ports in committed files

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| notebook launches but browser cannot connect | missing or wrong SSH tunnel | recreate the port forward and confirm the port number |
| VS Code remote session is slow or unstable | heavy processes on login node or quota pressure | move compute-heavy tasks into an allocation and clean caches |
| remote port is already in use | stale server or reused port | pick a new port and stop the old process |
| notebook dies when login session closes | launched in the wrong place or without allocation awareness | run it from a controlled interactive allocation |
