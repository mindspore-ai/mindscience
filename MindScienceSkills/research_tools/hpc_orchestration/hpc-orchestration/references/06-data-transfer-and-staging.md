# Data Transfer And Staging

## Purpose

Use this reference when the workflow moves data into or out of the cluster, synchronizes run directories, or reorganizes many files for production.

## Tool selection

| Situation | Preferred tool | Why |
| --- | --- | --- |
| one small file or a few files | `scp` | simplest path |
| recursive sync with incremental updates | `rsync` | only changed files move |
| many tiny files | archive first, then `rsync` or `scp` | avoids metadata storms |
| browser-friendly managed transfer path | Globus if the site supports it | resilient for large transfers |
| interactive inspection | `sftp` | low-volume manual browsing |

Portable default: use `rsync` for repeatable synchronization and archive huge small-file trees before transfer.

## Transfer hygiene

Guardrails:

- do not use compute nodes as generic transfer relays unless the site explicitly allows it
- prefer dedicated transfer nodes when the site provides them
- avoid launching many parallel `scp` processes without evidence it helps
- checksum important inputs and outputs when corruption would be expensive

## `rsync` patterns

Push local to cluster:

```bash
rsync -avh --partial --progress ./case/ user@login:/path/to/project/case/
```

Pull cluster to local:

```bash
rsync -avh --partial --progress user@login:/path/to/project/results/ ./results/
```

High-value flags:

- `-a`: archive mode
- `-v`: visibility
- `-h`: readable sizes
- `--partial`: keep partial files for interrupted transfers
- `--delete`: only for carefully controlled mirror workflows

Do not add `--delete` to ad hoc commands casually.

## Archive before transfer

When a dataset contains many tiny files:

```bash
tar -czf run-artifacts.tgz run-artifacts/
rsync -avh run-artifacts.tgz user@login:/path/to/archive/
```

Why:

- fewer metadata operations
- simpler restart if the transfer breaks
- easier archival and checksum workflows

## Staging strategy

Use a three-zone model:

1. source of truth on durable project storage
2. transient run directory on scratch
3. optional node-local workspace for the hottest IO phase

Recommended sequence:

1. sync authoritative inputs to project storage
2. copy or sync a run-specific subset to scratch
3. use node-local storage only for temporary high-IO phases
4. sync validated outputs back to durable storage

## What to stage

Stage in:

- solver input files
- restart files needed for continuation
- referenced meshes, pseudopotentials, force fields, or tables
- small helper scripts actually used by the run

Stage out:

- validated checkpoints
- final outputs
- key logs
- post-processed summaries

Usually leave behind:

- large temporary intermediates that can be regenerated
- duplicate cached inputs already retained elsewhere

## Restart-safe syncing

When syncing a run directory that may be resumed:

- do not overwrite authoritative checkpoints blindly
- keep restart files in predictable names or a dedicated subdirectory
- separate transient logs from restart-critical data
- verify timestamps and file sizes before replacing a last-known-good checkpoint

## Verification

For expensive data movement, capture at least one of:

- file count and total size before and after
- checksum manifest
- tar archive integrity check
- application-level validation on a sampled output

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| transfer is very slow for many small files | metadata overhead | archive before transfer |
| restarted sync keeps re-copying everything | path mismatch or timestamp drift | normalize path layout and inspect `rsync` options |
| outputs vanish after cleanup | durable and transient zones were mixed | separate project storage from scratch clearly |
| checkpoint is corrupted after sync | interrupted or blind overwrite | use partial-safe sync and validate before replacement |
