## Interactive Debugging And Profiling

## Purpose

Use this reference when a job must be diagnosed live, reproduced on a smaller resource shape, or profiled for performance evidence.

## Interactive allocation rules

Portable defaults:

- request an interactive allocation for live diagnosis
- use the site's debug queue when available
- keep the allocation as small and short as the problem allows
- reproduce on a reduced but representative case before using full-scale resources

Slurm pattern:

```bash
salloc --nodes=1 --ntasks=4 --cpus-per-task=2 --time=00:30:00
srun --pty bash
```

Inside the shell:

- load modules explicitly
- verify the executable path
- run a small test
- capture the exact failing command

## What belongs on a login node

Usually acceptable:

- editing files
- checking queues
- light compilation
- transferring data
- preparing scripts

Usually not acceptable:

- long solver runs
- heavy profiling
- large-memory preprocessing
- GPU workloads

If the site policy is not explicit, move the task to an allocation.

## Debugging sequence

1. reproduce the failure on the smallest representative case
2. capture the exact command, environment, and working directory
3. classify whether the failure is environment, path, resource shape, or solver logic
4. only then move to debuggers or profilers

## Low-level debugging

Useful tools by situation:

| Situation | Tool family | Notes |
| --- | --- | --- |
| crash with core dump | `gdb` or `lldb` | enable core dumps only when allowed |
| Python exception or hang | Python traceback, `faulthandler`, debugger | prefer a reduced reproduction |
| MPI hang | rank-local logs, timeout wrappers, small-rank reproduction | full debugger attachment is expensive |
| GPU kernel timeline | vendor profiler such as Nsight Systems | start with one GPU |

Portable guardrails:

- do not attach a heavyweight debugger to a full production allocation first
- keep the reproduction case as close as possible to the real failure mode
- record module list and executable path alongside the stack trace

## Profiling sequence

Profile in this order:

1. wall-time baseline
2. solver-provided timers
3. scheduler-level resource reports
4. system or vendor profiler only if needed

Why:

- many performance questions can be answered without invasive tracing
- heavyweight profilers can distort runtime or flood storage

## Scaling studies

For scaling or tuning work:

- vary one dimension at a time
- keep the problem size fixed for strong scaling
- grow the problem with resources for weak scaling
- store timing, node count, rank count, thread count, and key solver settings together

Minimum baseline table:

| case | nodes | ranks | threads | elapsed time | notes |
| --- | --- | --- | --- | --- | --- |

## Profiler families

Use the lightest tool that answers the question.

| Goal | Tool family | Typical use |
| --- | --- | --- |
| quick CPU hotspots | `perf` or site wrapper | single-node baseline |
| MPI and thread timeline | HPCToolkit, Score-P, vendor timeline tools | benchmark runs |
| GPU timeline | Nsight Systems or vendor equivalent | one-node GPU baseline |
| memory diagnostics | Valgrind-like tools or sanitizer builds | reduced test case only |

Run heavy profilers only on a trimmed case unless the performance question truly requires production scale.

## Evidence capture

For a useful tuning record, keep:

- exact resource request
- exact launch command
- timing summary
- solver timer output if available
- profiler command and output location
- one short conclusion about what changed and why

## Failure patterns

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| failure only appears in batch but not interactively | environment or path difference | capture module list and working directory in both modes |
| profiler output is enormous | profiled too large a run | reduce the case and rerun |
| MPI hang is impossible to inspect | reproduction is too large | reduce rank count and add rank-local logging |
| debug session blocks other users | used the wrong queue or oversized allocation | switch to debug or smaller interactive resources |
