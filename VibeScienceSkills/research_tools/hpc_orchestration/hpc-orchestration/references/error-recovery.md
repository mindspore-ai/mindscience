## HPC Orchestration Error Pattern Dictionary

## Contents

- Scheduler detection failures
- Batch script failures
- Queue-state failures
- Log-tracking failures

## Scheduler detection failures

### Pattern ID: `HPC_SCHEDULER_UNKNOWN`
- **Likely symptom**: auto-detection returns unknown
- **Root cause**: no supported scheduler command is visible in the environment
- **First checks**:
  - inspect `sbatch`, `qsub`, and `bsub` availability
  - inspect whether the task is really on a cluster environment
- **Primary fix**: switch to explicit scheduler selection or treat the run as local

## Batch script failures

### Pattern ID: `HPC_BATCH_DIRECTIVE_MISMATCH`
- **Likely symptom**: scheduler rejects the script immediately
- **Root cause**: directives do not match the scheduler family or resource syntax
- **First checks**:
  - inspect scheduler family
  - inspect queue, time, node, task, and memory syntax
- **Primary fix**: regenerate the script for the actual scheduler rather than patching unrelated run commands

### Pattern ID: `HPC_LAUNCH_COMMAND_MISMATCH`
- **Likely symptom**: allocation succeeds but the job command fails to start correctly
- **Root cause**: native launcher and MPI launcher assumptions are mismatched
- **First checks**:
  - inspect whether `srun` or `mpirun` is expected
  - inspect environment module setup
- **Primary fix**: change launch style while keeping resource requests coherent

## Queue-state failures

### Pattern ID: `HPC_JOB_STUCK_PENDING`
- **Likely symptom**: the job remains pending for too long
- **Root cause**: impossible resource request, wrong partition, or queue backlog
- **First checks**:
  - inspect nodes, tasks, walltime, and memory
  - inspect queue or partition name
- **Primary fix**: reduce or correct the resource request before resubmitting

### Pattern ID: `HPC_HOME_OR_PROJECT_QUOTA_EXCEEDED`
- **Likely symptom**: writes fail, builds stop, or jobs terminate with no-space or quota messages
- **Root cause**: caches, outputs, or build trees were written to a small quota-controlled filesystem
- **First checks**:
  - inspect home, project, or work usage
  - inspect whether scratch should have been used
- **Primary fix**: move transient data, caches, or build trees to the correct storage tier

### Pattern ID: `HPC_MODULE_STACK_CONFLICT`
- **Likely symptom**: executable exists but fails to launch or links to the wrong runtime libraries
- **Root cause**: compiler, MPI, and application modules come from incompatible stacks
- **First checks**:
  - inspect `module list`
  - inspect the intended compiler and MPI family
- **Primary fix**: purge and rebuild or relaunch with one consistent stack

## Log-tracking failures

### Pattern ID: `HPC_LOG_NEVER_CREATED`
- **Likely symptom**: log tracker times out waiting for a file
- **Root cause**: log filename, working directory, or scheduler output path is wrong, or the job never started
- **First checks**:
  - inspect script output settings
  - inspect whether the job reached `RUNNING`
- **Primary fix**: fix output path or queue execution before retrying

### Pattern ID: `HPC_STAGE_IN_INCOMPLETE`
- **Likely symptom**: job starts but quickly fails with missing mesh, table, pseudopotential, or restart-file errors
- **Root cause**: required inputs were not staged into the actual run directory
- **First checks**:
  - inspect run-directory contents
  - inspect stage-in script or sync command
- **Primary fix**: make staging explicit and revalidate the run directory before submission

### Pattern ID: `HPC_REMOTE_TOOL_RUNNING_ON_LOGIN`
- **Likely symptom**: notebook, language server, or heavy preprocessing degrades login-node behavior or violates policy
- **Root cause**: remote development tool was treated as a harmless login-node process
- **First checks**:
  - inspect process location
  - inspect whether an allocation exists
- **Primary fix**: move the workload into an interactive or batch allocation

### Pattern ID: `HPC_PORT_FORWARD_MISMATCH`
- **Likely symptom**: service starts remotely but the local browser or client cannot connect
- **Root cause**: forwarded port, host hop, or bind address does not match the launched service
- **First checks**:
  - inspect service port and bind address
  - inspect SSH tunnel path and local port
- **Primary fix**: recreate the tunnel with one documented end-to-end mapping

### Pattern ID: `HPC_REPEAT_DIVERGENCE_LOOP`
- **Likely symptom**: the same divergence signature recurs after resubmission
- **Root cause**: solver inputs were not actually repaired between runs
- **First checks**:
  - inspect the solver-specific error dictionary
  - inspect whether patched files were written
- **Primary fix**: repair solver inputs before resubmitting, not after
