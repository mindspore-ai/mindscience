# Error Recovery

## Diagnostic First Steps

Read the Gaussian output file (`.log`) from the **end** (where the job died).

## Input File Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Link 0 naming error` | Malformed % line | Check `%Chk=`, `%Mem=` syntax |
| `Bad command line` | Unrecognized directive | Remove unknown Link 0 |
| `Invalid route` | Wrong keyword in route | Check Gaussian documentation |
| `Charge/mult error` | Charge or multiplicity wrong | Verify electron count |

## SCF Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `SCF not converged` | SCF didn't reach self-consistency | Use `Guess=TCore`, increase cycles |
| `Lowadfadf` | Degenerate/diradical state | Use `Guess=Mix`, open-shell method |
| `Bad convergence` | Geometry far from equilibrium | Optimize geometry first |
| `Mayer indices bad` | Basis set issue | Check basis set compatibility |

### SCF Recovery Sequence

1. Try `Guess=TCore` (tighter initial guess)
2. Try `SCF=Conver=8` (tighter SCF convergence)
3. For open-shell: `Guess=Mix`
4. For diradicals: try multi-reference method

## Optimization Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `Optimization failed` | Bad starting geometry | Use better starting structure |
| `TS not found` | Poor TS guess | Use scan or different approach |
| `Too many steps` | Optimization stuck | Use `Opt=CalcFC` or better guess |
| `Backed up` | Disk space issue | Free scratch space |

### Optimization Recovery

1. Check last geometry in output
2. Restart from checkpoint with same or tighter opt settings
3. If TS: verify with frequency (one imaginary)

## Frequency Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `Freq job did not converge` | SCF issues | Fix SCF first |
| `Negative frequency` | Not a minimum | If expected (TS), this is correct |
| `No imaginary freq` | Not a TS | If searching TS, try different starting point |

## Checkpoint Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot open checkpoint` | Wrong path | Use absolute path |
| `Oldchk not found` | File missing | Check file exists |
| `Checkpoint incompatible` | Different method or basis | Use compatible checkpoint |

## Runtime Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `File size limit exceeded` | Output too large | Reduce print level |
| `Bad memory allocation` | %Mem too high for system | Reduce %Mem |
| `Segmentation fault` | Input error or bug | Simplify input |
| `Linda not supported` | Cluster doesn't have Linda | Use shared memory only |

## Scratch Issues

If scratch runs out:

```bash
export GAUSS_SCRDIR="/scratch/${USER}/gaussian_${SLURM_JOB_ID}"
mkdir -p "$GAUSS_SCRDIR"
```

Set explicitly in input:

```gjf
%RWF=/scratch/${USER}/scratch/rwf
```

## Geometry Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Z-matrix invalid index` | Reference to non-existent atom | Check atom numbers |
| `Blank needed` | Missing blank line in input | Add blank lines between sections |
| `Unknown element` | Unrecognized atom symbol | Check spelling |

## Method Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Basis not found` | Wrong basis set name | Check Gaussian basis names |
| `Method not supported` | Incompatible method/basis | Change method or basis |
| `Polarizability failed` | Method doesn't support it | Use appropriate method |

## Recovery Workflow

```
1. Job fails
   ↓
2. Read error from end of output
   ↓
3. Classify: Input / SCF / Optimization / Runtime / Checkpoint
   ↓
4. For input: validate format, simplify route
   ↓
5. For SCF: try different guess or method
   ↓
6. For optimization: check last geometry, restart
   ↓
7. For runtime: check scratch, memory, paths
   ↓
8. Test with smaller system before resubmitting
```
