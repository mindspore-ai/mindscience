# Checkpoint Files and Restart

## Checkpoint Files

Gaussian stores wavefunction and geometry in `.Chk` files:

```gjf
%Chk=mycalc.chk
#p B3LYP/6-311G(d,p) Opt Freq

...
```

The checkpoint file is essential for:
- Continuing optimizations
- Restarting frequency calculations
- Single point calculations at optimized geometry
- Creating cube files

## Restart from Checkpoint

### Restart an Optimization

```gjf
%OldChk=mycalc.chk
%Chk=mycalc_restart.chk
#p B3LYP/6-311G(d,p) Opt

Restart from checkpoint
--link1--
%OldChk=mycalc.chk
%Chk=mycalc_restart.chk
#p B3LYP/6-311G(d,p) Freq NoUseSymm
```

Use `NoUseSymm` if symmetry was disabled in the original job.

### Restart with New Method

```gjf
%OldChk=opt.chk
%Chk=sp_ccsdt.chk
#p CCSD(T)/cc-pVTZ SP
```

This performs CCSD(T) single point at the B3LYP optimized geometry.

## Formatted Checkpoint (fChk)

For post-processing, convert checkpoint to formatted checkpoint:

```bash
formchk mycalc.chk mycalc.fchk
```

The `.fchk` file is ASCII and can be read by other programs.

## Converting to Cube Files

For visualization in VMD, Avogadro, etc.:

```bash
# From checkpoint
formchk mycalc.chk mycalc.fchk
cubeadd mycalc.fchk -pl 140    # Plot orbital 140

# Direct from checkpoint (newer Gaussian)
formchk mycalc.chk
%plot
   16 130    # HOMO-3 to LUMO+3
```

## Managing Large Checkpoint Files

Checkpoints can be large. For big molecules:

```gjf
%Chk=large_calc.chk
%Mem=32GB
%NProcShared=16
#p B3LYP/6-311G(d,p) SP
```

Or save only what you need:

```gjf
#p B3LYP/6-311G(d,p) GFInput Test
```

`GFInput` prints basis set information. `Test` prevents writing unnecessary data.

## Restart After Crash

If a job was running but the queue expired:

```gjf
%OldChk=mycalc.chk
%Chk=mycalc_restart.chk
#p B3LYP/6-311G(d,p) Opt
```

Gaussian will read the last geometry from the checkpoint and continue optimization.

## Checkpoint in Multi-Step Jobs

Use `Link1` to connect steps:

```gjf
%Chk=step1.chk
#p B3LYP/6-31G(d,p) Opt

Step 1: Optimization

0 1
...geometry...

--link1--
%OldChk=step1.chk
%Chk=step2.chk
#p B3LYP/6-31G(d,p) Freq

Step 2: Frequency

--
%OldChk=step1.chk
%Chk=step3.chk
#p CCSD(T)/cc-pVTZ SP

Step 3: High-level single point
```

## Archive and Retrieval

Save checkpoints:

```bash
# Archive important checkpoints
gzip mycalc.chk
mv mycalc.chk.gz /project/backups/

# Retrieve for restart
gunzip mycalc.chk.gz
```

## Common Checkpoint Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `File not found` | %OldChk path wrong | Verify file path |
| `Incompatible checkpoint` | Method mismatch | Use appropriate method |
| `No geometry in checkpoint` | Job didn't reach geometry | Re-run from beginning |
| `Cannot open .Chk` | Permissions | Check file permissions |

## Checkpoint File Size

| System Size | Typical .Chk Size |
|------------|-------------------|
| < 20 atoms | < 1 MB |
| 20-50 atoms | 1-10 MB |
| 50-100 atoms | 10-100 MB |
| > 100 atoms | 100 MB+ |

For very large systems, use `Tight` or `VeryTight` options sparingly.
