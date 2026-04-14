# Error Recovery

## grompp Failures

`grompp` (preprocessing) failures occur before the simulation starts.

| Error | Cause | Fix |
|-------|-------|-----|
| "File not found: *.itp" | Missing include file | Check paths in .top |
| "ERROR: 1-4 interaction not found" | Missing `[ pairs ]` in topology | Add 1-4 pairs |
| "Invalid command line argument" | Wrong .mdp key | Check GROMACS documentation |
| "System has non-zero total charge" | Unneutralized system | Add ions with `genion -neutral` |
| "Atom count does not match" | .gro/.top mismatch | Rebuild system or fix .top |
| "Unknown dimension" | .gro file malformed | Check .gro format |

### grompp Warnings

**Never ignore warnings.** Common warnings:

| Warning | Meaning | Action |
|---------|---------|--------|
| "Can not have 0 atoms in Solute" | Empty group | Fix group selection |
| "distance of 0 to ..." | Overlapping atoms | Fix structure |
| "Long bond" | Bad geometry | Return to structure building |
| "Long pressure deviation" | Box may be wrong | Check NPT equilibration |

## mdrun Failures

| Error | Cause | Fix |
|-------|-------|-----|
| "Atoms crashed" | System exploded | Return to NPT, verify structure |
| "LINCS warnings" | Constraint failure | Reduce dt, check structure |
| "shake_toler" | Bad geometry | Return to EM |
| "Segmentation fault" | Bad .tpr or memory | Verify .tpr with `gmx check` |
| "Frozen atom" | Group set to frozen incorrectly | Check .ndx or .mdp |
| " PME nodes must be >= 0" | Bad -npme setting | Remove -npme or set correctly |

## Unstable Dynamics

| Symptom | Cause | Fix |
|---------|-------|-----|
| Energy increasing rapidly | Timestep too large | Reduce dt to 0.001 |
| Temperature spiking | Bad thermostat | Check tau-t, use v-rescale |
| Pressure oscillating | Barostat issue | Increase tau-p |
| "Atoms crashed" at step 0 | Bad starting structure | Return to EM |
| Crash after N steps | Growing instability | Check trajectory before crash |

### Step-by-Step Troubleshooting

1. **Reduce timestep:** `dt = 0.001` (1 fs)
2. **Verify EM:** Check EM finished with F_max < emtol
3. **Check NVT:** Temperature stable at target
4. **Check NPT:** Pressure and density stable
5. **Inspect structure:** Visualize the frame before crash

## Topology Issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "No such moleculetype" | Missing .itp include | Add `#include` in .top |
| Wrong atom names | PDB naming mismatch | Map to GROMACS naming |
| Missing dihedrals | .top incomplete | Add parameters manually |
| "Duplicate atom indices" | .itp error | Fix .itp file |

### Common Force Field Problems

- **CHARMM36 + TIP3P:** Use CHARMM-compatible water
- **AMBER + SPC/E:** Don't mix AMBER force fields with GROMOS water
- **Water model mismatch:** Most errors from force field + water incompatibility

## Analysis Errors

| Error | Cause | Fix |
|-------|-------|-----|
| "Can not read ... without a tpr" | Wrong file order | `gmx analysis -s md.tpr -f md.xtc` |
| "Invalid argument" | Wrong group index | Use `make_ndx` to check groups |
| Trajectory shows molecules jumping | PBC not handled | Use `gmx trjconv -pbc mol` |
| Wrong RMSD values | Not fitted | Fit trajectory first |

### PBC Issues

If molecules appear to fly apart or form "planets":

```bash
# Wrap into box, center on protein
gmx trjconv -f md.xtc -s md.tpr -o md_pbc.xtc -pbc mol -center yes

# Select protein for centering, whole system for output
```

## Recovery Workflow

```
1. Job fails with error
   ↓
2. Check if grompp or mdrun failed
   ↓
3. For grompp: fix topology or .mdp
   ↓
4. For mdrun: check log for crash step
   ↓
5. Visualize frame before crash
   ↓
6. Return to appropriate stage:
   - Step 0 → Fix EM
   - After NVT → Fix NVT
   - After NPT → Fix NPT
   ↓
7. Re-run from validated stage
```
