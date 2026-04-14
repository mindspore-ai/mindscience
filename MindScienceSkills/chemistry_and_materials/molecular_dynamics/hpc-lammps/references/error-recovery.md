# Error Recovery

## Lost Atoms

**The most common LAMMPS error.**

| Signature | Meaning |
|-----------|---------|
| `Lost atoms: ...` | Atoms moved beyond the simulation box |
| `Out of range atoms` | Similar — atoms outside valid domain |

### Recovery Sequence

1. **Add minimization before dynamics:**
   ```lammps
   minimize 1.0e-4 1.0e-6 1000 10000
   ```

2. **Reduce timestep:**
   ```lammps
   timestep 0.0005  # metal units, try 0.5 fs
   ```

3. **Force frequent neighbor rebuilds:**
   ```lammps
   neighbor 0.5 bin
   neigh_modify every 1 delay 0 check yes
   ```

4. **Check boundary conditions:**
   - Non-periodic boundaries need careful handling
   - `boundary p p p` is safest for unknown systems

5. **Check for initial overlaps:**
   - Visualize the starting structure
   - Reduce initial velocity if system is hot

**Repeated lost atoms = bad starting structure, not just settings.**

## Potential and Path Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `Cannot open potential file` | Wrong path or filename | Verify file exists in working directory |
| `Incorrect args for pair_coeff` | Wrong format for potential type | Match the expected format |
| `Substitution level 0` | Potential file not found | Check path and spelling |
| `Invalid weight` | EAM file element mismatch | Verify element names |

### Verification Steps

1. Confirm potential file exists:
   ```bash
   ls -la Cu_u3.eam
   ```

2. Verify atom types match potential:
   ```lammps
   # Data file: atom type 1 = Cu, 2 = Ni
   # Potential file must contain Cu and Ni
   pair_style eam
   pair_coeff 1 2 CuNi.eam
   ```

## Neighbor List Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Neighbor list overflow` | Too many neighbors per atom | Increase skin, use larger cutoff |
| `Too many neighbor bins` | Extreme density variation | Use `nsq` neighbor style |
| `Dangerous builds > 0` | Atoms moved too far between rebuilds | Increase skin or rebuild frequency |

### Neighbor Troubleshooting

```lammps
# For hot systems:
neighbor 0.5 bin
neigh_modify every 1 delay 0 check yes

# For dense systems:
neighbor 0.3 bin
neigh_modify every 1 delay 0 check yes
```

## Non-Numeric Output (NaN)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Non-numeric pressure` | Bad forces or box collapse | Reduce Pdamp, minimize |
| `Non-numeric temperature` | Zero velocities or bad integration | Set velocities properly |
| `NaN` everywhere | Division by zero or bad potential | Check pair coefficients |

### Recovery Steps

1. **Increase thermostat damping:**
   ```lammps
   fix 1 all nvt temp 300.0 300.0 500.0  # Tdamp = 500*dt
   ```

2. **Increase barostat damping:**
   ```lammps
   fix 1 all npt temp 300.0 300.0 100.0 iso 1.0 1.0 5000.0
   ```

3. **Stage the run properly:**
   - Minimize → NVT → NPT → Production

4. **Reduce timestep:**
   ```lammps
   timestep 0.001  # metal units
   ```

## Command Order Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Pair style not set` | `pair_coeff` before `pair_style` | Move `pair_style` first |
| `Bond style not set` | `bond_coeff` before `bond_style` | Add `bond_style` |
| `No fixes defined` | `run` before any `fix` | Add at least one `fix` |

## Data File Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `Invalid atom type` | Type number exceeds declared types | Fix numbering |
| `Bond atom missing` | Bond references non-existent atom | Check bonds section |
| `Incorrect # of atoms` | Count mismatch | Verify counts |
| `Atoms charge = 0.0` | Forgot to set charges | Check `set` commands |

## KSpace Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `KSpace style not set` | PPPM used without `kspace_style` | Add `kspace_style pppm` |
| `Cannot compute PPPM` | Mismatch between pair and kspace | Use matching pair style |
| `Real precision not met` | Ewald/PPPM tolerance too tight | Increase `ewald_coeff` or loosen tolerance |

## Recovery Workflow

```
1. Job fails
   ↓
2. Read error from end of log
   ↓
3. Classify: Lost atoms / Potential / Neighbor / NaN / Order
   ↓
4. For lost atoms: minimize, reduce dt, check boundaries
   ↓
5. For potential: verify file and atom type mapping
   ↓
6. For NaN: increase damping, stage run, reduce dt
   ↓
7. For command order: fix script structure
   ↓
8. Isolate with minimal test case
   ↓
9. Apply fix and re-run
```
