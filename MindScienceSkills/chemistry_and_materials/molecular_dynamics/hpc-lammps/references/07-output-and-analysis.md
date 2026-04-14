# Output and Analysis

## Thermo Output

Thermodynamic output (energy, temperature, pressure) printed during a run:

```lammps
thermo 100
thermo_style multi
```

### thermo_style Options

| Style | Output | Use |
|-------|--------|-----|
| `one` | Step, CPU, E, etc. | Simple |
| `multi` | Many properties | Detailed |
| `custom` | Specific variables | Custom |

```lammps
thermo_style custom step temp pe ke etotal press vol
```

### Common Thermo Variables

| Variable | Meaning |
|----------|---------|
| `step` | Timestep number |
| `temp` | Temperature |
| `pe` | Potential energy |
| `ke` | Kinetic energy |
| `etotal` | Total energy |
| `press` | Pressure |
| `vol` | Volume |
| `density` | Mass density |
| `enthalpy` | Enthalpy |

## Dump Files

Save atomic configurations for visualization:

```lammps
dump 1 all atom 1000 dump.lammpstrj
dump_modify 1 format "%d %d %.6f %.6f %.6f"
```

### Dump Frequency

| Frequency | Total Steps | Use |
|-----------|------------|-----|
| 1000 | 10000 | 10 frames for quick check |
| 10000 | 10000 | 1 frame (final) |
| 100 | 10000 | 100 frames for analysis |

### Dump for Analysis

```lammps
dump 1 all atom 1000 dump.lammpstrj
dump_modify 1 sort id
```

Sort by ID ensures consistent atom ordering for analysis.

## Compute Commands

Compute quantities for output or further analysis:

```lammps
compute myRDF all rdf 100
fix 1 all ave/time 10 1 100 c_myRDF file rdf.txt mode vector
```

### Useful Computes

| Compute | What it Does | Output |
|---------|-------------|--------|
| `compute msd` | Mean Square Displacement | 4 values (MSD, etc.) |
| `compute rdf` | Radial Distribution Function | g(r) |
| `compute pe/atom` | Per-atom potential energy | Per atom |
| `compute stress/atom` | Per-atom stress tensor | 6 values/atom |
| `compute gyration` | Radius of gyration | scalar |
| `compute voronoi` | Voronoi analysis | Per atom volume |

## Fix ave/time

Averaging over time:

```lammps
fix 1 all ave/time 10 1 100 c_thermo_temp file temp_avg.txt
```

Format: `fix ID group ave/time N_input N_block N_running compute_ID`

## Restart Files

Save state for continuation:

```lammps
write_restart restart.equil
```

### Read Restart

```lammps
read_restart restart.equil
```

Modify settings after restart:

```lammps
pair_style lj/cut 3.0
pair_coeff * * 1.0 1.0
```

## Trajectory Analysis with Ovito

Export to PDB or LAMMPS dump for Ovito:

```bash
# In LAMMPS
dump 1 all atom 1000 dump.lammpstrj

# Or for PDB (smaller)
dump 1 all dcd 1000 trajectory.dcd
```

### Ovito Analysis

In Ovito:
- Load dump file
- Calculate Centro-symmetry (for defects)
- Calculate Common Neighbor Analysis (for crystal structure)
- Calculate Displacement vectors

## Python Analysis

Post-process with Python + MDAnalysis or PyTraj:

```python
import mdanalysis as mda
u = mda.Universe("system.data", "trajectory.dcd")

# RMSD
from MDAnalysis.analysis import rms
R = rms.RMSD(u.select_atoms("name CA"), u.select_atoms("name CA"))
R.run()
```

## Common Output Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| "Cannot open dump file" | Wrong path | Check directory exists |
| "Too many dumps" | Output too large | Reduce dump frequency |
| "Invalid fix ave/time arguments" | Wrong format | Check fix ave/time syntax |
| Trajectory shows molecules flying apart | PBC not handled | Use `unwrap` in dump |
| Wrong RMSD | Not fitted | Fit before calculating |

## Log Analysis

Extract data from the log file:

```bash
# Extract temperature and pressure
grep -E "^(Step|Temp|Press|Energy)" log.lammps
```

Or use `extract` command in Python to parse the log programmatically.
